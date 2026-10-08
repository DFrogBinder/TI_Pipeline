#!/bin/bash
set -euo pipefail

PIPELINE_DIR="${PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
SIM_CPUS="${SIM_CPUS:-8}"
SIM_MEMORY="${SIM_MEMORY:-32G}"
SIM_TIME="${SIM_TIME:-08:00:00}"
SIM_MAX_CONCURRENT="${SIM_MAX_CONCURRENT:-50}"

MODE="${1:-preflight}"
if [ "$#" -gt 0 ]; then
    shift
fi
TARGET=""
DEPENDENCY=""
for argument in "$@"; do
    case "$argument" in
        left-hippocampus|right-m1)
            TARGET="$argument"
            ;;
        --dependency=*)
            DEPENDENCY="${argument#*=}"
            ;;
        *)
            echo "[ERROR] Unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    preflight|init|module-check|scaffold|remesh|post-remesh|fixed|all-remesh|all-fixed|status) ;;
    *)
        echo "[ERROR] Unsupported mode: $MODE" >&2
        echo "Usage: $0 {preflight|init|module-check|scaffold|remesh|post-remesh|fixed|all-remesh|all-fixed|status} [target] [--dependency=JOBID]" >&2
        exit 2
        ;;
esac
if [ -n "$DEPENDENCY" ] && ! [[ "$DEPENDENCY" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Dependency must be a numeric Slurm job ID." >&2
    exit 2
fi
if { [ "$MODE" = "remesh" ] || [ "$MODE" = "post-remesh" ] || [ "$MODE" = "fixed" ]; } && [ -z "$TARGET" ]; then
    echo "[ERROR] Mode $MODE requires left-hippocampus or right-m1." >&2
    exit 2
fi
if [ "$MODE" = "post-remesh" ] && [ -z "$DEPENDENCY" ]; then
    echo "[ERROR] post-remesh requires --dependency=<remesh-array-job-id>." >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIPELINE="$PIPELINE_DIR/pipeline.py"
MODULE_PREFLIGHT="$PIPELINE_DIR/hpc/module_preflight.slurm"
SCAFFOLD_ARRAY="$PIPELINE_DIR/hpc/scaffold_array.slurm"
SIM_ARRAY="$PIPELINE_DIR/hpc/simulation_array.slurm"
CONTROL="$PIPELINE_DIR/hpc/control.slurm"
ROI_ARRAY="$PIPELINE_DIR/hpc/optimizer_roi_array.slurm"
ROI_COLLECT="$PIPELINE_DIR/hpc/optimizer_roi_collect.slurm"
COMPAT_SHIM="$PIPELINE_DIR/hpc/activate_compat.sh"

for required in "$PIPELINE" "$MODULE_PREFLIGHT" "$SCAFFOLD_ARRAY" "$SIM_ARRAY" "$CONTROL" "$ROI_ARRAY" "$ROI_COLLECT" "$COMPAT_SHIM"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Missing pipeline file: $required" >&2
        exit 2
    fi
done

setting() {
    local expression="$1"
    "$PYTHON_BIN" -c "from settings import *; print($expression)"
}

SCAFFOLD_ROOT="$(setting 'SCAFFOLD_ROOT')"
MATLAB_MODULE="$(setting 'MATLAB_MODULE')"
PREFLIGHT_RECEIPT="$SCAFFOLD_ROOT/_simnibs326/module_preflight.json"

target_root() {
    local target="$1"
    "$PYTHON_BIN" -c 'from settings import target_settings; import sys; print(target_settings(sys.argv[1]).experiment_root)' "$target"
}

target_config() {
    local target="$1"
    local name="$2"
    printf '%s/_pipeline/configs/%s\n' "$(target_root "$target")" "$name"
}

metrics_root() {
    local target="$1"
    printf '%s/_post_processing/repeatability_optimizer_roi_metrics_simnibs326_v1\n' "$(target_root "$target")"
}

parse_job_id() {
    local raw="$1"
    local job_id="${raw%%;*}"
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Could not parse Slurm job ID from: $raw" >&2
        exit 2
    fi
    printf '%s\n' "$job_id"
}

submit_module_check() {
    local raw
    mkdir -p "$(dirname "$PREFLIGHT_RECEIPT")"
    raw="$(
        "$SBATCH_BIN" --parsable \
            --partition="$PARTITION" \
            --output="$SCAFFOLD_ROOT/_simnibs326/module-preflight-%j.out" \
            --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT,SIMNIBS326_MATLAB_MODULE=$MATLAB_MODULE" \
            "$MODULE_PREFLIGHT"
    )"
    parse_job_id "$raw"
}

submit_scaffold() {
    local after_job="${1:-}"
    local dep=()
    if [ -n "$after_job" ]; then
        dep+=("--dependency=afterok:$after_job")
    elif [ ! -f "$PREFLIGHT_RECEIPT" ]; then
        echo "[ERROR] Module preflight receipt is missing: $PREFLIGHT_RECEIPT" >&2
        exit 2
    fi
    local log_dir="$SCAFFOLD_ROOT/_simnibs326/logs/scaffold"
    mkdir -p "$log_dir"
    local raw
    raw="$(
        "$SBATCH_BIN" --parsable \
            --partition="$PARTITION" \
            --cpus-per-task="$SIM_CPUS" \
            --mem="$SIM_MEMORY" \
            --time="$SIM_TIME" \
            --array="0-9%10" \
            "${dep[@]}" \
            --output="$log_dir/slurm-%A_%a.out" \
            --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,LOG_DIR=$log_dir,SIMNIBS326_MATLAB_MODULE=$MATLAB_MODULE" \
            "$SCAFFOLD_ARRAY"
    )"
    parse_job_id "$raw"
}

submit_simulation() {
    local target="$1"
    local config_name="$2"
    local stage="$3"
    local after_job="${4:-}"
    local dep=()
    if [ -n "$after_job" ]; then
        dep+=("--dependency=afterok:$after_job")
    fi
    local root
    root="$(target_root "$target")"
    local config="$root/_pipeline/configs/$config_name"
    if [ -z "$after_job" ] && [ ! -f "$config" ]; then
        echo "[ERROR] Config is missing: $config" >&2
        exit 2
    fi
    local log_dir="$root/_simnibs326/logs/$stage"
    mkdir -p "$log_dir"
    local raw
    raw="$(
        "$SBATCH_BIN" --parsable \
            --job-name="ti326_${stage}" \
            --partition="$PARTITION" \
            --cpus-per-task="$SIM_CPUS" \
            --mem="$SIM_MEMORY" \
            --time="$SIM_TIME" \
            --array="0-399%$SIM_MAX_CONCURRENT" \
            "${dep[@]}" \
            --output="$log_dir/slurm-%A_%a.out" \
            --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,LOG_DIR=$log_dir,SIMNIBS326_MATLAB_MODULE=$MATLAB_MODULE" \
            "$SIM_ARRAY"
    )"
    parse_job_id "$raw"
}

submit_control() {
    local target="$1"
    local action="$2"
    local after_job="$3"
    local metrics_csv="${4:-}"
    local root
    root="$(target_root "$target")"
    local log_dir="$root/_simnibs326/logs/control"
    mkdir -p "$log_dir"
    local exports="ALL,PIPELINE_DIR=$PIPELINE_DIR,CONTROL_ACTION=$action,TARGET=$target"
    if [ -n "$metrics_csv" ]; then
        exports="$exports,METRICS_CSV=$metrics_csv"
    fi
    local raw
    raw="$(
        "$SBATCH_BIN" --parsable \
            --job-name="ti326_${action}" \
            --partition="$PARTITION" \
            --dependency="afterok:$after_job" \
            --output="$log_dir/${action}-%j.out" \
            --export="$exports" \
            "$CONTROL"
    )"
    parse_job_id "$raw"
}

submit_post_remesh() {
    local target="$1"
    local remesh_job="$2"
    local root
    root="$(target_root "$target")"
    local config="$root/_pipeline/configs/remesh_only.json"
    local metrics
    metrics="$(metrics_root "$target")"
    local metrics_csv="$metrics/optimizer_roi_metrics.csv"
    local log_dir="$root/_simnibs326/logs/optimizer_roi"
    mkdir -p "$log_dir" "$metrics/logs"

    local complete_job
    complete_job="$(submit_control "$target" complete-remesh "$remesh_job")"
    local roi_raw
    roi_raw="$(
        "$SBATCH_BIN" --parsable \
            --partition="$PARTITION" \
            --array="0-9%10" \
            --dependency="afterok:$complete_job" \
            --output="$log_dir/extract-%A_%a.out" \
            --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,METRICS_ROOT=$metrics" \
            "$ROI_ARRAY"
    )"
    local roi_job
    roi_job="$(parse_job_id "$roi_raw")"
    local collect_raw
    collect_raw="$(
        "$SBATCH_BIN" --parsable \
            --partition="$PARTITION" \
            --dependency="afterok:$roi_job" \
            --output="$log_dir/collect-%j.out" \
            --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,METRICS_ROOT=$metrics" \
            "$ROI_COLLECT"
    )"
    local collect_job
    collect_job="$(parse_job_id "$collect_raw")"
    local prepare_job
    prepare_job="$(submit_control "$target" prepare-fixed "$collect_job" "$metrics_csv")"
    printf '%s\t%s\t%s\t%s\n' "$complete_job" "$roi_job" "$collect_job" "$prepare_job"
}

submit_fixed_and_finalize() {
    local target="$1"
    local after_job="${2:-}"
    local fixed_job
    fixed_job="$(submit_simulation "$target" fixed_mesh_spherical_median.json fixed "$after_job")"
    local final_job
    final_job="$(submit_control "$target" complete-final "$fixed_job")"
    printf '%s\t%s\n' "$fixed_job" "$final_job"
}

print_scope() {
    "$PYTHON_BIN" "$PIPELINE" plan
    echo "  headreco MATLAB dependency: $MATLAB_MODULE"
    echo "Submission policy:"
    echo "  module check and v3 scaffold are explicit prerequisite stages"
    echo "  target arrays are chained sequentially, preserving a global 50-task cap"
    echo "  complete tasks are skipped; incomplete tasks use unlimited validation-gated requeue"
    echo "  no job is submitted by preflight or init"
}

case "$MODE" in
    preflight)
        print_scope
        "$PYTHON_BIN" "$PIPELINE" plan --check-paths --json >/dev/null
        echo "[INFO] Read-only campaign preflight passed; no files created and no jobs submitted."
        ;;
    init)
        print_scope
        "$PYTHON_BIN" "$PIPELINE" init
        echo "[INFO] Configs initialized; no jobs submitted."
        ;;
    module-check)
        print_scope
        job="$(submit_module_check)"
        echo "[INFO] Module-preflight job: $job"
        ;;
    scaffold)
        print_scope
        job="$(submit_scaffold "$DEPENDENCY")"
        echo "[INFO] Scaffold array job: $job"
        ;;
    remesh)
        print_scope
        job="$(submit_simulation "$TARGET" remesh_only.json remesh "$DEPENDENCY")"
        echo "[INFO] $TARGET remesh array job: $job"
        ;;
    post-remesh)
        print_scope
        jobs="$(submit_post_remesh "$TARGET" "$DEPENDENCY")"
        echo "[INFO] $TARGET post-remesh jobs (complete, ROI array, collect, prepare fixed): $jobs"
        ;;
    fixed)
        print_scope
        jobs="$(submit_fixed_and_finalize "$TARGET" "$DEPENDENCY")"
        echo "[INFO] $TARGET fixed/final jobs: $jobs"
        ;;
    all-remesh)
        print_scope
        left_config="$(target_config left-hippocampus remesh_only.json)"
        right_config="$(target_config right-m1 remesh_only.json)"
        if [ ! -f "$left_config" ] || [ ! -f "$right_config" ]; then
            echo "[ERROR] Run '$0 init' before all-remesh." >&2
            exit 2
        fi
        module_job="$(submit_module_check)"
        scaffold_job="$(submit_scaffold "$module_job")"
        left_remesh="$(submit_simulation left-hippocampus remesh_only.json remesh "$scaffold_job")"
        left_post="$(submit_post_remesh left-hippocampus "$left_remesh")"
        left_prepare="${left_post##*$'\t'}"
        right_remesh="$(submit_simulation right-m1 remesh_only.json remesh "$left_prepare")"
        right_post="$(submit_post_remesh right-m1 "$right_remesh")"
        echo "[INFO] Module check: $module_job"
        echo "[INFO] Scaffold array: $scaffold_job"
        echo "[INFO] Left remesh: $left_remesh"
        echo "[INFO] Left post-remesh: $left_post"
        echo "[INFO] Right remesh: $right_remesh"
        echo "[INFO] Right post-remesh: $right_post"
        echo "[INFO] Fixed arrays were not submitted; run all-fixed after both preparation jobs pass."
        ;;
    all-fixed)
        print_scope
        left_fixed="$(target_config left-hippocampus fixed_mesh_spherical_median.json)"
        right_fixed="$(target_config right-m1 fixed_mesh_spherical_median.json)"
        if [ ! -f "$left_fixed" ] || [ ! -f "$right_fixed" ]; then
            echo "[ERROR] Both prepared fixed configs are required before all-fixed." >&2
            exit 2
        fi
        left_jobs="$(submit_fixed_and_finalize left-hippocampus "$DEPENDENCY")"
        left_final="${left_jobs##*$'\t'}"
        right_jobs="$(submit_fixed_and_finalize right-m1 "$left_final")"
        echo "[INFO] Left fixed/final: $left_jobs"
        echo "[INFO] Right fixed/final: $right_jobs"
        ;;
    status)
        "$PYTHON_BIN" "$PIPELINE" status --target left-hippocampus
        "$PYTHON_BIN" "$PIPELINE" status --target right-m1
        ;;
esac
