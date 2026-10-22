#!/bin/bash
set -euo pipefail

PIPELINE_DIR="${PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PIPELINE_PARENT="$(dirname "$PIPELINE_DIR")"
export PYTHONPATH="$PIPELINE_PARENT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
SIM_CPUS="${SIM_CPUS:-8}"
SIM_MEMORY="${SIM_MEMORY:-32G}"
SIM_TIME="${SIM_TIME:-08:00:00}"
SIM_MAX_CONCURRENT="${SIM_MAX_CONCURRENT:-50}"
MAX_RETRIES="${SIMNIBS326H_MAX_RETRIES:-1}"
RENDER_PYTHON="${SIMNIBS326H_RENDER_PYTHON:-$HOME/.conda/envs/ti-post/bin/python}"

MODE="${1:-preflight}"
[ "$#" -eq 0 ] || shift
TARGET=""
DEPENDENCY=""
CONFIRM_ROOT=""
CONFIRM_ROOTS=""
MANIFEST_SHA256=""
CONFIRM_REVIEWED=0
for argument in "$@"; do
    case "$argument" in
        left-hippocampus|right-m1) TARGET="$argument" ;;
        --dependency=*) DEPENDENCY="${argument#*=}" ;;
        --confirm-root=*) CONFIRM_ROOT="${argument#*=}" ;;
        --confirm-roots=*) CONFIRM_ROOTS="${argument#*=}" ;;
        --manifest-sha256=*) MANIFEST_SHA256="${argument#*=}" ;;
        --confirm-reviewed) CONFIRM_REVIEWED=1 ;;
        *) echo "[ERROR] Unknown argument: $argument" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    preflight|init|module-check|stage-references|headmodels|promote|remesh|post-remesh|fixed|all-remesh|all-fixed|analysis|analysis-status|status) ;;
    *)
        echo "[ERROR] Unsupported mode: $MODE" >&2
        echo "Usage: $0 {preflight|init|module-check|stage-references|headmodels|promote|remesh|post-remesh|fixed|all-remesh|all-fixed|analysis|analysis-status|status} [target] [options]" >&2
        exit 2
        ;;
esac
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid retry cap." >&2; exit 2; }
[ -z "$DEPENDENCY" ] || [[ "$DEPENDENCY" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] Dependency must be a numeric Slurm job ID." >&2; exit 2;
}
if { [ "$MODE" = remesh ] || [ "$MODE" = post-remesh ] || [ "$MODE" = fixed ]; } && [ -z "$TARGET" ]; then
    echo "[ERROR] Mode $MODE requires a target." >&2; exit 2
fi
if [ "$MODE" = post-remesh ] && [ -z "$DEPENDENCY" ]; then
    echo "[ERROR] post-remesh requires --dependency=JOBID." >&2; exit 2
fi

setting_path() {
    local env_name="$1"
    local expression="$2"
    local override="${!env_name:-}"
    if [ -n "$override" ]; then printf '%s\n' "$override"; return; fi
    "$PYTHON_BIN" -c "from simnibs_326_corrected_label_repeatability.settings import campaign_paths_from_environment; print(campaign_paths_from_environment().$expression)"
}

HEADMODEL_ROOT="$(setting_path SIMNIBS326H_HEADMODEL_ROOT headmodel_root)"
LEFT_ROOT="$(setting_path SIMNIBS326H_LEFT_ROOT left_root)"
RIGHT_ROOT="$(setting_path SIMNIBS326H_RIGHT_ROOT right_root)"
PREFLIGHT_RECEIPT="$HEADMODEL_ROOT/_pipeline/module_preflight.json"
REFERENCE_RECEIPT="$HEADMODEL_ROOT/_pipeline/reference_stage_collection.json"
PROMOTION_RECEIPT="$HEADMODEL_ROOT/_pipeline/headmodels/promoted.json"
ANALYSIS_ROOT="$HEADMODEL_ROOT/_pipeline/analysis/repeatability_optimizer_roi_paired_v2"

MODULE_PREFLIGHT="$PIPELINE_DIR/hpc/module_preflight.slurm"
REFERENCE_ARRAY="$PIPELINE_DIR/hpc/reference_stage_array.slurm"
HEADMODEL_ARRAY="$PIPELINE_DIR/hpc/headmodel_array.slurm"
HEADMODEL_COLLECT="$PIPELINE_DIR/hpc/headmodel_collect.slurm"
SIM_ARRAY="$PIPELINE_DIR/hpc/simulation_array.slurm"
CONTROL="$PIPELINE_DIR/hpc/control.slurm"
ROI_ARRAY="$PIPELINE_DIR/hpc/optimizer_roi_array.slurm"
ROI_COLLECT="$PIPELINE_DIR/hpc/optimizer_roi_collect.slurm"
ROI_ANALYSIS="$PIPELINE_DIR/hpc/analyze_optimizer_roi.slurm"
for required in "$PIPELINE_DIR/pipeline.py" "$MODULE_PREFLIGHT" "$REFERENCE_ARRAY" \
    "$HEADMODEL_ARRAY" "$HEADMODEL_COLLECT" "$SIM_ARRAY" "$CONTROL" \
    "$ROI_ARRAY" "$ROI_COLLECT" "$ROI_ANALYSIS" "$PIPELINE_DIR/hpc/activate_compat.sh"
do
    [ -f "$required" ] || { echo "[ERROR] Missing workflow file: $required" >&2; exit 2; }
done

target_root() {
    case "$1" in
        left-hippocampus) printf '%s\n' "$LEFT_ROOT" ;;
        right-m1) printf '%s\n' "$RIGHT_ROOT" ;;
        *) echo "[ERROR] Unsupported target: $1" >&2; return 2 ;;
    esac
}
metrics_root() {
    printf '%s/_post_processing/repeatability_optimizer_roi_metrics_simnibs326h_v2\n' "$(target_root "$1")"
}
paired_metrics_root() {
    printf '%s/_post_processing/repeatability_optimizer_roi_metrics_simnibs326h_paired_v2\n' "$(target_root "$1")"
}
parse_job_id() {
    local job_id="${1%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid sbatch response: $1" >&2; exit 2; }
    printf '%s\n' "$job_id"
}
submit_job() {
    parse_job_id "$("$SBATCH_BIN" --parsable "$@")"
}
require_file() {
    [ -s "$1" ] || { echo "[ERROR] Required receipt/config is missing: $1" >&2; exit 2; }
}

print_scope() {
    echo "Scope:"
    echo "  study: corrected-label SimNIBS 3.2.6 balanced-10 repeatability"
    echo "  subjects: 10"
    echo "  targets: 2 (left hippocampus, right M1)"
    echo "  conditions: 2 (remesh, fixed_mesh)"
    echo "  repeats per condition: 40"
    echo "  reference-stage tasks: 10 (0-9%10)"
    echo "  head-model tasks: 10 (0-9%10)"
    echo "  remesh tasks: 800 (2 x 0-399%50)"
    echo "  fixed-mesh tasks: 800 (2 x 0-399%50)"
    echo "  total simulation tasks: 1600"
    echo "  component FEM solves: 3200"
    echo "  resources: sheffield; 8 CPU; 32G; 08:00:00; max 50 simulations"
    echo "  module: SimNIBS/3.2.6-foss-2023a"
    echo "  retries: $MAX_RETRIES; exact array-element requeue; append-only Slurm logs"
    echo "  MATLAB/CAT12 tasks: 0"
    echo "  submission default: none (read-only preflight)"
}

submit_module_check() {
    mkdir -p "$(dirname "$PREFLIGHT_RECEIPT")"
    submit_job --partition="$PARTITION" \
        --output="$HEADMODEL_ROOT/_pipeline/module-preflight-%j.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT" \
        "$MODULE_PREFLIGHT"
}
submit_reference_staging() {
    local log_dir="$HEADMODEL_ROOT/_pipeline/logs/reference_stage"
    mkdir -p "$log_dir"
    local array_job
    array_job="$(submit_job --partition="$PARTITION" --array="0-9%10" \
        --output="$log_dir/slurm-%A_%a.out" --open-mode=append \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,LOG_DIR=$log_dir,SIMNIBS326H_MAX_RETRIES=$MAX_RETRIES" \
        "$REFERENCE_ARRAY")"
    local collector
    collector="$(submit_job --partition="$PARTITION" --dependency="afterok:$array_job" \
        --output="$log_dir/collect-%j.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,CONTROL_ACTION=collect-references,TARGET=none" \
        "$CONTROL")"
    printf '%s\t%s\n' "$array_job" "$collector"
}
submit_headmodels() {
    require_file "$PREFLIGHT_RECEIPT"
    require_file "$REFERENCE_RECEIPT"
    "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability validate-module-receipt \
        --receipt "$PREFLIGHT_RECEIPT" >/dev/null
    "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status")=="complete" and p.get("subject_count")==10' \
        "$REFERENCE_RECEIPT"
    [ -x "$RENDER_PYTHON" ] || { echo "[ERROR] Renderer is not executable: $RENDER_PYTHON" >&2; exit 2; }
    local log_dir="$HEADMODEL_ROOT/_pipeline/logs/headmodels"
    mkdir -p "$log_dir"
    local array_job
    array_job="$(submit_job --partition="$PARTITION" --cpus-per-task=8 --mem=32G \
        --time=08:00:00 --array="0-9%10" --output="$log_dir/slurm-%A_%a.out" \
        --open-mode=append \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,LOG_DIR=$log_dir,PREFLIGHT_RECEIPT=$PREFLIGHT_RECEIPT,SIMNIBS326H_MAX_RETRIES=$MAX_RETRIES" \
        "$HEADMODEL_ARRAY")"
    local collector
    collector="$(submit_job --partition="$PARTITION" --dependency="afterok:$array_job" \
        --output="$log_dir/collect-%j.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,SIMNIBS326H_RENDER_PYTHON=$RENDER_PYTHON" \
        "$HEADMODEL_COLLECT")"
    printf '%s\t%s\n' "$array_job" "$collector"
}
submit_simulation() {
    local target="$1" config_name="$2" stage="$3" after_job="${4:-}"
    require_file "$PROMOTION_RECEIPT"
    local root config log_dir
    root="$(target_root "$target")"
    config="$root/_pipeline/configs/$config_name"
    require_file "$config"
    log_dir="$root/_simnibs326/logs/$stage"
    mkdir -p "$log_dir"
    local dependency_args=()
    [ -z "$after_job" ] || dependency_args+=("--dependency=afterok:$after_job")
    submit_job --job-name="ti326h_${stage}" --partition="$PARTITION" \
        --cpus-per-task="$SIM_CPUS" --mem="$SIM_MEMORY" --time="$SIM_TIME" \
        --array="0-399%$SIM_MAX_CONCURRENT" "${dependency_args[@]}" \
        --output="$log_dir/slurm-%A_%a.out" --open-mode=append \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,LOG_DIR=$log_dir,SIMNIBS326H_MAX_RETRIES=$MAX_RETRIES" \
        "$SIM_ARRAY"
}
submit_control() {
    local target="$1" action="$2" after_job="$3" metrics_csv="${4:-}"
    local root log_dir exports
    root="$(target_root "$target")"
    log_dir="$root/_simnibs326/logs/control"
    mkdir -p "$log_dir"
    exports="ALL,PIPELINE_DIR=$PIPELINE_DIR,CONTROL_ACTION=$action,TARGET=$target"
    [ -z "$metrics_csv" ] || exports="$exports,METRICS_CSV=$metrics_csv"
    submit_job --job-name="ti326h_${action}" --partition="$PARTITION" \
        --dependency="afterok:$after_job" --output="$log_dir/${action}-%j.out" \
        --export="$exports" "$CONTROL"
}
submit_post_remesh() {
    local target="$1" remesh_job="$2" root config metrics metrics_csv log_dir
    root="$(target_root "$target")"
    config="$root/_pipeline/configs/remesh_only.json"
    metrics="$(metrics_root "$target")"
    metrics_csv="$metrics/optimizer_roi_metrics.csv"
    log_dir="$root/_simnibs326/logs/optimizer_roi"
    mkdir -p "$log_dir" "$metrics/logs"
    local complete roi collect prepare
    complete="$(submit_control "$target" complete-remesh "$remesh_job")"
    roi="$(submit_job --partition="$PARTITION" --array="0-9%10" \
        --dependency="afterok:$complete" --output="$log_dir/extract-%A_%a.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,METRICS_ROOT=$metrics" "$ROI_ARRAY")"
    collect="$(submit_job --partition="$PARTITION" --dependency="afterok:$roi" \
        --output="$log_dir/collect-%j.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,METRICS_ROOT=$metrics" "$ROI_COLLECT")"
    prepare="$(submit_control "$target" prepare-fixed "$collect" "$metrics_csv")"
    printf '%s\t%s\t%s\t%s\n' "$complete" "$roi" "$collect" "$prepare"
}
submit_fixed_chain() {
    local target="$1" after_job="${2:-}" fixed final
    fixed="$(submit_simulation "$target" fixed_mesh_spherical_median.json fixed "$after_job")"
    final="$(submit_control "$target" complete-final "$fixed")"
    printf '%s\t%s\n' "$fixed" "$final"
}
submit_paired_roi() {
    local target="$1" root config metrics log_dir roi collect
    root="$(target_root "$target")"
    config="$root/_pipeline/configs/paired_analysis.json"
    metrics="$(paired_metrics_root "$target")"
    log_dir="$root/_simnibs326/logs/paired_optimizer_roi"
    require_file "$config"
    mkdir -p "$metrics" "$log_dir"
    roi="$(submit_job --partition="$PARTITION" --array="0-9%10" \
        --output="$log_dir/extract-%A_%a.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,METRICS_ROOT=$metrics" "$ROI_ARRAY")"
    collect="$(submit_job --partition="$PARTITION" --dependency="afterok:$roi" \
        --output="$log_dir/collect-%j.out" \
        --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,METRICS_ROOT=$metrics" "$ROI_COLLECT")"
    printf '%s\t%s\n' "$roi" "$collect"
}

print_scope
case "$MODE" in
    preflight)
        "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability preflight >/dev/null
        echo "[INFO] Read-only preflight passed; no job submitted."
        ;;
    status)
        "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability status
        ;;
    init)
        expected="$LEFT_ROOT,$RIGHT_ROOT"
        [ "$CONFIRM_ROOTS" = "$expected" ] || {
            echo "[ERROR] init requires --confirm-roots=$expected" >&2; exit 2;
        }
        "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability init
        ;;
    module-check)
        job="$(submit_module_check)"; echo "[INFO] Module-check job: $job"
        ;;
    stage-references)
        jobs="$(submit_reference_staging)"; echo "[INFO] Reference array/collector: $jobs"
        ;;
    headmodels)
        jobs="$(submit_headmodels)"; echo "[INFO] Head-model array/collector: $jobs"
        ;;
    promote)
        [ "$CONFIRM_ROOT" = "$HEADMODEL_ROOT" ] || {
            echo "[ERROR] promote requires --confirm-root=$HEADMODEL_ROOT" >&2; exit 2;
        }
        [ "$CONFIRM_REVIEWED" -eq 1 ] || { echo "[ERROR] Missing --confirm-reviewed." >&2; exit 2; }
        [[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "[ERROR] Invalid manifest hash." >&2; exit 2; }
        "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability promote-headmodels \
            --manifest-sha256 "$MANIFEST_SHA256" --confirm-reviewed
        ;;
    remesh)
        job="$(submit_simulation "$TARGET" remesh_only.json remesh "$DEPENDENCY")"
        echo "[INFO] Remesh array: $job"
        ;;
    post-remesh)
        jobs="$(submit_post_remesh "$TARGET" "$DEPENDENCY")"
        echo "[INFO] Post-remesh chain: $jobs"
        ;;
    fixed)
        jobs="$(submit_fixed_chain "$TARGET" "$DEPENDENCY")"
        echo "[INFO] Fixed/final chain: $jobs"
        ;;
    all-remesh)
        left_sim="$(submit_simulation left-hippocampus remesh_only.json remesh)"
        left_chain="$(submit_post_remesh left-hippocampus "$left_sim")"
        left_terminal="${left_chain##*$'\t'}"
        right_sim="$(submit_simulation right-m1 remesh_only.json remesh "$left_terminal")"
        right_chain="$(submit_post_remesh right-m1 "$right_sim")"
        echo "[INFO] Sequential remesh chains: left=$left_sim/$left_chain right=$right_sim/$right_chain"
        ;;
    all-fixed)
        left_chain="$(submit_fixed_chain left-hippocampus)"
        left_terminal="${left_chain##*$'\t'}"
        right_chain="$(submit_fixed_chain right-m1 "$left_terminal")"
        echo "[INFO] Sequential fixed chains: left=$left_chain right=$right_chain"
        ;;
    analysis)
        "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability validate-final \
            --target left-hippocampus >/dev/null
        "$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability validate-final \
            --target right-m1 >/dev/null
        left_jobs="$(submit_paired_roi left-hippocampus)"
        right_jobs="$(submit_paired_roi right-m1)"
        left_collect="${left_jobs##*$'\t'}"
        right_collect="${right_jobs##*$'\t'}"
        mkdir -p "$ANALYSIS_ROOT/logs"
        analysis_job="$(submit_job --partition="$PARTITION" \
            --dependency="afterok:$left_collect:$right_collect" \
            --output="$ANALYSIS_ROOT/logs/analysis-%j.out" \
            --export="ALL,PIPELINE_DIR=$PIPELINE_DIR,LEFT_METRICS_CSV=$(paired_metrics_root left-hippocampus)/optimizer_roi_metrics.csv,RIGHT_METRICS_CSV=$(paired_metrics_root right-m1)/optimizer_roi_metrics.csv,ANALYSIS_ROOT=$ANALYSIS_ROOT,SIMNIBS326H_RENDER_PYTHON=$RENDER_PYTHON" \
            "$ROI_ANALYSIS")"
        echo "[INFO] Analysis jobs: left=$left_jobs right=$right_jobs final=$analysis_job"
        ;;
    analysis-status)
        for file in \
            "$(paired_metrics_root left-hippocampus)/manifest.json" \
            "$(paired_metrics_root right-m1)/manifest.json" \
            "$ANALYSIS_ROOT/results/analysis_manifest.json"
        do
            echo "[$file]"; [ ! -s "$file" ] || cat "$file"
        done
        ;;
esac
