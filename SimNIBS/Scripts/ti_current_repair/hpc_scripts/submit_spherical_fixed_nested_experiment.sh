#!/bin/bash
set -euo pipefail

PIPELINE_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"

LEFT_SOURCE_ROOT="${LEFT_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10}"
RIGHT_SOURCE_ROOT="${RIGHT_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1}"
LEFT_METRICS_CSV="${LEFT_METRICS_CSV:-$LEFT_SOURCE_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv}"
RIGHT_METRICS_CSV="${RIGHT_METRICS_CSV:-$RIGHT_SOURCE_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv}"
LEFT_OUTPUT_ROOT="${LEFT_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_spherical_fixed_v1}"
RIGHT_OUTPUT_ROOT="${RIGHT_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1}"
NESTED_OUTPUT_ROOT="${NESTED_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1}"
SELECTION_SEED="${SELECTION_SEED:-20260831}"
NESTED_TARGET="${NESTED_TARGET:-left-hippocampus}"

SIM_CPUS="${SIM_CPUS:-8}"
SIM_MEMORY="${SIM_MEMORY:-32G}"
SIM_TIME="${SIM_TIME:-08:00:00}"
SIM_MAX_CONCURRENT="${SIM_MAX_CONCURRENT:-50}"
ROI_CPUS="${ROI_CPUS:-4}"
ROI_MEMORY="${ROI_MEMORY:-16G}"
ROI_TIME="${ROI_TIME:-02:00:00}"
ROI_MAX_CONCURRENT="${ROI_MAX_CONCURRENT:-10}"
RESUBMIT="${RESUBMIT:-0}"

WORKFLOW="$PIPELINE_DIR/pipeline/spherical_fixed_nested_experiment.py"
SIM_ARRAY="$PIPELINE_DIR/hpc_scripts/repeatability_experiment_array.slurm"
FINALIZER="$PIPELINE_DIR/hpc_scripts/spherical_fixed_nested_finalize.slurm"
ROI_ARRAY="$PIPELINE_DIR/hpc_scripts/repeatability_optimizer_roi_metrics_array.slurm"
ROI_COLLECT="$PIPELINE_DIR/hpc_scripts/repeatability_optimizer_roi_metrics_collect.slurm"
ROI_EXTRACTOR="$PIPELINE_DIR/post/extract_repeatability_optimizer_roi_metrics.py"
NESTED_ANALYSIS="$PIPELINE_DIR/hpc_scripts/nested_repeatability_analysis.slurm"

MODE="all"
PREFLIGHT_ONLY=0
PREPARE_ONLY=0
for argument in "$@"; do
    case "$argument" in
        all|fixed|nested)
            MODE="$argument"
            ;;
        --preflight)
            PREFLIGHT_ONLY=1
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            ;;
        *)
            echo "[ERROR] Unknown argument: $argument" >&2
            echo "Usage: $0 [all|fixed|nested] [--preflight|--prepare-only]" >&2
            exit 2
            ;;
    esac
done

if [ "$PREFLIGHT_ONLY" = "1" ] && [ "$PREPARE_ONLY" = "1" ]; then
    echo "[ERROR] Use only one of --preflight or --prepare-only." >&2
    exit 2
fi
if ! [[ "$SIM_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] SIM_MAX_CONCURRENT must be a positive integer." >&2
    exit 2
fi
if ! [[ "$ROI_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ROI_MAX_CONCURRENT must be a positive integer." >&2
    exit 2
fi
if ! [[ "$SELECTION_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] SELECTION_SEED must be an integer." >&2
    exit 2
fi
if [ "$NESTED_TARGET" != "left-hippocampus" ] && [ "$NESTED_TARGET" != "right-m1" ]; then
    echo "[ERROR] NESTED_TARGET must be left-hippocampus or right-m1." >&2
    exit 2
fi

for required in \
    "$WORKFLOW" \
    "$SIM_ARRAY" \
    "$FINALIZER" \
    "$ROI_ARRAY" \
    "$ROI_COLLECT" \
    "$ROI_EXTRACTOR" \
    "$NESTED_ANALYSIS"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required workflow file is missing: $required" >&2
        exit 2
    fi
done

echo "Scope:"
echo "  mode: $MODE"
if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    echo "  spherical fixed correction targets: 2"
    echo "  correction subjects: 10 per target (20 participant-target cases)"
    echo "  correction condition: fixed_mesh only"
    echo "  correction repeats: 40 per case"
    echo "  correction simulations: 800"
    echo "  correction arrays: 2 x 0-399%$SIM_MAX_CONCURRENT"
    echo "  correction expected TI outputs: 800"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    echo "  nested cases: 1 persisted random participant"
    echo "  nested prespecified target: $NESTED_TARGET"
    echo "  nested outer meshes: 40"
    echo "  nested repeats per mesh: 40"
    echo "  nested simulations: 1600"
    echo "  nested array: 0-1599%$SIM_MAX_CONCURRENT"
    echo "  nested expected TI outputs: 1600"
fi
if [ "$MODE" = "all" ]; then
    echo "  total new simulations: 2400"
    echo "  total expected TI outputs: 2400"
fi
echo "  execution: full requested scope; no smoke or reduced subset"
echo "  source remesh outputs: read-only"
echo "  retry policy: same persisted selection, skip complete tasks, unlimited task requeue"
echo "Resources:"
echo "  simulation module: SimNIBS/4.0.1-foss-2023a"
echo "  simulation partition: $PARTITION"
echo "  simulation CPU/memory/time: $SIM_CPUS / $SIM_MEMORY / $SIM_TIME"
echo "  spherical metric CPU/memory/time: $ROI_CPUS / $ROI_MEMORY / $ROI_TIME"

fixed_command() {
    local command="$1"
    local source_root="$2"
    local metrics_csv="$3"
    local output_root="$4"
    "$PYTHON_BIN" "$WORKFLOW" "$command" \
        --source-experiment-root "$source_root" \
        --optimizer-metrics-csv "$metrics_csv" \
        --output-root "$output_root" \
        --repeat-count 40
}

nested_command() {
    local command="$1"
    "$PYTHON_BIN" "$WORKFLOW" "$command" \
        --left-source-root "$LEFT_SOURCE_ROOT" \
        --right-source-root "$RIGHT_SOURCE_ROOT" \
        --left-metrics-csv "$LEFT_METRICS_CSV" \
        --right-metrics-csv "$RIGHT_METRICS_CSV" \
        --output-root "$NESTED_OUTPUT_ROOT" \
        --selection-seed "$SELECTION_SEED" \
        --nested-target "$NESTED_TARGET" \
        --outer-repeat-count 40 \
        --inner-repeat-count 40
}

if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    echo "[INFO] Preflight: left-hippocampus spherical-median fixed correction"
    fixed_command preflight-fixed "$LEFT_SOURCE_ROOT" "$LEFT_METRICS_CSV" "$LEFT_OUTPUT_ROOT"
    echo "[INFO] Preflight: right-M1 spherical-median fixed correction"
    fixed_command preflight-fixed "$RIGHT_SOURCE_ROOT" "$RIGHT_METRICS_CSV" "$RIGHT_OUTPUT_ROOT"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    echo "[INFO] Preflight: persisted random 40x40 nested case"
    nested_command preflight-nested
fi

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "[INFO] Full-scope preflight passed; no files created and no jobs submitted."
    exit 0
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    echo "[INFO] Preparing isolated left-hippocampus correction root"
    fixed_command prepare-fixed "$LEFT_SOURCE_ROOT" "$LEFT_METRICS_CSV" "$LEFT_OUTPUT_ROOT"
    echo "[INFO] Preparing isolated right-M1 correction root"
    fixed_command prepare-fixed "$RIGHT_SOURCE_ROOT" "$RIGHT_METRICS_CSV" "$RIGHT_OUTPUT_ROOT"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    echo "[INFO] Persisting/reusing the nested case and preparing 40 mesh caches"
    nested_command prepare-nested
fi

if [ "$PREPARE_ONLY" = "1" ]; then
    echo "[INFO] Preparation completed; no jobs submitted."
    exit 0
fi

check_submission_receipt() {
    local root="$1"
    local receipt="$root/_pipeline/submitted_job_ids.tsv"
    if [ -f "$receipt" ] && [ "$RESUBMIT" != "1" ]; then
        echo "[ERROR] Submission receipt already exists: $receipt" >&2
        echo "[INFO] Inspect active jobs first. Set RESUBMIT=1 only for a deliberate resumable relaunch." >&2
        exit 2
    fi
}

if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    check_submission_receipt "$LEFT_OUTPUT_ROOT"
    check_submission_receipt "$RIGHT_OUTPUT_ROOT"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    check_submission_receipt "$NESTED_OUTPUT_ROOT"
fi

submit_study() {
    local label="$1"
    local root="$2"
    local config="$3"
    local simulation_tasks="$4"
    local subjects="$5"
    local is_nested="$6"
    local receipt="$root/_pipeline/submitted_job_ids.tsv"
    local log_root="$root/_pipeline/logs"
    local optimizer_root="$root/_post_processing/optimizer_roi_metrics_v1"
    local optimizer_archive="$optimizer_root/${label}_optimizer_roi_metrics_v1.tar.gz"
    local sim_export
    local sim_job
    local finalize_export
    local finalize_job
    local roi_export
    local roi_job
    local collect_job
    local analysis_job=""

    if [ -f "$receipt" ] && [ "$RESUBMIT" != "1" ]; then
        echo "[ERROR] Submission receipt already exists: $receipt" >&2
        echo "[INFO] Inspect active jobs first. Set RESUBMIT=1 only for a deliberate resumable relaunch." >&2
        exit 2
    fi
    if [ -f "$receipt" ]; then
        mv "$receipt" "${receipt%.tsv}.previous.$(date -u +%Y%m%dT%H%M%SZ).tsv"
    fi
    mkdir -p "$log_root" "$optimizer_root/logs"

    sim_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,LOG_DIR=$log_root/simulation,TI_MESH_TIMEOUT_HOURS=4,TI_MESH_MAX_RETRIES=0,OVERWRITE_OUTPUT=0,FORCE_MESH=0"
    sim_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}" \
            --partition="$PARTITION" \
            --cpus-per-task="$SIM_CPUS" \
            --mem="$SIM_MEMORY" \
            --time="$SIM_TIME" \
            --array="0-$((simulation_tasks - 1))%$SIM_MAX_CONCURRENT" \
            --output="$log_root/simulation-%A_%a.out" \
            --error="$log_root/simulation-%A_%a.err" \
            --export="$sim_export" \
            "$SIM_ARRAY"
    )"

    finalize_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config"
    finalize_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_finalize" \
            --partition="$PARTITION" \
            --cpus-per-task=1 \
            --mem=8G \
            --time=01:00:00 \
            --dependency="afterok:$sim_job" \
            --output="$log_root/finalize-%j.out" \
            --error="$log_root/finalize-%j.err" \
            --export="$finalize_export" \
            "$FINALIZER"
    )"

    roi_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,OUTPUT_ROOT=$optimizer_root,EXTRACTOR=$ROI_EXTRACTOR,ARCHIVE_PATH=$optimizer_archive"
    roi_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_roi" \
            --partition="$PARTITION" \
            --cpus-per-task="$ROI_CPUS" \
            --mem="$ROI_MEMORY" \
            --time="$ROI_TIME" \
            --array="0-$((subjects - 1))%$ROI_MAX_CONCURRENT" \
            --dependency="afterok:$finalize_job" \
            --output="$optimizer_root/logs/extract-%A_%a.out" \
            --error="$optimizer_root/logs/extract-%A_%a.err" \
            --export="$roi_export" \
            "$ROI_ARRAY"
    )"
    collect_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_${label}_collect" \
            --partition="$PARTITION" \
            --cpus-per-task="$ROI_CPUS" \
            --mem="$ROI_MEMORY" \
            --time=01:00:00 \
            --dependency="afterok:$roi_job" \
            --output="$optimizer_root/logs/collect-%j.out" \
            --error="$optimizer_root/logs/collect-%j.err" \
            --export="$roi_export" \
            "$ROI_COLLECT"
    )"

    if [ "$is_nested" = "1" ]; then
        local nested_output="$root/_analysis/nested_variance"
        local selection="$root/_pipeline/nested_case_selection.json"
        local analysis_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,METRICS_CSV=$optimizer_root/optimizer_roi_metrics.csv,SELECTION_MANIFEST=$selection,OUTPUT_DIR=$nested_output"
        analysis_job="$(
            "$SBATCH_BIN" \
                --parsable \
                --job-name="ti_${label}_analysis" \
                --partition="$PARTITION" \
                --cpus-per-task=1 \
                --mem=8G \
                --time=01:00:00 \
                --dependency="afterok:$collect_job" \
                --output="$log_root/nested-analysis-%j.out" \
                --error="$log_root/nested-analysis-%j.err" \
                --export="$analysis_export" \
                "$NESTED_ANALYSIS"
        )"
    fi

    {
        printf "stage\tjob_id\tdependency\n"
        printf "simulation\t%s\t\n" "$sim_job"
        printf "finalize\t%s\tafterok:%s\n" "$finalize_job" "$sim_job"
        printf "optimizer_roi_extract\t%s\tafterok:%s\n" "$roi_job" "$finalize_job"
        printf "optimizer_roi_collect\t%s\tafterok:%s\n" "$collect_job" "$roi_job"
        if [ -n "$analysis_job" ]; then
            printf "nested_analysis\t%s\tafterok:%s\n" "$analysis_job" "$collect_job"
        fi
    } > "$receipt"

    echo "[INFO] $label simulation array: $sim_job"
    echo "[INFO] $label finalizer:        $finalize_job"
    echo "[INFO] $label ROI extraction:   $roi_job"
    echo "[INFO] $label ROI collector:    $collect_job"
    if [ -n "$analysis_job" ]; then
        echo "[INFO] $label nested analysis:  $analysis_job"
    fi
    echo "[INFO] $label submission log:   $receipt"
}

if [ "$MODE" = "all" ] || [ "$MODE" = "fixed" ]; then
    submit_study \
        "spherical_fixed_lh" \
        "$LEFT_OUTPUT_ROOT" \
        "$LEFT_OUTPUT_ROOT/_pipeline/configs/fixed_mesh_spherical_median.json" \
        400 \
        10 \
        0
    submit_study \
        "spherical_fixed_rm1" \
        "$RIGHT_OUTPUT_ROOT" \
        "$RIGHT_OUTPUT_ROOT/_pipeline/configs/fixed_mesh_spherical_median.json" \
        400 \
        10 \
        0
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "nested" ]; then
    submit_study \
        "nested_40x40" \
        "$NESTED_OUTPUT_ROOT" \
        "$NESTED_OUTPUT_ROOT/_pipeline/configs/nested_40x40.json" \
        1600 \
        1 \
        1
fi
