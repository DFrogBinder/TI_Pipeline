#!/bin/bash
set -euo pipefail

PIPELINE_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
SUBJECT_COUNT="${SUBJECT_COUNT:-10}"

LEFT_ROOT="${LEFT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10}"
RIGHT_ROOT="${RIGHT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1}"
LEFT_OUTPUT_ROOT="${LEFT_OUTPUT_ROOT:-$LEFT_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1}"
RIGHT_OUTPUT_ROOT="${RIGHT_OUTPUT_ROOT:-$RIGHT_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1}"

EXTRACTOR="$PIPELINE_DIR/post/extract_repeatability_optimizer_roi_metrics.py"
ARRAY_SCRIPT="$PIPELINE_DIR/hpc_scripts/repeatability_optimizer_roi_metrics_array.slurm"
COLLECT_SCRIPT="$PIPELINE_DIR/hpc_scripts/repeatability_optimizer_roi_metrics_collect.slurm"

MODE="all"
PREFLIGHT_ONLY=0
for argument in "$@"; do
    case "$argument" in
        all|left-hippocampus|right-m1)
            MODE="$argument"
            ;;
        --preflight)
            PREFLIGHT_ONLY=1
            ;;
        *)
            echo "[ERROR] Unknown argument: $argument" >&2
            echo "Usage: $0 [all|left-hippocampus|right-m1] [--preflight]" >&2
            exit 2
            ;;
    esac
done

for required in "$EXTRACTOR" "$ARRAY_SCRIPT" "$COLLECT_SCRIPT"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file is missing: $required" >&2
        exit 2
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    command -v module >/dev/null 2>&1 || {
        echo "[ERROR] The environment-modules command is unavailable." >&2
        exit 2
    }
    module purge
    module use "$HOME/modules"
    module load SimNIBS/4.0.1-foss-2023a
    PYTHON_BIN=python
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_CONCURRENT must be a positive integer." >&2
    exit 2
fi
if ! [[ "$SUBJECT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] SUBJECT_COUNT must be a positive integer." >&2
    exit 2
fi
if [ "$MODE" = "all" ] && [ "$LEFT_OUTPUT_ROOT" = "$RIGHT_OUTPUT_ROOT" ]; then
    echo "[ERROR] Left and right output roots must be different." >&2
    exit 2
fi

ARRAY_MAX=$((SUBJECT_COUNT - 1))
ARRAY_CONCURRENT="$MAX_CONCURRENT"
if [ "$ARRAY_CONCURRENT" -gt "$SUBJECT_COUNT" ]; then
    ARRAY_CONCURRENT="$SUBJECT_COUNT"
fi
FIELDS_PER_TARGET=$((SUBJECT_COUNT * 2 * 40))

declare -a LABELS=()
declare -a ROOTS=()
declare -a OUTPUT_ROOTS=()
if [ "$MODE" = "all" ] || [ "$MODE" = "left-hippocampus" ]; then
    LABELS+=("left_hippocampus")
    ROOTS+=("$LEFT_ROOT")
    OUTPUT_ROOTS+=("$LEFT_OUTPUT_ROOT")
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "right-m1" ]; then
    LABELS+=("right_m1")
    ROOTS+=("$RIGHT_ROOT")
    OUTPUT_ROOTS+=("$RIGHT_OUTPUT_ROOT")
fi

echo "Scope:"
echo "  mode: $MODE"
echo "  datasets/targets: ${#ROOTS[@]}"
echo "  subjects per target: $SUBJECT_COUNT"
echo "  conditions per subject: 2 (remesh, fixed_mesh)"
echo "  runs per condition: 40"
echo "  existing TI NIfTIs per target: $FIELDS_PER_TARGET"
echo "  existing TI NIfTIs total: $((FIELDS_PER_TARGET * ${#ROOTS[@]}))"
echo "  extraction arrays: ${#ROOTS[@]} x 0-$ARRAY_MAX%$ARRAY_CONCURRENT"
echo "  subject extraction tasks: $((SUBJECT_COUNT * ${#ROOTS[@]}))"
echo "  collector jobs: ${#ROOTS[@]}"
echo "  scheduler tasks total: $(((SUBJECT_COUNT + 1) * ${#ROOTS[@]}))"
echo "  expected combined metric rows: $((FIELDS_PER_TARGET * ${#ROOTS[@]}))"
echo "  target definitions: 200 mm3 left hippocampus, 100 mm3 right M1"
echo "  meshing tasks: 0"
echo "  FEM simulation tasks: 0"
echo "  execution: full requested scope"
echo "  source policy: read-only; isolated repeatability_optimizer_roi_metrics_v1 outputs"
echo "Resources:"
echo "  module: SimNIBS/4.0.1-foss-2023a"
echo "  partition: $PARTITION"
echo "  CPUs/task: $CPUS_PER_TASK"
echo "  memory/task: $MEMORY"
echo "  time/task: $TIME_LIMIT"

for index in "${!ROOTS[@]}"; do
    label="${LABELS[$index]}"
    root="${ROOTS[$index]}"
    config="$root/_pipeline/configs/paired_analysis.json"
    output_root="${OUTPUT_ROOTS[$index]}"
    echo "[INFO] Preflight: $label"
    "$PYTHON_BIN" "$EXTRACTOR" preflight \
        --config "$config" \
        --output-root "$output_root"
done

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash $0 $MODE"
    exit 0
fi

for index in "${!ROOTS[@]}"; do
    label="${LABELS[$index]}"
    root="${ROOTS[$index]}"
    config="$root/_pipeline/configs/paired_analysis.json"
    output_root="${OUTPUT_ROOTS[$index]}"
    log_root="$output_root/logs"
    archive="$output_root/${label}_repeatability_optimizer_roi_metrics_v1.tar.gz"
    mkdir -p "$log_root"

    export_spec="ALL,PIPELINE_DIR=$PIPELINE_DIR,EXPERIMENT_CONFIG=$config,OUTPUT_ROOT=$output_root,EXTRACTOR=$EXTRACTOR,ARCHIVE_PATH=$archive"
    array_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_roi_${label}" \
            --partition="$PARTITION" \
            --cpus-per-task="$CPUS_PER_TASK" \
            --mem="$MEMORY" \
            --time="$TIME_LIMIT" \
            --array="0-$ARRAY_MAX%$ARRAY_CONCURRENT" \
            --output="$log_root/extract-%A_%a.out" \
            --error="$log_root/extract-%A_%a.err" \
            --export="$export_spec" \
            "$ARRAY_SCRIPT"
    )"
    collector_job="$(
        "$SBATCH_BIN" \
            --parsable \
            --job-name="ti_roi_collect_${label}" \
            --partition="$PARTITION" \
            --cpus-per-task="$CPUS_PER_TASK" \
            --mem="$MEMORY" \
            --time="01:00:00" \
            --dependency="afterok:$array_job" \
            --output="$log_root/collect-%j.out" \
            --error="$log_root/collect-%j.err" \
            --export="$export_spec" \
            "$COLLECT_SCRIPT"
    )"
    receipt="$output_root/submitted_job_ids.tsv"
    printf "stage\tjob_id\tdependency\nextract\t%s\t\ncollect\t%s\tafterok:%s\n" \
        "$array_job" "$collector_job" "$array_job" > "$receipt"
    echo "[INFO] $label extraction array: $array_job"
    echo "[INFO] $label collector:        $collector_job"
    echo "[INFO] $label archive:          $archive"
    echo "[INFO] $label submission log:   $receipt"
done
