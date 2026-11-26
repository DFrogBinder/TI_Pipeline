#!/bin/bash
set -euo pipefail

PIPELINE_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_historical_component_analysis_v1}"
V4_ATLAS_DIR="${V4_ATLAS_DIR:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
PREFLIGHT_ONLY=1

if [ "${1:-}" = "--submit" ]; then
    PREFLIGHT_ONLY=0
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "Usage: bash $0 [--submit]" >&2
    exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_CONCURRENT must be a positive integer." >&2
    exit 2
fi

CLI=(python -m ti_current_repair.repeatability_v2.historical_cli)
WORKER="$PIPELINE_DIR/ti_current_repair/repeatability_v2/historical_hpc/extract_array.slurm"
COLLECTOR="$PIPELINE_DIR/ti_current_repair/repeatability_v2/historical_hpc/collect.slurm"
ANALYSIS="$PIPELINE_DIR/ti_current_repair/repeatability_v2/historical_hpc/analyse.slurm"
for required in "$WORKER" "$COLLECTOR" "$ANALYSIS"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file is missing: $required" >&2
        exit 2
    fi
done

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a
export PYTHONNOUSERSITE=1

echo "Scope:"
echo "  existing simulations: 3200"
echo "  expected metric rows: 9600"
echo "  versions: SimNIBS 3.2.6 and 4.0.1"
echo "  targets: left hippocampus and right M1"
echo "  conditions: remesh and fixed_mesh"
echo "  extraction arrays: 2 x 0-19%${MAX_CONCURRENT}"
echo "  subject extraction tasks: 40"
echo "  collector jobs: 2"
echo "  combined preliminary analysis jobs: 1"
echo "  meshing tasks: 0"
echo "  FEM tasks: 0"
echo "  source policy: read-only"
echo "Resources (candidate until this preflight/pilot succeeds):"
echo "  module: SimNIBS/4.0.1-foss-2023a"
echo "  partition: $PARTITION"
echo "  CPUs/task: $CPUS_PER_TASK"
echo "  memory/task: $MEMORY"
echo "  time/task: $TIME_LIMIT"

cd "$PIPELINE_DIR"
"${CLI[@]}" preflight \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --v4-atlas-dir "$V4_ATLAS_DIR"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "[INFO] Preflight passed without creating outputs or submitting jobs."
    echo "[INFO] Submit explicitly with: bash $0 --submit"
    exit 0
fi

"${CLI[@]}" initialize-output \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/logs"

common_export="ALL,PIPELINE_DIR=$PIPELINE_DIR,DATASET_ROOT=$DATASET_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,V4_ATLAS_DIR=$V4_ATLAS_DIR"
v3_array="$($SBATCH_BIN --parsable \
    --job-name=hist_native_326 \
    --partition="$PARTITION" --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" --time="$TIME_LIMIT" \
    --array="0-19%$MAX_CONCURRENT" \
    --output="$OUTPUT_ROOT/logs/extract-326-%A_%a.out" \
    --error="$OUTPUT_ROOT/logs/extract-326-%A_%a.err" \
    --export="$common_export,HISTORICAL_VERSION=simnibs326" "$WORKER")"
v4_array="$($SBATCH_BIN --parsable \
    --job-name=hist_native_401 \
    --partition="$PARTITION" --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" --time="$TIME_LIMIT" \
    --array="0-19%$MAX_CONCURRENT" \
    --output="$OUTPUT_ROOT/logs/extract-401-%A_%a.out" \
    --error="$OUTPUT_ROOT/logs/extract-401-%A_%a.err" \
    --export="$common_export,HISTORICAL_VERSION=simnibs4" "$WORKER")"
v3_collect="$($SBATCH_BIN --parsable \
    --job-name=hist_collect_326 --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --time=01:00:00 \
    --dependency="afterok:$v3_array" \
    --output="$OUTPUT_ROOT/logs/collect-326-%j.out" \
    --error="$OUTPUT_ROOT/logs/collect-326-%j.err" \
    --export="$common_export,HISTORICAL_VERSION=simnibs326" "$COLLECTOR")"
v4_collect="$($SBATCH_BIN --parsable \
    --job-name=hist_collect_401 --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --time=01:00:00 \
    --dependency="afterok:$v4_array" \
    --output="$OUTPUT_ROOT/logs/collect-401-%j.out" \
    --error="$OUTPUT_ROOT/logs/collect-401-%j.err" \
    --export="$common_export,HISTORICAL_VERSION=simnibs4" "$COLLECTOR")"
analysis_job="$($SBATCH_BIN --parsable \
    --job-name=hist_prelim_analysis --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --time=01:00:00 \
    --dependency="afterok:$v3_collect:$v4_collect" \
    --output="$OUTPUT_ROOT/logs/analysis-%j.out" \
    --error="$OUTPUT_ROOT/logs/analysis-%j.err" \
    --export="$common_export" "$ANALYSIS")"

receipt="$OUTPUT_ROOT/submitted_job_ids.tsv"
{
    printf "stage\tjob_id\tdependency\n"
    printf "extract_simnibs326\t%s\t\n" "$v3_array"
    printf "extract_simnibs401\t%s\t\n" "$v4_array"
    printf "collect_simnibs326\t%s\tafterok:%s\n" "$v3_collect" "$v3_array"
    printf "collect_simnibs401\t%s\tafterok:%s\n" "$v4_collect" "$v4_array"
    printf "preliminary_analysis\t%s\tafterok:%s:%s\n" \
        "$analysis_job" "$v3_collect" "$v4_collect"
} > "$receipt"
echo "[INFO] SimNIBS 3.2.6 array: $v3_array"
echo "[INFO] SimNIBS 4.0.1 array: $v4_array"
echo "[INFO] Preliminary analysis: $analysis_job"
echo "[INFO] Submission receipt: $receipt"
