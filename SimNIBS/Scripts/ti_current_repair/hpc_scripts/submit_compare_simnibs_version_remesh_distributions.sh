#!/bin/bash
set -euo pipefail

PIPELINE_DIR_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$HOME/.conda/envs/ti-post/bin/python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

V3_LEFT_ROOT="${V3_LEFT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1}"
V3_RIGHT_ROOT="${V3_RIGHT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_right_m1_v1}"
V4_LEFT_ROOT="${V4_LEFT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10}"
V4_RIGHT_ROOT="${V4_RIGHT_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1}"

V3_LEFT_HIPPOCAMPUS_CSV="${V3_LEFT_HIPPOCAMPUS_CSV:-$V3_LEFT_ROOT/_post_processing/repeatability_optimizer_roi_metrics_simnibs326_paired_v1/optimizer_roi_metrics.csv}"
V3_RIGHT_M1_CSV="${V3_RIGHT_M1_CSV:-$V3_RIGHT_ROOT/_post_processing/repeatability_optimizer_roi_metrics_simnibs326_paired_v1/optimizer_roi_metrics.csv}"
V4_LEFT_HIPPOCAMPUS_CSV="${V4_LEFT_HIPPOCAMPUS_CSV:-$V4_LEFT_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv}"
V4_RIGHT_M1_CSV="${V4_RIGHT_M1_CSV:-$V4_RIGHT_ROOT/_post_processing/repeatability_optimizer_roi_metrics_v1/optimizer_roi_metrics.csv}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-/mnt/parscratch/users/cop23bi/final_132_repeatability_paper_update_v1/simnibs_v3_v4_remesh_distribution_comparison_v1}"

ANALYZER="$PIPELINE_DIR/post/compare_simnibs_version_remesh_distributions.py"
SLURM_SCRIPT="$PIPELINE_DIR/hpc_scripts/compare_simnibs_version_remesh_distributions.slurm"
MODE="${1:-scope}"

case "$MODE" in
    scope|preflight|run|status)
        ;;
    *)
        echo "Usage: bash $0 {scope|preflight|run|status}" >&2
        exit 2
        ;;
esac

for required in "$ANALYZER" "$SLURM_SCRIPT"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required workflow file is missing: $required" >&2
        exit 2
    fi
done

print_scope() {
    echo "Official SimNIBS-version remesh-distribution comparison"
    echo "  versions: 2 (v3.2.6 and v4)"
    echo "  targets: 2 (left hippocampus and right M1)"
    echo "  participants: 10"
    echo "  source tables: 4 canonical schema-v2 tables"
    echo "  source metric rows: 3,200 (remesh plus fixed-mesh)"
    echo "  selected remesh metric rows: 1,600"
    echo "  remesh repetitions: 40 per participant-target-version"
    echo "  outcomes: ROI median primary; ROI mean and p95 sensitivity"
    echo "  run-percentile rows: 2,400"
    echo "  primary participant-target rows: 20"
    echo "  participant-level tests: 6"
    echo "  inferential units: 10 participants"
    echo "  technical repeats: summarized within participant"
    echo "  scheduler jobs: 1 analysis job"
    echo "  meshing/FEM tasks: 0"
    echo "  source policy: read-only"
    echo "  output: $ANALYSIS_ROOT"
}

print_scope

if [ "$MODE" = "scope" ]; then
    exit 0
fi

if [ "$MODE" = "status" ]; then
    if [ -s "$ANALYSIS_ROOT/results/analysis_manifest.json" ]; then
        echo "[INFO] Complete-manifest candidate: $ANALYSIS_ROOT/results/analysis_manifest.json"
    else
        echo "[INFO] No complete-manifest candidate is present."
    fi
    if [ -d "$ANALYSIS_ROOT/logs" ]; then
        find "$ANALYSIS_ROOT/logs" -maxdepth 1 -type f -name 'analysis-*.out' -print | sort
    fi
    exit 0
fi

if [ ! -x "$ANALYSIS_PYTHON" ]; then
    echo "[ERROR] Analysis Python is missing or not executable: $ANALYSIS_PYTHON" >&2
    exit 2
fi
for input in \
    "$V3_LEFT_HIPPOCAMPUS_CSV" \
    "$V3_RIGHT_M1_CSV" \
    "$V4_LEFT_HIPPOCAMPUS_CSV" \
    "$V4_RIGHT_M1_CSV"; do
    if [ ! -s "$input" ]; then
        echo "[ERROR] Canonical metric input is missing or empty: $input" >&2
        exit 2
    fi
done

echo "[INFO] Validating dependencies and all four canonical inputs."
"$ANALYSIS_PYTHON" -c 'import matplotlib, numpy, pandas'
"$ANALYSIS_PYTHON" "$ANALYZER" preflight \
    --v3-left-hippocampus-csv "$V3_LEFT_HIPPOCAMPUS_CSV" \
    --v3-right-m1-csv "$V3_RIGHT_M1_CSV" \
    --v4-left-hippocampus-csv "$V4_LEFT_HIPPOCAMPUS_CSV" \
    --v4-right-m1-csv "$V4_RIGHT_M1_CSV" \
    --out-dir "$ANALYSIS_ROOT/results"

if [ "$MODE" = "preflight" ]; then
    echo "[INFO] Read-only preflight passed; no job was submitted and no output was created."
    exit 0
fi

mkdir -p "$ANALYSIS_ROOT/logs"
export_spec="ALL,PIPELINE_DIR=$PIPELINE_DIR,V3_LEFT_HIPPOCAMPUS_CSV=$V3_LEFT_HIPPOCAMPUS_CSV,V3_RIGHT_M1_CSV=$V3_RIGHT_M1_CSV,V4_LEFT_HIPPOCAMPUS_CSV=$V4_LEFT_HIPPOCAMPUS_CSV,V4_RIGHT_M1_CSV=$V4_RIGHT_M1_CSV,ANALYSIS_ROOT=$ANALYSIS_ROOT,ANALYSIS_PYTHON=$ANALYSIS_PYTHON"
job_raw="$(
    "$SBATCH_BIN" --parsable \
        --job-name="ti_v3v4_remesh" \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$MEMORY" \
        --time="$TIME_LIMIT" \
        --output="$ANALYSIS_ROOT/logs/analysis-%j.out" \
        --error="$ANALYSIS_ROOT/logs/analysis-%j.err" \
        --export="$export_spec" \
        "$SLURM_SCRIPT"
)"
job_id="${job_raw%%;*}"
if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse scheduler job ID from: $job_raw" >&2
    exit 2
fi
printf 'stage\tjob_id\toutput_root\nanalysis\t%s\t%s\n' "$job_id" "$ANALYSIS_ROOT" \
    > "$ANALYSIS_ROOT/submitted_job_id.tsv"
echo "[INFO] Submitted analysis job: $job_id"
echo "[INFO] Submission receipt: $ANALYSIS_ROOT/submitted_job_id.tsv"
