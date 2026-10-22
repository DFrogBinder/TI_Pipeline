#!/bin/bash
#SBATCH --job-name=ti326h_analysis
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
module purge
export PYTHONNOUSERSITE=1
: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${LEFT_METRICS_CSV:?Set LEFT_METRICS_CSV.}"
: "${RIGHT_METRICS_CSV:?Set RIGHT_METRICS_CSV.}"
: "${ANALYSIS_ROOT:?Set ANALYSIS_ROOT.}"
ANALYSIS_PYTHON="${SIMNIBS326H_RENDER_PYTHON:?Set SIMNIBS326H_RENDER_PYTHON.}"
[ -x "$ANALYSIS_PYTHON" ] || { echo "[ERROR] Analysis interpreter is not executable." >&2; exit 2; }
ANALYZER="$PIPELINE_DIR/../ti_current_repair/post/analyze_repeatability_optimizer_roi_metrics.py"
RESULTS="$ANALYSIS_ROOT/results"
mkdir -p "$RESULTS" "$ANALYSIS_ROOT/.mplconfig"
export MPLCONFIGDIR="$ANALYSIS_ROOT/.mplconfig"
"$ANALYSIS_PYTHON" -u "$ANALYZER" --left-csv "$LEFT_METRICS_CSV" \
    --right-csv "$RIGHT_METRICS_CSV" --out-dir "$RESULTS" \
    --seed 20260803 --rank-draws 20000
tar -C "$RESULTS" -czf "$ANALYSIS_ROOT/simnibs326h_repeatability_analysis.tar.gz" .
sha256sum "$ANALYSIS_ROOT/simnibs326h_repeatability_analysis.tar.gz" \
    > "$ANALYSIS_ROOT/simnibs326h_repeatability_analysis.tar.gz.sha256"
