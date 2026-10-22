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
MANIFEST="$RESULTS/analysis_manifest.json"
mkdir -p "$ANALYSIS_ROOT" "$ANALYSIS_ROOT/.mplconfig"
export MPLCONFIGDIR="$ANALYSIS_ROOT/.mplconfig"
if [ -s "$MANIFEST" ]; then
    "$ANALYSIS_PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "complete" or int(payload.get("rows", -1)) != 1600:
    raise SystemExit("Existing analysis manifest is not a complete 1600-row result")
PY
else
    [ ! -e "$RESULTS" ] || {
        echo "[ERROR] Refusing to merge into an incomplete result directory: $RESULTS" >&2
        exit 2
    }
    ATTEMPT="$ANALYSIS_ROOT/.results-${SLURM_JOB_ID:-manual}"
    [ ! -e "$ATTEMPT" ] || { echo "[ERROR] Attempt directory exists: $ATTEMPT" >&2; exit 2; }
    "$ANALYSIS_PYTHON" -u "$ANALYZER" --left-csv "$LEFT_METRICS_CSV" \
        --right-csv "$RIGHT_METRICS_CSV" --out-dir "$ATTEMPT" \
        --seed 20260803 --rank-draws 20000
    "$ANALYSIS_PYTHON" - "$ATTEMPT/analysis_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "complete" or int(payload.get("rows", -1)) != 1600:
    raise SystemExit("New analysis manifest is not a complete 1600-row result")
PY
    mv "$ATTEMPT" "$RESULTS"
fi
ARCHIVE="$ANALYSIS_ROOT/simnibs326h_repeatability_analysis.tar.gz"
TEMP_ARCHIVE="$ANALYSIS_ROOT/.analysis-${SLURM_JOB_ID:-manual}.tar.gz"
tar -C "$RESULTS" -czf "$TEMP_ARCHIVE" .
mv "$TEMP_ARCHIVE" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
