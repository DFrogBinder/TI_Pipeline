#!/bin/bash
#SBATCH --job-name=ti326_analysis
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${LEFT_METRICS_CSV:?Set LEFT_METRICS_CSV.}"
: "${RIGHT_METRICS_CSV:?Set RIGHT_METRICS_CSV.}"
: "${ANALYSIS_ROOT:?Set ANALYSIS_ROOT.}"
ANALYSIS_PYTHON="${SIMNIBS326_ANALYSIS_PYTHON:-$HOME/.conda/envs/ti-post/bin/python}"
if [ ! -x "$ANALYSIS_PYTHON" ]; then
    echo "[ERROR] Analysis Python is missing or not executable: $ANALYSIS_PYTHON" >&2
    exit 2
fi

ANALYZER="$PIPELINE_DIR/../ti_current_repair/post/analyze_repeatability_optimizer_roi_metrics.py"
RESULTS_DIR="$ANALYSIS_ROOT/results"
ARCHIVE="$ANALYSIS_ROOT/simnibs326_repeatability_analysis_v1.tar.gz"
MANIFEST="$RESULTS_DIR/analysis_manifest.json"

for required in "$ANALYZER" "$LEFT_METRICS_CSV" "$RIGHT_METRICS_CSV"; do
    if [ ! -s "$required" ]; then
        echo "[ERROR] Required analysis input is missing or empty: $required" >&2
        exit 2
    fi
done

mkdir -p "$ANALYSIS_ROOT" "$ANALYSIS_ROOT/.mplconfig"
export MPLCONFIGDIR="$ANALYSIS_ROOT/.mplconfig"

if [ -s "$MANIFEST" ]; then
    "$ANALYSIS_PYTHON" - "$MANIFEST" "$LEFT_METRICS_CSV" "$RIGHT_METRICS_CSV" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_path, left_path, right_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "complete" or int(manifest.get("rows", -1)) != 1600:
    raise SystemExit("Existing analysis manifest is not complete")
for key, path in (("left_csv", left_path), ("right_csv", right_path)):
    recorded = manifest.get("inputs", {}).get(key, {}).get("sha256")
    if recorded != sha256(path):
        raise SystemExit(f"Existing analysis input hash differs for {key}")
for artifact in manifest.get("artifacts", []):
    path = manifest_path.parent / artifact["path"]
    if not path.is_file():
        raise SystemExit(f"Existing analysis artifact is missing: {path}")
    if int(artifact["size_bytes"]) != path.stat().st_size:
        raise SystemExit(f"Existing analysis artifact size differs: {path}")
    if artifact["sha256"] != sha256(path):
        raise SystemExit(f"Existing analysis artifact hash differs: {path}")
print("[INFO] Existing complete analysis matches the paired metric inputs; skipping recomputation.")
PY
else
    if [ -e "$RESULTS_DIR" ]; then
        echo "[ERROR] Refusing to merge with an incomplete analysis directory: $RESULTS_DIR" >&2
        exit 2
    fi
    ATTEMPT_DIR="$ANALYSIS_ROOT/.results-${SLURM_JOB_ID:-manual}"
    if [ -e "$ATTEMPT_DIR" ]; then
        echo "[ERROR] Attempt directory already exists: $ATTEMPT_DIR" >&2
        exit 2
    fi

    "$ANALYSIS_PYTHON" -c 'import matplotlib, numpy, pandas, scipy'
    "$ANALYSIS_PYTHON" -u "$ANALYZER" \
        --left-csv "$LEFT_METRICS_CSV" \
        --right-csv "$RIGHT_METRICS_CSV" \
        --out-dir "$ATTEMPT_DIR" \
        --seed 20260803 \
        --rank-draws 20000

    "$ANALYSIS_PYTHON" - "$ATTEMPT_DIR/analysis_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("status") != "complete":
    raise SystemExit("Analysis did not produce a complete manifest")
if int(manifest.get("rows", -1)) != 1600:
    raise SystemExit("Analysis manifest does not contain exactly 1600 rows")
if set(manifest.get("targets", [])) != {"left_hippocampus", "right_m1"}:
    raise SystemExit("Analysis manifest target set is incorrect")
PY
    mv "$ATTEMPT_DIR" "$RESULTS_DIR"
fi

TEMP_ARCHIVE="$ANALYSIS_ROOT/.simnibs326_repeatability_analysis_v1.tar.gz.${SLURM_JOB_ID:-manual}.tmp"
tar -C "$RESULTS_DIR" -czf "$TEMP_ARCHIVE" .
mv "$TEMP_ARCHIVE" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "[INFO] Analysis manifest: $MANIFEST"
echo "[INFO] Download archive:  $ARCHIVE"
echo "[INFO] Source policy:     read-only"
echo "[INFO] Meshing/FEM tasks: 0"
