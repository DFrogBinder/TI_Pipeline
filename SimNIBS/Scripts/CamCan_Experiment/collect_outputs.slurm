#!/bin/bash
#SBATCH --job-name=collect_outputs
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=collect_outputs_%j.out
#SBATCH --error=collect_outputs_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"
echo

# ----------------------------
# EDIT THESE PATHS
# ----------------------------
SRC="/mnt/parscratch/users/cop23bi/full-ti-dataset"
DST="/mnt/parscratch/users/cop23bi/export_sim_outputs"

SCRIPT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/collect_sim_outputs.py"
FILTER_RULES="/mnt/parscratch/users/cop23bi/export_filter.rules"   # optional artifact to inspect later

# ----------------------------
# Environment setup
# ----------------------------
module purge

# If your cluster provides a Python module, load it.
# If not, remove these lines and use your venv/conda/absolute python.
module load Python/3.11.3-GCCcore-12.3.0 || true

# Optional: ensure user site-packages don't interfere
export PYTHONNOUSERSITE=1

# ----------------------------
# (Optional) quick preflight checks
# ----------------------------
echo "[INFO] Python: $(python --version 2>&1 || true)"
echo "[INFO] rsync:  $(rsync --version | head -n 1 || true)"
echo

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Script not found: $SCRIPT" >&2
  exit 2
fi

# ----------------------------
# Run a dry-run first (logged), then the real run
# ----------------------------
echo "[INFO] Dry-run (for log visibility):"
python "$SCRIPT" \
  --src "$SRC" \
  --dst "$DST" \
  --dry-run \
  --no-verbose \
#  --write-filter-to "$FILTER_RULES"

echo
echo "[INFO] Real export run:"
python "$SCRIPT" \
  --src "$SRC" \
  --dst "$DST" \
  --no-verbose \
#  --write-filter-to "$FILTER_RULES"

echo
echo "[INFO] Finished: $(date)"
echo "[INFO] Export directory size:"
du -sh "$DST" || true

