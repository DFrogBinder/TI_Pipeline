#!/bin/bash
#SBATCH --job-name=ti_post
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=post_processing_%j.out
#SBATCH --error=post_processing_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"
echo

# ------------------------------------------------------------------
# This wrapper launches the config-driven batch runner in
# post/run_post_processing.py on a single node.
#
# Before submitting:
#   1) Edit the config block at the bottom of post/run_post_processing.py
#      so post.root / fastsurfer_root / population settings point at HPC paths.
#   2) If needed, replace the Python setup below with your venv/conda env.
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_DIR}/post/run_post_processing.py"

# ----------------------------
# Environment setup
# ----------------------------
module purge
module load Python/3.11.3-GCCcore-12.3.0 || true

# Example alternatives:
# source /path/to/venv/bin/activate
# PYTHON=/path/to/env/bin/python
PYTHON="${PYTHON:-python}"

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

# Avoid nested BLAS/OpenMP threading inside each subject worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# run_post_processing.py will use this when post.max_workers is None.
export POST_MAX_WORKERS="${POST_MAX_WORKERS:-${SLURM_CPUS_PER_TASK}}"

echo "[INFO] Repository: ${REPO_DIR}"
echo "[INFO] Runner:     ${RUNNER}"
echo "[INFO] Python:     $(${PYTHON} --version 2>&1 || true)"
echo "[INFO] Workers:    ${POST_MAX_WORKERS}"
echo

if [[ ! -f "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    exit 2
fi

"${PYTHON}" - <<'PY'
import importlib.util
import sys

required = ["numpy", "pandas", "nibabel", "scipy", "nilearn"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")

print("[INFO] Python dependency check passed.")
PY

echo
echo "[INFO] Using pipeline config embedded in post/run_post_processing.py"
cd "${REPO_DIR}"
"${PYTHON}" -u "${RUNNER}"

echo
echo "[INFO] Finished: $(date)"
