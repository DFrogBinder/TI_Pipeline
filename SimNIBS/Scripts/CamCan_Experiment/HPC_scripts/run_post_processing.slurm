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
#   2) Point the job at a Python environment that contains the post-processing
#      stack, using one of:
#        - POST_CONDA_ENV=<name-or-prefix> (Stanage docs path; optionally set
#          POST_ANACONDA_MODULE, defaults to Anaconda3/2022.05)
#        - POST_VENV=/path/to/venv
#        - POST_CONDA_ENV=<name-or-prefix> and POST_CONDA_SH=/path/to/conda.sh
#          for a personal miniconda install
#        - PYTHON=/path/to/env/bin/python
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_DIR}/post/run_post_processing.py"
POST_REQUIREMENTS_FILE="${POST_REQUIREMENTS_FILE:-${REPO_DIR}/requirements-post.txt}"

# ----------------------------
# Environment setup
# ----------------------------
export SLURM_EXPORT_ENV=ALL
module purge

activate_post_python() {
    if [[ -n "${POST_CONDA_ENV:-}" && -z "${POST_CONDA_SH:-}" ]]; then
        module load "${POST_ANACONDA_MODULE:-Anaconda3/2022.05}" || true
        if ! command -v conda >/dev/null 2>&1; then
            echo "[ERROR] Failed to load Conda from module ${POST_ANACONDA_MODULE:-Anaconda3/2022.05}." >&2
            echo "[ERROR] Set POST_ANACONDA_MODULE to another Stanage Anaconda3 module or use POST_CONDA_SH." >&2
            exit 2
        fi
        source activate "${POST_CONDA_ENV}"
        return
    fi

    if [[ -n "${POST_VENV:-}" ]]; then
        if [[ ! -f "${POST_VENV}/bin/activate" ]]; then
            echo "[ERROR] POST_VENV does not look like a virtualenv: ${POST_VENV}" >&2
            exit 2
        fi
        # shellcheck disable=SC1090
        source "${POST_VENV}/bin/activate"
        return
    fi

    if [[ -n "${POST_CONDA_ENV:-}" ]]; then
        local conda_sh="${POST_CONDA_SH:-}"

        if [[ -z "${conda_sh}" ]] && command -v conda >/dev/null 2>&1; then
            local conda_base
            conda_base="$(conda info --base 2>/dev/null || true)"
            if [[ -n "${conda_base}" ]]; then
                conda_sh="${conda_base}/etc/profile.d/conda.sh"
            fi
        fi

        if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
            echo "[ERROR] POST_CONDA_ENV is set but conda.sh could not be found." >&2
            echo "[ERROR] Set POST_CONDA_SH=/path/to/conda.sh or use POST_VENV/PYTHON instead." >&2
            exit 2
        fi

        # shellcheck disable=SC1090
        source "${conda_sh}"
        if command -v conda >/dev/null 2>&1; then
            conda activate "${POST_CONDA_ENV}"
        else
            source activate "${POST_CONDA_ENV}"
        fi
        return
    fi

    module load Python/3.11.3-GCCcore-12.3.0 || true
    if [[ -z "${PYTHON:-}" && -f "${REPO_DIR}/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${REPO_DIR}/.venv/bin/activate"
    fi
}

activate_post_python
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
echo "[INFO] Python exe: $(${PYTHON} -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
echo "[INFO] Virtualenv: ${VIRTUAL_ENV:-<none>}"
echo "[INFO] Conda env:  ${CONDA_PREFIX:-<none>}"
echo "[INFO] Workers:    ${POST_MAX_WORKERS}"
echo

if [[ ! -f "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    exit 2
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON}" >&2
    exit 2
fi

"${PYTHON}" - <<'PY'
import importlib.util
import os
from pathlib import Path
import sys

required = ["numpy", "pandas", "nibabel", "scipy", "nilearn", "matplotlib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    req_file = Path(os.environ.get("POST_REQUIREMENTS_FILE", "")).expanduser()
    lines = [f"Missing Python packages: {', '.join(missing)}"]
    if req_file.is_file():
        lines.append(f"Install them into your HPC env with: pip install -r {req_file}")
    lines.append(
        "Then resubmit with one of POST_VENV=/path/to/venv, "
        "POST_CONDA_ENV=<name-or-prefix>, or PYTHON=/path/to/env/bin/python."
    )
    raise SystemExit("\n".join(lines))

print("[INFO] Python dependency check passed.")
PY

echo
echo "[INFO] Using pipeline config embedded in post/run_post_processing.py"
cd "${REPO_DIR}"
"${PYTHON}" -u "${RUNNER}"

echo
echo "[INFO] Finished: $(date)"
