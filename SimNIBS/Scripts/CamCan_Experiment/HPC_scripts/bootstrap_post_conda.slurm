#!/bin/bash
#SBATCH --job-name=ti_post_bootstrap
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=bootstrap_post_conda_%j.out
#SBATCH --error=bootstrap_post_conda_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

POST_CONDA_ENV="${POST_CONDA_ENV:-ti-post}"
POST_PYTHON_VERSION="${POST_PYTHON_VERSION:-3.11}"
POST_REQUIREMENTS_FILE="${POST_REQUIREMENTS_FILE:-${REPO_DIR}/requirements-post.txt}"

export SLURM_EXPORT_ENV=ALL
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

module purge
module load "${POST_ANACONDA_MODULE:-Anaconda3/2022.05}" || true

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Failed to load Conda from module ${POST_ANACONDA_MODULE:-Anaconda3/2022.05}." >&2
    echo "[ERROR] Set POST_ANACONDA_MODULE to a valid Anaconda module before resubmitting." >&2
    exit 2
fi

if [[ ! -f "${POST_REQUIREMENTS_FILE}" ]]; then
    echo "[ERROR] Requirements file not found: ${POST_REQUIREMENTS_FILE}" >&2
    exit 2
fi

conda_env_exists() {
    conda env list --json | python - "$POST_CONDA_ENV" <<'PY'
import json
import os
import sys

target = sys.argv[1]
envs = json.load(sys.stdin).get("envs", [])

if os.path.isabs(target):
    target_norm = os.path.normpath(target)
    exists = any(os.path.normpath(env) == target_norm for env in envs)
else:
    exists = any(os.path.basename(env) == target for env in envs)

raise SystemExit(0 if exists else 1)
PY
}

create_conda_env() {
    if [[ "${POST_CONDA_ENV}" = /* ]]; then
        mkdir -p "$(dirname "${POST_CONDA_ENV}")"
        conda create -y -p "${POST_CONDA_ENV}" "python=${POST_PYTHON_VERSION}" \
            numpy pandas nibabel scipy matplotlib pip
    else
        conda create -y -n "${POST_CONDA_ENV}" "python=${POST_PYTHON_VERSION}" \
            numpy pandas nibabel scipy matplotlib pip
    fi
}

activate_conda_env() {
    source activate "${POST_CONDA_ENV}"
}

python_stack_ok() {
    python - <<'PY'
import importlib.util

required = ["numpy", "pandas", "nibabel", "scipy", "nilearn", "matplotlib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
print("[INFO] Python dependency check passed.")
PY
}

echo "[INFO] Repository:        ${REPO_DIR}"
echo "[INFO] Conda module:      ${POST_ANACONDA_MODULE:-Anaconda3/2022.05}"
echo "[INFO] Conda env target:  ${POST_CONDA_ENV}"
echo "[INFO] Python version:    ${POST_PYTHON_VERSION}"
echo "[INFO] Requirements file: ${POST_REQUIREMENTS_FILE}"
echo

if conda_env_exists; then
    echo "[INFO] Conda env already exists: ${POST_CONDA_ENV}"
else
    echo "[INFO] Creating Conda env: ${POST_CONDA_ENV}"
    create_conda_env
fi

activate_conda_env

if ! python_stack_ok; then
    echo "[INFO] Installing or repairing Python packages from ${POST_REQUIREMENTS_FILE}"
    python -m pip install --upgrade pip
    python -m pip install -r "${POST_REQUIREMENTS_FILE}"
    python_stack_ok
fi

echo
echo "[INFO] Bootstrap complete."
echo "[INFO] Submit post-processing with:"
echo "sbatch --export=ALL,POST_CONDA_ENV=\"${POST_CONDA_ENV}\" ${REPO_DIR}/HPC_scripts/run_post_processing.slurm"
echo
echo "[INFO] Finished: $(date)"
