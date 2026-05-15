#!/bin/bash
#SBATCH --job-name=collect_post_outputs
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=collect_post_outputs_%j.out
#SBATCH --error=collect_post_outputs_%j.err

set -euo pipefail

# ------------------------------------------------------------------
# User config
#
# Edit these defaults directly, or override them at submission time with:
# sbatch --export=ALL,POST_COLLECT_ROOT=/path/to/root,POST_COLLECT_DEST=/path/to/export \
#   HPC_scripts/collect_post_outputs.slurm
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/post/utils/collect_post_outputs.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/../post/utils/collect_post_outputs.py" ]]; then
        REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

# Environment. Set only one of PYTHON, POST_VENV, or POST_CONDA_ENV if needed.
PYTHON="${PYTHON:-}"
POST_VENV="${POST_VENV:-}"
POST_CONDA_ENV="${POST_CONDA_ENV:-ti-post}"
POST_CONDA_SH="${POST_CONDA_SH:-}"
POST_ANACONDA_MODULE="${POST_ANACONDA_MODULE:-Anaconda3/2022.05}"

# Collector input/output.
POST_COLLECT_ROOT="${POST_COLLECT_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Runs}"
POST_COLLECT_DEST="${POST_COLLECT_DEST:-/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data/Left_Hippocampus_Post_Export}"
POST_COLLECT_DATASET_GLOB="${POST_COLLECT_DATASET_GLOB:-*_Data_*}"
POST_COLLECT_REPEATS="${POST_COLLECT_REPEATS:-01 02 03 04 05 06 07 08 09 10}"
POST_COLLECT_SUMMARY_GLOB="${POST_COLLECT_SUMMARY_GLOB:-post_processing_batch_summary*.json}"
POST_COLLECT_INCLUDE_POPULATION="${POST_COLLECT_INCLUDE_POPULATION:-1}"
POST_COLLECT_DRY_RUN="${POST_COLLECT_DRY_RUN:-0}"
POST_COLLECT_WORKERS="${POST_COLLECT_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<not-slurm>}"
echo "[INFO] Start: $(date)"
echo

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[ERROR] REPO_DIR does not exist: ${REPO_DIR}" >&2
    exit 2
fi

REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
COLLECT_POST_OUTPUTS_PY="${REPO_DIR}/post/utils/collect_post_outputs.py"

if [[ ! -f "${COLLECT_POST_OUTPUTS_PY}" ]]; then
    echo "[ERROR] Collector script not found: ${COLLECT_POST_OUTPUTS_PY}" >&2
    exit 2
fi

if [[ -z "${POST_COLLECT_ROOT}" || "${POST_COLLECT_ROOT}" == "/path/to/root" ]]; then
    echo "[ERROR] Set POST_COLLECT_ROOT to the repeat-batch root or single dataset root." >&2
    exit 2
fi

if [[ -z "${POST_COLLECT_DEST}" || "${POST_COLLECT_DEST}" == "/path/to/export" ]]; then
    echo "[ERROR] Set POST_COLLECT_DEST to the export destination directory." >&2
    exit 2
fi

if [[ ! -d "${POST_COLLECT_ROOT}" ]]; then
    echo "[ERROR] POST_COLLECT_ROOT does not exist: ${POST_COLLECT_ROOT}" >&2
    exit 2
fi

export SLURM_EXPORT_ENV=ALL
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export POST_EXPORT_WORKERS="${POST_COLLECT_WORKERS}"

module purge

activate_post_python() {
    if [[ -n "${PYTHON}" ]]; then
        return
    fi

    if [[ -n "${POST_VENV}" ]]; then
        if [[ ! -f "${POST_VENV}/bin/activate" ]]; then
            echo "[ERROR] POST_VENV does not look like a virtualenv: ${POST_VENV}" >&2
            exit 2
        fi
        # shellcheck disable=SC1090
        source "${POST_VENV}/bin/activate"
        return
    fi

    if [[ -n "${POST_CONDA_ENV}" && -z "${POST_CONDA_SH}" ]]; then
        module load "${POST_ANACONDA_MODULE}" || true
        if command -v conda >/dev/null 2>&1; then
            source activate "${POST_CONDA_ENV}"
            return
        fi
    fi

    if [[ -n "${POST_CONDA_ENV}" ]]; then
        local conda_sh="${POST_CONDA_SH}"

        if [[ -z "${conda_sh}" ]] && command -v conda >/dev/null 2>&1; then
            local conda_base
            conda_base="$(conda info --base 2>/dev/null || true)"
            if [[ -n "${conda_base}" ]]; then
                conda_sh="${conda_base}/etc/profile.d/conda.sh"
            fi
        fi

        if [[ -n "${conda_sh}" && -f "${conda_sh}" ]]; then
            # shellcheck disable=SC1090
            source "${conda_sh}"
            conda activate "${POST_CONDA_ENV}"
            return
        fi
    fi

    module load Python/3.11.3-GCCcore-12.3.0 || true
    if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${REPO_DIR}/.venv/bin/activate"
    fi
}

activate_post_python
PYTHON="${PYTHON:-python}"

echo "[INFO] Repository:       ${REPO_DIR}"
echo "[INFO] Python:           $(${PYTHON} --version 2>&1 || true)"
echo "[INFO] Python exe:       $(${PYTHON} -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
echo "[INFO] Virtualenv:       ${VIRTUAL_ENV:-<none>}"
echo "[INFO] Conda env:        ${CONDA_PREFIX:-<none>}"
echo "[INFO] Root:             ${POST_COLLECT_ROOT}"
echo "[INFO] Destination:      ${POST_COLLECT_DEST}"
echo "[INFO] Dataset glob:     ${POST_COLLECT_DATASET_GLOB}"
echo "[INFO] Repeats:          ${POST_COLLECT_REPEATS:-<all matched repeats>}"
echo "[INFO] Summary glob:     ${POST_COLLECT_SUMMARY_GLOB:-<disabled>}"
echo "[INFO] Include pop:      ${POST_COLLECT_INCLUDE_POPULATION}"
echo "[INFO] Dry run:          ${POST_COLLECT_DRY_RUN}"
echo "[INFO] Copy workers:     ${POST_COLLECT_WORKERS}"
echo

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON}" >&2
    exit 2
fi

"${PYTHON}" - <<'PY'
import importlib.util

required = ["numpy", "pandas", "nibabel", "scipy", "nilearn", "matplotlib", "rich"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Python package(s): "
        + ", ".join(missing)
        + "\nInstall into the selected HPC Python env, for example: python -m pip install -r requirements-post.txt"
    )

print("[INFO] Python dependency check passed.")
PY

args=(
    --root "${POST_COLLECT_ROOT}"
    --dest "${POST_COLLECT_DEST}"
    --dataset-glob "${POST_COLLECT_DATASET_GLOB}"
    --workers "${POST_COLLECT_WORKERS}"
)

if [[ -n "${POST_COLLECT_SUMMARY_GLOB}" ]]; then
    args+=(--summary-glob "${POST_COLLECT_SUMMARY_GLOB}")
else
    args+=(--summary-glob "")
fi

if [[ "${POST_COLLECT_INCLUDE_POPULATION}" != "1" ]]; then
    args+=(--no-population)
fi

if [[ "${POST_COLLECT_DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
fi

if [[ -n "${POST_COLLECT_REPEATS}" ]]; then
    read -r -a repeat_args <<< "${POST_COLLECT_REPEATS}"
    args+=(--repeats "${repeat_args[@]}")
fi

echo "[INFO] Running collector:"
printf ' %q' "${PYTHON}" "${COLLECT_POST_OUTPUTS_PY}" "${args[@]}"
echo
echo

"${PYTHON}" -u "${COLLECT_POST_OUTPUTS_PY}" "${args[@]}"

echo
echo "[INFO] Finished: $(date)"
echo "[INFO] Export directory size:"
du -sh "${POST_COLLECT_DEST}" || true
