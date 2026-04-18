#!/bin/bash
#SBATCH --job-name=ti_post_batch
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=post_processing_batch_%j.out
#SBATCH --error=post_processing_batch_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"
echo

# ------------------------------------------------------------------
# This wrapper launches the repeat-aware batch runner in
# post/run_post_processing_batch.py on a single node.
#
# Before submitting:
#   1) Edit the config block near the bottom of
#      post/run_post_processing_batch.py so the shared pipeline settings point
#      at your HPC paths, especially:
#        - pipeline_template.post.fastsurfer_root
#        - pipeline_template.population.enabled / related population settings
#      For your current atlas layout, the batch runner expects the shared atlas
#      directory to contain files such as:
#        /mnt/parscratch/users/cop23bi/ZIPs/atlases/sub-CC110056.nii
#      and it also accepts .nii.gz if some subjects are compressed.
#   2) Set BATCH_ROOT at submission time to the parent directory containing
#      repeat folders such as:
#        /path/to/rootDIR/Left_Hippocampus_Data_01
#        /path/to/rootDIR/Left_Hippocampus_Data_02
#        ...
#   3) One-time cluster-side setup option:
#        sbatch HPC_scripts/bootstrap_post_conda.slurm
#
# Submission example:
#   sbatch HPC_scripts/run_post_processing_batch.slurm \
#     --rootDirPath /path/to/rootDIR
#
# Optional script arguments:
#   --dataset-glob "*_Data_*"
#   --repeats "01 02 03 04 05 06 07 08 09 10"
#   --summary-filename post_processing_batch_summary.json
#   --atlas-filename mri/aparc.DKTatlas+aseg.deep.nii.gz
#   --stop-on-error
# Only use --atlas-filename when you want to override the default flat
# shared-atlas layout. For your current setup, it is not needed because
# fastsurfer_root already points at the atlas folder.
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_DIR}/post/run_post_processing_batch.py"
POST_REQUIREMENTS_FILE="${POST_REQUIREMENTS_FILE:-${REPO_DIR}/requirements-post.txt}"
POST_CONDA_ENV="${POST_CONDA_ENV:-ti-post}"

BATCH_ROOT="${BATCH_ROOT:-}"
BATCH_DATASET_GLOB="${BATCH_DATASET_GLOB:-}"
BATCH_REPEATS="${BATCH_REPEATS:-}"
BATCH_SUMMARY_FILENAME="${BATCH_SUMMARY_FILENAME:-}"
BATCH_ATLAS_FILENAME="${BATCH_ATLAS_FILENAME:-}"
BATCH_STOP_ON_ERROR="${BATCH_STOP_ON_ERROR:-0}"

usage() {
    cat <<'EOF'
Usage:
  sbatch run_post_processing_batch.slurm --rootDirPath /path/to/rootDIR

Optional arguments:
  --dataset-glob PATTERN
  --repeats "01 02 03 04 05 06 07 08 09 10"
  --summary-filename PATH
  --atlas-filename PATH
  --stop-on-error
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rootDirPath|--root-dir-path|--batch-root)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            BATCH_ROOT="$2"
            shift 2
            ;;
        --dataset-glob)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            BATCH_DATASET_GLOB="$2"
            shift 2
            ;;
        --repeats)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            BATCH_REPEATS="$2"
            shift 2
            ;;
        --summary-filename)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            BATCH_SUMMARY_FILENAME="$2"
            shift 2
            ;;
        --atlas-filename)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            BATCH_ATLAS_FILENAME="$2"
            shift 2
            ;;
        --stop-on-error)
            BATCH_STOP_ON_ERROR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

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

# The batch runner delegates to run_post_processing.py, which uses this when
# post.max_workers is None.
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

if [[ -z "${BATCH_ROOT:-}" ]]; then
    echo "[ERROR] BATCH_ROOT is required." >&2
    usage >&2
    exit 2
fi

"${PYTHON}" - <<'PY'
import importlib.util
import os
from pathlib import Path

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

ARGS=(--batch-root "${BATCH_ROOT}")

if [[ -n "${BATCH_DATASET_GLOB:-}" ]]; then
    ARGS+=(--dataset-glob "${BATCH_DATASET_GLOB}")
fi

if [[ -n "${BATCH_REPEATS:-}" ]]; then
    # shellcheck disable=SC2206
    repeat_values=(${BATCH_REPEATS})
    ARGS+=(--repeats "${repeat_values[@]}")
fi

if [[ -n "${BATCH_SUMMARY_FILENAME:-}" ]]; then
    ARGS+=(--summary-filename "${BATCH_SUMMARY_FILENAME}")
fi

if [[ -n "${BATCH_ATLAS_FILENAME:-}" ]]; then
    ARGS+=(--atlas-filename "${BATCH_ATLAS_FILENAME}")
fi

if [[ "${BATCH_STOP_ON_ERROR:-0}" == "1" ]]; then
    ARGS+=(--stop-on-error)
fi

echo
echo "[INFO] Batch root:   ${BATCH_ROOT}"
echo "[INFO] Dataset glob: ${BATCH_DATASET_GLOB:-*_Data_*}"
echo "[INFO] Repeats:      ${BATCH_REPEATS:-<runner default>}"
echo "[INFO] Summary:      ${BATCH_SUMMARY_FILENAME:-<runner default>}"
echo "[INFO] Atlas file:   ${BATCH_ATLAS_FILENAME:-<runner default>}"
echo "[INFO] Stop on err:  ${BATCH_STOP_ON_ERROR:-0}"
echo

cd "${REPO_DIR}"
"${PYTHON}" -u "${RUNNER}" "${ARGS[@]}"

echo
echo "[INFO] Finished: $(date)"
