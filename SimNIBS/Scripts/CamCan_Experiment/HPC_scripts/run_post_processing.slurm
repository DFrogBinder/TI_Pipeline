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
# User config
#
# Edit the #SBATCH lines above for queue, CPU, memory, and walltime.
# Edit the variables below for repo/data/env/pipeline settings.
#
# This wrapper can be submitted from anywhere. It can also live outside the
# repository as long as REPO_DIR points at the CamCan_Experiment checkout.
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repository / environment
REPO_DIR="/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment"
POST_REQUIREMENTS_FILE=""  # blank -> use <REPO_DIR>/requirements-post.txt
POST_CONDA_ENV="ti-post"
POST_CONDA_SH=""  # set when using a personal conda install
POST_VENV=""  # set instead of POST_CONDA_ENV when using venv
PYTHON=""  # set to an explicit interpreter to skip activation logic
POST_ANACONDA_MODULE="Anaconda3/2022.05"

# Single-dataset discovery
# Example below is configured for one local left-hippocampus repeat.
DATASET_ROOT="/home/boyan/sandbox/Jake_Data/Simulation-Target-Data/Left-Hoppocampus-Post-Data/Left_Hippocampus_Data_01"

# Subject-level pipeline template
PIPELINE_MODE="single"
PIPELINE_FASTSURFER_ROOT="/home/boyan/sandbox/Jake_Data/atlases"
PIPELINE_FASTSURFER_ATLAS_FILENAME=""  # blank -> flat <FASTSURFER_ROOT>/<subject>.nii(.gz); set a relative path like mri/aparc.a2009s+aseg.nii.gz for nested Destrieux layouts, or mri/aparc.DKTatlas+aseg.deep.nii.gz for legacy DKT layouts
PIPELINE_ATLAS_MODE="fastsurfer"
PIPELINE_T1_PATH=""  # blank -> infer T1 per subject
PIPELINE_SUBJECTS=""  # blank -> all subjects in the dataset
PIPELINE_MAX_WORKERS=""  # blank -> use POST_MAX_WORKERS / SLURM_CPUS_PER_TASK
PIPELINE_PLOT_ROI=""  # blank -> infer ROI from the dataset name
PIPELINE_PERCENTILE="95.0"
PIPELINE_HARD_THRESHOLD="0.2"
PIPELINE_OFFTARGET_THRESHOLD="0.2"
PIPELINE_MNI_BASELINE_ROOT="/home/boyan/sandbox/Jake_Data/MNI152-data/MNI152_Hippocampus"
PIPELINE_MNI_FIXED_ATLAS_PATH="/home/boyan/sandbox/Jake_Data/atlases/sub-mni152.nii.gz"
PIPELINE_NEIGHBOR_DILATION_ITER="1"
PIPELINE_WRITE_NEIGHBOR_VISUALIZATION="1"  # write fixed-neighbor masks and overlay PNGs
PIPELINE_CSF_LABELS="24"  # space-separated label ids
PIPELINE_SKULL_LABELS=""  # space-separated label ids, blank -> skip skull-distance metric
PIPELINE_ELECTRODE_CSV=""  # optional CSV with columns subject,electrode,x,y,z
PIPELINE_ELECTRODE_DATASET_DIR=""  # optional directory with <roi>/<subject>/electrodes.csv files
PIPELINE_ELECTRODE_NAMES=""  # optional space-separated EEG position names
PIPELINE_EEG_POSITIONS_PATH_TEMPLATE=""  # optional template, e.g. /path/{subject}/eeg_positions.csv
PIPELINE_FORCE="0"
PIPELINE_QUIET="0"

# Within-run population aggregation
PIPELINE_POPULATION_ENABLED="1"
PIPELINE_POPULATION_OUT_DIR=""  # blank -> default population_analysis under the dataset
PIPELINE_POPULATION_TEMPLATE_REGION_CSV=""

if [[ $# -gt 0 ]]; then
    echo "[ERROR] This wrapper now reads configuration from the top of the file." >&2
    echo "[ERROR] Remove script arguments and edit the config block instead." >&2
    exit 2
fi

if [[ -z "${REPO_DIR}" ]]; then
    REPO_DIR="${SCRIPT_DIR}/.."
elif [[ "${REPO_DIR}" != /* ]]; then
    REPO_DIR="${SCRIPT_DIR}/${REPO_DIR}"
fi

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[ERROR] REPO_DIR does not exist: ${REPO_DIR}" >&2
    exit 2
fi

REPO_DIR="$(cd "${REPO_DIR}" && pwd)"
RUNNER="${REPO_DIR}/post/run_full_post_pipeline.py"

if [[ -z "${POST_REQUIREMENTS_FILE}" ]]; then
    POST_REQUIREMENTS_FILE="${REPO_DIR}/requirements-post.txt"
elif [[ "${POST_REQUIREMENTS_FILE}" != /* ]]; then
    POST_REQUIREMENTS_FILE="${REPO_DIR}/${POST_REQUIREMENTS_FILE}"
fi

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

# run_full_post_pipeline.py will use this when max_workers is unset.
export POST_MAX_WORKERS="${POST_MAX_WORKERS:-${SLURM_CPUS_PER_TASK}}"

echo "[INFO] Repository: ${REPO_DIR}"
echo "[INFO] Runner:     ${RUNNER}"
echo "[INFO] Python:     $(${PYTHON} --version 2>&1 || true)"
echo "[INFO] Python exe: $(${PYTHON} -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
echo "[INFO] Virtualenv: ${VIRTUAL_ENV:-<none>}"
echo "[INFO] Conda env:  ${CONDA_PREFIX:-<none>}"
echo "[INFO] Dataset root: ${DATASET_ROOT}"
echo "[INFO] ROI:          ${PIPELINE_PLOT_ROI:-<infer from dataset>}"
echo "[INFO] Atlas mode:   ${PIPELINE_ATLAS_MODE}"
echo "[INFO] FastSurfer:   ${PIPELINE_FASTSURFER_ROOT:-<none>}"
echo "[INFO] Atlas file:   ${PIPELINE_FASTSURFER_ATLAS_FILENAME:-<flat subject atlas lookup>}"
echo "[INFO] MNI root:     ${PIPELINE_MNI_BASELINE_ROOT:-<none>}"
echo "[INFO] MNI atlas:    ${PIPELINE_MNI_FIXED_ATLAS_PATH:-<none>}"
echo "[INFO] Population:   ${PIPELINE_POPULATION_ENABLED}"
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

if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "[ERROR] DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
    exit 2
fi

declare -a CMD
CMD=(
    "${PYTHON}" -u "${RUNNER}"
    --root "${DATASET_ROOT}"
    --mode "${PIPELINE_MODE}"
    --atlas-mode "${PIPELINE_ATLAS_MODE}"
    --percentile "${PIPELINE_PERCENTILE}"
    --hard-threshold "${PIPELINE_HARD_THRESHOLD}"
    --offtarget-threshold "${PIPELINE_OFFTARGET_THRESHOLD}"
    --mni-baseline-root "${PIPELINE_MNI_BASELINE_ROOT}"
    --mni-fixed-atlas-path "${PIPELINE_MNI_FIXED_ATLAS_PATH}"
    --neighbor-dilation-iter "${PIPELINE_NEIGHBOR_DILATION_ITER}"
)

if [[ -n "${PIPELINE_FASTSURFER_ROOT}" ]]; then
    CMD+=(--fastsurfer-root "${PIPELINE_FASTSURFER_ROOT}")
fi
if [[ -n "${PIPELINE_FASTSURFER_ATLAS_FILENAME}" ]]; then
    CMD+=(--atlas-filename "${PIPELINE_FASTSURFER_ATLAS_FILENAME}")
fi
if [[ -n "${PIPELINE_T1_PATH}" ]]; then
    CMD+=(--t1-path "${PIPELINE_T1_PATH}")
fi
if [[ -n "${PIPELINE_MAX_WORKERS}" ]]; then
    CMD+=(--max-workers "${PIPELINE_MAX_WORKERS}")
fi
if [[ -n "${PIPELINE_PLOT_ROI}" ]]; then
    CMD+=(--roi "${PIPELINE_PLOT_ROI}")
fi
if [[ -n "${PIPELINE_SUBJECTS}" ]]; then
    # shellcheck disable=SC2206
    SUBJECT_ARGS=(${PIPELINE_SUBJECTS})
    CMD+=(--subjects "${SUBJECT_ARGS[@]}")
fi
if [[ -n "${PIPELINE_CSF_LABELS}" ]]; then
    # shellcheck disable=SC2206
    CSF_ARGS=(${PIPELINE_CSF_LABELS})
    CMD+=(--csf-labels "${CSF_ARGS[@]}")
fi
if [[ -n "${PIPELINE_SKULL_LABELS}" ]]; then
    # shellcheck disable=SC2206
    SKULL_ARGS=(${PIPELINE_SKULL_LABELS})
    CMD+=(--skull-labels "${SKULL_ARGS[@]}")
fi
if [[ -n "${PIPELINE_ELECTRODE_CSV}" ]]; then
    CMD+=(--electrode-csv "${PIPELINE_ELECTRODE_CSV}")
fi
if [[ -n "${PIPELINE_ELECTRODE_DATASET_DIR}" ]]; then
    CMD+=(--electrode-dataset-dir "${PIPELINE_ELECTRODE_DATASET_DIR}")
fi
if [[ -n "${PIPELINE_ELECTRODE_NAMES}" ]]; then
    # shellcheck disable=SC2206
    ELECTRODE_ARGS=(${PIPELINE_ELECTRODE_NAMES})
    CMD+=(--electrode-names "${ELECTRODE_ARGS[@]}")
fi
if [[ -n "${PIPELINE_EEG_POSITIONS_PATH_TEMPLATE}" ]]; then
    CMD+=(--eeg-positions-path-template "${PIPELINE_EEG_POSITIONS_PATH_TEMPLATE}")
fi
if [[ "${PIPELINE_WRITE_NEIGHBOR_VISUALIZATION}" == "0" ]]; then
    CMD+=(--no-neighbor-visualization)
fi
if [[ "${PIPELINE_POPULATION_ENABLED}" == "0" ]]; then
    CMD+=(--no-population)
fi
if [[ -n "${PIPELINE_POPULATION_OUT_DIR}" ]]; then
    CMD+=(--population-output-dir "${PIPELINE_POPULATION_OUT_DIR}")
fi
if [[ -n "${PIPELINE_POPULATION_TEMPLATE_REGION_CSV}" ]]; then
    CMD+=(--template-region-csv "${PIPELINE_POPULATION_TEMPLATE_REGION_CSV}")
fi
if [[ "${PIPELINE_FORCE}" == "1" ]]; then
    CMD+=(--force)
fi
if [[ "${PIPELINE_QUIET}" == "1" ]]; then
    CMD+=(--quiet)
fi

echo
echo "[INFO] Running single-dataset pipeline via run_full_post_pipeline.py"
cd "${REPO_DIR}"
"${CMD[@]}"

echo
echo "[INFO] Finished: $(date)"
