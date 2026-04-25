#!/bin/bash
#SBATCH --job-name=ti_post_batch
#SBATCH --partition=sheffield  # TODO: replace if this partition is not valid on your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12  # TODO: lower if this request does not fit your partition
#SBATCH --mem=24G  # TODO: lower if this request does not fit your partition
#SBATCH --time=12:00:00  # TODO: 08:00:00 may be safer if 12h is rejected
#SBATCH --output=post_processing_batch_%j.out  # TODO: set an absolute log path if desired
#SBATCH --error=post_processing_batch_%j.err  # TODO: set an absolute log path if desired

set -euo pipefail

# ------------------------------------------------------------------
# User config
#
# Edit the #SBATCH lines above for queue, CPU, memory, and walltime.
# Edit the variables below for repo/data/env/pipeline settings.
#
# This script can be submitted from anywhere. It can also live outside the
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

# Batch discovery
# Example below is configured for the local left-hippocampus repeatability dataset.
BATCH_ROOT="/home/boyan/sandbox/Jake_Data/Simulation-Target-Data/Left-Hoppocampus-Post-Data"
BATCH_DATASET_GLOB="*_Data_*"
BATCH_REPEATS="01 02 03 04 05 06 07 08 09 10"
BATCH_SUMMARY_FILENAME="post_processing_batch_summary.json"  # blank -> disable summary
BATCH_STOP_ON_ERROR="0"

# Subject-level pipeline template
PIPELINE_ATLAS_MODE="fastsurfer"
PIPELINE_FASTSURFER_ROOT="/home/boyan/sandbox/Jake_Data/atlases"
PIPELINE_FASTSURFER_ATLAS_FILENAME=""  # blank -> flat <FASTSURFER_ROOT>/<subject>.nii(.gz); set a relative path like mri/aparc.DKTatlas+aseg.deep.nii.gz for nested per-subject layouts
PIPELINE_FS_MRI_PATH=""  # shared atlas path reused for every subject
PIPELINE_T1_PATH=""  # blank -> infer T1 per subject
PIPELINE_SUBJECTS=""  # blank -> all subjects in each dataset
PIPELINE_MAX_WORKERS=""  # blank -> use POST_MAX_WORKERS / SLURM_CPUS_PER_TASK
PIPELINE_PLOT_ROI=""  # blank -> infer ROI from each dataset name
PIPELINE_PERCENTILE="95.0"
PIPELINE_HARD_THRESHOLD="0.2"
PIPELINE_OVERLAY_Z_OFFSET_MM="0.0"
PIPELINE_OVERLAY_FULL_FIELD="1"
PIPELINE_WRITE_REGION_TABLE="1"
PIPELINE_REGION_PERCENTILE="95.0"
PIPELINE_OFFTARGET_THRESHOLD="0.2"
PIPELINE_MNI_BASELINE_ROOT="/home/boyan/sandbox/Jake_Data/MNI152-data/MNI152_Hippocampus"
PIPELINE_MNI_FIXED_ATLAS_PATH="/home/boyan/sandbox/Jake_Data/atlases/sub-mni152.nii.gz"
PIPELINE_NEIGHBOR_DILATION_ITER="1"
PIPELINE_CSF_LABELS="24"  # space-separated label ids
PIPELINE_SKULL_LABELS=""  # space-separated label ids, blank -> skip skull-distance metric
PIPELINE_ELECTRODE_CSV=""  # optional CSV with columns subject,electrode,x,y,z; leave blank until a study-specific table is available
PIPELINE_ELECTRODE_NAMES=""  # optional space-separated EEG position names
PIPELINE_EEG_POSITIONS_PATH_TEMPLATE=""  # optional template, e.g. /path/{subject}/eeg_positions.csv
PIPELINE_WRITE_NEIGHBOR_TABLE="1"
PIPELINE_WRITE_ELECTRODE_TABLE="1"
PIPELINE_FORCE="0"
PIPELINE_VERBOSE="1"

# Population aggregation
PIPELINE_POPULATION_ENABLED="1"
PIPELINE_POPULATION_OUT_DIR=""  # blank -> default population_analysis under each dataset
PIPELINE_POPULATION_REGION_FILENAME="region_stats_fastsurfer.csv"
PIPELINE_POPULATION_METRICS_FILENAME="subject_metrics.json"
PIPELINE_POPULATION_PEAK_THRESHOLD="0.2"
PIPELINE_POPULATION_TARGET_ROI=""  # blank -> reuse resolved post ROI
PIPELINE_POPULATION_TEMPLATE_REGION_CSV=""

# Across-repeat repeatability
PIPELINE_REPEATABILITY_ENABLED="1"
PIPELINE_REPEATABILITY_OUTPUT_DIR="repeatability_analysis"
PIPELINE_REPEATABILITY_LOGS_ROOT=""  # optional directory containing slurm-*.out files for the failure audit

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

RUN_POST_BATCH_PY="${REPO_DIR}/post/run_post_processing_batch.py"
RUN_POST_BATCH_ENV_PY="${REPO_DIR}/post/run_post_processing_batch_env.py"
RUN_POST_PY="${REPO_DIR}/post/run_post_processing.py"

if [[ ! -f "${RUN_POST_BATCH_PY}" || ! -f "${RUN_POST_BATCH_ENV_PY}" || ! -f "${RUN_POST_PY}" ]]; then
    echo "[ERROR] REPO_DIR does not look like the CamCan_Experiment checkout: ${REPO_DIR}" >&2
    echo "[ERROR] Expected to find post/run_post_processing_batch.py, post/run_post_processing_batch_env.py, and post/run_post_processing.py." >&2
    exit 2
fi

if [[ -z "${POST_REQUIREMENTS_FILE}" ]]; then
    POST_REQUIREMENTS_FILE="${REPO_DIR}/requirements-post.txt"
elif [[ "${POST_REQUIREMENTS_FILE}" != /* ]]; then
    POST_REQUIREMENTS_FILE="${REPO_DIR}/${POST_REQUIREMENTS_FILE}"
fi

if [[ -z "${PIPELINE_MAX_WORKERS}" ]]; then
    export POST_MAX_WORKERS="${POST_MAX_WORKERS:-${SLURM_CPUS_PER_TASK}}"
else
    export POST_MAX_WORKERS="${PIPELINE_MAX_WORKERS}"
fi

export REPO_DIR
export POST_REQUIREMENTS_FILE
export BATCH_ROOT
export BATCH_DATASET_GLOB
export BATCH_REPEATS
export BATCH_SUMMARY_FILENAME
export BATCH_STOP_ON_ERROR
export PIPELINE_ATLAS_MODE
export PIPELINE_FASTSURFER_ROOT
export PIPELINE_FASTSURFER_ATLAS_FILENAME
export PIPELINE_FS_MRI_PATH
export PIPELINE_T1_PATH
export PIPELINE_SUBJECTS
export PIPELINE_MAX_WORKERS
export PIPELINE_PLOT_ROI
export PIPELINE_PERCENTILE
export PIPELINE_HARD_THRESHOLD
export PIPELINE_OVERLAY_Z_OFFSET_MM
export PIPELINE_OVERLAY_FULL_FIELD
export PIPELINE_WRITE_REGION_TABLE
export PIPELINE_REGION_PERCENTILE
export PIPELINE_OFFTARGET_THRESHOLD
export PIPELINE_MNI_BASELINE_ROOT
export PIPELINE_MNI_FIXED_ATLAS_PATH
export PIPELINE_NEIGHBOR_DILATION_ITER
export PIPELINE_CSF_LABELS
export PIPELINE_SKULL_LABELS
export PIPELINE_ELECTRODE_CSV
export PIPELINE_ELECTRODE_NAMES
export PIPELINE_EEG_POSITIONS_PATH_TEMPLATE
export PIPELINE_WRITE_NEIGHBOR_TABLE
export PIPELINE_WRITE_ELECTRODE_TABLE
export PIPELINE_FORCE
export PIPELINE_VERBOSE
export PIPELINE_POPULATION_ENABLED
export PIPELINE_POPULATION_OUT_DIR
export PIPELINE_POPULATION_REGION_FILENAME
export PIPELINE_POPULATION_METRICS_FILENAME
export PIPELINE_POPULATION_PEAK_THRESHOLD
export PIPELINE_POPULATION_TARGET_ROI
export PIPELINE_POPULATION_TEMPLATE_REGION_CSV
export PIPELINE_REPEATABILITY_ENABLED
export PIPELINE_REPEATABILITY_OUTPUT_DIR
export PIPELINE_REPEATABILITY_LOGS_ROOT

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"
echo

# ----------------------------
# Environment setup
# ----------------------------
export SLURM_EXPORT_ENV=ALL
module purge

activate_post_python() {
    if [[ -n "${POST_CONDA_ENV:-}" && -z "${POST_CONDA_SH:-}" ]]; then
        module load "${POST_ANACONDA_MODULE}" || true
        if ! command -v conda >/dev/null 2>&1; then
            echo "[ERROR] Failed to load Conda from module ${POST_ANACONDA_MODULE}." >&2
            echo "[ERROR] Set POST_ANACONDA_MODULE to another Anaconda3 module or use POST_CONDA_SH." >&2
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
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Avoid nested BLAS/OpenMP threading inside each subject worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[INFO] Repository:          ${REPO_DIR}"
echo "[INFO] Script location:     ${SCRIPT_DIR}"
echo "[INFO] Python:              $(${PYTHON} --version 2>&1 || true)"
echo "[INFO] Python exe:          $(${PYTHON} -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
echo "[INFO] Virtualenv:          ${VIRTUAL_ENV:-<none>}"
echo "[INFO] Conda env:           ${CONDA_PREFIX:-<none>}"
echo "[INFO] Batch root:          ${BATCH_ROOT}"
echo "[INFO] Dataset glob:        ${BATCH_DATASET_GLOB}"
echo "[INFO] Repeats:             ${BATCH_REPEATS:-<all matched repeats>}"
echo "[INFO] Summary filename:    ${BATCH_SUMMARY_FILENAME:-<disabled>}"
echo "[INFO] Atlas mode:          ${PIPELINE_ATLAS_MODE}"
echo "[INFO] FastSurfer root:     ${PIPELINE_FASTSURFER_ROOT:-<none>}"
echo "[INFO] Atlas filename:      ${PIPELINE_FASTSURFER_ATLAS_FILENAME:-<none>}"
echo "[INFO] Explicit atlas path: ${PIPELINE_FS_MRI_PATH:-<none>}"
echo "[INFO] MNI baseline root:   ${PIPELINE_MNI_BASELINE_ROOT:-<none>}"
echo "[INFO] Fixed MNI atlas:     ${PIPELINE_MNI_FIXED_ATLAS_PATH:-<none>}"
echo "[INFO] Electrode CSV:       ${PIPELINE_ELECTRODE_CSV:-<none>}"
echo "[INFO] Population enabled:  ${PIPELINE_POPULATION_ENABLED}"
echo "[INFO] Repeatability:       ${PIPELINE_REPEATABILITY_ENABLED}"
echo "[INFO] Repeatability out:   ${PIPELINE_REPEATABILITY_OUTPUT_DIR:-<default>}"
echo "[INFO] Repeatability logs:  ${PIPELINE_REPEATABILITY_LOGS_ROOT:-<none>}"
echo "[INFO] Workers:             ${POST_MAX_WORKERS}"
echo

if [[ "${BATCH_ROOT}" == "/path/to/rootDIR" || -z "${BATCH_ROOT}" ]]; then
    echo "[ERROR] Set BATCH_ROOT near the top of this file to the parent directory containing repeat datasets." >&2
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

required = ["numpy", "pandas", "nibabel", "scipy", "nilearn", "matplotlib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    req_file = Path(os.environ.get("POST_REQUIREMENTS_FILE", "")).expanduser()
    lines = [f"Missing Python packages: {', '.join(missing)}"]
    if req_file.is_file():
        lines.append(f"Install them into your HPC env with: pip install -r {req_file}")
    lines.append(
        "Then rerun after updating POST_CONDA_ENV, POST_VENV, POST_CONDA_SH, or PYTHON "
        "near the top of this Slurm file."
    )
    raise SystemExit("\n".join(lines))

print("[INFO] Python dependency check passed.")
PY

"${PYTHON}" -u "${RUN_POST_BATCH_ENV_PY}"

echo
echo "[INFO] Finished: $(date)"
