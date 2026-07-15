#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${REPO_DIR:?Set REPO_DIR to the CamCan_Experiment checkout.}"
: "${BATCH_ROOT:?Set BATCH_ROOT to the repeat-batch parent root.}"

POST_REQUIREMENTS_FILE="${POST_REQUIREMENTS_FILE:-}"
POST_CONDA_ENV="${POST_CONDA_ENV:-ti-post}"
POST_CONDA_SH="${POST_CONDA_SH:-}"
POST_VENV="${POST_VENV:-}"
PYTHON="${PYTHON:-}"
POST_ANACONDA_MODULE="${POST_ANACONDA_MODULE:-Anaconda3/2022.05}"

BATCH_DATASET_GLOB="${BATCH_DATASET_GLOB:-*_Data_*}"
BATCH_REPEATS="${BATCH_REPEATS:-01 02 03 04 05 06 07 08 09 10}"
BATCH_SUMMARY_FILENAME="${BATCH_SUMMARY_FILENAME:-post_processing_batch_summary.json}"
BATCH_STOP_ON_ERROR="${BATCH_STOP_ON_ERROR:-0}"

PIPELINE_ATLAS_MODE="${PIPELINE_ATLAS_MODE:-fastsurfer}"
PIPELINE_FASTSURFER_ROOT="${PIPELINE_FASTSURFER_ROOT:-}"
PIPELINE_FASTSURFER_ATLAS_FILENAME="${PIPELINE_FASTSURFER_ATLAS_FILENAME:-}"
PIPELINE_FS_MRI_PATH="${PIPELINE_FS_MRI_PATH:-}"
PIPELINE_T1_PATH="${PIPELINE_T1_PATH:-}"
PIPELINE_SUBJECTS="${PIPELINE_SUBJECTS:-}"
PIPELINE_MAX_WORKERS="${PIPELINE_MAX_WORKERS:-}"
PIPELINE_PLOT_ROI="${PIPELINE_PLOT_ROI:-}"
PIPELINE_PERCENTILE="${PIPELINE_PERCENTILE:-95.0}"
PIPELINE_HARD_THRESHOLD="${PIPELINE_HARD_THRESHOLD:-0.2}"
PIPELINE_OVERLAY_Z_OFFSET_MM="${PIPELINE_OVERLAY_Z_OFFSET_MM:-0.0}"
PIPELINE_OVERLAY_FULL_FIELD="${PIPELINE_OVERLAY_FULL_FIELD:-1}"
PIPELINE_WRITE_REGION_TABLE="${PIPELINE_WRITE_REGION_TABLE:-1}"
PIPELINE_REGION_PERCENTILE="${PIPELINE_REGION_PERCENTILE:-95.0}"
PIPELINE_OFFTARGET_THRESHOLD="${PIPELINE_OFFTARGET_THRESHOLD:-0.2}"
PIPELINE_MNI_BASELINE_ROOT="${PIPELINE_MNI_BASELINE_ROOT:-}"
PIPELINE_MNI_FIXED_ATLAS_PATH="${PIPELINE_MNI_FIXED_ATLAS_PATH:-}"
PIPELINE_NEIGHBOR_DILATION_ITER="${PIPELINE_NEIGHBOR_DILATION_ITER:-1}"
PIPELINE_CSF_LABELS="${PIPELINE_CSF_LABELS:-24}"
PIPELINE_SKULL_LABELS="${PIPELINE_SKULL_LABELS:-}"
PIPELINE_ELECTRODE_CSV="${PIPELINE_ELECTRODE_CSV:-}"
PIPELINE_ELECTRODE_DATASET_DIR="${PIPELINE_ELECTRODE_DATASET_DIR:-}"
PIPELINE_ELECTRODE_NAMES="${PIPELINE_ELECTRODE_NAMES:-}"
PIPELINE_EEG_POSITIONS_PATH_TEMPLATE="${PIPELINE_EEG_POSITIONS_PATH_TEMPLATE:-}"
PIPELINE_CAMCAN_TARGETS_CSV="${PIPELINE_CAMCAN_TARGETS_CSV:-}"
PIPELINE_EXPECTED_TARGETS_SHA256="${PIPELINE_EXPECTED_TARGETS_SHA256:-}"
PIPELINE_WRITE_NEIGHBOR_TABLE="${PIPELINE_WRITE_NEIGHBOR_TABLE:-1}"
PIPELINE_WRITE_NEIGHBOR_VISUALIZATION="${PIPELINE_WRITE_NEIGHBOR_VISUALIZATION:-1}"
PIPELINE_WRITE_ELECTRODE_TABLE="${PIPELINE_WRITE_ELECTRODE_TABLE:-1}"
PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
PIPELINE_VERBOSE="${PIPELINE_VERBOSE:-1}"

PIPELINE_POPULATION_ENABLED="${PIPELINE_POPULATION_ENABLED:-1}"
PIPELINE_POPULATION_OUT_DIR="${PIPELINE_POPULATION_OUT_DIR:-}"
PIPELINE_POPULATION_REGION_FILENAME="${PIPELINE_POPULATION_REGION_FILENAME:-region_stats_fastsurfer.csv}"
PIPELINE_POPULATION_METRICS_FILENAME="${PIPELINE_POPULATION_METRICS_FILENAME:-subject_metrics.json}"
PIPELINE_POPULATION_PEAK_THRESHOLD="${PIPELINE_POPULATION_PEAK_THRESHOLD:-0.2}"
PIPELINE_POPULATION_TARGET_ROI="${PIPELINE_POPULATION_TARGET_ROI:-}"
PIPELINE_POPULATION_TEMPLATE_REGION_CSV="${PIPELINE_POPULATION_TEMPLATE_REGION_CSV:-}"

PIPELINE_REPEATABILITY_ENABLED="${PIPELINE_REPEATABILITY_ENABLED:-1}"
PIPELINE_REPEATABILITY_OUTPUT_DIR="${PIPELINE_REPEATABILITY_OUTPUT_DIR:-}"
PIPELINE_REPEATABILITY_LOGS_ROOT="${PIPELINE_REPEATABILITY_LOGS_ROOT:-}"
PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY="${PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY:-1}"

PIPELINE_FIGURE_GENERATION_ENABLED="${PIPELINE_FIGURE_GENERATION_ENABLED:-1}"
PIPELINE_FIGURE_OUTPUT_DIR="${PIPELINE_FIGURE_OUTPUT_DIR:-}"

if [[ "${REPO_DIR}" != /* ]]; then
    REPO_DIR="${SCRIPT_DIR}/../${REPO_DIR}"
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
export PIPELINE_ELECTRODE_DATASET_DIR
export PIPELINE_ELECTRODE_NAMES
export PIPELINE_EEG_POSITIONS_PATH_TEMPLATE
export PIPELINE_CAMCAN_TARGETS_CSV
export PIPELINE_EXPECTED_TARGETS_SHA256
export PIPELINE_WRITE_NEIGHBOR_TABLE
export PIPELINE_WRITE_NEIGHBOR_VISUALIZATION
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
export PIPELINE_COMPLETE_REPEAT_SUBJECTS_ONLY
export PIPELINE_FIGURE_GENERATION_ENABLED
export PIPELINE_FIGURE_OUTPUT_DIR

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID:-<none>}"
echo "[INFO] Start: $(date)"
echo

export SLURM_EXPORT_ENV=ALL
module purge

activate_post_python() {
    if [[ -n "${POST_CONDA_ENV:-}" && -z "${POST_CONDA_SH:-}" ]]; then
        module load "${POST_ANACONDA_MODULE}" || true
        if ! command -v conda >/dev/null 2>&1; then
            echo "[ERROR] Failed to load Conda from module ${POST_ANACONDA_MODULE}." >&2
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
echo "[INFO] Atlas mode:          ${PIPELINE_ATLAS_MODE}"
echo "[INFO] FastSurfer root:     ${PIPELINE_FASTSURFER_ROOT:-<none>}"
echo "[INFO] Atlas filename:      ${PIPELINE_FASTSURFER_ATLAS_FILENAME:-<none>}"
echo "[INFO] Explicit atlas path: ${PIPELINE_FS_MRI_PATH:-<none>}"
echo "[INFO] MNI baseline root:   ${PIPELINE_MNI_BASELINE_ROOT:-<none>}"
echo "[INFO] Fixed MNI atlas:     ${PIPELINE_MNI_FIXED_ATLAS_PATH:-<none>}"
echo "[INFO] CamCan targets.csv:  ${PIPELINE_CAMCAN_TARGETS_CSV:-<disabled>}"
echo "[INFO] Expected target SHA: ${PIPELINE_EXPECTED_TARGETS_SHA256:-<none>}"
echo "[INFO] Population enabled:  ${PIPELINE_POPULATION_ENABLED}"
echo "[INFO] Repeatability:       ${PIPELINE_REPEATABILITY_ENABLED}"
echo "[INFO] Figure generation:   ${PIPELINE_FIGURE_GENERATION_ENABLED}"
echo "[INFO] Workers:             ${POST_MAX_WORKERS}"
echo

if [[ "${BATCH_ROOT}" == "/path/to/rootDIR" || -z "${BATCH_ROOT}" ]]; then
    echo "[ERROR] Set BATCH_ROOT to the parent directory containing repeat datasets." >&2
    exit 2
fi

if [[ "${PIPELINE_ATLAS_MODE}" == "fastsurfer" \
      && -z "${PIPELINE_FS_MRI_PATH}" \
      && -z "${PIPELINE_FASTSURFER_ATLAS_FILENAME}" \
      && -n "${PIPELINE_FASTSURFER_ROOT}" \
      && -n "${PIPELINE_SUBJECTS}" ]]; then
    missing_subject_atlases=()
    for subject in ${PIPELINE_SUBJECTS}; do
        if [[ ! -f "${PIPELINE_FASTSURFER_ROOT}/${subject}.nii" \
              && ! -f "${PIPELINE_FASTSURFER_ROOT}/${subject}.nii.gz" ]]; then
            missing_subject_atlases+=("${PIPELINE_FASTSURFER_ROOT}/${subject}.nii[.gz]")
        fi
    done

    if (( ${#missing_subject_atlases[@]} > 0 )); then
        echo "[ERROR] FastSurfer atlas mode needs a subject-space atlas for each subject." >&2
        echo "[ERROR] Missing atlas path(s):" >&2
        printf '  - %s\n' "${missing_subject_atlases[@]}" >&2
        echo "[ERROR] Do not point this at the raw MNI atlas; the atlas must be aligned to the subject T1/TI grid." >&2
        exit 2
    fi
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON}" >&2
    exit 2
fi

"${PYTHON}" - <<'PY'
import importlib.util
import os
from pathlib import Path

required = ["numpy", "pandas", "nibabel", "scipy", "nilearn", "matplotlib", "PIL"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    req_file = Path(os.environ.get("POST_REQUIREMENTS_FILE", "")).expanduser()
    lines = [f"Missing Python packages: {', '.join(missing)}"]
    if req_file.is_file():
        lines.append(f"Install them into your HPC env with: pip install -r {req_file}")
    raise SystemExit("\n".join(lines))

print("[INFO] Python dependency check passed.")
PY

"${PYTHON}" -u "${RUN_POST_BATCH_ENV_PY}"

echo
echo "[INFO] Finished: $(date)"
