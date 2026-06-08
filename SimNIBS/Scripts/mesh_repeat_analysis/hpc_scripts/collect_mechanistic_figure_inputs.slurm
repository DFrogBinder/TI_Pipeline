#!/bin/bash
#SBATCH --job-name=collect_mech_fig_inputs
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=collect_mech_fig_inputs_%j.out
#SBATCH --error=collect_mech_fig_inputs_%j.err

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

set -euo pipefail

# ---------------------------------------------------------------------------
# Mechanistic figure input collection settings
#
# Edit this block once on the HPC, then submit with:
#
#   sbatch hpc_scripts/collect_mechanistic_figure_inputs.slurm
#
# To write manifests without copying files:
#
#   sbatch hpc_scripts/collect_mechanistic_figure_inputs.slurm --dry-run
#
# Blank optional values are ignored. Environment variables with the same names
# still override these defaults, which is useful for one-off smoke tests.
# ---------------------------------------------------------------------------
PIPELINE_DIR_CONFIG="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis"
BASE_OUTPUT_DIR_CONFIG="/mnt/parscratch/users/cop23bi/balaned-repeatability-outputs"
OUT_DIR_CONFIG="/mnt/parscratch/users/cop23bi/mechanistic_figure_bundle"

# Runtime controls.
COPY_MODE_CONFIG="copy"
THRESHOLD_CONFIG="0.2"
MAX_TOTAL_BYTES_CONFIG="5000000000"
ALLOW_LARGE_CONFIG="0"

# Optional selectors. Leave COVERAGE_CSV_CONFIG blank to compute coverage from
# roi_mask_on_reference_ti.nii.gz and remesh ti_brain_only.nii.gz files.
COVERAGE_CSV_CONFIG=""
PREFERRED_SUBJECT_CONFIG=""
FIXED_REFERENCE_REPEAT_CONFIG="repeat_001"

# Python/runtime paths. The SimNIBS module above should provide nibabel/numpy.
# Set VENV_PATH_CONFIG only if you intentionally want to activate a separate
# Python virtual environment before running the collector.
PYTHON_BIN_CONFIG="python"
VENV_PATH_CONFIG=""
COLLECTOR_CONFIG=""

# Logs. Leave LOG_DIR_CONFIG blank to use <pipeline>/logs.
LOG_DIR_CONFIG=""
LOG_FILE_CONFIG=""

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${BASE_OUTPUT_DIR_CONFIG}}"
OUT_DIR="${OUT_DIR:-${OUT_DIR_CONFIG}}"
COPY_MODE="${COPY_MODE:-${COPY_MODE_CONFIG}}"
THRESHOLD="${THRESHOLD:-${THRESHOLD_CONFIG}}"
MAX_TOTAL_BYTES="${MAX_TOTAL_BYTES:-${MAX_TOTAL_BYTES_CONFIG}}"
ALLOW_LARGE="${ALLOW_LARGE:-${ALLOW_LARGE_CONFIG}}"
COVERAGE_CSV="${COVERAGE_CSV:-${COVERAGE_CSV_CONFIG}}"
PREFERRED_SUBJECT="${PREFERRED_SUBJECT:-${PREFERRED_SUBJECT_CONFIG}}"
FIXED_REFERENCE_REPEAT="${FIXED_REFERENCE_REPEAT:-${FIXED_REFERENCE_REPEAT_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_CONFIG}}"
VENV_PATH="${VENV_PATH:-${VENV_PATH_CONFIG}}"
COLLECTOR="${COLLECTOR:-${COLLECTOR_CONFIG:-${PIPELINE_DIR}/post/collect_mechanistic_figure_inputs.py}}"

usage() {
    echo "Usage: sbatch hpc_scripts/collect_mechanistic_figure_inputs.slurm [--dry-run]"
}

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            COPY_MODE="dry-run"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $arg"
            usage
            exit 1
            ;;
    esac
done

case "${COPY_MODE}" in
    copy|dry-run) ;;
    *)
        echo "[ERROR] COPY_MODE must be copy or dry-run: ${COPY_MODE}"
        exit 1
        ;;
esac

case "${ALLOW_LARGE}" in
    0|1) ;;
    *)
        echo "[ERROR] ALLOW_LARGE must be 0 or 1: ${ALLOW_LARGE}"
        exit 1
        ;;
esac

if ! [[ "${MAX_TOTAL_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MAX_TOTAL_BYTES must be a non-negative integer: ${MAX_TOTAL_BYTES}"
    exit 1
fi

if [ -z "${PIPELINE_DIR}" ]; then
    echo "[ERROR] PIPELINE_DIR is empty. Edit PIPELINE_DIR_CONFIG near the top of this script."
    exit 1
fi
if [ ! -d "${PIPELINE_DIR}" ]; then
    echo "[ERROR] Pipeline directory does not exist: ${PIPELINE_DIR}"
    exit 1
fi
if [ -z "${BASE_OUTPUT_DIR}" ]; then
    echo "[ERROR] BASE_OUTPUT_DIR is empty. Edit BASE_OUTPUT_DIR_CONFIG near the top of this script."
    exit 1
fi
if [ ! -d "${BASE_OUTPUT_DIR}" ]; then
    echo "[ERROR] Base output directory does not exist: ${BASE_OUTPUT_DIR}"
    exit 1
fi
if [ -z "${OUT_DIR}" ]; then
    echo "[ERROR] OUT_DIR is empty. Edit OUT_DIR_CONFIG near the top of this script."
    exit 1
fi
if [ ! -f "${COLLECTOR}" ]; then
    echo "[ERROR] Collector script does not exist: ${COLLECTOR}"
    exit 1
fi
if [ -n "${COVERAGE_CSV}" ] && [ ! -f "${COVERAGE_CSV}" ]; then
    echo "[ERROR] Coverage CSV does not exist: ${COVERAGE_CSV}"
    exit 1
fi

if [ -n "${VENV_PATH}" ]; then
    if [ ! -f "${VENV_PATH}/bin/activate" ]; then
        echo "[ERROR] VENV_PATH does not contain bin/activate: ${VENV_PATH}"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

cd "${PIPELINE_DIR}"

LOG_DIR="${LOG_DIR:-${LOG_DIR_CONFIG:-${PIPELINE_DIR}/logs}}"
mkdir -p "${LOG_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="${LOG_FILE:-${LOG_FILE_CONFIG:-${LOG_DIR}/collect_mechanistic_figure_inputs_${JOB_ID}.log}}"

echo "[INFO] Mechanistic figure input collection"
echo "[INFO] Job ID:                 ${JOB_ID}"
echo "[INFO] Host:                   $(hostname)"
echo "[INFO] Submit dir:             ${SLURM_SUBMIT_DIR:-n/a}"
echo "[INFO] Pipeline root:          ${PIPELINE_DIR}"
echo "[INFO] Collector:              ${COLLECTOR}"
echo "[INFO] Python:                 ${PYTHON_BIN}"
echo "[INFO] Virtualenv:             ${VENV_PATH:-module environment only}"
echo "[INFO] CPUs:                   ${SLURM_CPUS_PER_TASK:-1}"
echo "[INFO] Log file:               ${LOG_FILE}"
echo "[INFO] Base output dir:        ${BASE_OUTPUT_DIR}"
echo "[INFO] Output dir:             ${OUT_DIR}"
echo "[INFO] Copy mode:              ${COPY_MODE}"
echo "[INFO] Threshold:              ${THRESHOLD}"
echo "[INFO] Max total bytes:        ${MAX_TOTAL_BYTES}"
echo "[INFO] Allow large bundle:     ${ALLOW_LARGE}"
echo "[INFO] Coverage CSV:           ${COVERAGE_CSV:-compute from TI volumes}"
echo "[INFO] Preferred subject:      ${PREFERRED_SUBJECT:-auto-select largest coverage range}"
echo "[INFO] Fixed reference repeat: ${FIXED_REFERENCE_REPEAT}"

if [ -z "${COVERAGE_CSV}" ]; then
    echo "[INFO] Checking NIfTI Python dependencies before computing coverage."
    "${PYTHON_BIN}" -u "${COLLECTOR}" --check-nifti-dependencies
fi

PYTHON_CMD=(
    "${PYTHON_BIN}"
    -u
    "${COLLECTOR}"
    --base-output-dir "${BASE_OUTPUT_DIR}"
    --out-dir "${OUT_DIR}"
    --threshold "${THRESHOLD}"
    --max-total-bytes "${MAX_TOTAL_BYTES}"
    --fixed-reference-repeat "${FIXED_REFERENCE_REPEAT}"
)

if [ -n "${COVERAGE_CSV}" ]; then
    PYTHON_CMD+=(--coverage-csv "${COVERAGE_CSV}")
fi
if [ -n "${PREFERRED_SUBJECT}" ]; then
    PYTHON_CMD+=(--preferred-subject "${PREFERRED_SUBJECT}")
fi
if [ "${ALLOW_LARGE}" = "1" ]; then
    PYTHON_CMD+=(--allow-large)
fi
if [ "${COPY_MODE}" = "copy" ]; then
    PYTHON_CMD+=(--copy)
else
    PYTHON_CMD+=(--dry-run)
fi

echo "[INFO] Command:"
printf '  %q' "${PYTHON_CMD[@]}"
printf '\n'

"${PYTHON_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
