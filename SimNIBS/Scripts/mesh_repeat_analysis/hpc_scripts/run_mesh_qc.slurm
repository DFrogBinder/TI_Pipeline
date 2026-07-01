#!/bin/bash
#SBATCH --job-name=mesh_qc
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=mesh_qc_%j.out
#SBATCH --error=mesh_qc_%j.err

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1

# The QC/render code parallelizes across processes. Keep each worker effectively
# single-threaded so BLAS/OpenMP does not multiply CPU usage inside the pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

# ---------------------------------------------------------------------------
# Mesh QC batch settings
#
# Edit this block once on the HPC, then submit with:
#
#   sbatch hpc_scripts/run_mesh_qc.slurm
#
# Blank optional values are ignored. Environment variables with the same names
# still override these defaults for one-off runs.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR_CONFIG="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MESH_QC_ROOT_CONFIG="/mnt/parscratch/users/cop23bi/ZIPs/Analised-Data"
MESH_QC_OUT_CONFIG="/mnt/parscratch/users/cop23bi/mesh-wall"
MESH_GLOB_CONFIG=""
RENDERER_CONFIG="auto"
CHECK_COMPONENTS_CONFIG="0"
ROI_WALLS_CONFIG="0"
PROGRESS_CONFIG="tqdm"
WORKERS_CONFIG="0"
MESH_QC_STAGE_CONFIG="full"
IMAGE_SIZE_CONFIG="1200"
TILE_SIZE_CONFIG="220"
COLS_CONFIG=""
DISCOVERY_PROGRESS_SECONDS_CONFIG="5"
PROGRESS_EVERY_CONFIG="25"

# Logs. Leave LOG_DIR_CONFIG blank to use <pipeline>/logs.
LOG_DIR_CONFIG=""
LOG_FILE_CONFIG=""

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
MESH_QC_ROOT="${MESH_QC_ROOT:-${MESH_QC_ROOT_CONFIG}}"
MESH_QC_OUT="${MESH_QC_OUT:-${MESH_QC_OUT_CONFIG}}"
MESH_GLOB="${MESH_GLOB:-${MESH_GLOB_CONFIG}}"
RENDERER="${RENDERER:-${RENDERER_CONFIG}}"
CHECK_COMPONENTS="${CHECK_COMPONENTS:-${CHECK_COMPONENTS_CONFIG}}"
ROI_WALLS="${ROI_WALLS:-${ROI_WALLS_CONFIG}}"
PROGRESS_MODE="${PROGRESS_MODE:-${PROGRESS_CONFIG}}"
WORKERS="${WORKERS:-${WORKERS_CONFIG}}"
MESH_QC_STAGE="${MESH_QC_STAGE:-${MESH_QC_STAGE_CONFIG}}"
IMAGE_SIZE="${IMAGE_SIZE:-${IMAGE_SIZE_CONFIG}}"
TILE_SIZE="${TILE_SIZE:-${TILE_SIZE_CONFIG}}"
COLS="${COLS:-${COLS_CONFIG}}"
DISCOVERY_PROGRESS_SECONDS="${DISCOVERY_PROGRESS_SECONDS:-${DISCOVERY_PROGRESS_SECONDS_CONFIG}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-${PROGRESS_EVERY_CONFIG}}"

if [ -z "${PIPELINE_DIR}" ] || [ ! -d "${PIPELINE_DIR}" ]; then
    echo "[ERROR] Pipeline directory does not exist: ${PIPELINE_DIR}" >&2
    exit 1
fi
if [ -z "${MESH_QC_ROOT}" ] || [ ! -d "${MESH_QC_ROOT}" ]; then
    echo "[ERROR] Mesh QC root does not exist: ${MESH_QC_ROOT}" >&2
    exit 1
fi
if [ -z "${MESH_QC_OUT}" ]; then
    echo "[ERROR] Mesh QC output directory is empty." >&2
    exit 1
fi

cd "${PIPELINE_DIR}"

LOG_DIR="${LOG_DIR:-${LOG_DIR_CONFIG:-${PIPELINE_DIR}/logs}}"
mkdir -p "${LOG_DIR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="${LOG_FILE:-${LOG_FILE_CONFIG:-${LOG_DIR}/mesh_qc_${JOB_ID}.log}}"

echo "[INFO] Mesh QC batch run"
echo "[INFO] Job ID:         ${JOB_ID}"
echo "[INFO] Host:           $(hostname)"
echo "[INFO] Submit dir:     ${SLURM_SUBMIT_DIR:-n/a}"
echo "[INFO] Pipeline root:  ${PIPELINE_DIR}"
echo "[INFO] Mesh root:      ${MESH_QC_ROOT}"
echo "[INFO] Output dir:     ${MESH_QC_OUT}"
echo "[INFO] CPUs alloc:     ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[INFO] Workers:        ${WORKERS}"
echo "[INFO] Stage:          ${MESH_QC_STAGE}"
echo "[INFO] Renderer:       ${RENDERER}"
echo "[INFO] Progress mode:  ${PROGRESS_MODE}"
echo "[INFO] Check comps:    ${CHECK_COMPONENTS}"
echo "[INFO] ROI walls:      ${ROI_WALLS}"
echo "[INFO] Log file:       ${LOG_FILE}"

PYTHON_CMD=(
    python
    -u
    "${PIPELINE_DIR}/mesh_repeat_analysis/post/mesh_qc/run_mesh_qc.py"
    --root "${MESH_QC_ROOT}"
    --out "${MESH_QC_OUT}"
    --renderer "${RENDERER}"
    --workers "${WORKERS}"
    --progress "${PROGRESS_MODE}"
    --image-size "${IMAGE_SIZE}"
    --tile-size "${TILE_SIZE}"
    --discovery-progress-seconds "${DISCOVERY_PROGRESS_SECONDS}"
    --progress-every "${PROGRESS_EVERY}"
)

if [ -n "${MESH_GLOB}" ]; then
    PYTHON_CMD+=(--mesh-glob "${MESH_GLOB}")
fi
if [ -n "${COLS}" ]; then
    PYTHON_CMD+=(--cols "${COLS}")
fi
if [ "${CHECK_COMPONENTS}" = "1" ]; then
    PYTHON_CMD+=(--check-components)
fi
if [ "${ROI_WALLS}" = "1" ]; then
    PYTHON_CMD+=(--roi-walls)
fi
case "${MESH_QC_STAGE}" in
    full)
        ;;
    qc)
        PYTHON_CMD+=(--qc-only)
        ;;
    render)
        PYTHON_CMD+=(--render-only)
        ;;
    *)
        echo "[ERROR] Unsupported MESH_QC_STAGE: ${MESH_QC_STAGE}" >&2
        echo "[ERROR] Use one of: full, qc, render" >&2
        exit 1
        ;;
esac

echo "[INFO] Command:"
printf '  %q' "${PYTHON_CMD[@]}"
printf '\n'

"${PYTHON_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
