#!/bin/bash
#SBATCH --job-name=mesh_charm_maps
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=mesh-collected-charm-%A_%a.out
#SBATCH --requeue

set -uo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array." >&2
    exit 1
fi

: "${TI_CHARM_MESH_MANIFEST:?Missing TI_CHARM_MESH_MANIFEST.}"
: "${TI_CHARM_MESH_WORKFLOW_PY:?Missing TI_CHARM_MESH_WORKFLOW_PY.}"
: "${TI_CHARM_MESH_LOG_DIR:?Missing TI_CHARM_MESH_LOG_DIR.}"

MAX_RETRIES="${TI_CHARM_MESH_MAX_RETRIES:-2}"
STATE_DIR="${TI_CHARM_MESH_STATE_DIR:-${TI_CHARM_MESH_LOG_DIR}/retry_state}"
SETTINGS_PATH="${TI_CHARM_MESH_SETTINGS:-}"

if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_CHARM_MESH_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi
if [ ! -f "${TI_CHARM_MESH_MANIFEST}" ] || [ ! -f "${TI_CHARM_MESH_WORKFLOW_PY}" ]; then
    echo "[ERROR] Mesh manifest or workflow script is missing." >&2
    exit 2
fi

mkdir -p "${TI_CHARM_MESH_LOG_DIR}" "${STATE_DIR}"

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_mesh_charm_maps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SHIM_DIR}"
cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass
EOF
export PYTHONPATH="${SHIM_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
TASK_LINE=$(sed -n "$((TASK_INDEX + 2))p" "${TI_CHARM_MESH_MANIFEST}")
IFS=$'\t' read -r TASK_ID SUBJECT LABEL_PATH LABEL_HASH LABEL_BYTES MESH_PATH RESULT_PATH STATUS MESSAGE <<< "${TASK_LINE}"
if [ -z "${TASK_LINE}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "[ERROR] No valid manifest row for task index ${TASK_INDEX}." >&2
    exit 2
fi
if [ "${TASK_ID}" != "${TASK_INDEX}" ] || [ "${STATUS}" != "ready" ]; then
    echo "[ERROR] Manifest row ${TASK_INDEX} is not runnable: task=${TASK_ID}, status=${STATUS}, message=${MESSAGE}." >&2
    exit 2
fi

TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_DIR}/${TASK_KEY}.retry"
if [ -f "${RETRY_FILE}" ]; then
    RETRY_COUNT=$(tr -d '[:space:]' < "${RETRY_FILE}")
else
    RETRY_COUNT=0
fi
if ! [[ "${RETRY_COUNT}" =~ ^[0-9]+$ ]]; then
    RETRY_COUNT=0
fi
TASK_LOG="${TI_CHARM_MESH_LOG_DIR}/${SUBJECT}.log"

echo "[INFO] Job ID:        ${SLURM_JOB_ID}"
echo "[INFO] Array index:   ${TASK_INDEX}"
echo "[INFO] Subject:       ${SUBJECT}"
echo "[INFO] Label:         ${LABEL_PATH}"
echo "[INFO] Label SHA-256: ${LABEL_HASH}"
echo "[INFO] Mesh:          ${MESH_PATH}"
echo "[INFO] Operation:     direct CHARM create_mesh; no segmentation or simulation"
echo "[INFO] Retry:         ${RETRY_COUNT}/${MAX_RETRIES}"

COMMAND=(
    python -u "${TI_CHARM_MESH_WORKFLOW_PY}"
    run-task
    --manifest "${TI_CHARM_MESH_MANIFEST}"
    --task-index "${TASK_INDEX}"
)
if [ -n "${SETTINGS_PATH}" ]; then
    COMMAND+=(--settings "${SETTINGS_PATH}")
fi

set +e
"${COMMAND[@]}" 2>&1 | tee -a "${TASK_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi

if [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Mesh task failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Mesh task failed after ${RETRY_COUNT} retries; see ${TASK_LOG}." >&2
exit "${RUN_EXIT}"
