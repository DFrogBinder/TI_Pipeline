#!/bin/bash
#SBATCH --job-name=charm_only_remesh
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=charm-only-remesh-%A_%a.out
#SBATCH --requeue

set -uo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array." >&2
    exit 1
fi

: "${TI_CHARM_REMESH_MANIFEST:?Missing TI_CHARM_REMESH_MANIFEST.}"
: "${TI_CHARM_REMESH_RESULT_DIR:?Missing TI_CHARM_REMESH_RESULT_DIR.}"
: "${TI_CHARM_REMESH_WORKFLOW_PY:?Missing TI_CHARM_REMESH_WORKFLOW_PY.}"

TASK_OFFSET="${TASK_OFFSET:-0}"
MAX_RETRIES="${TI_CHARM_REMESH_MAX_RETRIES:-2}"
LOG_DIR="${TI_CHARM_REMESH_LOG_DIR:-${TI_CHARM_REMESH_RESULT_DIR}/logs}"
STATE_DIR="${TI_CHARM_REMESH_STATE_DIR:-${TI_CHARM_REMESH_RESULT_DIR}/retry_state}"

if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TASK_OFFSET must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_CHARM_REMESH_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi
if [ ! -f "${TI_CHARM_REMESH_MANIFEST}" ] || [ ! -f "${TI_CHARM_REMESH_WORKFLOW_PY}" ]; then
    echo "[ERROR] Remesh manifest or workflow script is missing." >&2
    exit 2
fi

TASK_INDEX=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
mkdir -p "${LOG_DIR}" "${STATE_DIR}" "${TI_CHARM_REMESH_RESULT_DIR}"

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_charm_remesh_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

TASK_LINE=$(sed -n "$((TASK_INDEX + 2))p" "${TI_CHARM_REMESH_MANIFEST}")
IFS=$'\t' read -r TASK_ID DATASET_NAME ROI_PREFIX REPEAT_ID DATASET_ROOT SUBJECT ANAT_DIR M2M_DIR LABEL_PATH SOURCE_LABEL_PATH LABEL_HASH MESH_PATH STATUS MESSAGE SOURCE_LABEL_MODE <<< "${TASK_LINE}"
if [ -z "${TASK_LINE}" ] || [ -z "${SUBJECT:-}" ]; then
    echo "[ERROR] No valid manifest row for task index ${TASK_INDEX}." >&2
    exit 2
fi
SAFE_DATASET="${DATASET_NAME//[^A-Za-z0-9_.-]/_}"
TASK_LOG="${LOG_DIR}/${SAFE_DATASET}__${SUBJECT}.log"

echo "[INFO] Job ID:          ${SLURM_JOB_ID}"
echo "[INFO] Global index:    ${TASK_INDEX}"
echo "[INFO] Dataset:         ${DATASET_NAME}"
echo "[INFO] Subject:         ${SUBJECT}"
echo "[INFO] CHARM label:     ${LABEL_PATH}"
echo "[INFO] Source label:    ${SOURCE_LABEL_PATH}"
echo "[INFO] Source mode:     ${SOURCE_LABEL_MODE:-external_verified}"
echo "[INFO] Label SHA-256:   ${LABEL_HASH}"
echo "[INFO] Operation:       charm ${SUBJECT} --mesh"
echo "[INFO] Retry:           ${RETRY_COUNT}/${MAX_RETRIES}"

if [ "${STATUS}" != "ready" ]; then
    echo "[ERROR] Manifest task is not ready: ${MESSAGE}" >&2
    exit 2
fi

set +e
python -u "${TI_CHARM_REMESH_WORKFLOW_PY}" run-task \
    --manifest "${TI_CHARM_REMESH_MANIFEST}" \
    --task-index "${TASK_INDEX}" \
    --result-dir "${TI_CHARM_REMESH_RESULT_DIR}" 2>&1 | tee -a "${TASK_LOG}"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi

if [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Remesh failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Remesh failed after ${RETRY_COUNT} retries; see ${TASK_LOG}." >&2
exit "${RUN_EXIT}"
