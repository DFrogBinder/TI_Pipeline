#!/bin/bash
#SBATCH --job-name=charm_seg_cleanup
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=charm-seg-cleanup-%A_%a.out
#SBATCH --requeue

set -uo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array." >&2
    exit 2
fi

: "${TI_CHARM_CLEANUP_MANIFEST:?Missing TI_CHARM_CLEANUP_MANIFEST.}"
: "${TI_CHARM_CLEANUP_WORKFLOW_PY:?Missing TI_CHARM_CLEANUP_WORKFLOW_PY.}"
: "${TI_CHARM_CLEANUP_LOG_DIR:?Missing TI_CHARM_CLEANUP_LOG_DIR.}"

MAX_RETRIES="${TI_CHARM_CLEANUP_MAX_RETRIES:-2}"
STATE_DIR="${TI_CHARM_CLEANUP_STATE_DIR:-${TI_CHARM_CLEANUP_LOG_DIR}/retry_state}"
CSF_RADIUS="${TI_CHARM_CLEANUP_CSF_RADIUS:-5}"
SKIN_RADIUS="${TI_CHARM_CLEANUP_SKIN_RADIUS:-10}"
CSF_COMPONENT_POLICY="${TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY:-none}"
COMPONENT_POLICY="${TI_CHARM_CLEANUP_COMPONENT_POLICY:-largest}"
WM_MIN_COMPONENT_VOXELS="${TI_CHARM_CLEANUP_WM_MIN_COMPONENT_VOXELS:-0}"
GM_MIN_COMPONENT_VOXELS="${TI_CHARM_CLEANUP_GM_MIN_COMPONENT_VOXELS:-0}"
CONNECTIVITY="${TI_CHARM_CLEANUP_CONNECTIVITY:-26}"

for value_name in MAX_RETRIES CSF_RADIUS SKIN_RADIUS WM_MIN_COMPONENT_VOXELS GM_MIN_COMPONENT_VOXELS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${COMPONENT_POLICY}" != "largest" ] && [ "${COMPONENT_POLICY}" != "min-size" ]; then
    echo "[ERROR] COMPONENT_POLICY must be largest or min-size." >&2
    exit 2
fi
if [ "${CSF_COMPONENT_POLICY}" != "none" ] && [ "${CSF_COMPONENT_POLICY}" != "largest" ]; then
    echo "[ERROR] CSF_COMPONENT_POLICY must be none or largest." >&2
    exit 2
fi
if [ "${CONNECTIVITY}" != "6" ] && [ "${CONNECTIVITY}" != "18" ] && [ "${CONNECTIVITY}" != "26" ]; then
    echo "[ERROR] CONNECTIVITY must be 6, 18, or 26." >&2
    exit 2
fi
if [ ! -f "${TI_CHARM_CLEANUP_MANIFEST}" ] || [ ! -f "${TI_CHARM_CLEANUP_WORKFLOW_PY}" ]; then
    echo "[ERROR] Cleanup manifest or workflow is missing." >&2
    exit 2
fi

mkdir -p "${TI_CHARM_CLEANUP_LOG_DIR}" "${STATE_DIR}"

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
TASK_LINE=$(sed -n "$((TASK_INDEX + 2))p" "${TI_CHARM_CLEANUP_MANIFEST}")
IFS=$'\t' read -r TASK_ID SUBJECT SOURCE_MAP SOURCE_HASH SOURCE_BYTES CORRECTED_MAP RESULT_PATH STATUS MESSAGE <<< "${TASK_LINE}"
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
TASK_LOG="${TI_CHARM_CLEANUP_LOG_DIR}/${SUBJECT}.log"

echo "[INFO] Job ID:          ${SLURM_JOB_ID}"
echo "[INFO] Array index:     ${TASK_INDEX}"
echo "[INFO] Subject:         ${SUBJECT}"
echo "[INFO] Source map:      ${SOURCE_MAP}"
echo "[INFO] Source SHA-256:  ${SOURCE_HASH}"
echo "[INFO] Source bytes:    ${SOURCE_BYTES}"
echo "[INFO] Corrected map:   ${CORRECTED_MAP}"
echo "[INFO] Result marker:   ${RESULT_PATH}"
echo "[INFO] CSF radius:      ${CSF_RADIUS} voxels"
echo "[INFO] CSF components:  ${CSF_COMPONENT_POLICY}"
echo "[INFO] Skin radius:     ${SKIN_RADIUS} voxels"
echo "[INFO] Component mode:  ${COMPONENT_POLICY}"
echo "[INFO] Connectivity:    ${CONNECTIVITY}"
echo "[INFO] Retry:           ${RETRY_COUNT}/${MAX_RETRIES}"
echo "[INFO] Source mutation: forbidden"

COMMAND=(
    python -u "${TI_CHARM_CLEANUP_WORKFLOW_PY}"
    run-task
    --manifest "${TI_CHARM_CLEANUP_MANIFEST}"
    --task-index "${TASK_INDEX}"
    --csf-radius "${CSF_RADIUS}"
    --skin-radius "${SKIN_RADIUS}"
    --csf-component-policy "${CSF_COMPONENT_POLICY}"
    --component-policy "${COMPONENT_POLICY}"
    --wm-min-component-voxels "${WM_MIN_COMPONENT_VOXELS}"
    --gm-min-component-voxels "${GM_MIN_COMPONENT_VOXELS}"
    --connectivity "${CONNECTIVITY}"
)

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
    echo "[WARN] Cleanup failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Cleanup failed after ${RETRY_COUNT} retries; see ${TASK_LOG}." >&2
exit "${RUN_EXIT}"
