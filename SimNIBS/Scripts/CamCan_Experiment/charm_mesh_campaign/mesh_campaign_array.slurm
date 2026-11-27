#!/bin/bash
#SBATCH --job-name=charm_mesh_campaign
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=charm-mesh-%A_%a.out
#SBATCH --requeue

set -uo pipefail

: "${SLURM_ARRAY_TASK_ID:?Submit this worker as a Slurm array.}"
: "${TI_CHARM_MESH_CONFIG:?Missing TI_CHARM_MESH_CONFIG.}"
: "${TI_CHARM_MESH_PACKAGE_ROOT:?Missing TI_CHARM_MESH_PACKAGE_ROOT.}"
: "${TI_CHARM_MESH_SIMNIBS_MODULE:?Missing TI_CHARM_MESH_SIMNIBS_MODULE.}"
: "${TI_CHARM_MESH_PYTHON:?Missing TI_CHARM_MESH_PYTHON.}"
: "${TI_CHARM_MESH_TASKS:?Missing TI_CHARM_MESH_TASKS.}"

MAX_RETRIES="${TI_CHARM_MESH_MAX_RETRIES:-2}"
WORKERS="${TI_CHARM_MESH_WORKERS_PER_ARRAY_TASK:-1}"
THREADS="${TI_CHARM_MESH_THREADS_PER_WORKER:-${SLURM_CPUS_PER_TASK:-1}}"
LOCAL_STAGING="${TI_CHARM_MESH_LOCAL_STAGING:-0}"
LOG_ROOT="${TI_CHARM_MESH_LOG_ROOT:-$(dirname "${TI_CHARM_MESH_CONFIG}")/logs}"
STATE_ROOT="${LOG_ROOT}/retry_state"
mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"

module purge
module use "$HOME/modules"
module load "${TI_CHARM_MESH_SIMNIBS_MODULE}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${TI_CHARM_MESH_PACKAGE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TI_CHARM_MESH_THREADS_PER_WORKER="${THREADS}"

FIRST_TASK=$((SLURM_ARRAY_TASK_ID * WORKERS))
LAST_TASK=$((FIRST_TASK + WORKERS - 1))
if [ "${LAST_TASK}" -ge "${TI_CHARM_MESH_TASKS}" ]; then
    LAST_TASK=$((TI_CHARM_MESH_TASKS - 1))
fi
if [ "${FIRST_TASK}" -gt "${LAST_TASK}" ]; then
    echo "[ERROR] Array element starts beyond configured tasks." >&2
    exit 2
fi

TASK_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="${STATE_ROOT}/${TASK_KEY}.retry"
RETRY_COUNT=0
if [ -f "${RETRY_FILE}" ]; then
    RETRY_COUNT="$(tr -d '[:space:]' < "${RETRY_FILE}")"
fi

run_one() {
    local task_index="$1"
    local -a command
    command=(
        "${TI_CHARM_MESH_PYTHON}" -u -m charm_mesh_campaign.cli run-task
        --config "${TI_CHARM_MESH_CONFIG}"
        --task-index "${task_index}"
    )
    if [ "${LOCAL_STAGING}" = "1" ]; then
        local staging_root="${SLURM_TMPDIR:-/tmp}/${USER}_charm_mesh_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        mkdir -p "${staging_root}"
        command+=(--staging-root "${staging_root}")
    fi
    "${command[@]}" 2>&1 | tee -a "${LOG_ROOT}/task-${task_index}.log"
}

PIDS=()
for ((task_index = FIRST_TASK; task_index <= LAST_TASK; task_index++)); do
    run_one "${task_index}" &
    PIDS+=("$!")
done
RUN_EXIT=0
for pid in "${PIDS[@]}"; do
    if wait "${pid}"; then
        :
    else
        TASK_EXIT=$?
        if [ "${RUN_EXIT}" -eq 0 ]; then
            RUN_EXIT="${TASK_EXIT}"
        fi
    fi
done
if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi
if [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    printf '%s\n' "$((RETRY_COUNT + 1))" > "${RETRY_FILE}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi
exit "${RUN_EXIT}"
