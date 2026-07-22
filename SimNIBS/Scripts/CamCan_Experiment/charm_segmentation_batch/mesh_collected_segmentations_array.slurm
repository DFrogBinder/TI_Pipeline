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
WORKERS_PER_ARRAY_TASK="${TI_CHARM_MESH_WORKERS_PER_ARRAY_TASK:-1}"
THREADS_PER_WORKER="${TI_CHARM_MESH_THREADS_PER_WORKER:-${SLURM_CPUS_PER_TASK:-1}}"
LOCAL_STAGING="${TI_CHARM_MESH_LOCAL_STAGING:-0}"
TASKSET_BIN="${TASKSET_BIN:-taskset}"

for value_name in MAX_RETRIES WORKERS_PER_ARRAY_TASK THREADS_PER_WORKER; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${WORKERS_PER_ARRAY_TASK}" -lt 1 ] || [ "${THREADS_PER_WORKER}" -lt 1 ]; then
    echo "[ERROR] Worker and thread counts must be positive." >&2
    exit 2
fi
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ $((WORKERS_PER_ARRAY_TASK * THREADS_PER_WORKER)) -gt "${SLURM_CPUS_PER_TASK}" ]; then
    echo "[ERROR] Packed workers request more threads than SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}." >&2
    exit 2
fi
if [ "${LOCAL_STAGING}" != "0" ] && [ "${LOCAL_STAGING}" != "1" ]; then
    echo "[ERROR] TI_CHARM_MESH_LOCAL_STAGING must be 0 or 1." >&2
    exit 2
fi
if [ "${WORKERS_PER_ARRAY_TASK}" -gt 1 ] && ! command -v "${TASKSET_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Packed meshing requires taskset for per-worker CPU affinity." >&2
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
export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export TI_CHARM_MESH_THREADS_PER_WORKER="${THREADS_PER_WORKER}"

CPU_IDS=()
if [ "${WORKERS_PER_ARRAY_TASK}" -gt 1 ]; then
    read -r -a CPU_IDS <<< "$(python -c 'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
    REQUIRED_CPUS=$((WORKERS_PER_ARRAY_TASK * THREADS_PER_WORKER))
    if [ "${#CPU_IDS[@]}" -lt "${REQUIRED_CPUS}" ]; then
        echo "[ERROR] CPU affinity exposes ${#CPU_IDS[@]} CPUs, but packed workers require ${REQUIRED_CPUS}." >&2
        exit 2
    fi
fi

TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${TI_CHARM_MESH_MANIFEST}")
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
FIRST_TASK_INDEX=$((SHARD_INDEX * WORKERS_PER_ARRAY_TASK))
LAST_TASK_INDEX=$((FIRST_TASK_INDEX + WORKERS_PER_ARRAY_TASK - 1))
if [ "${FIRST_TASK_INDEX}" -ge "${TOTAL_TASKS}" ]; then
    echo "[ERROR] Array element ${SHARD_INDEX} starts beyond ${TOTAL_TASKS} manifest tasks." >&2
    exit 2
fi
if [ "${LAST_TASK_INDEX}" -ge "${TOTAL_TASKS}" ]; then
    LAST_TASK_INDEX=$((TOTAL_TASKS - 1))
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

echo "[INFO] Job ID:             ${SLURM_JOB_ID}"
echo "[INFO] Array element:      ${SHARD_INDEX}"
echo "[INFO] Manifest tasks:     ${FIRST_TASK_INDEX}-${LAST_TASK_INDEX}"
echo "[INFO] Worker processes:   ${WORKERS_PER_ARRAY_TASK} maximum"
echo "[INFO] CPUs allocated:     ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[INFO] Threads per worker: ${THREADS_PER_WORKER}"
echo "[INFO] Local staging:      ${LOCAL_STAGING}"
echo "[INFO] Operation:          direct CHARM create_mesh; no segmentation or simulation"
echo "[INFO] Retry:              ${RETRY_COUNT}/${MAX_RETRIES}"

run_one_task() {
    local task_index="$1"
    local task_line task_id subject label_path label_hash label_bytes mesh_path result_path status message
    local task_log staging_root worker_position cpu_offset cpu_list cpu_position
    local -a command launch_command

    task_line=$(sed -n "$((task_index + 2))p" "${TI_CHARM_MESH_MANIFEST}")
    IFS=$'\t' read -r task_id subject label_path label_hash label_bytes mesh_path result_path status message <<< "${task_line}"
    if [ -z "${task_line}" ] || [ -z "${subject:-}" ]; then
        echo "[ERROR] No valid manifest row for task index ${task_index}." >&2
        return 2
    fi
    if [ "${task_id}" != "${task_index}" ] || [ "${status}" != "ready" ]; then
        echo "[ERROR] Manifest row ${task_index} is not runnable: task=${task_id}, status=${status}, message=${message}." >&2
        return 2
    fi
    task_log="${TI_CHARM_MESH_LOG_DIR}/${subject}.log"
    command=(
        python -u "${TI_CHARM_MESH_WORKFLOW_PY}"
        run-task
        --manifest "${TI_CHARM_MESH_MANIFEST}"
        --task-index "${task_index}"
    )
    if [ -n "${SETTINGS_PATH}" ]; then
        command+=(--settings "${SETTINGS_PATH}")
    fi
    if [ "${LOCAL_STAGING}" = "1" ]; then
        staging_root="${SLURM_TMPDIR:-/tmp}/${USER}_mesh_charm_stage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        mkdir -p "${staging_root}"
        command+=(--staging-root "${staging_root}")
    fi
    launch_command=("${command[@]}")
    if [ "${WORKERS_PER_ARRAY_TASK}" -gt 1 ]; then
        worker_position=$((task_index - FIRST_TASK_INDEX))
        cpu_offset=$((worker_position * THREADS_PER_WORKER))
        cpu_list="${CPU_IDS[$cpu_offset]}"
        for ((cpu_position = 1; cpu_position < THREADS_PER_WORKER; cpu_position++)); do
            cpu_list+=",${CPU_IDS[$((cpu_offset + cpu_position))]}"
        done
        launch_command=("${TASKSET_BIN}" --cpu-list "${cpu_list}" "${command[@]}")
    else
        cpu_list="all allocated CPUs"
    fi

    {
        echo "[INFO] Manifest task:      ${task_index}"
        echo "[INFO] Subject:            ${subject}"
        echo "[INFO] Label:              ${label_path}"
        echo "[INFO] Label SHA-256:      ${label_hash}"
        echo "[INFO] Label bytes:        ${label_bytes}"
        echo "[INFO] Mesh:               ${mesh_path}"
        echo "[INFO] Result marker:      ${result_path}"
        echo "[INFO] Worker CPU set:      ${cpu_list}"
        "${launch_command[@]}"
    } 2>&1 | tee -a "${task_log}"
}

PIDS=()
TASK_INDICES=()
for ((task_index = FIRST_TASK_INDEX; task_index <= LAST_TASK_INDEX; task_index++)); do
    run_one_task "${task_index}" &
    PIDS+=("$!")
    TASK_INDICES+=("${task_index}")
done

RUN_EXIT=0
for position in "${!PIDS[@]}"; do
    if wait "${PIDS[$position]}"; then
        echo "[INFO] Manifest task ${TASK_INDICES[$position]} completed."
    else
        TASK_EXIT=$?
        echo "[ERROR] Manifest task ${TASK_INDICES[$position]} failed with exit ${TASK_EXIT}." >&2
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
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Mesh shard failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Mesh shard failed after ${RETRY_COUNT} retries; see per-subject logs under ${TI_CHARM_MESH_LOG_DIR}." >&2
exit "${RUN_EXIT}"
