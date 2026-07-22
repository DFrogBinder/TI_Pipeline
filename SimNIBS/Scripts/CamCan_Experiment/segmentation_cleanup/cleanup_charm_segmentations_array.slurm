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
WORKERS_PER_ARRAY_TASK="${TI_CHARM_CLEANUP_WORKERS_PER_ARRAY_TASK:-1}"
CSF_RADIUS="${TI_CHARM_CLEANUP_CSF_RADIUS:-5}"
CSF_OPENING_RADIUS="${TI_CHARM_CLEANUP_CSF_OPENING_RADIUS:-0}"
SKIN_RADIUS="${TI_CHARM_CLEANUP_SKIN_RADIUS:-10}"
CSF_INCLUDE_BLOOD="${TI_CHARM_CLEANUP_CSF_INCLUDE_BLOOD:-0}"
CSF_COMPONENT_POLICY="${TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY:-none}"
COMPONENT_POLICY="${TI_CHARM_CLEANUP_COMPONENT_POLICY:-largest}"
WM_MIN_COMPONENT_VOXELS="${TI_CHARM_CLEANUP_WM_MIN_COMPONENT_VOXELS:-0}"
GM_MIN_COMPONENT_VOXELS="${TI_CHARM_CLEANUP_GM_MIN_COMPONENT_VOXELS:-0}"
CONNECTIVITY="${TI_CHARM_CLEANUP_CONNECTIVITY:-26}"

for value_name in MAX_RETRIES WORKERS_PER_ARRAY_TASK CSF_RADIUS CSF_OPENING_RADIUS SKIN_RADIUS WM_MIN_COMPONENT_VOXELS GM_MIN_COMPONENT_VOXELS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${WORKERS_PER_ARRAY_TASK}" -lt 1 ]; then
    echo "[ERROR] WORKERS_PER_ARRAY_TASK must be positive." >&2
    exit 2
fi
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "${WORKERS_PER_ARRAY_TASK}" -gt "${SLURM_CPUS_PER_TASK}" ]; then
    echo "[ERROR] ${WORKERS_PER_ARRAY_TASK} workers exceed SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}." >&2
    exit 2
fi
if [ "${COMPONENT_POLICY}" != "largest" ] && [ "${COMPONENT_POLICY}" != "min-size" ]; then
    echo "[ERROR] COMPONENT_POLICY must be largest or min-size." >&2
    exit 2
fi
if [ "${CSF_COMPONENT_POLICY}" != "none" ] && [ "${CSF_COMPONENT_POLICY}" != "largest" ]; then
    echo "[ERROR] CSF_COMPONENT_POLICY must be none or largest." >&2
    exit 2
fi
if [ "${CSF_INCLUDE_BLOOD}" != "0" ] && [ "${CSF_INCLUDE_BLOOD}" != "1" ]; then
    echo "[ERROR] CSF_INCLUDE_BLOOD must be 0 or 1." >&2
    exit 2
fi
if [ "${CSF_INCLUDE_BLOOD}" = "1" ] && [ "${CSF_OPENING_RADIUS}" -gt 0 ]; then
    echo "[ERROR] CSF opening cannot be combined with legacy blood inclusion." >&2
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
# Each worker is an independent process and must remain single-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${TI_CHARM_CLEANUP_MANIFEST}")
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
echo "[INFO] Threads per worker: 1"
echo "[INFO] Retry:              ${RETRY_COUNT}/${MAX_RETRIES}"
echo "[INFO] Source mutation:    forbidden"

run_one_task() {
    local task_index="$1"
    local task_line task_id subject source_map source_hash source_bytes corrected_map result_path status message
    local task_log
    local -a command

    task_line=$(sed -n "$((task_index + 2))p" "${TI_CHARM_CLEANUP_MANIFEST}")
    IFS=$'\t' read -r task_id subject source_map source_hash source_bytes corrected_map result_path status message <<< "${task_line}"
    if [ -z "${task_line}" ] || [ -z "${subject:-}" ]; then
        echo "[ERROR] No valid manifest row for task index ${task_index}." >&2
        return 2
    fi
    if [ "${task_id}" != "${task_index}" ] || [ "${status}" != "ready" ]; then
        echo "[ERROR] Manifest row ${task_index} is not runnable: task=${task_id}, status=${status}, message=${message}." >&2
        return 2
    fi
    task_log="${TI_CHARM_CLEANUP_LOG_DIR}/${subject}.log"
    command=(
        python -u "${TI_CHARM_CLEANUP_WORKFLOW_PY}"
        run-task
        --manifest "${TI_CHARM_CLEANUP_MANIFEST}"
        --task-index "${task_index}"
        --csf-radius "${CSF_RADIUS}"
        --csf-opening-radius "${CSF_OPENING_RADIUS}"
        --skin-radius "${SKIN_RADIUS}"
        --csf-component-policy "${CSF_COMPONENT_POLICY}"
        --component-policy "${COMPONENT_POLICY}"
        --wm-min-component-voxels "${WM_MIN_COMPONENT_VOXELS}"
        --gm-min-component-voxels "${GM_MIN_COMPONENT_VOXELS}"
        --connectivity "${CONNECTIVITY}"
    )
    if [ "${CSF_INCLUDE_BLOOD}" = "1" ]; then
        command+=(--include-blood-in-csf)
    fi

    {
        echo "[INFO] Manifest task:      ${task_index}"
        echo "[INFO] Subject:            ${subject}"
        echo "[INFO] Source map:         ${source_map}"
        echo "[INFO] Source SHA-256:     ${source_hash}"
        echo "[INFO] Source bytes:       ${source_bytes}"
        echo "[INFO] Corrected map:      ${corrected_map}"
        echo "[INFO] Result marker:      ${result_path}"
        echo "[INFO] CSF closing:        ${CSF_RADIUS} voxels"
        echo "[INFO] CSF opening:        ${CSF_OPENING_RADIUS} voxels"
        echo "[INFO] CSF includes blood: ${CSF_INCLUDE_BLOOD}"
        echo "[INFO] CSF components:     ${CSF_COMPONENT_POLICY}"
        echo "[INFO] Skin closing:       ${SKIN_RADIUS} voxels"
        echo "[INFO] Component mode:     ${COMPONENT_POLICY}"
        echo "[INFO] Connectivity:       ${CONNECTIVITY}"
        "${command[@]}"
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
    echo "[WARN] Cleanup shard failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Cleanup shard failed after ${RETRY_COUNT} retries; see per-subject logs under ${TI_CHARM_CLEANUP_LOG_DIR}." >&2
exit "${RUN_EXIT}"
