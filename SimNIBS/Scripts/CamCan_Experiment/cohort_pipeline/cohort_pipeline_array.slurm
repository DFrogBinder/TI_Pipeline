#!/bin/bash
#SBATCH --job-name=camcan_cohort
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=camcan-cohort-%A_%a.out
#SBATCH --requeue

set -uo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array." >&2
    exit 2
fi

: "${TI_COHORT_STAGE:?Missing TI_COHORT_STAGE.}"
: "${TI_COHORT_MANIFEST:?Missing TI_COHORT_MANIFEST.}"
: "${TI_COHORT_WORKFLOW_PY:?Missing TI_COHORT_WORKFLOW_PY.}"
: "${TI_COHORT_LOG_DIR:?Missing TI_COHORT_LOG_DIR.}"

case "${TI_COHORT_STAGE}" in
    scaffold|mesh|simulate)
        ;;
    *)
        echo "[ERROR] TI_COHORT_STAGE must be scaffold, mesh, or simulate." >&2
        exit 2
        ;;
esac

if [ "${TI_COHORT_STAGE}" = "simulate" ]; then
    : "${TI_TARGETS_CSV:?Missing TI_TARGETS_CSV.}"
    : "${TI_EXPECTED_TARGETS_SHA256:?Missing TI_EXPECTED_TARGETS_SHA256.}"
    : "${TI_SIM_RUNNER_PY:?Missing TI_SIM_RUNNER_PY.}"
    : "${TI_COMPLETION_CHECK_PY:?Missing TI_COMPLETION_CHECK_PY.}"
fi

MAX_RETRIES="${TI_COHORT_MAX_RETRIES:-unlimited}"
STATE_DIR="${TI_COHORT_STATE_DIR:-${TI_COHORT_LOG_DIR}/retry_state}"
TASK_OFFSET="${TASK_OFFSET:-0}"
ELEMENT_OFFSET="${ELEMENT_OFFSET:-0}"
MESH_WORKERS="${TI_COHORT_MESH_WORKERS_PER_ELEMENT:-2}"
THREADS_PER_MESH_WORKER="${TI_COHORT_MESH_THREADS_PER_WORKER:-4}"
LOCAL_STAGING="${TI_COHORT_MESH_LOCAL_STAGING:-1}"
TASKSET_BIN="${TASKSET_BIN:-taskset}"

for value_name in TASK_OFFSET ELEMENT_OFFSET MESH_WORKERS THREADS_PER_MESH_WORKER; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${MESH_WORKERS}" -lt 1 ] || [ "${THREADS_PER_MESH_WORKER}" -lt 1 ]; then
    echo "[ERROR] Mesh worker and thread counts must be positive." >&2
    exit 2
fi
if [ "${MAX_RETRIES}" != "unlimited" ] && ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TI_COHORT_MAX_RETRIES must be unlimited or a non-negative integer." >&2
    exit 2
fi
if [ "${LOCAL_STAGING}" != "0" ] && [ "${LOCAL_STAGING}" != "1" ]; then
    echo "[ERROR] TI_COHORT_MESH_LOCAL_STAGING must be 0 or 1." >&2
    exit 2
fi
if [ ! -f "${TI_COHORT_MANIFEST}" ] || [ ! -f "${TI_COHORT_WORKFLOW_PY}" ]; then
    echo "[ERROR] Cohort manifest or workflow is missing." >&2
    exit 2
fi

mkdir -p "${TI_COHORT_LOG_DIR}" "${STATE_DIR}"

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_cohort_pipeline_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SHIM_DIR}"
cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass

import builtins
_real_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    try:
        if name == "simnibs" or name.startswith("simnibs."):
            from simnibs import sim_struct as _sim_struct
            if not getattr(_sim_struct.SESSION, "_gmsh_disabled", False):
                _orig_init = _sim_struct.SESSION.__init__
                def _new_init(self, *a, **k):
                    _orig_init(self, *a, **k)
                    try:
                        self.open_in_gmsh = False
                    except Exception:
                        pass
                _sim_struct.SESSION.__init__ = _new_init
                _sim_struct.SESSION._gmsh_disabled = True
    except Exception:
        pass
    return mod

builtins.__import__ = _patched_import
EOF
export PYTHONPATH="${SHIM_DIR}:${PYTHONPATH:-}"

CPU_IDS=()
if [ "${TI_COHORT_STAGE}" = "mesh" ]; then
    REQUIRED_CPUS=$((MESH_WORKERS * THREADS_PER_MESH_WORKER))
    if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "${REQUIRED_CPUS}" -gt "${SLURM_CPUS_PER_TASK}" ]; then
        echo "[ERROR] Packed mesh workers require ${REQUIRED_CPUS} CPUs; allocation exposes ${SLURM_CPUS_PER_TASK}." >&2
        exit 2
    fi
    if [ "${MESH_WORKERS}" -gt 1 ] && ! command -v "${TASKSET_BIN}" >/dev/null 2>&1; then
        echo "[ERROR] Packed meshing requires taskset." >&2
        exit 2
    fi
    read -r -a CPU_IDS <<< "$(python -c 'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
    if [ "${#CPU_IDS[@]}" -lt "${REQUIRED_CPUS}" ]; then
        echo "[ERROR] CPU affinity exposes ${#CPU_IDS[@]} CPUs; ${REQUIRED_CPUS} are required." >&2
        exit 2
    fi
    export OMP_NUM_THREADS="${THREADS_PER_MESH_WORKER}"
    export MKL_NUM_THREADS="${THREADS_PER_MESH_WORKER}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_MESH_WORKER}"
    export NUMEXPR_NUM_THREADS="${THREADS_PER_MESH_WORKER}"
    export TI_CHARM_MESH_THREADS_PER_WORKER="${THREADS_PER_MESH_WORKER}"
else
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
fi

TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${TI_COHORT_MANIFEST}")
if [ "${TI_COHORT_STAGE}" = "mesh" ]; then
    GLOBAL_ELEMENT=$((ELEMENT_OFFSET + SLURM_ARRAY_TASK_ID))
    FIRST_TASK=$((GLOBAL_ELEMENT * MESH_WORKERS))
    LAST_TASK=$((FIRST_TASK + MESH_WORKERS - 1))
else
    FIRST_TASK=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))
    LAST_TASK="${FIRST_TASK}"
fi
if [ "${FIRST_TASK}" -ge "${TOTAL_TASKS}" ]; then
    echo "[ERROR] Array element starts beyond ${TOTAL_TASKS} manifest rows." >&2
    exit 2
fi
if [ "${LAST_TASK}" -ge "${TOTAL_TASKS}" ]; then
    LAST_TASK=$((TOTAL_TASKS - 1))
fi

TASK_KEY="${TI_COHORT_STAGE}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
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
echo "[INFO] Stage:              ${TI_COHORT_STAGE}"
echo "[INFO] Manifest tasks:     ${FIRST_TASK}-${LAST_TASK}"
echo "[INFO] Total tasks:        ${TOTAL_TASKS}"
echo "[INFO] Mesh workers:       ${MESH_WORKERS}"
echo "[INFO] Threads/mesh worker:${THREADS_PER_MESH_WORKER}"
echo "[INFO] Local mesh staging: ${LOCAL_STAGING}"
echo "[INFO] Retry:              ${RETRY_COUNT}/${MAX_RETRIES}"

run_one_task() {
    local task_index="$1"
    local worker_position cpu_offset cpu_list cpu_position
    local task_line task_id subject task_log staging_root
    local -a command launch_command

    task_line=$(sed -n "$((task_index + 2))p" "${TI_COHORT_MANIFEST}")
    if [ -z "${task_line}" ]; then
        echo "[ERROR] Missing manifest row ${task_index}." >&2
        return 2
    fi
    task_id="${task_line%%$'\t'*}"
    if [ "${TI_COHORT_STAGE}" = "scaffold" ]; then
        subject=$(printf '%s\n' "${task_line}" | cut -f2)
    else
        subject=$(printf '%s\n' "${task_line}" | cut -f7)
    fi
    if [ "${task_id}" != "${task_index}" ] || [ -z "${subject}" ]; then
        echo "[ERROR] Malformed or misindexed manifest row ${task_index}." >&2
        return 2
    fi
    task_log="${TI_COHORT_LOG_DIR}/${TI_COHORT_STAGE}__${task_index}__${subject}.log"
    command=(python -u "${TI_COHORT_WORKFLOW_PY}")
    case "${TI_COHORT_STAGE}" in
        scaffold)
            command+=(
                scaffold-task
                --manifest "${TI_COHORT_MANIFEST}"
                --task-index "${task_index}"
            )
            ;;
        mesh)
            command+=(
                mesh-task
                --manifest "${TI_COHORT_MANIFEST}"
                --task-index "${task_index}"
            )
            if [ -n "${TI_COHORT_MESH_SETTINGS:-}" ]; then
                command+=(--settings "${TI_COHORT_MESH_SETTINGS}")
            fi
            if [ "${LOCAL_STAGING}" = "1" ]; then
                staging_root="${SLURM_TMPDIR:-/tmp}/${USER}_cohort_mesh_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
                mkdir -p "${staging_root}"
                command+=(--staging-root "${staging_root}")
            fi
            ;;
        simulate)
            command+=(
                simulation-task
                --manifest "${TI_COHORT_MANIFEST}"
                --task-index "${task_index}"
                --targets-csv "${TI_TARGETS_CSV}"
                --expected-targets-sha256 "${TI_EXPECTED_TARGETS_SHA256}"
                --simulation-runner "${TI_SIM_RUNNER_PY}"
                --simulation-validator "${TI_COMPLETION_CHECK_PY}"
                --python-bin python
            )
            if [ -n "${TI_INDIVIDUALIZED_TARGETS_CSV:-}" ]; then
                : "${TI_EXPECTED_INDIVIDUALIZED_TARGETS_SHA256:?Missing TI_EXPECTED_INDIVIDUALIZED_TARGETS_SHA256.}"
                command+=(
                    --individualized-targets-csv "${TI_INDIVIDUALIZED_TARGETS_CSV}"
                    --expected-individualized-targets-sha256 "${TI_EXPECTED_INDIVIDUALIZED_TARGETS_SHA256}"
                )
            fi
            ;;
    esac
    launch_command=("${command[@]}")
    cpu_list="all allocated CPUs"
    if [ "${TI_COHORT_STAGE}" = "mesh" ] && [ "${MESH_WORKERS}" -gt 1 ]; then
        worker_position=$((task_index - FIRST_TASK))
        cpu_offset=$((worker_position * THREADS_PER_MESH_WORKER))
        cpu_list="${CPU_IDS[$cpu_offset]}"
        for ((cpu_position = 1; cpu_position < THREADS_PER_MESH_WORKER; cpu_position++)); do
            cpu_list+=",${CPU_IDS[$((cpu_offset + cpu_position))]}"
        done
        launch_command=("${TASKSET_BIN}" --cpu-list "${cpu_list}" "${command[@]}")
    fi
    {
        echo "[INFO] Task index:         ${task_index}"
        echo "[INFO] Subject:            ${subject}"
        echo "[INFO] Worker CPU set:      ${cpu_list}"
        "${launch_command[@]}"
    } 2>&1 | tee -a "${task_log}"
}

PIDS=()
TASK_INDICES=()
for ((task_index = FIRST_TASK; task_index <= LAST_TASK; task_index++)); do
    run_one_task "${task_index}" &
    PIDS+=("$!")
    TASK_INDICES+=("${task_index}")
done

RUN_EXIT=0
for position in "${!PIDS[@]}"; do
    if wait "${PIDS[$position]}"; then
        echo "[INFO] Task ${TASK_INDICES[$position]} completed."
    else
        TASK_EXIT=$?
        echo "[ERROR] Task ${TASK_INDICES[$position]} failed with exit ${TASK_EXIT}." >&2
        if [ "${RUN_EXIT}" -eq 0 ]; then
            RUN_EXIT="${TASK_EXIT}"
        fi
    fi
done

if [ "${RUN_EXIT}" -eq 0 ]; then
    rm -f "${RETRY_FILE}"
    exit 0
fi
if [ "${MAX_RETRIES}" = "unlimited" ] || [ "${RETRY_COUNT}" -lt "${MAX_RETRIES}" ]; then
    NEXT_RETRY=$((RETRY_COUNT + 1))
    printf '%s\n' "${NEXT_RETRY}" > "${RETRY_FILE}"
    echo "[WARN] Cohort ${TI_COHORT_STAGE} shard failed with exit ${RUN_EXIT}; requeueing (${NEXT_RETRY}/${MAX_RETRIES})."
    scontrol requeue "${SLURM_JOB_ID}"
    exit $?
fi

echo "[ERROR] Cohort task failed after ${RETRY_COUNT} retries." >&2
exit "${RUN_EXIT}"
