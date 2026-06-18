#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Simulation submission controls
#
# Edit this block once on the HPC, then submit with:
#
#   bash hpc_scripts/submit_repeatability_experiment.sh
#
# Environment variables with the same runtime names still override these values
# for one-off submissions, but normal use should only require this block.
# ---------------------------------------------------------------------------
PIPELINE_DIR_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_CONFIG_CONFIG="${PIPELINE_DIR_CONFIG}/paired_repeatability_experiment.json"
LOG_DIR_CONFIG="${PIPELINE_DIR_CONFIG}/logs"

# Slurm resources and array shape.
JOB_NAME_CONFIG="ti_repeat_experiment"
PARTITION_CONFIG="sheffield"
CPUS_PER_TASK_CONFIG="8"
MEMORY_CONFIG="32G"
TIME_LIMIT_CONFIG="12:00:00"
MAX_CONCURRENT_TASKS_CONFIG="16"
# Leave blank to write Slurm stdout/stderr under LOG_DIR.
SLURM_OUTPUT_CONFIG=""

# Retry behavior. 0 or a negative value means unlimited retries until task
# validation passes. Retries automatically rerun incomplete outputs.
MESH_TIMEOUT_HOURS_CONFIG="4"
MESH_MAX_RETRIES_CONFIG="0"

# Rerun controls. Leave these at 0 for normal recovery submissions.
OVERWRITE_OUTPUT_CONFIG="0"
FORCE_MESH_CONFIG="0"

# Command/script paths. These normally do not need changing.
PYTHON_BIN_CONFIG="python"
SBATCH_BIN_CONFIG="sbatch"
SLURM_SCRIPT_CONFIG="${PIPELINE_DIR_CONFIG}/hpc_scripts/repeatability_experiment_array.slurm"
COMPLETION_CHECK_PY_CONFIG="${PIPELINE_DIR_CONFIG}/validate_repeatability_task.py"

PIPELINE_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_CONFIG}}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${EXPERIMENT_CONFIG_CONFIG}}"
LOG_DIR="${LOG_DIR:-${LOG_DIR_CONFIG}}"
JOB_NAME="${JOB_NAME:-${JOB_NAME_CONFIG}}"
PARTITION="${PARTITION:-${PARTITION_CONFIG}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${CPUS_PER_TASK_CONFIG}}"
MEMORY="${MEMORY:-${MEMORY_CONFIG}}"
TIME_LIMIT="${TIME_LIMIT:-${TIME_LIMIT_CONFIG}}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-${MAX_CONCURRENT_TASKS_CONFIG}}"
SLURM_OUTPUT="${SLURM_OUTPUT:-${SLURM_OUTPUT_CONFIG}}"
MESH_TIMEOUT_HOURS="${MESH_TIMEOUT_HOURS:-${TI_MESH_TIMEOUT_HOURS:-${MESH_TIMEOUT_HOURS_CONFIG}}}"
MESH_MAX_RETRIES="${MESH_MAX_RETRIES:-${TI_MESH_MAX_RETRIES:-${MESH_MAX_RETRIES_CONFIG}}}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-${OVERWRITE_OUTPUT_CONFIG}}"
FORCE_MESH="${FORCE_MESH:-${FORCE_MESH_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_CONFIG}}"
SBATCH_BIN="${SBATCH_BIN:-${SBATCH_BIN_CONFIG}}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SLURM_SCRIPT_CONFIG}}"
COMPLETION_CHECK_PY="${COMPLETION_CHECK_PY:-${COMPLETION_CHECK_PY_CONFIG}}"

if [ "$#" -ne 0 ]; then
    echo "[ERROR] This helper is configured from the controls at the top of the file."
    echo "        Edit the *_CONFIG values there, then run:"
    echo "        bash hpc_scripts/submit_repeatability_experiment.sh"
    exit 1
fi

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

require_file() {
    local label="$1"
    local path="$2"
    if [ ! -f "${path}" ]; then
        echo "[ERROR] ${label} does not exist: ${path}"
        exit 1
    fi
}

PIPELINE_DIR="$(resolve_path "${PIPELINE_DIR}")"
EXPERIMENT_CONFIG="$(resolve_path "${EXPERIMENT_CONFIG}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
COMPLETION_CHECK_PY="$(resolve_path "${COMPLETION_CHECK_PY}")"
RUNNER_SCRIPT="${PIPELINE_DIR}/simulation_runners/repeatability_experiment.py"
mkdir -p "${LOG_DIR}"
if [ -z "${SLURM_OUTPUT}" ]; then
    SLURM_OUTPUT="${LOG_DIR}/slurm-%A_%a.out"
fi

if ! [[ "${MAX_CONCURRENT_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_CONCURRENT_TASKS must be a positive integer: ${MAX_CONCURRENT_TASKS}"
    exit 1
fi
if ! [[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] CPUS_PER_TASK must be a positive integer: ${CPUS_PER_TASK}"
    exit 1
fi
if [ -z "${JOB_NAME}" ]; then
    echo "[ERROR] JOB_NAME must not be blank."
    exit 1
fi
if [ -z "${MEMORY}" ]; then
    echo "[ERROR] MEMORY must not be blank."
    exit 1
fi
if [ -z "${TIME_LIMIT}" ]; then
    echo "[ERROR] TIME_LIMIT must not be blank."
    exit 1
fi

if ! [[ "${MESH_MAX_RETRIES}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] MESH_MAX_RETRIES must be an integer: ${MESH_MAX_RETRIES}"
    exit 1
fi

if [ "${MESH_MAX_RETRIES}" -gt 0 ]; then
    MESH_RETRY_LIMIT_LABEL="${MESH_MAX_RETRIES}"
else
    MESH_RETRY_LIMIT_LABEL="unlimited"
fi

case "${OVERWRITE_OUTPUT}" in
    0|1) ;;
    *)
        echo "[ERROR] OVERWRITE_OUTPUT must be 0 or 1: ${OVERWRITE_OUTPUT}"
        exit 1
        ;;
esac

case "${FORCE_MESH}" in
    0|1) ;;
    *)
        echo "[ERROR] FORCE_MESH must be 0 or 1: ${FORCE_MESH}"
        exit 1
        ;;
esac

require_file "Experiment config" "${EXPERIMENT_CONFIG}"
require_file "Repeatability runner" "${RUNNER_SCRIPT}"
require_file "Slurm script" "${SLURM_SCRIPT}"
require_file "Completion check" "${COMPLETION_CHECK_PY}"

TASK_COUNT="$("${PYTHON_BIN}" "${RUNNER_SCRIPT}" \
    show-plan \
    --config "${EXPERIMENT_CONFIG}" \
    --count-only)"

if ! [[ "${TASK_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not resolve task count from config: ${EXPERIMENT_CONFIG}"
    exit 1
fi

if [ "${TASK_COUNT}" -lt 1 ]; then
    echo "[ERROR] Config produced zero tasks: ${EXPERIMENT_CONFIG}"
    exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT_TASKS}"
EXPORT_VARS="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},PIPELINE_DIR=${PIPELINE_DIR},LOG_DIR=${LOG_DIR},OVERWRITE_OUTPUT=${OVERWRITE_OUTPUT},FORCE_MESH=${FORCE_MESH},COMPLETION_CHECK_PY=${COMPLETION_CHECK_PY},TI_MESH_TIMEOUT_HOURS=${MESH_TIMEOUT_HOURS},TI_MESH_MAX_RETRIES=${MESH_MAX_RETRIES}"

echo "[INFO] Pipeline root:     ${PIPELINE_DIR}"
echo "[INFO] Experiment config: ${EXPERIMENT_CONFIG}"
echo "[INFO] Task count:        ${TASK_COUNT}"
echo "[INFO] Max concurrent:    ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Array spec:        ${ARRAY_SPEC}"
echo "[INFO] Job name:          ${JOB_NAME}"
echo "[INFO] Partition:         ${PARTITION:-cluster default}"
echo "[INFO] CPUs per task:     ${CPUS_PER_TASK}"
echo "[INFO] Memory:            ${MEMORY}"
echo "[INFO] Time limit:        ${TIME_LIMIT}"
echo "[INFO] Slurm output:      ${SLURM_OUTPUT:-script default}"
echo "[INFO] Log dir:           ${LOG_DIR}"
echo "[INFO] Overwrite output:  ${OVERWRITE_OUTPUT}"
echo "[INFO] Force mesh:        ${FORCE_MESH}"
echo "[INFO] Completion check:  ${COMPLETION_CHECK_PY}"
echo "[INFO] Mesh timeout:      ${MESH_TIMEOUT_HOURS} hour(s)"
echo "[INFO] Task retries:      ${MESH_RETRY_LIMIT_LABEL}"
echo "[INFO] Slurm script:      ${SLURM_SCRIPT}"

SBATCH_CMD=(
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --array="${ARRAY_SPEC}"
    --export="${EXPORT_VARS}"
)
if [ -n "${PARTITION}" ]; then
    SBATCH_CMD+=(--partition="${PARTITION}")
fi
if [ -n "${SLURM_OUTPUT}" ]; then
    SBATCH_CMD+=(--output="${SLURM_OUTPUT}")
fi
SBATCH_CMD+=("${SLURM_SCRIPT}")

"${SBATCH_CMD[@]}"
