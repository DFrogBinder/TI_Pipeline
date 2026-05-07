#!/bin/bash
set -euo pipefail

# Submission settings. Edit these values, then submit with:
#   bash hpc_scripts/submit_repeatability_experiment.sh
PIPELINE_DIR="${PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${PIPELINE_DIR}/paired_repeatability_experiment.example.json}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-16}"
LOG_DIR="${LOG_DIR:-${PIPELINE_DIR}/logs}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
FORCE_MESH="${FORCE_MESH:-0}"

# Command/script settings. These normally do not need changing.
PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${PIPELINE_DIR}/hpc_scripts/repeatability_experiment_array.slurm}"
SBATCH_OPTIONS=()

if [ "$#" -ne 0 ]; then
    echo "[ERROR] This helper is configured from the variables at the top of the file."
    echo "        Edit EXPERIMENT_CONFIG/MAX_CONCURRENT_TASKS there, then run:"
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
RUNNER_SCRIPT="${PIPELINE_DIR}/simulation_runners/repeatability_experiment.py"

if ! [[ "${MAX_CONCURRENT_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_CONCURRENT_TASKS must be a positive integer: ${MAX_CONCURRENT_TASKS}"
    exit 1
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
EXPORT_VARS="ALL,EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},PIPELINE_DIR=${PIPELINE_DIR},LOG_DIR=${LOG_DIR},OVERWRITE_OUTPUT=${OVERWRITE_OUTPUT},FORCE_MESH=${FORCE_MESH}"

echo "[INFO] Pipeline root:     ${PIPELINE_DIR}"
echo "[INFO] Experiment config: ${EXPERIMENT_CONFIG}"
echo "[INFO] Task count:        ${TASK_COUNT}"
echo "[INFO] Max concurrent:    ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Array spec:        ${ARRAY_SPEC}"
echo "[INFO] Log dir:           ${LOG_DIR}"
echo "[INFO] Overwrite output:  ${OVERWRITE_OUTPUT}"
echo "[INFO] Force mesh:        ${FORCE_MESH}"
echo "[INFO] Slurm script:      ${SLURM_SCRIPT}"

"${SBATCH_BIN}" \
    "${SBATCH_OPTIONS[@]}" \
    --array="${ARRAY_SPEC}" \
    --export="${EXPORT_VARS}" \
    "${SLURM_SCRIPT}"
