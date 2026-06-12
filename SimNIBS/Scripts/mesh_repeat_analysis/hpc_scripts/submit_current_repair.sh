#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG to the Repeatability JSON config.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the corrected repair root or comparison output root.}"
: "${MODE:?Set MODE to scaled, pair2-rerun, or compare.}"
: "${LOG_DIR:?Set LOG_DIR for Slurm and repair logs.}"

if [ "${MODE}" = "compare" ]; then
    : "${SCALED_ROOT:?Set SCALED_ROOT for compare mode.}"
    : "${PAIR2_RERUN_ROOT:?Set PAIR2_RERUN_ROOT for compare mode.}"
else
    : "${ORIGINAL_ROOT:?Set ORIGINAL_ROOT to the original Repeatability experiment root.}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
REPAIR_CLI="${REPAIR_CLI:-${SCRIPT_DIR}/simulation_runners/repair_current_bug.py}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/hpc_scripts/current_repair_array.slurm}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-16}"
JOB_NAME="${JOB_NAME:-ti_current_repair}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

EXPERIMENT_CONFIG="$(resolve_path "${EXPERIMENT_CONFIG}")"
OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
REPAIR_CLI="$(resolve_path "${REPAIR_CLI}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
mkdir -p "${LOG_DIR}"

CLI_COUNT=("${PYTHON_BIN}" "${REPAIR_CLI}" --mode "${MODE}" --config "${EXPERIMENT_CONFIG}" --output-root "${OUTPUT_ROOT}" --count-only)
if [ "${MODE}" = "compare" ]; then
    SCALED_ROOT="$(resolve_path "${SCALED_ROOT}")"
    PAIR2_RERUN_ROOT="$(resolve_path "${PAIR2_RERUN_ROOT}")"
    CLI_COUNT+=(--scaled-root "${SCALED_ROOT}" --pair2-rerun-root "${PAIR2_RERUN_ROOT}")
else
    ORIGINAL_ROOT="$(resolve_path "${ORIGINAL_ROOT}")"
    CLI_COUNT+=(--original-root "${ORIGINAL_ROOT}")
fi

TASK_COUNT="$("${CLI_COUNT[@]}")"
if ! [[ "${TASK_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not resolve task count; CLI returned: ${TASK_COUNT}" >&2
    exit 1
fi
if [ "${TASK_COUNT}" -lt 1 ]; then
    echo "[INFO] No current-repair tasks found for MODE=${MODE}."
    exit 0
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT_TASKS}"
SLURM_OUTPUT="${SLURM_OUTPUT:-${LOG_DIR}/current-repair-%A_%a.out}"
EXPORT_VARS="ALL,REPAIR_CLI=${REPAIR_CLI},EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG},ORIGINAL_ROOT=${ORIGINAL_ROOT:-},OUTPUT_ROOT=${OUTPUT_ROOT},MODE=${MODE},SCALED_ROOT=${SCALED_ROOT:-},PAIR2_RERUN_ROOT=${PAIR2_RERUN_ROOT:-},LOG_DIR=${LOG_DIR},OVERWRITE_OUTPUT=${OVERWRITE_OUTPUT}"

echo "[INFO] Mode:              ${MODE}"
echo "[INFO] Task count:        ${TASK_COUNT}"
echo "[INFO] Array spec:        ${ARRAY_SPEC}"
echo "[INFO] Original root:     ${ORIGINAL_ROOT:-n/a}"
echo "[INFO] Output root:       ${OUTPUT_ROOT}"
echo "[INFO] Scaled root:       ${SCALED_ROOT:-n/a}"
echo "[INFO] Pair2-rerun root:  ${PAIR2_RERUN_ROOT:-n/a}"
echo "[INFO] Log dir:           ${LOG_DIR}"

SBATCH_CMD=(
    "${SBATCH_BIN}"
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --array="${ARRAY_SPEC}"
    --output="${SLURM_OUTPUT}"
    --export="${EXPORT_VARS}"
    "${SLURM_SCRIPT}"
)

"${SBATCH_CMD[@]}"
