#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the corrected repair root or comparison output root.}"
: "${MODE:?Set MODE to scaled, pair2-rerun, or compare.}"
: "${LOG_DIR:?Set LOG_DIR for Slurm and repair logs.}"
: "${MONTAGE_PRESET:?Set MONTAGE_PRESET, e.g. left-hippocampus.}"

if [ "${MODE}" = "compare" ]; then
    : "${SCALED_ROOT:?Set SCALED_ROOT for compare mode.}"
    : "${PAIR2_RERUN_ROOT:?Set PAIR2_RERUN_ROOT for compare mode.}"
else
    : "${ORIGINAL_ROOT:?Set ORIGINAL_ROOT to the original CamCan batch root.}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
REPAIR_CLI="${REPAIR_CLI:-${SCRIPT_DIR}/simulation/repair_current_bug.py}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/HPC_scripts/current_repair_jobArray.slurm}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-8}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
JOB_NAME="${JOB_NAME:-ti_camcan_current_repair}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-0}"
INCLUDE_UNAFFECTED="${INCLUDE_UNAFFECTED:-0}"
DATASET_GLOB="${DATASET_GLOB:-*_Data_*}"

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
REPAIR_CLI="$(resolve_path "${REPAIR_CLI}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
mkdir -p "${LOG_DIR}"

CLI_COUNT=("${PYTHON_BIN}" "${REPAIR_CLI}" --mode "${MODE}" --output-root "${OUTPUT_ROOT}" --montage-preset "${MONTAGE_PRESET}" --dataset-glob "${DATASET_GLOB}" --count-only)
if [ "${INCLUDE_UNAFFECTED}" = "1" ]; then
    CLI_COUNT+=(--include-unaffected)
fi
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
    echo "[INFO] No CamCan current-repair tasks found for MODE=${MODE}, MONTAGE_PRESET=${MONTAGE_PRESET}."
    exit 0
fi
if ! [[ "${MAX_ARRAY_TASKS}" =~ ^[0-9]+$ ]] || [ "${MAX_ARRAY_TASKS}" -lt 1 ]; then
    echo "[ERROR] MAX_ARRAY_TASKS must be a positive integer, got: ${MAX_ARRAY_TASKS}" >&2
    exit 1
fi

SLURM_OUTPUT="${SLURM_OUTPUT:-${LOG_DIR}/camcan-current-repair-%A_%a.out}"
BASE_EXPORT_VARS="ALL,REPAIR_CLI=${REPAIR_CLI},ORIGINAL_ROOT=${ORIGINAL_ROOT:-},OUTPUT_ROOT=${OUTPUT_ROOT},MODE=${MODE},SCALED_ROOT=${SCALED_ROOT:-},PAIR2_RERUN_ROOT=${PAIR2_RERUN_ROOT:-},LOG_DIR=${LOG_DIR},MONTAGE_PRESET=${MONTAGE_PRESET},OVERWRITE_OUTPUT=${OVERWRITE_OUTPUT},INCLUDE_UNAFFECTED=${INCLUDE_UNAFFECTED},DATASET_GLOB=${DATASET_GLOB}"

echo "[INFO] Mode:              ${MODE}"
echo "[INFO] Montage preset:    ${MONTAGE_PRESET}"
echo "[INFO] Task count:        ${TASK_COUNT}"
echo "[INFO] Max array tasks:   ${MAX_ARRAY_TASKS}"
echo "[INFO] Original root:     ${ORIGINAL_ROOT:-n/a}"
echo "[INFO] Output root:       ${OUTPUT_ROOT}"
echo "[INFO] Scaled root:       ${SCALED_ROOT:-n/a}"
echo "[INFO] Pair2-rerun root:  ${PAIR2_RERUN_ROOT:-n/a}"
echo "[INFO] Log dir:           ${LOG_DIR}"

TASK_OFFSET=0
SUBMITTED_ARRAYS=0
while [ "${TASK_OFFSET}" -lt "${TASK_COUNT}" ]; do
    CHUNK_COUNT="${MAX_ARRAY_TASKS}"
    REMAINING=$((TASK_COUNT - TASK_OFFSET))
    if [ "${REMAINING}" -lt "${CHUNK_COUNT}" ]; then
        CHUNK_COUNT="${REMAINING}"
    fi
    ARRAY_SPEC="0-$((CHUNK_COUNT - 1))%${MAX_CONCURRENT_TASKS}"
    GLOBAL_END=$((TASK_OFFSET + CHUNK_COUNT - 1))
    EXPORT_VARS="${BASE_EXPORT_VARS},TASK_OFFSET=${TASK_OFFSET}"

    echo "[INFO] Submitting chunk: local ${ARRAY_SPEC}, global ${TASK_OFFSET}-${GLOBAL_END}"
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
    TASK_OFFSET=$((TASK_OFFSET + CHUNK_COUNT))
    SUBMITTED_ARRAYS=$((SUBMITTED_ARRAYS + 1))
done

echo "[INFO] Submitted ${SUBMITTED_ARRAYS} Slurm array chunk(s)."
