#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${ROI_ROOT:?Set ROI_ROOT to the CamCan ROI root containing *_Data_* repeat folders.}"
: "${MANIFEST:?Set MANIFEST to the preflight task manifest TSV.}"
: "${MONTAGE_PRESET:?Set MONTAGE_PRESET, e.g. left-m1 or right-dlpc.}"
: "${LOG_DIR:?Set LOG_DIR for Slurm and task logs.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/HPC_scripts/camcan_inplace_rerun_array.slurm}"
SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-${SCRIPT_DIR}/simulation/TI_runner_multi-core.py}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-${SCRIPT_DIR}/simulation/validate_simulation_outputs.py}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-8}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
START_TASK_OFFSET="${START_TASK_OFFSET:-0}"
MAX_SUBMITTED_CHUNKS="${MAX_SUBMITTED_CHUNKS:-0}"
JOB_NAME="${JOB_NAME:-ti_camcan_inplace_rerun}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
TASK_MAX_RETRIES="${TI_TASK_MAX_RETRIES:-${TI_MESH_MAX_RETRIES:-0}}"

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

ROI_ROOT="$(resolve_path "${ROI_ROOT}")"
MANIFEST="$(resolve_path "${MANIFEST}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
SIM_RUNNER_PY="$(resolve_path "${SIM_RUNNER_PY}")"
COMPLETION_CHECK_PY="$(resolve_path "${COMPLETION_CHECK_PY}")"
mkdir -p "${LOG_DIR}"

if [ ! -d "${ROI_ROOT}" ]; then
    echo "[ERROR] ROI root not found: ${ROI_ROOT}" >&2
    exit 1
fi
if [ ! -f "${MANIFEST}" ]; then
    echo "[ERROR] Manifest not found: ${MANIFEST}" >&2
    exit 1
fi
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "[ERROR] Slurm array script not found: ${SLURM_SCRIPT}" >&2
    exit 1
fi
if [ ! -f "${SIM_RUNNER_PY}" ]; then
    echo "[ERROR] Simulation runner not found: ${SIM_RUNNER_PY}" >&2
    exit 1
fi
if [ ! -f "${COMPLETION_CHECK_PY}" ]; then
    echo "[ERROR] Completion check script not found: ${COMPLETION_CHECK_PY}" >&2
    exit 1
fi

awk -F '\t' 'NR == 1 { exit !(($1 == "task_id") && ($4 == "dataset_root") && ($5 == "subject") && ($8 == "status")) }' "$MANIFEST" || {
    echo "[ERROR] Manifest has an unexpected header: ${MANIFEST}" >&2
    exit 1
}

FIRST_BLANK_LINE=$(awk 'NR > 1 && NF == 0 { print NR; exit }' "$MANIFEST")
if [ -n "$FIRST_BLANK_LINE" ]; then
    echo "[ERROR] Blank line found in manifest at line ${FIRST_BLANK_LINE}: ${MANIFEST}" >&2
    exit 1
fi

TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "$MANIFEST")
READY_TASKS=$(awk -F '\t' 'NR > 1 && NF > 0 && $8 == "ready" { count++ } END { print count + 0 }' "$MANIFEST")
BLOCKED_TASKS=$(awk -F '\t' 'NR > 1 && NF > 0 && $8 != "ready" { count++ } END { print count + 0 }' "$MANIFEST")

if [ "$TOTAL_TASKS" -le 0 ]; then
    echo "[INFO] Manifest contains no task rows: ${MANIFEST}"
    exit 0
fi
if [ "$READY_TASKS" -le 0 ]; then
    echo "[INFO] Manifest contains no ready tasks: ${MANIFEST}"
    echo "[INFO] Blocked tasks: ${BLOCKED_TASKS}"
    exit 0
fi
if ! [[ "${MAX_ARRAY_TASKS}" =~ ^[0-9]+$ ]] || [ "${MAX_ARRAY_TASKS}" -lt 1 ]; then
    echo "[ERROR] MAX_ARRAY_TASKS must be a positive integer; got '${MAX_ARRAY_TASKS}'." >&2
    exit 1
fi
if ! [[ "${MAX_CONCURRENT_TASKS}" =~ ^[0-9]+$ ]] || [ "${MAX_CONCURRENT_TASKS}" -lt 1 ]; then
    echo "[ERROR] MAX_CONCURRENT_TASKS must be a positive integer; got '${MAX_CONCURRENT_TASKS}'." >&2
    exit 1
fi
if ! [[ "${START_TASK_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] START_TASK_OFFSET must be a non-negative integer; got '${START_TASK_OFFSET}'." >&2
    exit 1
fi
if [ "${START_TASK_OFFSET}" -ge "${TOTAL_TASKS}" ]; then
    echo "[INFO] START_TASK_OFFSET=${START_TASK_OFFSET} is at or beyond total rows (${TOTAL_TASKS}); nothing to submit."
    exit 0
fi
if ! [[ "${MAX_SUBMITTED_CHUNKS}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MAX_SUBMITTED_CHUNKS must be a non-negative integer; got '${MAX_SUBMITTED_CHUNKS}'." >&2
    exit 1
fi
if ! [[ "${TASK_MAX_RETRIES}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TI_TASK_MAX_RETRIES/TI_MESH_MAX_RETRIES must be an integer; got '${TASK_MAX_RETRIES}'." >&2
    exit 1
fi

SLURM_OUTPUT="${SLURM_OUTPUT:-${LOG_DIR}/camcan-inplace-rerun-%A_%a.out}"
BASE_EXPORT_VARS="ALL,TI_INPLACE_RERUN_MANIFEST=${MANIFEST},TI_MONTAGE_PRESET=${MONTAGE_PRESET},TI_SIM_RUNNER_PY=${SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${COMPLETION_CHECK_PY},LOG_DIR=${LOG_DIR},TI_TASK_MAX_RETRIES=${TASK_MAX_RETRIES}"

echo "[INFO] ROI root:          ${ROI_ROOT}"
echo "[INFO] Manifest:          ${MANIFEST}"
echo "[INFO] Montage preset:    ${MONTAGE_PRESET}"
echo "[INFO] Total rows:        ${TOTAL_TASKS}"
echo "[INFO] Ready tasks:       ${READY_TASKS}"
echo "[INFO] Blocked tasks:     ${BLOCKED_TASKS}"
echo "[INFO] Start task offset: ${START_TASK_OFFSET}"
echo "[INFO] Max array tasks:   ${MAX_ARRAY_TASKS}"
echo "[INFO] Max concurrent:    ${MAX_CONCURRENT_TASKS}"
if [ "${MAX_SUBMITTED_CHUNKS}" -gt 0 ]; then
    echo "[INFO] Max chunks/call:   ${MAX_SUBMITTED_CHUNKS}"
else
    echo "[INFO] Max chunks/call:   unlimited"
fi
echo "[INFO] Slurm script:      ${SLURM_SCRIPT}"
echo "[INFO] Runner:            ${SIM_RUNNER_PY}"
echo "[INFO] Completion check:  ${COMPLETION_CHECK_PY}"
echo "[INFO] Log dir:           ${LOG_DIR}"

TASK_OFFSET="${START_TASK_OFFSET}"
SUBMITTED_ARRAYS=0
while [ "${TASK_OFFSET}" -lt "${TOTAL_TASKS}" ]; do
    if [ "${MAX_SUBMITTED_CHUNKS}" -gt 0 ] && [ "${SUBMITTED_ARRAYS}" -ge "${MAX_SUBMITTED_CHUNKS}" ]; then
        echo "[INFO] Reached MAX_SUBMITTED_CHUNKS=${MAX_SUBMITTED_CHUNKS}; stopping at TASK_OFFSET=${TASK_OFFSET}."
        echo "[INFO] Next resume command: START_TASK_OFFSET=${TASK_OFFSET} MAX_SUBMITTED_CHUNKS=${MAX_SUBMITTED_CHUNKS} ROI_ROOT=\"${ROI_ROOT}\" MANIFEST=\"${MANIFEST}\" MONTAGE_PRESET=\"${MONTAGE_PRESET}\" LOG_DIR=\"${LOG_DIR}\" bash ${BASH_SOURCE[0]}"
        break
    fi

    CHUNK_COUNT="${MAX_ARRAY_TASKS}"
    REMAINING=$((TOTAL_TASKS - TASK_OFFSET))
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
    set +e
    SBATCH_OUTPUT="$("${SBATCH_CMD[@]}" 2>&1)"
    SBATCH_EXIT=$?
    set -e
    echo "${SBATCH_OUTPUT}"
    if [ "${SBATCH_EXIT}" -ne 0 ]; then
        echo "[ERROR] sbatch failed for global task range ${TASK_OFFSET}-${GLOBAL_END}." >&2
        echo "[ERROR] Resume after job limits clear with:" >&2
        echo "[ERROR] START_TASK_OFFSET=${TASK_OFFSET} MAX_SUBMITTED_CHUNKS=${MAX_SUBMITTED_CHUNKS:-0} ROI_ROOT=\"${ROI_ROOT}\" MANIFEST=\"${MANIFEST}\" MONTAGE_PRESET=\"${MONTAGE_PRESET}\" LOG_DIR=\"${LOG_DIR}\" bash ${BASH_SOURCE[0]}" >&2
        exit "${SBATCH_EXIT}"
    fi
    TASK_OFFSET=$((TASK_OFFSET + CHUNK_COUNT))
    SUBMITTED_ARRAYS=$((SUBMITTED_ARRAYS + 1))
done

echo "[INFO] Submitted ${SUBMITTED_ARRAYS} Slurm array chunk(s)."
