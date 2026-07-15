#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MANIFEST:?Set MANIFEST to the ready CHARM-only remesh manifest.}"
: "${RESULT_DIR:?Set RESULT_DIR for immutable per-task remesh results.}"
: "${LOG_DIR:?Set LOG_DIR for Slurm output and task logs.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/HPC_scripts/charm_only_remesh_array.slurm}"
WORKFLOW_PY="${TI_CHARM_REMESH_WORKFLOW_PY:-${SCRIPT_DIR}/charm_only_remesh/workflow.py}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-8}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
START_TASK_OFFSET="${START_TASK_OFFSET:-0}"
MAX_SUBMITTED_CHUNKS="${MAX_SUBMITTED_CHUNKS:-0}"
MAX_RETRIES="${TI_CHARM_REMESH_MAX_RETRIES:-2}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
JOB_NAME="${JOB_NAME:-charm_only_remesh}"

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

MANIFEST="$(resolve_path "${MANIFEST}")"
RESULT_DIR="$(resolve_path "${RESULT_DIR}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
WORKFLOW_PY="$(resolve_path "${WORKFLOW_PY}")"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

for path in "${MANIFEST}" "${SLURM_SCRIPT}" "${WORKFLOW_PY}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 2
    fi
done

awk -F '\t' 'NR == 1 { exit !(($1 == "task_id") && ($2 == "dataset_name") && ($6 == "subject") && ($13 == "status")) }' "${MANIFEST}" || {
    echo "[ERROR] Manifest has an unexpected header: ${MANIFEST}" >&2
    exit 2
}
TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${MANIFEST}")
READY_TASKS=$(awk -F '\t' 'NR > 1 && NF > 0 && $13 == "ready" { count++ } END { print count + 0 }' "${MANIFEST}")
if [ "${TOTAL_TASKS}" -le 0 ] || [ "${READY_TASKS}" -ne "${TOTAL_TASKS}" ]; then
    echo "[ERROR] Refusing submission: ready=${READY_TASKS}, total=${TOTAL_TASKS}." >&2
    exit 2
fi

for value_name in MAX_CONCURRENT_TASKS MAX_ARRAY_TASKS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer." >&2
        exit 2
    fi
done
for value_name in START_TASK_OFFSET MAX_SUBMITTED_CHUNKS MAX_RETRIES; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${START_TASK_OFFSET}" -ge "${TOTAL_TASKS}" ]; then
    echo "[INFO] Nothing to submit: offset ${START_TASK_OFFSET}, total ${TOTAL_TASKS}."
    exit 0
fi

echo "[INFO] CHARM-only manifest: ${MANIFEST}"
echo "[INFO] Ready tasks:        ${READY_TASKS}/${TOTAL_TASKS}"
echo "[INFO] Result directory:   ${RESULT_DIR}"
echo "[INFO] Resource profile:   SimNIBS/4.0.1, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Concurrency:        ${MAX_CONCURRENT_TASKS} (candidate; validate with one-subject smoke test first)"

BASE_EXPORT="ALL,TI_CHARM_REMESH_MANIFEST=${MANIFEST},TI_CHARM_REMESH_RESULT_DIR=${RESULT_DIR},TI_CHARM_REMESH_LOG_DIR=${LOG_DIR},TI_CHARM_REMESH_WORKFLOW_PY=${WORKFLOW_PY},TI_CHARM_REMESH_MAX_RETRIES=${MAX_RETRIES}"
TASK_OFFSET="${START_TASK_OFFSET}"
SUBMITTED=0
while [ "${TASK_OFFSET}" -lt "${TOTAL_TASKS}" ]; do
    if [ "${MAX_SUBMITTED_CHUNKS}" -gt 0 ] && [ "${SUBMITTED}" -ge "${MAX_SUBMITTED_CHUNKS}" ]; then
        echo "[INFO] Stopped at TASK_OFFSET=${TASK_OFFSET}; use this offset to resume."
        break
    fi
    COUNT="${MAX_ARRAY_TASKS}"
    REMAINING=$((TOTAL_TASKS - TASK_OFFSET))
    if [ "${REMAINING}" -lt "${COUNT}" ]; then COUNT="${REMAINING}"; fi
    ARRAY_SPEC="0-$((COUNT - 1))%${MAX_CONCURRENT_TASKS}"
    END=$((TASK_OFFSET + COUNT - 1))
    echo "[INFO] Submitting global tasks ${TASK_OFFSET}-${END} as ${ARRAY_SPEC}."
    "${SBATCH_BIN}" \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="${ARRAY_SPEC}" \
        --output="${LOG_DIR}/charm-only-remesh-%A_%a.out" \
        --export="${BASE_EXPORT},TASK_OFFSET=${TASK_OFFSET}" \
        "${SLURM_SCRIPT}"
    TASK_OFFSET=$((TASK_OFFSET + COUNT))
    SUBMITTED=$((SUBMITTED + 1))
done
echo "[INFO] Submitted ${SUBMITTED} array chunk(s)."
