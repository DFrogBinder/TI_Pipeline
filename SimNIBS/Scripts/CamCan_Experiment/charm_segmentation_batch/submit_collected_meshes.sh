#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MANIFEST:?Set MANIFEST to the ready collected-CHARM mesh manifest.}"
: "${LOG_DIR:?Set LOG_DIR for Slurm output and persistent task logs.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/mesh_collected_segmentations_array.slurm}"
WORKFLOW_PY="${TI_CHARM_MESH_WORKFLOW_PY:-${SCRIPT_DIR}/mesh_collected_segmentations.py}"
EXPECTED_TASKS="${EXPECTED_TASKS:-474}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MAX_RETRIES="${TI_CHARM_MESH_MAX_RETRIES:-2}"
WORKERS_PER_ARRAY_TASK="${TI_CHARM_MESH_WORKERS_PER_ARRAY_TASK:-1}"
LOCAL_STAGING="${TI_CHARM_MESH_LOCAL_STAGING:-0}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
JOB_NAME="${JOB_NAME:-mesh_charm_maps_${EXPECTED_TASKS}}"
SETTINGS_PATH="${TI_CHARM_MESH_SETTINGS:-}"

resolve_path() {
    "${PYTHON_BIN}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

MANIFEST="$(resolve_path "${MANIFEST}")"
LOG_DIR="$(resolve_path "${LOG_DIR}")"
SLURM_SCRIPT="$(resolve_path "${SLURM_SCRIPT}")"
WORKFLOW_PY="$(resolve_path "${WORKFLOW_PY}")"
mkdir -p "${LOG_DIR}"

for path in "${MANIFEST}" "${SLURM_SCRIPT}" "${WORKFLOW_PY}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Required file not found: ${path}" >&2
        exit 2
    fi
done
for value_name in EXPECTED_TASKS MAX_CONCURRENT_TASKS CPUS_PER_TASK MAX_RETRIES WORKERS_PER_ARRAY_TASK; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || { [ "${value_name}" != "MAX_RETRIES" ] && [ "${value}" -lt 1 ]; }; then
        echo "[ERROR] ${value_name} has an invalid value: ${value}." >&2
        exit 2
    fi
done
if [ "${CPUS_PER_TASK}" -lt "${WORKERS_PER_ARRAY_TASK}" ]; then
    echo "[ERROR] CPUS_PER_TASK=${CPUS_PER_TASK} is smaller than workers per array task ${WORKERS_PER_ARRAY_TASK}." >&2
    exit 2
fi
if [ $((CPUS_PER_TASK % WORKERS_PER_ARRAY_TASK)) -ne 0 ]; then
    echo "[ERROR] CPUS_PER_TASK=${CPUS_PER_TASK} must divide evenly across ${WORKERS_PER_ARRAY_TASK} workers." >&2
    exit 2
fi
if [ "${LOCAL_STAGING}" != "0" ] && [ "${LOCAL_STAGING}" != "1" ]; then
    echo "[ERROR] TI_CHARM_MESH_LOCAL_STAGING must be 0 or 1." >&2
    exit 2
fi

awk -F '\t' 'NR == 1 { exit !(($1 == "task_id") && ($2 == "subject") && ($8 == "status")) }' "${MANIFEST}" || {
    echo "[ERROR] Manifest has an unexpected header: ${MANIFEST}" >&2
    exit 2
}
TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${MANIFEST}")
READY_TASKS=$(awk -F '\t' 'NR > 1 && NF > 0 && $8 == "ready" { count++ } END { print count + 0 }' "${MANIFEST}")
UNIQUE_SUBJECTS=$(awk -F '\t' 'NR > 1 && NF > 0 { seen[$2]=1 } END { for (subject in seen) count++; print count + 0 }' "${MANIFEST}")
if [ "${TOTAL_TASKS}" -ne "${EXPECTED_TASKS}" ] || [ "${READY_TASKS}" -ne "${EXPECTED_TASKS}" ] || [ "${UNIQUE_SUBJECTS}" -ne "${EXPECTED_TASKS}" ]; then
    echo "[ERROR] Refusing submission: total=${TOTAL_TASKS}, ready=${READY_TASKS}, unique_subjects=${UNIQUE_SUBJECTS}, expected=${EXPECTED_TASKS}." >&2
    exit 2
fi

ARRAY_TASKS=$(( (EXPECTED_TASKS + WORKERS_PER_ARRAY_TASK - 1) / WORKERS_PER_ARRAY_TASK ))
ARRAY_SPEC="0-$((ARRAY_TASKS - 1))%${MAX_CONCURRENT_TASKS}"
THREADS_PER_WORKER=$((CPUS_PER_TASK / WORKERS_PER_ARRAY_TASK))
MAX_SUBJECT_CONCURRENCY=$((MAX_CONCURRENT_TASKS * WORKERS_PER_ARRAY_TASK))
echo "Scope:"
echo "  dataset: CamCan collected CHARM segmentations"
echo "  subjects: ${UNIQUE_SUBJECTS}"
echo "  tasks: ${TOTAL_TASKS}"
echo "  array elements: ${ARRAY_TASKS}"
echo "  array: ${ARRAY_SPEC}"
echo "  workers per element: ${WORKERS_PER_ARRAY_TASK}"
echo "  threads per worker: ${THREADS_PER_WORKER}"
echo "  maximum simultaneous subjects: ${MAX_SUBJECT_CONCURRENCY}"
echo "  expected outputs: ${EXPECTED_TASKS} tetrahedral .msh files and ${EXPECTED_TASKS} provenance JSON files"
echo "  execution: full requested ${EXPECTED_TASKS}-subject mesh-generation run"
echo "[INFO] Manifest:          ${MANIFEST}"
echo "[INFO] Log directory:     ${LOG_DIR}"
echo "[INFO] Resource profile:  SimNIBS/4.0.1-foss-2023a, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Array concurrency: ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Worker processes:  ${WORKERS_PER_ARRAY_TASK} per array element"
echo "[INFO] Threads/worker:    ${THREADS_PER_WORKER}"
echo "[INFO] Subject slots:     ${MAX_SUBJECT_CONCURRENCY}"
echo "[INFO] Retries:           ${MAX_RETRIES}"
echo "[INFO] Meshing mode:      direct CHARM create_mesh from collected maps"
echo "[INFO] Local staging:     ${LOCAL_STAGING}"
echo "[INFO] Segmentation rerun: no"
echo "[INFO] Simulations:        no"

EXPORT_VARS="ALL,TI_CHARM_MESH_MANIFEST=${MANIFEST},TI_CHARM_MESH_WORKFLOW_PY=${WORKFLOW_PY},TI_CHARM_MESH_LOG_DIR=${LOG_DIR},TI_CHARM_MESH_MAX_RETRIES=${MAX_RETRIES},TI_CHARM_MESH_WORKERS_PER_ARRAY_TASK=${WORKERS_PER_ARRAY_TASK},TI_CHARM_MESH_THREADS_PER_WORKER=${THREADS_PER_WORKER},TI_CHARM_MESH_LOCAL_STAGING=${LOCAL_STAGING}"
if [ -n "${SETTINGS_PATH}" ]; then
    EXPORT_VARS="${EXPORT_VARS},TI_CHARM_MESH_SETTINGS=${SETTINGS_PATH}"
fi

SUBMISSION=$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="${ARRAY_SPEC}" \
        --output="${LOG_DIR}/mesh-collected-charm-%A_%a.out" \
        --export="${EXPORT_VARS}" \
        "${SLURM_SCRIPT}"
)
JOB_ID="${SUBMISSION%%;*}"
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse array job ID from: ${SUBMISSION}" >&2
    exit 2
fi
echo "[INFO] Submitted full collected-CHARM mesh array: ${JOB_ID}"
