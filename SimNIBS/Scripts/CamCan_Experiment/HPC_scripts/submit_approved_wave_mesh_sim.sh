#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBJECTS_FILE="${SUBJECTS_FILE:-${SCRIPT_DIR}/approved_wave/accepted_wave_1_subjects.txt}"
SOURCE_ROOT_1="${SOURCE_ROOT_1:-/mnt/parscratch/users/cop23bi/CamCanMRI}"
SOURCE_ROOT_2="${SOURCE_ROOT_2:-/mnt/parscratch/users/cop23bi/ti_dataset}"
MAP_ROOT="${MAP_ROOT:-/mnt/parscratch/users/cop23bi/charm_segmentations/maps}"
STUDY_ROOT="${STUDY_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Approved_Wave_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STUDY_ROOT}/Left_Hippocampus_Runs}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${STUDY_ROOT}/campaign/Left_Hippocampus}"
RESULT_DIR="${RESULT_DIR:-${CAMPAIGN_ROOT}/task_results}"
LOG_DIR="${LOG_DIR:-${CAMPAIGN_ROOT}/logs}"
MANIFEST="${MANIFEST:-${CAMPAIGN_ROOT}/tasks.tsv}"
SUMMARY="${SUMMARY:-${CAMPAIGN_ROOT}/preflight.json}"

WORKFLOW_PY="${TI_APPROVED_WAVE_WORKFLOW_PY:-${SCRIPT_DIR}/approved_wave/workflow.py}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/HPC_scripts/approved_wave_mesh_sim_array.slurm}"
SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-${SCRIPT_DIR}/simulation/TI_runner_multi-core.py}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-${SCRIPT_DIR}/simulation/validate_simulation_outputs.py}"
MONTAGE_VALIDATOR_PY="${TI_MONTAGE_VALIDATOR_PY:-${SCRIPT_DIR}/simulation/validate_montage_selection.py}"
TARGETS_CSV="${TI_TARGETS_CSV:-${PIPELINE_DIR}/utils/targets.csv}"
EXPECTED_TARGETS_SHA256="${TI_EXPECTED_TARGETS_SHA256:-97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6}"

EXPECTED_SUBJECTS="${EXPECTED_SUBJECTS:-89}"
EXPECTED_TASKS="${EXPECTED_TASKS:-890}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MAX_RETRIES="${TI_APPROVED_WAVE_MAX_RETRIES:-2}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
JOB_NAME="${JOB_NAME:-approved_wave_lhip}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"

for required in \
    "${SUBJECTS_FILE}" \
    "${WORKFLOW_PY}" \
    "${SLURM_SCRIPT}" \
    "${SIM_RUNNER_PY}" \
    "${COMPLETION_CHECK_PY}" \
    "${MONTAGE_VALIDATOR_PY}" \
    "${TARGETS_CSV}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required file is missing: ${required}" >&2
        exit 2
    fi
done
for required_dir in "${SOURCE_ROOT_1}" "${SOURCE_ROOT_2}" "${MAP_ROOT}"
do
    if [ ! -d "${required_dir}" ]; then
        echo "[ERROR] Required directory is missing: ${required_dir}" >&2
        exit 2
    fi
done
for value_name in EXPECTED_SUBJECTS EXPECTED_TASKS MAX_CONCURRENT_TASKS MAX_RETRIES CPUS_PER_TASK
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer; got '${value}'." >&2
        exit 2
    fi
done

mkdir -p "${CAMPAIGN_ROOT}" "${RESULT_DIR}" "${LOG_DIR}"

python3 "${WORKFLOW_PY}" preflight \
    --subjects-file "${SUBJECTS_FILE}" \
    --source-root "${SOURCE_ROOT_1}" \
    --source-root "${SOURCE_ROOT_2}" \
    --map-root "${MAP_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --result-dir "${RESULT_DIR}" \
    --manifest "${MANIFEST}" \
    --summary "${SUMMARY}" \
    --dataset-prefix Left_Hippocampus \
    --repeats 01 02 03 04 05 06 07 08 09 10 \
    --expected-subjects "${EXPECTED_SUBJECTS}" \
    --expected-tasks "${EXPECTED_TASKS}"

TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${MANIFEST}")
READY_TASKS=$(awk -F '\t' 'NR > 1 && $18 == "ready" { count++ } END { print count + 0 }' "${MANIFEST}")
UNIQUE_SUBJECTS=$(awk -F '\t' 'NR > 1 { seen[$5]=1 } END { for (subject in seen) count++; print count + 0 }' "${MANIFEST}")
UNIQUE_REPEATS=$(awk -F '\t' 'NR > 1 { seen[$3]=1 } END { for (repeat in seen) count++; print count + 0 }' "${MANIFEST}")
if [ "${TOTAL_TASKS}" -ne "${EXPECTED_TASKS}" ] || \
   [ "${READY_TASKS}" -ne "${EXPECTED_TASKS}" ] || \
   [ "${UNIQUE_SUBJECTS}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${UNIQUE_REPEATS}" -ne 10 ]
then
    echo "[ERROR] Preflight scope gate failed." >&2
    printf 'tasks=%s ready=%s subjects=%s repeats=%s\n' \
        "${TOTAL_TASKS}" "${READY_TASKS}" "${UNIQUE_SUBJECTS}" "${UNIQUE_REPEATS}" >&2
    exit 2
fi

TARGETS_SHA256=$(sha256sum "${TARGETS_CSV}" | awk '{print $1}')
if [ "${TARGETS_SHA256}" != "${EXPECTED_TARGETS_SHA256}" ]; then
    echo "[ERROR] targets.csv hash mismatch." >&2
    echo "[ERROR] actual=${TARGETS_SHA256} expected=${EXPECTED_TARGETS_SHA256}" >&2
    exit 2
fi
python3 "${MONTAGE_VALIDATOR_PY}" \
    --manifest "${MANIFEST}" \
    --preset left-hippocampus \
    --targets-csv "${TARGETS_CSV}" \
    --expected-targets-sha256 "${TARGETS_SHA256}"

ARRAY_END=$((EXPECTED_TASKS - 1))
ARRAY_SPEC="0-${ARRAY_END}%${MAX_CONCURRENT_TASKS}"
if command -v "${SCONTROL_BIN}" >/dev/null 2>&1
then
    set +e
    SCONTROL_CONFIG=$("${SCONTROL_BIN}" show config 2>&1)
    SCONTROL_EXIT=$?
    set -e
    if [ "${SCONTROL_EXIT}" -ne 0 ]; then
        echo "[ERROR] Could not inspect the live Slurm MaxArraySize." >&2
        echo "${SCONTROL_CONFIG}" >&2
        exit "${SCONTROL_EXIT}"
    fi
    MAX_ARRAY_SIZE=$(printf '%s\n' "${SCONTROL_CONFIG}" | awk -F '=' '$1 ~ /^[[:space:]]*MaxArraySize/ && !found { gsub(/[[:space:]]/, "", $2); print $2; found=1 }')
    if [ -n "${MAX_ARRAY_SIZE}" ] && [ "${EXPECTED_TASKS}" -gt "${MAX_ARRAY_SIZE}" ]; then
        echo "[ERROR] ${EXPECTED_TASKS} tasks exceed Slurm MaxArraySize=${MAX_ARRAY_SIZE}." >&2
        exit 2
    fi
fi

printf '%s\n' \
    'Scope:' \
    '  dataset/ROI: Left_Hippocampus approved wave 1' \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    '  repeats: 10 (01-10)' \
    "  tasks: ${EXPECTED_TASKS}" \
    "  array: ${ARRAY_SPEC}" \
    "  expected meshes: ${EXPECTED_TASKS}" \
    "  expected validated simulations: ${EXPECTED_TASKS}" \
    '  execution: full requested wave; not a smoke or subset' \
    '  per task: CHARM prerequisites -> exact approved label -> mesh -> optimized simulation -> validation'

echo "[INFO] Manifest:          ${MANIFEST}"
echo "[INFO] Output root:       ${OUTPUT_ROOT}"
echo "[INFO] Result directory:  ${RESULT_DIR}"
echo "[INFO] Log directory:     ${LOG_DIR}"
echo "[INFO] targets.csv:       ${TARGETS_CSV}"
echo "[INFO] targets SHA-256:   ${TARGETS_SHA256}"
echo "[INFO] Resource profile:  SimNIBS/4.0.1-foss-2023a, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Retry limit:       ${MAX_RETRIES}"
echo "[INFO] Existing CamCan scaffolding is not read, modified, or deleted by this submission."

EXPORT_VARS="ALL,TI_APPROVED_WAVE_MANIFEST=${MANIFEST},TI_APPROVED_WAVE_WORKFLOW_PY=${WORKFLOW_PY},TI_APPROVED_WAVE_RESULT_DIR=${RESULT_DIR},TI_APPROVED_WAVE_LOG_DIR=${LOG_DIR},TI_SIM_RUNNER_PY=${SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${COMPLETION_CHECK_PY},TI_TARGETS_CSV=${TARGETS_CSV},TI_EXPECTED_TARGETS_SHA256=${TARGETS_SHA256},TI_MONTAGE_PRESET=left-hippocampus,TI_APPROVED_WAVE_MAX_RETRIES=${MAX_RETRIES}"
SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/approved-wave-lhip-%A_%a.out" \
    --export="${EXPORT_VARS}" \
    "${SLURM_SCRIPT}")
JOB_ID="${SUBMISSION%%;*}"
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse Slurm job ID from: ${SUBMISSION}" >&2
    exit 2
fi
echo "[INFO] Submitted approved wave-1 array job: ${JOB_ID}"
