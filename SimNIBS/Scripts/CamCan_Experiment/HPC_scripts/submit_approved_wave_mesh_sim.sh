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
PREP_RESULT_DIR="${PREP_RESULT_DIR:-${CAMPAIGN_ROOT}/prep_results}"
SIMULATION_RESULT_DIR="${SIMULATION_RESULT_DIR:-${CAMPAIGN_ROOT}/simulation_results}"
LOG_DIR="${LOG_DIR:-${CAMPAIGN_ROOT}/logs}"
PREP_MANIFEST="${PREP_MANIFEST:-${CAMPAIGN_ROOT}/prep_tasks.tsv}"
SIMULATION_MANIFEST="${SIMULATION_MANIFEST:-${CAMPAIGN_ROOT}/simulation_tasks.tsv}"
SUMMARY="${SUMMARY:-${CAMPAIGN_ROOT}/preflight.json}"

WORKFLOW_PY="${TI_APPROVED_WAVE_WORKFLOW_PY:-${SCRIPT_DIR}/approved_wave/workflow.py}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${SCRIPT_DIR}/HPC_scripts/approved_wave_mesh_sim_array.slurm}"
SIM_RUNNER_PY="${TI_SIM_RUNNER_PY:-${SCRIPT_DIR}/simulation/TI_runner_multi-core.py}"
COMPLETION_CHECK_PY="${TI_COMPLETION_CHECK_PY:-${SCRIPT_DIR}/simulation/validate_simulation_outputs.py}"
MONTAGE_VALIDATOR_PY="${TI_MONTAGE_VALIDATOR_PY:-${SCRIPT_DIR}/simulation/validate_montage_selection.py}"
TARGETS_CSV="${TI_TARGETS_CSV:-${PIPELINE_DIR}/utils/targets.csv}"
EXPECTED_TARGETS_SHA256="${TI_EXPECTED_TARGETS_SHA256:-97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6}"

EXPECTED_SUBJECTS="${EXPECTED_SUBJECTS:-89}"
EXPECTED_PREP_TASKS="${EXPECTED_PREP_TASKS:-${EXPECTED_SUBJECTS}}"
EXPECTED_SIMULATION_TASKS="${EXPECTED_SIMULATION_TASKS:-${EXPECTED_TASKS:-890}}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MAX_RETRIES="${TI_APPROVED_WAVE_MAX_RETRIES:-2}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
PREP_JOB_NAME="${PREP_JOB_NAME:-approved_wave_lhip_prep}"
SIMULATION_JOB_NAME="${SIMULATION_JOB_NAME:-approved_wave_lhip_sim}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"

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
for required_dir in "${SOURCE_ROOT_1}" "${SOURCE_ROOT_2}" "${MAP_ROOT}"; do
    if [ ! -d "${required_dir}" ]; then
        echo "[ERROR] Required directory is missing: ${required_dir}" >&2
        exit 2
    fi
done
for value_name in \
    EXPECTED_SUBJECTS \
    EXPECTED_PREP_TASKS \
    EXPECTED_SIMULATION_TASKS \
    MAX_CONCURRENT_TASKS \
    CPUS_PER_TASK
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer; got '${value}'." >&2
        exit 2
    fi
done
if ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi

mkdir -p \
    "${CAMPAIGN_ROOT}" \
    "${PREP_RESULT_DIR}" \
    "${SIMULATION_RESULT_DIR}" \
    "${LOG_DIR}"

python3 "${WORKFLOW_PY}" preflight \
    --subjects-file "${SUBJECTS_FILE}" \
    --source-root "${SOURCE_ROOT_1}" \
    --source-root "${SOURCE_ROOT_2}" \
    --map-root "${MAP_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --prep-result-dir "${PREP_RESULT_DIR}" \
    --simulation-result-dir "${SIMULATION_RESULT_DIR}" \
    --prep-manifest "${PREP_MANIFEST}" \
    --simulation-manifest "${SIMULATION_MANIFEST}" \
    --summary "${SUMMARY}" \
    --dataset-prefix Left_Hippocampus \
    --repeats 01 02 03 04 05 06 07 08 09 10 \
    --expected-subjects "${EXPECTED_SUBJECTS}" \
    --expected-prep-tasks "${EXPECTED_PREP_TASKS}" \
    --expected-simulation-tasks "${EXPECTED_SIMULATION_TASKS}"

PREP_TOTAL=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${PREP_MANIFEST}")
PREP_READY=$(awk -F '\t' 'NR > 1 && $16 == "ready" { count++ } END { print count + 0 }' "${PREP_MANIFEST}")
PREP_SUBJECTS=$(awk -F '\t' 'NR > 1 { seen[$3]=1 } END { for (subject in seen) count++; print count + 0 }' "${PREP_MANIFEST}")
SIM_TOTAL=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${SIMULATION_MANIFEST}")
SIM_READY=$(awk -F '\t' 'NR > 1 && $12 == "ready" { count++ } END { print count + 0 }' "${SIMULATION_MANIFEST}")
SIM_SUBJECTS=$(awk -F '\t' 'NR > 1 { seen[$5]=1 } END { for (subject in seen) count++; print count + 0 }' "${SIMULATION_MANIFEST}")
SIM_REPEATS=$(awk -F '\t' 'NR > 1 { seen[$3]=1 } END { for (repeat in seen) count++; print count + 0 }' "${SIMULATION_MANIFEST}")
if [ "${PREP_TOTAL}" -ne "${EXPECTED_PREP_TASKS}" ] || \
   [ "${PREP_READY}" -ne "${EXPECTED_PREP_TASKS}" ] || \
   [ "${PREP_SUBJECTS}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${SIM_TOTAL}" -ne "${EXPECTED_SIMULATION_TASKS}" ] || \
   [ "${SIM_READY}" -ne "${EXPECTED_SIMULATION_TASKS}" ] || \
   [ "${SIM_SUBJECTS}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${SIM_REPEATS}" -ne 10 ]
then
    echo "[ERROR] Two-stage preflight scope gate failed." >&2
    printf 'prep=%s prep_ready=%s prep_subjects=%s sim=%s sim_ready=%s sim_subjects=%s repeats=%s\n' \
        "${PREP_TOTAL}" \
        "${PREP_READY}" \
        "${PREP_SUBJECTS}" \
        "${SIM_TOTAL}" \
        "${SIM_READY}" \
        "${SIM_SUBJECTS}" \
        "${SIM_REPEATS}" >&2
    exit 2
fi

TARGETS_SHA256=$(sha256sum "${TARGETS_CSV}" | awk '{print $1}')
if [ "${TARGETS_SHA256}" != "${EXPECTED_TARGETS_SHA256}" ]; then
    echo "[ERROR] targets.csv hash mismatch." >&2
    echo "[ERROR] actual=${TARGETS_SHA256} expected=${EXPECTED_TARGETS_SHA256}" >&2
    exit 2
fi
python3 "${MONTAGE_VALIDATOR_PY}" \
    --manifest "${SIMULATION_MANIFEST}" \
    --preset left-hippocampus \
    --targets-csv "${TARGETS_CSV}" \
    --expected-targets-sha256 "${TARGETS_SHA256}"

LARGEST_ARRAY_TASKS="${EXPECTED_SIMULATION_TASKS}"
if [ "${EXPECTED_PREP_TASKS}" -gt "${LARGEST_ARRAY_TASKS}" ]; then
    LARGEST_ARRAY_TASKS="${EXPECTED_PREP_TASKS}"
fi
if command -v "${SCONTROL_BIN}" >/dev/null 2>&1; then
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
    if [ -n "${MAX_ARRAY_SIZE}" ] && [ "${LARGEST_ARRAY_TASKS}" -gt "${MAX_ARRAY_SIZE}" ]; then
        echo "[ERROR] ${LARGEST_ARRAY_TASKS} tasks exceed Slurm MaxArraySize=${MAX_ARRAY_SIZE}." >&2
        exit 2
    fi
fi

PREP_ARRAY="0-$((EXPECTED_PREP_TASKS - 1))%${MAX_CONCURRENT_TASKS}"
SIMULATION_ARRAY="0-$((EXPECTED_SIMULATION_TASKS - 1))%${MAX_CONCURRENT_TASKS}"
TOTAL_SCHEDULER_TASKS=$((EXPECTED_PREP_TASKS + EXPECTED_SIMULATION_TASKS))
printf '%s\n' \
    'Scope:' \
    '  dataset/ROI: Left_Hippocampus approved wave 1' \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    '  repeats: 10 (01-10)' \
    "  preparation tasks: ${EXPECTED_PREP_TASKS}" \
    "  simulation tasks: ${EXPECTED_SIMULATION_TASKS}" \
    "  scheduler tasks total: ${TOTAL_SCHEDULER_TASKS}" \
    "  preparation array: ${PREP_ARRAY}" \
    "  simulation array: ${SIMULATION_ARRAY} (afterok preparation)" \
    "  expected CHARM support builds: ${EXPECTED_SUBJECTS}" \
    "  expected independent meshes: ${EXPECTED_SIMULATION_TASKS}" \
    "  expected validated FEM simulations: ${EXPECTED_SIMULATION_TASKS}" \
    '  execution: full requested wave; not a smoke or subset' \
    '  architecture: repeat 01 builds the scaffold/first mesh; repeats 02-10 copy the scaffold and remesh independently'

echo "[INFO] Preparation manifest: ${PREP_MANIFEST}"
echo "[INFO] Simulation manifest:  ${SIMULATION_MANIFEST}"
echo "[INFO] Output root:          ${OUTPUT_ROOT}"
echo "[INFO] Preparation results:  ${PREP_RESULT_DIR}"
echo "[INFO] Simulation results:   ${SIMULATION_RESULT_DIR}"
echo "[INFO] Log directory:        ${LOG_DIR}"
echo "[INFO] targets.csv:          ${TARGETS_CSV}"
echo "[INFO] targets SHA-256:      ${TARGETS_SHA256}"
echo "[INFO] Resource profile:     SimNIBS/4.0.1-foss-2023a, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Retry limit:          ${MAX_RETRIES}"
echo "[INFO] ROAST involvement:    none"
echo "[INFO] Existing CamCan scaffolding is not read, modified, or deleted."

COMMON_EXPORTS="TI_APPROVED_WAVE_WORKFLOW_PY=${WORKFLOW_PY},TI_APPROVED_WAVE_LOG_DIR=${LOG_DIR},TI_TARGETS_CSV=${TARGETS_CSV},TI_EXPECTED_TARGETS_SHA256=${TARGETS_SHA256},TI_MONTAGE_PRESET=left-hippocampus,TI_APPROVED_WAVE_MAX_RETRIES=${MAX_RETRIES}"
PREP_EXPORTS="ALL,${COMMON_EXPORTS},TI_APPROVED_WAVE_STAGE=prepare,TI_APPROVED_WAVE_MANIFEST=${PREP_MANIFEST},TI_APPROVED_WAVE_RESULT_DIR=${PREP_RESULT_DIR}"
SIM_EXPORTS="ALL,${COMMON_EXPORTS},TI_APPROVED_WAVE_STAGE=simulate,TI_APPROVED_WAVE_MANIFEST=${SIMULATION_MANIFEST},TI_APPROVED_WAVE_RESULT_DIR=${SIMULATION_RESULT_DIR},TI_SIM_RUNNER_PY=${SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${COMPLETION_CHECK_PY}"

set +e
PREP_SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --job-name="${PREP_JOB_NAME}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --array="${PREP_ARRAY}" \
    --output="${LOG_DIR}/approved-wave-prepare-%A_%a.out" \
    --export="${PREP_EXPORTS}" \
    "${SLURM_SCRIPT}" 2>&1)
PREP_EXIT=$?
set -e
echo "${PREP_SUBMISSION}"
if [ "${PREP_EXIT}" -ne 0 ]; then
    echo "[ERROR] Preparation array submission failed." >&2
    exit "${PREP_EXIT}"
fi
PREP_JOB_ID="${PREP_SUBMISSION%%;*}"
if ! [[ "${PREP_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse preparation job ID: ${PREP_SUBMISSION}" >&2
    exit 2
fi

set +e
SIM_SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --dependency="afterok:${PREP_JOB_ID}" \
    --job-name="${SIMULATION_JOB_NAME}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --array="${SIMULATION_ARRAY}" \
    --output="${LOG_DIR}/approved-wave-simulate-%A_%a.out" \
    --export="${SIM_EXPORTS}" \
    "${SLURM_SCRIPT}" 2>&1)
SIM_EXIT=$?
set -e
echo "${SIM_SUBMISSION}"
if [ "${SIM_EXIT}" -ne 0 ]; then
    echo "[ERROR] Simulation array submission failed; cancelling preparation job ${PREP_JOB_ID}." >&2
    "${SCANCEL_BIN}" "${PREP_JOB_ID}" || true
    exit "${SIM_EXIT}"
fi
SIM_JOB_ID="${SIM_SUBMISSION%%;*}"
if ! [[ "${SIM_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse simulation job ID: ${SIM_SUBMISSION}" >&2
    "${SCANCEL_BIN}" "${PREP_JOB_ID}" || true
    exit 2
fi

echo "[INFO] Submitted preparation array job: ${PREP_JOB_ID}"
echo "[INFO] Submitted dependent simulation array job: ${SIM_JOB_ID}"
echo "[INFO] Simulation dependency: afterok:${PREP_JOB_ID}"
