#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ "$#" -ne 1 ]; then
    echo "Usage: bash $0 {Left_M1|Right_DLPC|Right_Thalamus}" >&2
    exit 2
fi

ROI_PREFIX="$1"
case "${ROI_PREFIX}" in
    Left_M1)
        MONTAGE_PRESET="left-m1"
        ROI_SLUG="left_m1"
        ;;
    Right_DLPC)
        MONTAGE_PRESET="right-dlpfc"
        ROI_SLUG="right_dlpc"
        ;;
    Right_Thalamus)
        MONTAGE_PRESET="right-thalamus"
        ROI_SLUG="right_thalamus"
        ;;
    *)
        echo "[ERROR] Unsupported ROI prefix: ${ROI_PREFIX}" >&2
        echo "[ERROR] Expected Left_M1, Right_DLPC, or Right_Thalamus." >&2
        exit 2
        ;;
esac

SUBJECTS_FILE="${SUBJECTS_FILE:-${SCRIPT_DIR}/approved_wave/accepted_wave_1_subjects.txt}"
STUDY_ROOT="${STUDY_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Approved_Wave_1}"
CANONICAL_CAMPAIGN_ROOT="${CANONICAL_CAMPAIGN_ROOT:-${STUDY_ROOT}/campaign/Left_Hippocampus}"
CANONICAL_PREP_MANIFEST="${CANONICAL_PREP_MANIFEST:-${CANONICAL_CAMPAIGN_ROOT}/prep_tasks.tsv}"
CANONICAL_DATASET_NAME="${CANONICAL_DATASET_NAME:-Left_Hippocampus_Data_01}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STUDY_ROOT}/${ROI_PREFIX}_Runs}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${STUDY_ROOT}/campaign/${ROI_PREFIX}}"
SIMULATION_RESULT_DIR="${SIMULATION_RESULT_DIR:-${CAMPAIGN_ROOT}/simulation_results}"
LOG_DIR="${LOG_DIR:-${CAMPAIGN_ROOT}/logs}"
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
EXPECTED_SIMULATION_TASKS="${EXPECTED_SIMULATION_TASKS:-890}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MAX_RETRIES="${TI_APPROVED_WAVE_MAX_RETRIES:-unlimited}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
JOB_NAME="${JOB_NAME:-approved_wave_${ROI_SLUG}}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"

for required in \
    "${SUBJECTS_FILE}" \
    "${CANONICAL_PREP_MANIFEST}" \
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

for value_name in \
    EXPECTED_SUBJECTS \
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
if [ "${MAX_RETRIES}" != "unlimited" ] && ! [[ "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] MAX_RETRIES must be 'unlimited' or a non-negative integer." >&2
    exit 2
fi

mkdir -p \
    "${CAMPAIGN_ROOT}" \
    "${SIMULATION_RESULT_DIR}" \
    "${LOG_DIR}"

python3 "${WORKFLOW_PY}" scaffold-preflight \
    --subjects-file "${SUBJECTS_FILE}" \
    --canonical-prep-manifest "${CANONICAL_PREP_MANIFEST}" \
    --canonical-dataset-name "${CANONICAL_DATASET_NAME}" \
    --output-root "${OUTPUT_ROOT}" \
    --simulation-result-dir "${SIMULATION_RESULT_DIR}" \
    --simulation-manifest "${SIMULATION_MANIFEST}" \
    --summary "${SUMMARY}" \
    --dataset-prefix "${ROI_PREFIX}" \
    --montage-preset "${MONTAGE_PRESET}" \
    --targets-csv "${TARGETS_CSV}" \
    --expected-targets-sha256 "${EXPECTED_TARGETS_SHA256}" \
    --repeats 01 02 03 04 05 06 07 08 09 10 \
    --expected-subjects "${EXPECTED_SUBJECTS}" \
    --expected-simulation-tasks "${EXPECTED_SIMULATION_TASKS}"

SIM_TOTAL=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${SIMULATION_MANIFEST}")
SIM_READY=$(awk -F '\t' 'NR > 1 && $12 == "ready" { count++ } END { print count + 0 }' "${SIMULATION_MANIFEST}")
SIM_SUBJECTS=$(awk -F '\t' 'NR > 1 { seen[$5]=1 } END { for (subject in seen) count++; print count + 0 }' "${SIMULATION_MANIFEST}")
SIM_REPEATS=$(awk -F '\t' 'NR > 1 { seen[$3]=1 } END { for (repeat in seen) count++; print count + 0 }' "${SIMULATION_MANIFEST}")
CANONICAL_SCAFFOLDS=$(awk -F '\t' 'NR > 1 { seen[$8]=1 } END { for (path in seen) count++; print count + 0 }' "${SIMULATION_MANIFEST}")
if [ "${SIM_TOTAL}" -ne "${EXPECTED_SIMULATION_TASKS}" ] || \
   [ "${SIM_READY}" -ne "${EXPECTED_SIMULATION_TASKS}" ] || \
   [ "${SIM_SUBJECTS}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${SIM_REPEATS}" -ne 10 ] || \
   [ "${CANONICAL_SCAFFOLDS}" -ne "${EXPECTED_SUBJECTS}" ]
then
    echo "[ERROR] External-scaffold scope gate failed." >&2
    printf 'tasks=%s ready=%s subjects=%s repeats=%s scaffolds=%s\n' \
        "${SIM_TOTAL}" \
        "${SIM_READY}" \
        "${SIM_SUBJECTS}" \
        "${SIM_REPEATS}" \
        "${CANONICAL_SCAFFOLDS}" >&2
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
    --preset "${MONTAGE_PRESET}" \
    --targets-csv "${TARGETS_CSV}" \
    --expected-targets-sha256 "${TARGETS_SHA256}"

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
    if [ -n "${MAX_ARRAY_SIZE}" ] && [ "${EXPECTED_SIMULATION_TASKS}" -gt "${MAX_ARRAY_SIZE}" ]; then
        echo "[ERROR] ${EXPECTED_SIMULATION_TASKS} tasks exceed Slurm MaxArraySize=${MAX_ARRAY_SIZE}." >&2
        exit 2
    fi
fi

SIMULATION_ARRAY="0-$((EXPECTED_SIMULATION_TASKS - 1))%${MAX_CONCURRENT_TASKS}"
printf '%s\n' \
    'Scope:' \
    "  dataset/ROI: ${ROI_PREFIX} approved wave 1" \
    "  canonical scaffold dataset: ${CANONICAL_DATASET_NAME}" \
    "  canonical subjects/scaffolds: ${EXPECTED_SUBJECTS}" \
    '  repeats: 10 (01-10)' \
    "  simulation tasks: ${EXPECTED_SIMULATION_TASKS}" \
    "  array: ${SIMULATION_ARRAY}" \
    '  expected CHARM segmentation runs: 0' \
    "  expected physical scaffold copies: ${EXPECTED_SIMULATION_TASKS}" \
    "  expected independent meshes: ${EXPECTED_SIMULATION_TASKS}" \
    "  expected validated FEM simulations: ${EXPECTED_SIMULATION_TASKS}" \
    '  execution: full requested ROI; not a smoke or subset' \
    '  architecture: every repeat copies the validated scaffold and remeshes independently'

echo "[INFO] Canonical preparation: ${CANONICAL_PREP_MANIFEST}"
echo "[INFO] Simulation manifest:  ${SIMULATION_MANIFEST}"
echo "[INFO] Output root:          ${OUTPUT_ROOT}"
echo "[INFO] Simulation results:   ${SIMULATION_RESULT_DIR}"
echo "[INFO] Log directory:        ${LOG_DIR}"
echo "[INFO] Montage preset:       ${MONTAGE_PRESET}"
echo "[INFO] targets.csv:          ${TARGETS_CSV}"
echo "[INFO] targets SHA-256:      ${TARGETS_SHA256}"
echo "[INFO] Resource profile:     SimNIBS/4.0.1-foss-2023a, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Retry limit:          ${MAX_RETRIES}"
echo "[INFO] CHARM segmentation:   none"
echo "[INFO] ROAST involvement:    none"

EXPORTS="ALL,TI_APPROVED_WAVE_WORKFLOW_PY=${WORKFLOW_PY},TI_APPROVED_WAVE_LOG_DIR=${LOG_DIR},TI_TARGETS_CSV=${TARGETS_CSV},TI_EXPECTED_TARGETS_SHA256=${TARGETS_SHA256},TI_MONTAGE_PRESET=${MONTAGE_PRESET},TI_APPROVED_WAVE_MAX_RETRIES=${MAX_RETRIES},TI_APPROVED_WAVE_STAGE=simulate,TI_APPROVED_WAVE_MANIFEST=${SIMULATION_MANIFEST},TI_APPROVED_WAVE_RESULT_DIR=${SIMULATION_RESULT_DIR},TI_SIM_RUNNER_PY=${SIM_RUNNER_PY},TI_COMPLETION_CHECK_PY=${COMPLETION_CHECK_PY}"

set +e
SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --array="${SIMULATION_ARRAY}" \
    --output="${LOG_DIR}/approved-wave-${ROI_SLUG}-%A_%a.out" \
    --export="${EXPORTS}" \
    "${SLURM_SCRIPT}" 2>&1)
SUBMIT_EXIT=$?
set -e
echo "${SUBMISSION}"
if [ "${SUBMIT_EXIT}" -ne 0 ]; then
    echo "[ERROR] ${ROI_PREFIX} array submission failed." >&2
    exit "${SUBMIT_EXIT}"
fi
JOB_ID="${SUBMISSION%%;*}"
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse job ID: ${SUBMISSION}" >&2
    exit 2
fi

echo "[INFO] Submitted ${ROI_PREFIX} independent-remesh/FEM array job: ${JOB_ID}"
