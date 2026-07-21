#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPS_ROOT="${MAPS_ROOT:-/mnt/parscratch/users/cop23bi/charm_segmentations/maps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/charm_segmentations_corrected_v1}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${OUTPUT_ROOT}/campaign}"
MANIFEST="${MANIFEST:-${CAMPAIGN_ROOT}/cleanup_manifest.tsv}"
PREFLIGHT_SUMMARY="${PREFLIGHT_SUMMARY:-${CAMPAIGN_ROOT}/preflight.json}"
VALIDATION="${VALIDATION:-${CAMPAIGN_ROOT}/validation.tsv}"
VALIDATION_SUMMARY="${VALIDATION_SUMMARY:-${CAMPAIGN_ROOT}/validation.json}"
COLLECTION_DIR="${COLLECTION_DIR:-${OUTPUT_ROOT}/collection}"
COLLECTION_MANIFEST="${COLLECTION_MANIFEST:-${COLLECTION_DIR}/charm_segmentation_manifest.tsv}"
CHECKSUMS="${CHECKSUMS:-${COLLECTION_DIR}/sha256sums.txt}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

WORKFLOW_PY="${TI_CHARM_CLEANUP_WORKFLOW_PY:-${SCRIPT_DIR}/workflow.py}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-${SCRIPT_DIR}/cleanup_charm_segmentations_array.slurm}"
COLLECTOR_SCRIPT="${COLLECTOR_SCRIPT:-${SCRIPT_DIR}/collect_corrected_segmentations.slurm}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOAD_SIMNIBS_MODULE="${LOAD_SIMNIBS_MODULE:-1}"

MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-50}"
MAX_RETRIES="${TI_CHARM_CLEANUP_MAX_RETRIES:-2}"
CSF_RADIUS="${TI_CHARM_CLEANUP_CSF_RADIUS:-5}"
SKIN_RADIUS="${TI_CHARM_CLEANUP_SKIN_RADIUS:-10}"
CSF_COMPONENT_POLICY="${TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY:-none}"
COMPONENT_POLICY="${TI_CHARM_CLEANUP_COMPONENT_POLICY:-largest}"
WM_MIN_COMPONENT_VOXELS="${TI_CHARM_CLEANUP_WM_MIN_COMPONENT_VOXELS:-0}"
GM_MIN_COMPONENT_VOXELS="${TI_CHARM_CLEANUP_GM_MIN_COMPONENT_VOXELS:-0}"
CONNECTIVITY="${TI_CHARM_CLEANUP_CONNECTIVITY:-26}"
PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-2}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-16G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-charm_seg_cleanup_v1}"
COLLECTOR_JOB_NAME="${COLLECTOR_JOB_NAME:-collect_charm_seg_cleanup_v1}"

mkdir -p "${CAMPAIGN_ROOT}" "${COLLECTION_DIR}" "${LOG_DIR}"

for path in "${MAPS_ROOT}" "${WORKFLOW_PY}" "${ARRAY_SCRIPT}" "${COLLECTOR_SCRIPT}"; do
    if [ ! -e "${path}" ]; then
        echo "[ERROR] Required path is missing: ${path}" >&2
        exit 2
    fi
done
if [ "$(realpath -m "${MAPS_ROOT}")" = "$(realpath -m "${OUTPUT_ROOT}/maps")" ]; then
    echo "[ERROR] Output maps directory must differ from the source maps directory." >&2
    exit 2
fi

for value_name in MAX_CONCURRENT_TASKS MAX_RETRIES CSF_RADIUS SKIN_RADIUS WM_MIN_COMPONENT_VOXELS GM_MIN_COMPONENT_VOXELS CPUS_PER_TASK COLLECTOR_CPUS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${value_name} must be a non-negative integer." >&2
        exit 2
    fi
done
if [ "${MAX_CONCURRENT_TASKS}" -lt 1 ] || [ "${CPUS_PER_TASK}" -lt 1 ] || [ "${COLLECTOR_CPUS}" -lt 1 ]; then
    echo "[ERROR] Concurrency and CPU counts must be positive." >&2
    exit 2
fi
if [ "${COMPONENT_POLICY}" != "largest" ] && [ "${COMPONENT_POLICY}" != "min-size" ]; then
    echo "[ERROR] COMPONENT_POLICY must be largest or min-size." >&2
    exit 2
fi
if [ "${CSF_COMPONENT_POLICY}" != "none" ] && [ "${CSF_COMPONENT_POLICY}" != "largest" ]; then
    echo "[ERROR] CSF_COMPONENT_POLICY must be none or largest." >&2
    exit 2
fi
if [ "${COMPONENT_POLICY}" = "min-size" ] && { [ "${WM_MIN_COMPONENT_VOXELS}" -lt 1 ] || [ "${GM_MIN_COMPONENT_VOXELS}" -lt 1 ]; }; then
    echo "[ERROR] min-size policy requires positive WM and GM thresholds." >&2
    exit 2
fi
if [ "${CONNECTIVITY}" != "6" ] && [ "${CONNECTIVITY}" != "18" ] && [ "${CONNECTIVITY}" != "26" ]; then
    echo "[ERROR] CONNECTIVITY must be 6, 18, or 26." >&2
    exit 2
fi
if [ "${LOAD_SIMNIBS_MODULE}" = "1" ]; then
    module purge
    module use "$HOME/modules"
    module load SimNIBS/4.0.1-foss-2023a
    export PYTHONNOUSERSITE=1
elif [ "${LOAD_SIMNIBS_MODULE}" != "0" ]; then
    echo "[ERROR] LOAD_SIMNIBS_MODULE must be 0 or 1." >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

DISCOVERED_SUBJECTS=$(find "${MAPS_ROOT}" -maxdepth 1 -type f -name 'sub-*_CHARM_tissue_labeling_upsampled.nii.gz' | wc -l)
EXPECTED_SUBJECTS="${EXPECTED_SUBJECTS:-652}"
if ! [[ "${EXPECTED_SUBJECTS}" =~ ^[0-9]+$ ]] || [ "${EXPECTED_SUBJECTS}" -lt 1 ]; then
    echo "[ERROR] EXPECTED_SUBJECTS must be positive." >&2
    exit 2
fi
if [ "${DISCOVERED_SUBJECTS}" -ne "${EXPECTED_SUBJECTS}" ]; then
    echo "[ERROR] Discovered ${DISCOVERED_SUBJECTS} maps but expected ${EXPECTED_SUBJECTS}." >&2
    exit 2
fi

"${PYTHON_BIN}" "${WORKFLOW_PY}" preflight \
    --maps-root "${MAPS_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --manifest "${MANIFEST}" \
    --summary "${PREFLIGHT_SUMMARY}" \
    --expected-subjects "${EXPECTED_SUBJECTS}"

TOTAL_TASKS=$(awk 'NR > 1 && NF > 0 { count++ } END { print count + 0 }' "${MANIFEST}")
READY_TASKS=$(awk -F '\t' 'NR > 1 && $8 == "ready" { count++ } END { print count + 0 }' "${MANIFEST}")
UNIQUE_SUBJECTS=$(awk -F '\t' 'NR > 1 { seen[$2]=1 } END { for (subject in seen) count++; print count + 0 }' "${MANIFEST}")
if [ "${TOTAL_TASKS}" -ne "${EXPECTED_SUBJECTS}" ] || [ "${READY_TASKS}" -ne "${EXPECTED_SUBJECTS}" ] || [ "${UNIQUE_SUBJECTS}" -ne "${EXPECTED_SUBJECTS}" ]; then
    echo "[ERROR] Cleanup scope gate failed: total=${TOTAL_TASKS}, ready=${READY_TASKS}, subjects=${UNIQUE_SUBJECTS}, expected=${EXPECTED_SUBJECTS}." >&2
    exit 2
fi

if command -v "${SCONTROL_BIN}" >/dev/null 2>&1; then
    MAX_ARRAY_SIZE=$("${SCONTROL_BIN}" show config | awk -F '=' '$1 ~ /^[[:space:]]*MaxArraySize/ && !found { gsub(/[[:space:]]/, "", $2); print $2; found=1 }')
    if [ -n "${MAX_ARRAY_SIZE}" ] && [ "${EXPECTED_SUBJECTS}" -gt "${MAX_ARRAY_SIZE}" ]; then
        echo "[ERROR] ${EXPECTED_SUBJECTS} tasks exceed Slurm MaxArraySize=${MAX_ARRAY_SIZE}." >&2
        exit 2
    fi
fi

ARRAY_SPEC="0-$((EXPECTED_SUBJECTS - 1))%${MAX_CONCURRENT_TASKS}"
printf '%s\n' \
    'Scope:' \
    '  dataset: complete collected CamCan CHARM segmentation cohort' \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    "  correction tasks: ${EXPECTED_SUBJECTS}" \
    "  array: ${ARRAY_SPEC}" \
    "  expected corrected maps: ${EXPECTED_SUBJECTS}" \
    "  expected provenance JSON files: ${EXPECTED_SUBJECTS}" \
    '  expected collection manifests: 1' \
    '  execution: full discovered cohort; not a smoke or subset' \
    '  source-map mutation: forbidden' \
    "  CSF closing: ${CSF_RADIUS} voxels" \
    "  CSF components: ${CSF_COMPONENT_POLICY}, connectivity ${CONNECTIVITY}" \
    "  skin closing: ${SKIN_RADIUS} voxels" \
    "  WM/GM components: ${COMPONENT_POLICY}, connectivity ${CONNECTIVITY}"

echo "[INFO] Source maps:       ${MAPS_ROOT}"
echo "[INFO] Corrected root:    ${OUTPUT_ROOT}"
echo "[INFO] Manifest:          ${MANIFEST}"
echo "[INFO] Logs:              ${LOG_DIR}"
echo "[INFO] Preflight Python:  $("${PYTHON_BIN}" --version 2>&1)"
echo "[INFO] Module bootstrap:  ${LOAD_SIMNIBS_MODULE} (SimNIBS/4.0.1-foss-2023a when enabled)"
echo "[INFO] Resource profile:  SimNIBS/4.0.1-foss-2023a, ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Concurrency:       ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Retries:           ${MAX_RETRIES}"

EXPORTS="ALL,TI_CHARM_CLEANUP_MANIFEST=${MANIFEST},TI_CHARM_CLEANUP_WORKFLOW_PY=${WORKFLOW_PY},TI_CHARM_CLEANUP_LOG_DIR=${LOG_DIR},TI_CHARM_CLEANUP_MAX_RETRIES=${MAX_RETRIES},TI_CHARM_CLEANUP_CSF_RADIUS=${CSF_RADIUS},TI_CHARM_CLEANUP_SKIN_RADIUS=${SKIN_RADIUS},TI_CHARM_CLEANUP_CSF_COMPONENT_POLICY=${CSF_COMPONENT_POLICY},TI_CHARM_CLEANUP_COMPONENT_POLICY=${COMPONENT_POLICY},TI_CHARM_CLEANUP_WM_MIN_COMPONENT_VOXELS=${WM_MIN_COMPONENT_VOXELS},TI_CHARM_CLEANUP_GM_MIN_COMPONENT_VOXELS=${GM_MIN_COMPONENT_VOXELS},TI_CHARM_CLEANUP_CONNECTIVITY=${CONNECTIVITY}"
ARRAY_SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --array="${ARRAY_SPEC}" \
    --output="${LOG_DIR}/charm-seg-cleanup-%A_%a.out" \
    --export="${EXPORTS}" \
    "${ARRAY_SCRIPT}")
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
if ! [[ "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse cleanup array job ID: ${ARRAY_SUBMISSION}" >&2
    exit 2
fi

COLLECT_EXPORTS="ALL,TI_CHARM_CLEANUP_MANIFEST=${MANIFEST},TI_CHARM_CLEANUP_WORKFLOW_PY=${WORKFLOW_PY},TI_CHARM_CLEANUP_VALIDATION=${VALIDATION},TI_CHARM_CLEANUP_SUMMARY=${VALIDATION_SUMMARY},TI_CHARM_CLEANUP_COLLECTION_MANIFEST=${COLLECTION_MANIFEST},TI_CHARM_CLEANUP_CHECKSUMS=${CHECKSUMS}"
set +e
COLLECT_SUBMISSION=$("${SBATCH_BIN}" \
    --parsable \
    --job-name="${COLLECTOR_JOB_NAME}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${COLLECTOR_CPUS}" \
    --mem="${COLLECTOR_MEMORY}" \
    --time="${COLLECTOR_TIME}" \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --output="${LOG_DIR}/collect-charm-seg-cleanup-%j.out" \
    --export="${COLLECT_EXPORTS}" \
    "${COLLECTOR_SCRIPT}" 2>&1)
COLLECT_EXIT=$?
set -e
if [ "${COLLECT_EXIT}" -ne 0 ]; then
    echo "[ERROR] Cleanup array ${ARRAY_JOB_ID} was submitted, but collector submission failed: ${COLLECT_SUBMISSION}" >&2
    exit "${COLLECT_EXIT}"
fi
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
if ! [[ "${COLLECT_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not parse collector job ID: ${COLLECT_SUBMISSION}" >&2
    exit 2
fi

echo "[INFO] Submitted cleanup array job: ${ARRAY_JOB_ID}"
echo "[INFO] Submitted afterany collector job: ${COLLECT_JOB_ID}"
echo "[INFO] Corrected collection manifest: ${COLLECTION_MANIFEST}"
