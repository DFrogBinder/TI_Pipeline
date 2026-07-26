#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMCAN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="$(cd "${CAMCAN_DIR}/.." && pwd)"
# shellcheck source=cohort_post_common.sh
source "${SCRIPT_DIR}/cohort_post_common.sh"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash $0 COHORT_ID [--preflight]" >&2
    exit 2
fi
COHORT_ID="$1"
PREFLIGHT_ONLY=0
if [ "$#" -eq 2 ]; then
    if [ "$2" != "--preflight" ]; then
        echo "[ERROR] Unknown option: $2" >&2
        exit 2
    fi
    PREFLIGHT_ONLY=1
fi
if ! [[ "${COHORT_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[ERROR] Invalid cohort ID: ${COHORT_ID}." >&2
    exit 2
fi

STUDY_CONFIG="${STUDY_CONFIG:-${SCRIPT_DIR}/studies/corrected_v4_four_roi.json}"
COHORT_CONFIG="${COHORT_CONFIG:-${SCRIPT_DIR}/cohorts/${COHORT_ID}/cohort.json}"
STUDY_ROOT="${STUDY_ROOT:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hpc_study_root"])' "${STUDY_CONFIG}")}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${STUDY_ROOT}/campaigns/${COHORT_ID}}"
POST_CAMPAIGN_ROOT="${POST_CAMPAIGN_ROOT:-${CAMPAIGN_ROOT}/post_processing}"
SUBJECTS_FILE="${SUBJECTS_FILE:-${SCRIPT_DIR}/cohorts/${COHORT_ID}/subjects.txt}"
SIMULATION_MANIFEST="${SIMULATION_MANIFEST:-${CAMPAIGN_ROOT}/simulation_tasks.tsv}"
CHAIN_RECEIPT="${CHAIN_RECEIPT:-${CAMPAIGN_ROOT}/release_state/chain_complete.tsv}"
WORKFLOW_PY="${WORKFLOW_PY:-${SCRIPT_DIR}/workflow.py}"
SUBJECT_SLURM="${SUBJECT_SLURM:-${SCRIPT_DIR}/cohort_post_subjects.slurm}"
COLLECT_SLURM="${COLLECT_SLURM:-${SCRIPT_DIR}/cohort_post_collect.slurm}"
FASTSURFER_ROOT="${FASTSURFER_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
MNI_FIXED_ATLAS_PATH="${MNI_FIXED_ATLAS_PATH:-${FASTSURFER_ROOT}/sub-mni152.nii.gz}"
MNI_BASELINE_PARENT="${MNI_BASELINE_PARENT:-/mnt/parscratch/users/cop23bi/ZIPs/MNI152-data}"
TARGETS_CSV="${TARGETS_CSV:-${PIPELINE_DIR}/utils/targets.csv}"
EXPECTED_TARGETS_SHA256="${EXPECTED_TARGETS_SHA256:-97a8c7a72faf88d9af9e4facbdf628fba1a130d327da778bcbd00af66f2916e6}"
PYTHON="${PYTHON:-/users/cop23bi/.conda/envs/ti-post/bin/python}"

PARTITION="${PARTITION:-sheffield}"
POST_WORKERS="${POST_WORKERS:-12}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAX_CONCURRENT_DATASETS="${MAX_CONCURRENT_DATASETS:-20}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-12}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-24G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-08:00:00}"
MAX_CONCURRENT_COLLECTORS="${MAX_CONCURRENT_COLLECTORS:-4}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"

export STUDY_ROOT SUBJECTS_FILE POST_CAMPAIGN_ROOT PIPELINE_DIR
export FASTSURFER_ROOT MNI_FIXED_ATLAS_PATH MNI_BASELINE_PARENT
export TARGETS_CSV EXPECTED_TARGETS_SHA256 PYTHON POST_WORKERS

for required_file in \
    "${STUDY_CONFIG}" \
    "${COHORT_CONFIG}" \
    "${SUBJECTS_FILE}" \
    "${SIMULATION_MANIFEST}" \
    "${CHAIN_RECEIPT}" \
    "${WORKFLOW_PY}" \
    "${SUBJECT_SLURM}" \
    "${COLLECT_SLURM}" \
    "${TARGETS_CSV}" \
    "${MNI_FIXED_ATLAS_PATH}"
do
    if [ ! -f "${required_file}" ]; then
        echo "[ERROR] Required file is missing: ${required_file}" >&2
        exit 2
    fi
done
for value_name in \
    POST_WORKERS \
    CPUS_PER_TASK \
    MAX_CONCURRENT_DATASETS \
    COLLECTOR_CPUS \
    MAX_CONCURRENT_COLLECTORS
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer; got ${value}." >&2
        exit 2
    fi
done
if [ "${POST_WORKERS}" -gt "${CPUS_PER_TASK}" ]; then
    echo "[ERROR] POST_WORKERS cannot exceed CPUS_PER_TASK." >&2
    exit 2
fi
if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] Explicit ti-post Python is not executable: ${PYTHON}" >&2
    exit 2
fi

EXPECTED_SUBJECTS="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_subjects"])' \
        "${COHORT_CONFIG}"
)"
SUBJECT_COUNT="$(
    awk 'NF > 0 && $1 !~ /^#/ { count++ } END { print count + 0 }' \
        "${SUBJECTS_FILE}"
)"
UNIQUE_SUBJECT_COUNT="$(
    awk 'NF > 0 && $1 !~ /^#/ { seen[$1]=1 } END { for (item in seen) count++; print count + 0 }' \
        "${SUBJECTS_FILE}"
)"
if [ "${SUBJECT_COUNT}" -ne "${EXPECTED_SUBJECTS}" ] || \
   [ "${UNIQUE_SUBJECT_COUNT}" -ne "${EXPECTED_SUBJECTS}" ]
then
    echo "[ERROR] Cohort subject list does not contain exactly ${EXPECTED_SUBJECTS} unique subjects." >&2
    exit 2
fi

if ! awk -F '\t' '$1 == "status" && $2 == "complete" { found=1 } END { exit !found }' \
    "${CHAIN_RECEIPT}"
then
    echo "[ERROR] The FEM release chain is not marked complete: ${CHAIN_RECEIPT}" >&2
    exit 2
fi

TARGETS_SHA256="$(sha256sum "${TARGETS_CSV}" | awk '{print $1}')"
if [ "${TARGETS_SHA256}" != "${EXPECTED_TARGETS_SHA256}" ]; then
    echo "[ERROR] targets.csv hash mismatch: ${TARGETS_SHA256}" >&2
    exit 2
fi

mkdir -p \
    "${POST_CAMPAIGN_ROOT}/logs" \
    "${POST_CAMPAIGN_ROOT}/subject_summaries" \
    "${POST_CAMPAIGN_ROOT}/validation"

python3 "${WORKFLOW_PY}" validate \
    --stage simulations \
    --manifest "${SIMULATION_MANIFEST}" \
    --summary "${POST_CAMPAIGN_ROOT}/validation/simulations.tsv" \
    --skip-hashes

VALIDATION_JSON="${POST_CAMPAIGN_ROOT}/validation/simulations.json"
VALIDATION_STATUS="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "${VALIDATION_JSON}"
)"
VALIDATION_COMPLETE="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["complete"])' \
        "${VALIDATION_JSON}"
)"
EXPECTED_POST_SUBJECT_TASKS=$((EXPECTED_SUBJECTS * 4 * 10))
if [ "${VALIDATION_STATUS}" != "complete" ] || \
   [ "${VALIDATION_COMPLETE}" -ne "${EXPECTED_POST_SUBJECT_TASKS}" ]
then
    echo "[ERROR] Simulation validation did not confirm all ${EXPECTED_POST_SUBJECT_TASKS} inputs." >&2
    exit 2
fi

MISSING_ATLASES=0
while read -r subject; do
    [ -n "${subject}" ] || continue
    if [ ! -s "${FASTSURFER_ROOT}/${subject}.nii" ] && \
       [ ! -s "${FASTSURFER_ROOT}/${subject}.nii.gz" ]
    then
        echo "[ERROR] Missing subject-space atlas: ${FASTSURFER_ROOT}/${subject}.nii[.gz]" >&2
        MISSING_ATLASES=$((MISSING_ATLASES + 1))
    fi
done < "${SUBJECTS_FILE}"
if [ "${MISSING_ATLASES}" -ne 0 ]; then
    echo "[ERROR] Missing ${MISSING_ATLASES} subject-space atlas file(s)." >&2
    exit 2
fi

ROIS=(Left_Hippocampus Left_M1 Right_DLPC Right_Thalamus)
for roi in "${ROIS[@]}"; do
    ROI_ROOT="${STUDY_ROOT}/runs/${roi}_Runs"
    if [ ! -d "${ROI_ROOT}" ]; then
        echo "[ERROR] ROI repeat root is missing: ${ROI_ROOT}" >&2
        exit 2
    fi
    DATASET_COUNT="$(
        find "${ROI_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "${roi}_Data_*" | wc -l
    )"
    if [ "${DATASET_COUNT}" -ne 10 ]; then
        echo "[ERROR] ${roi} has ${DATASET_COUNT} repeat directories; expected 10." >&2
        exit 2
    fi
    BASELINE="$(post_mni_baseline_for_roi "${roi}")"
    if [ ! -d "${BASELINE}" ]; then
        echo "[ERROR] ROI-specific MNI baseline root is missing: ${BASELINE}" >&2
        exit 2
    fi
    BASELINE_TI_COUNT="$(
        find "${BASELINE}" -maxdepth 5 -type f -path '*/anat/SimNIBS/ti_brain_only.nii.gz' 2>/dev/null | wc -l
    )"
    if [ "${BASELINE_TI_COUNT}" -lt 1 ]; then
        echo "[ERROR] Missing ROI-specific MNI baseline TI output below: ${BASELINE}" >&2
        exit 2
    fi
done

"${PYTHON}" -c 'import matplotlib,nibabel,nilearn,numpy,pandas,PIL,scipy; print("ti_post_dependencies=ready")'

EXISTING_METRICS="$(
    find "${STUDY_ROOT}/runs" -type f -path '*/anat/post/subject_metrics.json' 2>/dev/null | wc -l
)"
EXPECTED_DATASETS=40
SUBJECT_ARRAY_ELEMENTS=40
COLLECTOR_ARRAY_ELEMENTS=4
SCHEDULER_TASKS=$((SUBJECT_ARRAY_ELEMENTS + COLLECTOR_ARRAY_ELEMENTS))

printf '%s\n' \
    'Scope:' \
    '  study: corrected-v4 CamCan final-cohort post-processing' \
    "  cohort: ${COHORT_ID}" \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    '  ROIs: 4 (Left_Hippocampus, Left_M1, Right_DLPC, Right_Thalamus)' \
    '  repeats per ROI: 10 (01-10)' \
    "  repeat datasets: ${EXPECTED_DATASETS}" \
    "  subject-level metric tasks: ${EXPECTED_POST_SUBJECT_TASKS}" \
    "  subject-stage array elements: ${SUBJECT_ARRAY_ELEMENTS} (0-39%${MAX_CONCURRENT_DATASETS})" \
    "  ROI collector array elements: ${COLLECTOR_ARRAY_ELEMENTS} (0-3%${MAX_CONCURRENT_COLLECTORS})" \
    "  scheduler tasks total: ${SCHEDULER_TASKS}" \
    "  expected subject_metrics.json files: ${EXPECTED_POST_SUBJECT_TASKS}" \
    '  expected complete-case population summaries: 40' \
    '  expected across-repeat analyses: 4' \
    '  expected static figure collections: 4' \
    '  execution: full requested cohort; not a smoke or subset' \
    '  architecture: 40 independent resumable dataset jobs, then 4 afterok ROI collectors'

echo "[INFO] Study root:        ${STUDY_ROOT}"
echo "[INFO] Post campaign:      ${POST_CAMPAIGN_ROOT}"
echo "[INFO] Subject list:       ${SUBJECTS_FILE}"
echo "[INFO] FastSurfer atlases: ${FASTSURFER_ROOT}"
echo "[INFO] MNI atlas:          ${MNI_FIXED_ATLAS_PATH}"
echo "[INFO] MNI baselines:      ${MNI_BASELINE_PARENT}"
echo "[INFO] Python:             ${PYTHON}"
echo "[INFO] Resource profile:   ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Subject workers:    ${POST_WORKERS} per dataset job"
echo "[INFO] Dataset concurrency:${MAX_CONCURRENT_DATASETS}"
echo "[INFO] Existing resumable subject metrics: ${EXISTING_METRICS}/${EXPECTED_POST_SUBJECT_TASKS}"

if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash CamCan_Experiment/cohort_pipeline/submit_cohort_post_processing.sh ${COHORT_ID}"
    exit 0
fi

JOB_ID_FILE="${POST_CAMPAIGN_ROOT}/submitted_job_ids.txt"
if [ -s "${JOB_ID_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_IDS="$(
        awk '/^[0-9]+$/ { values = values separator $1; separator = "," } END { print values }' \
            "${JOB_ID_FILE}"
    )"
    if [ -n "${PREVIOUS_IDS}" ] && \
       [ -n "$("${SQUEUE_BIN}" -h -j "${PREVIOUS_IDS}" -o '%i' 2>/dev/null || true)" ]
    then
        echo "[ERROR] An earlier post-processing submission is still active." >&2
        exit 2
    fi
fi

SUBJECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_${COHORT_ID}_subjects" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="0-39%${MAX_CONCURRENT_DATASETS}" \
        --output="${POST_CAMPAIGN_ROOT}/logs/subject-%A_%a.out" \
        --error="${POST_CAMPAIGN_ROOT}/logs/subject-%A_%a.err" \
        --export="ALL,STUDY_ROOT=${STUDY_ROOT},SUBJECTS_FILE=${SUBJECTS_FILE},POST_CAMPAIGN_ROOT=${POST_CAMPAIGN_ROOT},PIPELINE_DIR=${PIPELINE_DIR},FASTSURFER_ROOT=${FASTSURFER_ROOT},MNI_FIXED_ATLAS_PATH=${MNI_FIXED_ATLAS_PATH},MNI_BASELINE_PARENT=${MNI_BASELINE_PARENT},TARGETS_CSV=${TARGETS_CSV},EXPECTED_TARGETS_SHA256=${EXPECTED_TARGETS_SHA256},PYTHON=${PYTHON},POST_WORKERS=${POST_WORKERS}" \
        "${SUBJECT_SLURM}"
)"
SUBJECT_JOB="${SUBJECT_JOB%%;*}"
printf '%s\n' "${SUBJECT_JOB}" > "${JOB_ID_FILE}"

set +e
COLLECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_${COHORT_ID}_collect" \
        --partition="${PARTITION}" \
        --cpus-per-task="${COLLECTOR_CPUS}" \
        --mem="${COLLECTOR_MEMORY}" \
        --time="${COLLECTOR_TIME}" \
        --array="0-3%${MAX_CONCURRENT_COLLECTORS}" \
        --dependency="afterok:${SUBJECT_JOB}" \
        --output="${POST_CAMPAIGN_ROOT}/logs/collect-%A_%a.out" \
        --error="${POST_CAMPAIGN_ROOT}/logs/collect-%A_%a.err" \
        --export="ALL,STUDY_ROOT=${STUDY_ROOT},SUBJECTS_FILE=${SUBJECTS_FILE},POST_CAMPAIGN_ROOT=${POST_CAMPAIGN_ROOT},PIPELINE_DIR=${PIPELINE_DIR},FASTSURFER_ROOT=${FASTSURFER_ROOT},MNI_FIXED_ATLAS_PATH=${MNI_FIXED_ATLAS_PATH},MNI_BASELINE_PARENT=${MNI_BASELINE_PARENT},TARGETS_CSV=${TARGETS_CSV},EXPECTED_TARGETS_SHA256=${EXPECTED_TARGETS_SHA256},PYTHON=${PYTHON},POST_WORKERS=${COLLECTOR_CPUS}" \
        "${COLLECT_SLURM}"
)"
COLLECT_EXIT=$?
set -e
if [ "${COLLECT_EXIT}" -ne 0 ]; then
    echo "[ERROR] Collector submission failed; cancelling subject array ${SUBJECT_JOB}." >&2
    "${SCANCEL_BIN}" "${SUBJECT_JOB}" 2>/dev/null || true
    exit "${COLLECT_EXIT}"
fi
COLLECT_JOB="${COLLECT_JOB%%;*}"

printf '%s\n' "${COLLECT_JOB}" >> "${JOB_ID_FILE}"
printf '%s\n' \
    "[INFO] Submitted subject-level post array: ${SUBJECT_JOB}" \
    "[INFO] Submitted dependent ROI collectors: ${COLLECT_JOB}" \
    "[INFO] Collector dependency: afterok:${SUBJECT_JOB}" \
    "[INFO] Job IDs: ${JOB_ID_FILE}"
