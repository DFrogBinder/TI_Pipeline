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
ANALYSIS_SUBDIR="${ANALYSIS_SUBDIR:-optimizer_matched_analysis}"
ANALYSIS_ROOT="${POST_CAMPAIGN_ROOT}/${ANALYSIS_SUBDIR}"
SUBJECTS_FILE="${SUBJECTS_FILE:-${SCRIPT_DIR}/cohorts/${COHORT_ID}/subjects.txt}"
SIMULATION_MANIFEST="${SIMULATION_MANIFEST:-${CAMPAIGN_ROOT}/simulation_tasks.tsv}"
CHAIN_RECEIPT="${CHAIN_RECEIPT:-${CAMPAIGN_ROOT}/release_state/chain_complete.tsv}"
WORKFLOW_PY="${WORKFLOW_PY:-${SCRIPT_DIR}/workflow.py}"
ANALYSIS_PY="${ANALYSIS_PY:-${CAMCAN_DIR}/post/camcan_manuscript_analysis.py}"
SUBJECT_SLURM="${SUBJECT_SLURM:-${SCRIPT_DIR}/cohort_post_manuscript_subjects.slurm}"
COLLECT_SLURM="${COLLECT_SLURM:-${SCRIPT_DIR}/cohort_post_manuscript_collect.slurm}"
FASTSURFER_ROOT="${FASTSURFER_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
MNI_FIXED_ATLAS_PATH="${MNI_FIXED_ATLAS_PATH:-${FASTSURFER_ROOT}/sub-mni152.nii.gz}"
MNI_BASELINE_PARENT="${MNI_BASELINE_PARENT:-/mnt/parscratch/users/cop23bi/ZIPs/MNI152-data}"
PYTHON="${PYTHON:-/users/cop23bi/.conda/envs/ti-post/bin/python}"

PARTITION="${PARTITION:-sheffield}"
MANUSCRIPT_WORKERS="${MANUSCRIPT_WORKERS:-12}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAX_CONCURRENT_DATASETS="${MAX_CONCURRENT_DATASETS:-40}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-12}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-24G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-08:00:00}"
MANUSCRIPT_THRESHOLDS_COLON="${MANUSCRIPT_THRESHOLDS_COLON:-0.20:0.18:0.15}"
MANUSCRIPT_TOP_PERCENTILE="${MANUSCRIPT_TOP_PERCENTILE:-95.0}"
MANUSCRIPT_ROBUST_MAX_PERCENTILE="${MANUSCRIPT_ROBUST_MAX_PERCENTILE:-99.9}"
MANUSCRIPT_UPPER_TAIL_FRACTION="${MANUSCRIPT_UPPER_TAIL_FRACTION:-0.01}"
MANUSCRIPT_FORCE="${MANUSCRIPT_FORCE:-0}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"

for required_file in \
    "${STUDY_CONFIG}" \
    "${COHORT_CONFIG}" \
    "${SUBJECTS_FILE}" \
    "${SIMULATION_MANIFEST}" \
    "${CHAIN_RECEIPT}" \
    "${WORKFLOW_PY}" \
    "${ANALYSIS_PY}" \
    "${SUBJECT_SLURM}" \
    "${COLLECT_SLURM}" \
    "${MNI_FIXED_ATLAS_PATH}"
do
    if [ ! -f "${required_file}" ]; then
        echo "[ERROR] Required file is missing: ${required_file}" >&2
        exit 2
    fi
done
for value_name in \
    MANUSCRIPT_WORKERS \
    CPUS_PER_TASK \
    MAX_CONCURRENT_DATASETS \
    COLLECTOR_CPUS
do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${value_name} must be a positive integer; got ${value}." >&2
        exit 2
    fi
done
if [ "${MANUSCRIPT_WORKERS}" -gt "${CPUS_PER_TASK}" ]; then
    echo "[ERROR] MANUSCRIPT_WORKERS cannot exceed CPUS_PER_TASK." >&2
    exit 2
fi
if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] Explicit ti-post Python is not executable: ${PYTHON}" >&2
    exit 2
fi
if ! [[ "${MANUSCRIPT_THRESHOLDS_COLON}" =~ ^[0-9]+([.][0-9]+)?(:[0-9]+([.][0-9]+)?)*$ ]]; then
    echo "[ERROR] MANUSCRIPT_THRESHOLDS_COLON must look like 0.20:0.18:0.15." >&2
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

mkdir -p \
    "${ANALYSIS_ROOT}/logs" \
    "${ANALYSIS_ROOT}/dataset_summaries" \
    "${ANALYSIS_ROOT}/results"

python3 "${WORKFLOW_PY}" validate \
    --stage simulations \
    --manifest "${SIMULATION_MANIFEST}" \
    --summary "${ANALYSIS_ROOT}/simulations.tsv" \
    --skip-hashes
VALIDATION_JSON="${ANALYSIS_ROOT}/simulations.json"
EXPECTED_RECORDS=$((EXPECTED_SUBJECTS * 4 * 10))
VALIDATION_STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${VALIDATION_JSON}")"
VALIDATION_COMPLETE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["complete"])' "${VALIDATION_JSON}")"
if [ "${VALIDATION_STATUS}" != "complete" ] || \
   [ "${VALIDATION_COMPLETE}" -ne "${EXPECTED_RECORDS}" ]
then
    echo "[ERROR] Simulation validation did not confirm all ${EXPECTED_RECORDS} inputs." >&2
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
MISSING_TI=0
for roi in "${ROIS[@]}"; do
    for repeat_number in $(seq 1 10); do
        repeat_id="$(printf '%02d' "${repeat_number}")"
        dataset_root="${STUDY_ROOT}/runs/${roi}_Runs/${roi}_Data_${repeat_id}"
        if [ ! -d "${dataset_root}" ]; then
            echo "[ERROR] Missing repeat dataset: ${dataset_root}" >&2
            MISSING_TI=$((MISSING_TI + EXPECTED_SUBJECTS))
            continue
        fi
        while read -r subject; do
            [ -n "${subject}" ] || continue
            ti_path="${dataset_root}/${subject}/anat/SimNIBS/ti_brain_only.nii.gz"
            if [ ! -s "${ti_path}" ]; then
                echo "[ERROR] Missing whole-brain TI field: ${ti_path}" >&2
                MISSING_TI=$((MISSING_TI + 1))
            fi
        done < "${SUBJECTS_FILE}"
    done
    baseline="$(post_mni_baseline_for_roi "${roi}")"
    if [ ! -d "${baseline}" ] || \
       [ "$(find "${baseline}" -maxdepth 5 -type f -path '*/anat/SimNIBS/ti_brain_only.nii.gz' 2>/dev/null | wc -l)" -lt 1 ]
    then
        echo "[ERROR] Missing ROI-specific MNI baseline TI output below: ${baseline}" >&2
        exit 2
    fi
done
if [ "${MISSING_TI}" -ne 0 ]; then
    echo "[ERROR] Missing ${MISSING_TI} whole-brain subject TI NIfTI file(s)." >&2
    exit 2
fi

"${PYTHON}" -c 'import matplotlib,nibabel,numpy,pandas; print("manuscript_analysis_dependencies=ready")'
"${PYTHON}" "${ANALYSIS_PY}" validate-atlas --atlas "${MNI_FIXED_ATLAS_PATH}"

EXISTING_COMPLETE="$(
    find "${STUDY_ROOT}/runs" -type f -path '*/anat/post/optimizer_matched_metrics.json' \
        -exec grep -lF '"status": "complete"' {} + 2>/dev/null | wc -l
)"
MANUSCRIPT_THRESHOLDS_DISPLAY="${MANUSCRIPT_THRESHOLDS_COLON//:/ and }"
SCHEDULER_TASKS=41
printf '%s\n' \
    'Scope:' \
    '  study: corrected-v4 CamCan manuscript analysis' \
    "  cohort: ${COHORT_ID}" \
    "  subjects: ${EXPECTED_SUBJECTS}" \
    '  ROIs: 4 (Left_Hippocampus, Left_M1, Right_DLPC, Right_Thalamus)' \
    '  repeats per ROI: 10 (01-10)' \
    "  repeat-level metric records: ${EXPECTED_RECORDS}" \
    "  subject-stage array: 0-39%${MAX_CONCURRENT_DATASETS}" \
    '  collector jobs: 1' \
    "  scheduler tasks total: ${SCHEDULER_TASKS}" \
    "  thresholds: ${MANUSCRIPT_THRESHOLDS_DISPLAY} V/m" \
    "  robust maximum: P${MANUSCRIPT_ROBUST_MAX_PERCENTILE}" \
    "  robust sensitivity: median of upper $(python3 -c "print(${MANUSCRIPT_UPPER_TAIL_FRACTION} * 100)")%" \
    '  repeat aggregation: arithmetic mean after metric calculation' \
    '  primary ROI: anatomical-parcel-clipped optimizer sphere (100 mm3 cortical; 200 mm3 subcortical)' \
    '  sphere construction: centroid, 3.00 mm start, 0.01 mm steps, <10 mm cap' \
    '  secondary ROI: full anatomical parcel (anatomical_ metrics)' \
    '  primary spread: off-target volume excluding the optimizer-matched ROI' \
    '  companion spread: whole-brain volume including the ROI' \
    '  individualized optimization: excluded' \
    '  execution: resumable metric-only analysis; existing simulations are read-only'

echo "[INFO] Study root:          ${STUDY_ROOT}"
echo "[INFO] Post campaign:        ${POST_CAMPAIGN_ROOT}"
echo "[INFO] FastSurfer atlases:   ${FASTSURFER_ROOT}"
echo "[INFO] MNI atlas:            ${MNI_FIXED_ATLAS_PATH}"
echo "[INFO] MNI baselines:        ${MNI_BASELINE_PARENT}"
echo "[INFO] Python:               ${PYTHON}"
echo "[INFO] Resource profile:     ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Dataset concurrency:  ${MAX_CONCURRENT_DATASETS}"
echo "[INFO] Workers per dataset:  ${MANUSCRIPT_WORKERS}"
echo "[INFO] Existing markers:     ${EXISTING_COMPLETE}/${EXPECTED_RECORDS}"
echo "[INFO] Output:               ${ANALYSIS_ROOT}/results"

if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash CamCan_Experiment/cohort_pipeline/submit_cohort_manuscript_analysis.sh ${COHORT_ID}"
    exit 0
fi

JOB_ID_FILE="${ANALYSIS_ROOT}/submitted_job_ids.txt"
if [ -s "${JOB_ID_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_IDS="$(
        awk '/^[0-9]+$/ { values = values separator $1; separator = "," } END { print values }' \
            "${JOB_ID_FILE}"
    )"
    if [ -n "${PREVIOUS_IDS}" ] && \
       [ -n "$("${SQUEUE_BIN}" -h -j "${PREVIOUS_IDS}" -o '%i' 2>/dev/null || true)" ]
    then
        echo "[ERROR] An earlier manuscript-analysis submission is still active." >&2
        exit 2
    fi
fi

EXPORTS="ALL,COHORT_ID=${COHORT_ID},STUDY_ROOT=${STUDY_ROOT},SUBJECTS_FILE=${SUBJECTS_FILE},POST_CAMPAIGN_ROOT=${POST_CAMPAIGN_ROOT},ANALYSIS_ROOT=${ANALYSIS_ROOT},PIPELINE_DIR=${PIPELINE_DIR},FASTSURFER_ROOT=${FASTSURFER_ROOT},MNI_FIXED_ATLAS_PATH=${MNI_FIXED_ATLAS_PATH},MNI_BASELINE_PARENT=${MNI_BASELINE_PARENT},PYTHON=${PYTHON},MANUSCRIPT_WORKERS=${MANUSCRIPT_WORKERS},MANUSCRIPT_THRESHOLDS_COLON=${MANUSCRIPT_THRESHOLDS_COLON},MANUSCRIPT_TOP_PERCENTILE=${MANUSCRIPT_TOP_PERCENTILE},MANUSCRIPT_ROBUST_MAX_PERCENTILE=${MANUSCRIPT_ROBUST_MAX_PERCENTILE},MANUSCRIPT_UPPER_TAIL_FRACTION=${MANUSCRIPT_UPPER_TAIL_FRACTION},MANUSCRIPT_FORCE=${MANUSCRIPT_FORCE}"
SUBJECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_${COHORT_ID}_manuscript" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="0-39%${MAX_CONCURRENT_DATASETS}" \
        --output="${ANALYSIS_ROOT}/logs/subjects-%A_%a.out" \
        --error="${ANALYSIS_ROOT}/logs/subjects-%A_%a.err" \
        --export="${EXPORTS}" \
        "${SUBJECT_SLURM}"
)"
SUBJECT_JOB="${SUBJECT_JOB%%;*}"
printf '%s\n' "${SUBJECT_JOB}" > "${JOB_ID_FILE}"

set +e
COLLECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_${COHORT_ID}_manuscript_collect" \
        --partition="${PARTITION}" \
        --cpus-per-task="${COLLECTOR_CPUS}" \
        --mem="${COLLECTOR_MEMORY}" \
        --time="${COLLECTOR_TIME}" \
        --dependency="afterok:${SUBJECT_JOB}" \
        --output="${ANALYSIS_ROOT}/logs/collector-%j.out" \
        --error="${ANALYSIS_ROOT}/logs/collector-%j.err" \
        --export="${EXPORTS}" \
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
    "[INFO] Submitted manuscript metric array: ${SUBJECT_JOB}" \
    "[INFO] Submitted dependent manuscript collector: ${COLLECT_JOB}" \
    "[INFO] Collector dependency: afterok:${SUBJECT_JOB}" \
    "[INFO] Job IDs: ${JOB_ID_FILE}"
