#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMCAN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="$(cd "${CAMCAN_DIR}/.." && pwd)"

if [ "$#" -gt 1 ]; then
    echo "Usage: bash $0 [--preflight]" >&2
    exit 2
fi
PREFLIGHT_ONLY=0
if [ "$#" -eq 1 ]; then
    if [ "$1" != "--preflight" ]; then
        echo "[ERROR] Unknown option: $1" >&2
        exit 2
    fi
    PREFLIGHT_ONLY=1
fi

GENERIC_STUDY_ROOT="${GENERIC_STUDY_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI}"
PERSONALIZED_STUDY_ROOT="${PERSONALIZED_STUDY_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Individualized_Optimized}"
COHORT_ID="optimized_best_worst_7"
COHORT_DIR="${SCRIPT_DIR}/cohorts/${COHORT_ID}"
COHORT_CONFIG="${COHORT_CONFIG:-${COHORT_DIR}/cohort.json}"
INDIVIDUALIZED_TARGETS="${INDIVIDUALIZED_TARGETS:-${COHORT_DIR}/individualized_targets.csv}"
GENERIC_TARGETS="${GENERIC_TARGETS:-${PIPELINE_DIR}/utils/targets.csv}"
FASTSURFER_ROOT="${FASTSURFER_ROOT:-/mnt/parscratch/users/cop23bi/ZIPs/atlases}"
MNI_FIXED_ATLAS_PATH="${MNI_FIXED_ATLAS_PATH:-${FASTSURFER_ROOT}/sub-mni152.nii.gz}"
MNI_BASELINE_PARENT="${MNI_BASELINE_PARENT:-/mnt/parscratch/users/cop23bi/ZIPs/MNI152-data}"
GENERIC_CHAIN_RECEIPT="${GENERIC_CHAIN_RECEIPT:-${GENERIC_STUDY_ROOT}/campaigns/final_132/release_state/chain_complete.tsv}"
PERSONALIZED_CHAIN_RECEIPT="${PERSONALIZED_CHAIN_RECEIPT:-${PERSONALIZED_STUDY_ROOT}/campaigns/${COHORT_ID}/release_state/chain_complete.tsv}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${PERSONALIZED_STUDY_ROOT}/campaigns/${COHORT_ID}/post_processing/personalized_vs_generic}"
VISUAL_QC_ROOT="${VISUAL_QC_ROOT:-${COMPARISON_ROOT}/full_post_visual_qc}"
COMPARISON_PY="${COMPARISON_PY:-${CAMCAN_DIR}/post/camcan_personalized_comparison.py}"
VISUAL_QC_PY="${VISUAL_QC_PY:-${CAMCAN_DIR}/post/camcan_personalized_visual_qc.py}"
PAIR_SLURM="${PAIR_SLURM:-${SCRIPT_DIR}/cohort_personalized_visual_qc_pair.slurm}"
COLLECT_SLURM="${COLLECT_SLURM:-${SCRIPT_DIR}/cohort_personalized_visual_qc_collect.slurm}"
PYTHON="${PYTHON:-/users/cop23bi/.conda/envs/ti-post/bin/python}"

PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_CONCURRENT_PAIRS="${MAX_CONCURRENT_PAIRS:-8}"
PAIR_ARRAY_SPEC="${PAIR_ARRAY_SPEC:-0-7%${MAX_CONCURRENT_PAIRS}}"
VISUAL_QC_WORKERS="${VISUAL_QC_WORKERS:-4}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-2}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-8G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-02:00:00}"
VISUAL_QC_FORCE="${VISUAL_QC_FORCE:-0}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"

for path in \
    "${COHORT_CONFIG}" \
    "${INDIVIDUALIZED_TARGETS}" \
    "${GENERIC_TARGETS}" \
    "${GENERIC_CHAIN_RECEIPT}" \
    "${PERSONALIZED_CHAIN_RECEIPT}" \
    "${MNI_FIXED_ATLAS_PATH}" \
    "${COMPARISON_PY}" \
    "${VISUAL_QC_PY}" \
    "${PAIR_SLURM}" \
    "${COLLECT_SLURM}"
do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Required file is missing: ${path}" >&2
        exit 2
    fi
done
if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] Python is not executable: ${PYTHON}" >&2
    exit 2
fi
for receipt in "${GENERIC_CHAIN_RECEIPT}" "${PERSONALIZED_CHAIN_RECEIPT}"; do
    if ! awk -F '\t' '$1=="status" && $2=="complete"{ok=1} END{exit !ok}' "${receipt}"; then
        echo "[ERROR] FEM campaign is not complete: ${receipt}" >&2
        exit 2
    fi
done
for variable in \
    CPUS_PER_TASK \
    MAX_CONCURRENT_PAIRS \
    VISUAL_QC_WORKERS \
    COLLECTOR_CPUS
do
    value="${!variable}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${variable} must be a positive integer; got ${value}." >&2
        exit 2
    fi
done
if [ "${MAX_CONCURRENT_PAIRS}" -gt 8 ]; then
    echo "[ERROR] MAX_CONCURRENT_PAIRS cannot exceed the eight selected pairs." >&2
    exit 2
fi
if [[ "${PAIR_ARRAY_SPEC}" =~ ^[0-7]$ ]]; then
    SCHEDULER_TASKS_TOTAL=2
    EXECUTION_SCOPE="resumable recovery of pair ${PAIR_ARRAY_SPEC}; all existing valid products reused"
    SUBMIT_COMMAND="PAIR_ARRAY_SPEC=${PAIR_ARRAY_SPEC} bash CamCan_Experiment/cohort_pipeline/submit_personalized_vs_generic_visual_qc.sh"
elif [ "${PAIR_ARRAY_SPEC}" = "0-7%${MAX_CONCURRENT_PAIRS}" ]; then
    SCHEDULER_TASKS_TOTAL=9
    EXECUTION_SCOPE="full eight-pair resumable execution"
    SUBMIT_COMMAND="bash CamCan_Experiment/cohort_pipeline/submit_personalized_vs_generic_visual_qc.sh"
else
    echo "[ERROR] PAIR_ARRAY_SPEC must be one pair index (0-7) or the default 0-7%${MAX_CONCURRENT_PAIRS}; got ${PAIR_ARRAY_SPEC}." >&2
    exit 2
fi
if [ "${VISUAL_QC_WORKERS}" -gt "${CPUS_PER_TASK}" ]; then
    echo "[ERROR] VISUAL_QC_WORKERS cannot exceed CPUS_PER_TASK." >&2
    exit 2
fi

for roi_dir in \
    MNI152-left-hippocampus \
    MNI152-left-m1 \
    MNI152-right-dlpc \
    MNI152-right-thalamus
do
    if [ ! -s "${MNI_BASELINE_PARENT}/${roi_dir}/anat/SimNIBS/ti_brain_only.nii.gz" ]; then
        echo "[ERROR] Corrected MNI152 baseline field is missing: ${roi_dir}" >&2
        exit 2
    fi
done

mkdir -p \
    "${COMPARISON_ROOT}" \
    "${VISUAL_QC_ROOT}/logs" \
    "${VISUAL_QC_ROOT}/pair_summaries" \
    "${VISUAL_QC_ROOT}/post_records" \
    "${VISUAL_QC_ROOT}/post_products" \
    "${VISUAL_QC_ROOT}/paired_fields" \
    "${VISUAL_QC_ROOT}/pair_reports"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}_camcan_personalized_visual_qc_mpl}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" -c \
    'import matplotlib,nibabel,nilearn,numpy,pandas,PIL,scipy; print("personalized_visual_qc_dependencies=ready")'

# Rebuild and revalidate the exact eight-pair allowlist and all 160 source
# fields.  This is intentionally the same hard boundary as the manuscript
# comparison analysis.
"${PYTHON}" \
    "${COMPARISON_PY}" \
    prepare \
    --cohort-config "${COHORT_CONFIG}" \
    --individualized-targets "${INDIVIDUALIZED_TARGETS}" \
    --generic-targets "${GENERIC_TARGETS}" \
    --generic-study-root "${GENERIC_STUDY_ROOT}" \
    --personalized-study-root "${PERSONALIZED_STUDY_ROOT}" \
    --atlas-root "${FASTSURFER_ROOT}" \
    --out-dir "${COMPARISON_ROOT}"

PREFLIGHT_JSON="${COMPARISON_ROOT}/preflight.json"
GENERIC_TARGETS_SHA256="$(
    "${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["generic_targets_csv_sha256"])' \
        "${PREFLIGHT_JSON}"
)"
INDIVIDUALIZED_TARGETS_SHA256="$(
    "${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["individualized_targets_csv_sha256"])' \
        "${PREFLIGHT_JSON}"
)"
ALLOWLIST_SHA256="$(
    "${PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["selection_allowlist_sha256"])' \
        "${PREFLIGHT_JSON}"
)"
EXISTING_RECORDS="$(
    find "${VISUAL_QC_ROOT}/post_records" -type f -name 'repeat_*.json' \
        -exec grep -lF '"status": "complete"' {} + 2>/dev/null | wc -l
)"

printf '%s\n' \
    'Scope:' \
    '  purpose: full subject-level post-processing and E-field visual QC' \
    '  selected subject-ROI pairs: 8 (best and worst case for each of four ROIs)' \
    '  unique subjects: 7' \
    '  conditions per pair: 2 (generic, personalized)' \
    '  independently remeshed repeats per condition: 10' \
    '  full subject-level post-processing records: 160 (8 x 2 x 10)' \
    '  legacy single-field overlays expected: 1120 (7 per field)' \
    '  common-scale paired PNGs expected: 160 (2 per repeat)' \
    '  multipage pair reports expected: 16 (2 per pair)' \
    '  personalized simulations intentionally included: 80' \
    '  personalized simulations excluded as out of scope: 200' \
    "  pair-stage array: ${PAIR_ARRAY_SPEC}" \
    '  collector jobs: 1' \
    "  scheduler tasks total: ${SCHEDULER_TASKS_TOTAL}" \
    "  execution: ${EXECUTION_SCOPE}" \
    '  final validated product scope remains: 8 pairs, 160 post records, 160 paired PNGs, 16 reports' \
    '  source simulations modified: no' \
    '  existing source post directories modified: no' \
    '  output: isolated QC tree'

echo "[INFO] Generic study:          ${GENERIC_STUDY_ROOT}"
echo "[INFO] Personalized study:     ${PERSONALIZED_STUDY_ROOT}"
echo "[INFO] Atlases:                ${FASTSURFER_ROOT}"
echo "[INFO] MNI152 baselines:       ${MNI_BASELINE_PARENT}"
echo "[INFO] Selection allowlist:    ${COMPARISON_ROOT}/selection_allowlist.csv"
echo "[INFO] Allowlist SHA256:       ${ALLOWLIST_SHA256}"
echo "[INFO] Generic targets SHA:    ${GENERIC_TARGETS_SHA256}"
echo "[INFO] Personalized table SHA: ${INDIVIDUALIZED_TARGETS_SHA256}"
echo "[INFO] Existing valid records: ${EXISTING_RECORDS}/160"
echo "[INFO] Resource profile:       ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Isolated QC output:     ${VISUAL_QC_ROOT}"

if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: ${SUBMIT_COMMAND}"
    exit 0
fi

JOB_ID_FILE="${VISUAL_QC_ROOT}/submitted_job_ids.txt"
if [ -s "${JOB_ID_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_IDS="$(
        awk '/^[0-9]+$/ { values=values separator $1; separator="," } END{print values}' \
            "${JOB_ID_FILE}"
    )"
    if [ -n "${PREVIOUS_IDS}" ] && \
       [ -n "$("${SQUEUE_BIN}" -h -j "${PREVIOUS_IDS}" -o '%i' 2>/dev/null || true)" ]
    then
        echo "[ERROR] A previous visual-QC submission is still active." >&2
        exit 2
    fi
fi

EXPORTS="ALL,PIPELINE_DIR=${PIPELINE_DIR},GENERIC_STUDY_ROOT=${GENERIC_STUDY_ROOT},PERSONALIZED_STUDY_ROOT=${PERSONALIZED_STUDY_ROOT},COMPARISON_ROOT=${COMPARISON_ROOT},VISUAL_QC_ROOT=${VISUAL_QC_ROOT},FASTSURFER_ROOT=${FASTSURFER_ROOT},MNI_FIXED_ATLAS_PATH=${MNI_FIXED_ATLAS_PATH},MNI_BASELINE_PARENT=${MNI_BASELINE_PARENT},PYTHON=${PYTHON},GENERIC_TARGETS_SHA256=${GENERIC_TARGETS_SHA256},INDIVIDUALIZED_TARGETS_SHA256=${INDIVIDUALIZED_TARGETS_SHA256},VISUAL_QC_WORKERS=${VISUAL_QC_WORKERS},VISUAL_QC_FORCE=${VISUAL_QC_FORCE},MPLCONFIGDIR=${MPLCONFIGDIR}"
PAIR_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_personalized_visual_qc_pairs" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="${PAIR_ARRAY_SPEC}" \
        --output="${VISUAL_QC_ROOT}/logs/pairs-%A_%a.out" \
        --error="${VISUAL_QC_ROOT}/logs/pairs-%A_%a.err" \
        --export="${EXPORTS}" \
        "${PAIR_SLURM}"
)"
PAIR_JOB="${PAIR_JOB%%;*}"
printf '%s\n' "${PAIR_JOB}" > "${JOB_ID_FILE}"

set +e
COLLECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_personalized_visual_qc_collect" \
        --partition="${PARTITION}" \
        --cpus-per-task="${COLLECTOR_CPUS}" \
        --mem="${COLLECTOR_MEMORY}" \
        --time="${COLLECTOR_TIME}" \
        --dependency="afterok:${PAIR_JOB}" \
        --output="${VISUAL_QC_ROOT}/logs/collector-%j.out" \
        --error="${VISUAL_QC_ROOT}/logs/collector-%j.err" \
        --export="${EXPORTS}" \
        "${COLLECT_SLURM}"
)"
COLLECT_EXIT=$?
set -e
if [ "${COLLECT_EXIT}" -ne 0 ]; then
    echo "[ERROR] Collector submission failed; cancelling pair array ${PAIR_JOB}." >&2
    "${SCANCEL_BIN}" "${PAIR_JOB}" 2>/dev/null || true
    exit "${COLLECT_EXIT}"
fi
COLLECT_JOB="${COLLECT_JOB%%;*}"
printf '%s\n' "${COLLECT_JOB}" >> "${JOB_ID_FILE}"

printf '%s\n' \
    "[INFO] Submitted selected-pair full-post visual-QC array: ${PAIR_JOB}" \
    "[INFO] Submitted dependent visual-QC collector:          ${COLLECT_JOB}" \
    "[INFO] Collector dependency:                            afterok:${PAIR_JOB}" \
    "[INFO] Job IDs:                                         ${JOB_ID_FILE}"
