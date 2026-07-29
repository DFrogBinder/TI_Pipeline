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
GENERIC_CHAIN_RECEIPT="${GENERIC_CHAIN_RECEIPT:-${GENERIC_STUDY_ROOT}/campaigns/final_132/release_state/chain_complete.tsv}"
PERSONALIZED_CHAIN_RECEIPT="${PERSONALIZED_CHAIN_RECEIPT:-${PERSONALIZED_STUDY_ROOT}/campaigns/${COHORT_ID}/release_state/chain_complete.tsv}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${PERSONALIZED_STUDY_ROOT}/campaigns/${COHORT_ID}/post_processing/optimizer_matched_personalized_vs_generic_schema3}"
ANALYSIS_PY="${ANALYSIS_PY:-${CAMCAN_DIR}/post/camcan_personalized_comparison.py}"
PUBLICATION_PACKAGER="${PUBLICATION_PACKAGER:-${CAMCAN_DIR}/post/package_camcan_publication_inputs.py}"
PUBLICATION_RENDERER="${PUBLICATION_RENDERER:-${CAMCAN_DIR}/post/build_camcan_supervisor_revision_figures.py}"
MNI_THRESHOLD_TABLE="${MNI_THRESHOLD_TABLE:-${CAMCAN_DIR}/post/mni152_simnibs401_roi_thresholds.csv}"
PAIR_SLURM="${PAIR_SLURM:-${SCRIPT_DIR}/cohort_personalized_comparison_pair.slurm}"
COLLECT_SLURM="${COLLECT_SLURM:-${SCRIPT_DIR}/cohort_personalized_comparison_collect.slurm}"
PYTHON="${PYTHON:-/users/cop23bi/.conda/envs/ti-post/bin/python}"

PARTITION="${PARTITION:-sheffield}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-16G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MAX_CONCURRENT_PAIRS="${MAX_CONCURRENT_PAIRS:-8}"
COMPARISON_WORKERS="${COMPARISON_WORKERS:-4}"
COLLECTOR_CPUS="${COLLECTOR_CPUS:-4}"
COLLECTOR_MEMORY="${COLLECTOR_MEMORY:-16G}"
COLLECTOR_TIME="${COLLECTOR_TIME:-01:00:00}"
COMPARISON_THRESHOLDS_COLON="${COMPARISON_THRESHOLDS_COLON:-0.20:0.18:0.15}"
COMPARISON_TOP_PERCENTILE="${COMPARISON_TOP_PERCENTILE:-95.0}"
COMPARISON_ROBUST_MAX_PERCENTILE="${COMPARISON_ROBUST_MAX_PERCENTILE:-99.9}"
COMPARISON_UPPER_TAIL_FRACTION="${COMPARISON_UPPER_TAIL_FRACTION:-0.01}"
COMPARISON_FORCE="${COMPARISON_FORCE:-0}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SQUEUE_BIN="${SQUEUE_BIN:-squeue}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"

for path in \
    "${COHORT_CONFIG}" \
    "${INDIVIDUALIZED_TARGETS}" \
    "${GENERIC_TARGETS}" \
    "${GENERIC_CHAIN_RECEIPT}" \
    "${PERSONALIZED_CHAIN_RECEIPT}" \
    "${ANALYSIS_PY}" \
    "${PUBLICATION_PACKAGER}" \
    "${PUBLICATION_RENDERER}" \
    "${MNI_THRESHOLD_TABLE}" \
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
for variable in CPUS_PER_TASK MAX_CONCURRENT_PAIRS COMPARISON_WORKERS COLLECTOR_CPUS; do
    value="${!variable}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [ "${value}" -lt 1 ]; then
        echo "[ERROR] ${variable} must be a positive integer; got ${value}." >&2
        exit 2
    fi
done
if [ "${MAX_CONCURRENT_PAIRS}" -gt 28 ]; then
    echo "[ERROR] MAX_CONCURRENT_PAIRS cannot exceed the 28 subject/ROI configurations." >&2
    exit 2
fi
if [ "${COMPARISON_WORKERS}" -gt "${CPUS_PER_TASK}" ]; then
    echo "[ERROR] COMPARISON_WORKERS cannot exceed CPUS_PER_TASK." >&2
    exit 2
fi
if ! [[ "${COMPARISON_THRESHOLDS_COLON}" =~ ^[0-9]+([.][0-9]+)?(:[0-9]+([.][0-9]+)?)*$ ]]; then
    echo "[ERROR] COMPARISON_THRESHOLDS_COLON must look like 0.20:0.18:0.15." >&2
    exit 2
fi

mkdir -p \
    "${COMPARISON_ROOT}/logs" \
    "${COMPARISON_ROOT}/pair_summaries" \
    "${COMPARISON_ROOT}/repeat_records"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}_camcan_personalized_mpl}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" -c 'import matplotlib,nibabel,numpy,pandas; print("personalized_comparison_dependencies=ready")'
"${PYTHON}" \
    "${ANALYSIS_PY}" \
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
    "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["generic_targets_csv_sha256"])' "${PREFLIGHT_JSON}"
)"
INDIVIDUALIZED_TARGETS_SHA256="$(
    "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["individualized_targets_csv_sha256"])' "${PREFLIGHT_JSON}"
)"
ALLOWLIST_SHA256="$(
    "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection_allowlist_sha256"])' "${PREFLIGHT_JSON}"
)"
EXISTING_RECORDS="$(
    find "${COMPARISON_ROOT}/repeat_records" -type f -name 'repeat_*.json' \
        -exec grep -lF '"status": "complete"' {} + 2>/dev/null | wc -l
)"

THRESHOLDS_DISPLAY="${COMPARISON_THRESHOLDS_COLON//:/, }"
printf '%s\n' \
    'Scope:' \
    '  analysis: personalized Pareto montage vs MNI152-derived generic montage on the same subject head' \
    '  subject-ROI configurations: 28 (all 7 optimized subjects x all 4 ROIs)' \
    '  originally selected extremes: 8; cross-target configurations: 20' \
    '  unique subjects: 7' \
    '  conditions per pair: 2 (generic, personalized)' \
    '  independent remesh repeats per condition: 10' \
    '  required repeat-level metric inputs: 560 (28 x 2 x 10)' \
    '  personalized simulations intentionally included: 280' \
    '  personalized simulations excluded as out of scope: 0' \
    "  pair-stage array: 0-27%${MAX_CONCURRENT_PAIRS}" \
    '  collector jobs: 1' \
    '  scheduler tasks total: 29' \
    "  thresholds: ${THRESHOLDS_DISPLAY} V/m" \
    '  target-field summaries: mean and median primary; minimum and robust maximum retained for QC' \
    "  robust maximum: P${COMPARISON_ROBUST_MAX_PERCENTILE}" \
    '  aggregation: calculate every metric per repeat, then mean within condition' \
    '  primary ROI: MakeROIs.m-equivalent parcel-clipped optimizer sphere' \
    '  secondary ROI: full anatomical parcel (anatomical_ metrics)' \
    '  repeat pairing across conditions: none' \
    '  population inference: none; descriptive seven-subject comparison' \
    '  source simulations: read-only' \
    '  download contract: exact polished-figure source tables, SHA-256 checksums, instructions, and versioned renderer included' \
    '  reuse policy: comparison schema 3, manuscript schema 4, and the complete configuration fingerprint must match'

echo "[INFO] Generic study:          ${GENERIC_STUDY_ROOT}"
echo "[INFO] Personalized study:     ${PERSONALIZED_STUDY_ROOT}"
echo "[INFO] Atlases:                ${FASTSURFER_ROOT}"
echo "[INFO] Selection allowlist:    ${COMPARISON_ROOT}/selection_allowlist.csv"
echo "[INFO] Allowlist SHA256:       ${ALLOWLIST_SHA256}"
echo "[INFO] Generic targets SHA:    ${GENERIC_TARGETS_SHA256}"
echo "[INFO] Personalized table SHA: ${INDIVIDUALIZED_TARGETS_SHA256}"
echo "[INFO] Complete markers (any schema): ${EXISTING_RECORDS}/560"
echo "[INFO] Worker validation: comparison schema 3 + manuscript schema 4 + configuration fingerprint; older records are recomputed"
echo "[INFO] Resource profile:       ${PARTITION}, ${CPUS_PER_TASK} CPU, ${MEMORY}, ${TIME_LIMIT}"
echo "[INFO] Comparison output:      ${COMPARISON_ROOT}"

if [ "${PREFLIGHT_ONLY}" -eq 1 ]; then
    echo "[INFO] Preflight passed without submitting jobs."
    echo "[INFO] Submit with: bash CamCan_Experiment/cohort_pipeline/submit_personalized_vs_generic_analysis.sh"
    exit 0
fi

JOB_ID_FILE="${COMPARISON_ROOT}/submitted_job_ids.txt"
if [ -s "${JOB_ID_FILE}" ] && command -v "${SQUEUE_BIN}" >/dev/null 2>&1; then
    PREVIOUS_IDS="$(
        awk '/^[0-9]+$/ { values=values separator $1; separator="," } END{print values}' \
            "${JOB_ID_FILE}"
    )"
    if [ -n "${PREVIOUS_IDS}" ] && \
       [ -n "$("${SQUEUE_BIN}" -h -j "${PREVIOUS_IDS}" -o '%i' 2>/dev/null || true)" ]
    then
        echo "[ERROR] A previous personalized-comparison submission is still active." >&2
        exit 2
    fi
fi

EXPORTS="ALL,PIPELINE_DIR=${PIPELINE_DIR},GENERIC_STUDY_ROOT=${GENERIC_STUDY_ROOT},PERSONALIZED_STUDY_ROOT=${PERSONALIZED_STUDY_ROOT},COMPARISON_ROOT=${COMPARISON_ROOT},FASTSURFER_ROOT=${FASTSURFER_ROOT},PYTHON=${PYTHON},GENERIC_TARGETS_SHA256=${GENERIC_TARGETS_SHA256},INDIVIDUALIZED_TARGETS_SHA256=${INDIVIDUALIZED_TARGETS_SHA256},COMPARISON_WORKERS=${COMPARISON_WORKERS},COMPARISON_THRESHOLDS_COLON=${COMPARISON_THRESHOLDS_COLON},COMPARISON_TOP_PERCENTILE=${COMPARISON_TOP_PERCENTILE},COMPARISON_ROBUST_MAX_PERCENTILE=${COMPARISON_ROBUST_MAX_PERCENTILE},COMPARISON_UPPER_TAIL_FRACTION=${COMPARISON_UPPER_TAIL_FRACTION},COMPARISON_FORCE=${COMPARISON_FORCE},MPLCONFIGDIR=${MPLCONFIGDIR},PUBLICATION_PACKAGER=${PUBLICATION_PACKAGER},PUBLICATION_RENDERER=${PUBLICATION_RENDERER},MNI_THRESHOLD_TABLE=${MNI_THRESHOLD_TABLE}"
PAIR_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_personalized_vs_generic_pairs" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="0-27%${MAX_CONCURRENT_PAIRS}" \
        --output="${COMPARISON_ROOT}/logs/pairs-%A_%a.out" \
        --error="${COMPARISON_ROOT}/logs/pairs-%A_%a.err" \
        --export="${EXPORTS}" \
        "${PAIR_SLURM}"
)"
PAIR_JOB="${PAIR_JOB%%;*}"
printf '%s\n' "${PAIR_JOB}" > "${JOB_ID_FILE}"

set +e
COLLECT_JOB="$(
    "${SBATCH_BIN}" \
        --parsable \
        --job-name="post_personalized_vs_generic_collect" \
        --partition="${PARTITION}" \
        --cpus-per-task="${COLLECTOR_CPUS}" \
        --mem="${COLLECTOR_MEMORY}" \
        --time="${COLLECTOR_TIME}" \
        --dependency="afterok:${PAIR_JOB}" \
        --output="${COMPARISON_ROOT}/logs/collector-%j.out" \
        --error="${COMPARISON_ROOT}/logs/collector-%j.err" \
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
    "[INFO] Submitted all-configuration metric array: ${PAIR_JOB}" \
    "[INFO] Submitted dependent collector:       ${COLLECT_JOB}" \
    "[INFO] Collector dependency:                afterok:${PAIR_JOB}" \
    "[INFO] Job IDs:                             ${JOB_ID_FILE}"
