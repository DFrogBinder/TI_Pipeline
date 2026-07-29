#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMCAN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE_DIR="$(cd "${CAMCAN_DIR}/.." && pwd)"

MODE="${1:-all}"
PREFLIGHT="${2:-}"
if [[ "${MODE}" != "all" && "${MODE}" != "cohort" && "${MODE}" != "personalized" ]]; then
    echo "Usage: bash $0 [all|cohort|personalized] [--preflight]" >&2
    exit 2
fi
if [[ -n "${PREFLIGHT}" && "${PREFLIGHT}" != "--preflight" ]]; then
    echo "Usage: bash $0 [all|cohort|personalized] [--preflight]" >&2
    exit 2
fi

THRESHOLD_TABLE="${CAMCAN_DIR}/post/mni152_simnibs401_roi_thresholds.csv"
PACKAGER="${CAMCAN_DIR}/post/package_camcan_publication_inputs_v5.py"
RENDERER="${CAMCAN_DIR}/post/build_camcan_supervisor_revision_figures_v5.py"
MNI401_BASELINES="${MNI_BASELINE_PARENT:-/mnt/parscratch/users/cop23bi/MNI152_SimNIBS401_validation}"
PERSONALIZED_STUDY_ROOT="${PERSONALIZED_STUDY_ROOT:-/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Individualized_Optimized}"
COMPARISON_ROOT="${PERSONALIZED_STUDY_ROOT}/campaigns/optimized_best_worst_7/post_processing/optimizer_matched_personalized_vs_generic_mni_thresholds_v1"

for path in "${THRESHOLD_TABLE}" "${PACKAGER}" "${RENDERER}"; do
    if [ ! -s "${path}" ]; then
        echo "[ERROR] Required MNI-threshold figure input is missing: ${path}" >&2
        exit 2
    fi
done

THRESHOLDS="$(
    python3 -c \
        'import csv,sys; print(":".join(row["threshold_v_per_m"] for row in csv.DictReader(open(sys.argv[1]))))' \
        "${THRESHOLD_TABLE}"
)"
if [ "$(awk -F: '{print NF}' <<< "${THRESHOLDS}")" -ne 4 ]; then
    echo "[ERROR] Expected exactly four ROI-specific MNI thresholds." >&2
    exit 2
fi

printf '%s\n' \
    'MNI-threshold manuscript reanalysis scope:' \
    "  mode: ${MODE}" \
    "  thresholds (Left M1, Right DLPFC, Left hippocampus, Right thalamus): ${THRESHOLDS//:/, } V/m" \
    '  source TI images: read-only' \
    '  FEM simulations: none' \
    '  metric extraction: exact image-level coverage at all four thresholds' \
    '  output policy: isolated v1 directories; schema3/schema4 and v3/v4 figure outputs remain untouched' \
    "  renderer: ${RENDERER}" \
    "  threshold table: ${THRESHOLD_TABLE}"

EXTRA_ARGS=()
if [ "${PREFLIGHT}" = "--preflight" ]; then
    EXTRA_ARGS+=(--preflight)
fi

if [[ "${MODE}" == "all" || "${MODE}" == "cohort" ]]; then
    ANALYSIS_SUBDIR=optimizer_matched_analysis_mni_thresholds_v1 \
    MNI_BASELINE_PARENT="${MNI401_BASELINES}" \
    MANUSCRIPT_THRESHOLDS_COLON="${THRESHOLDS}" \
    PUBLICATION_PACKAGER="${PACKAGER}" \
    PUBLICATION_RENDERER="${RENDERER}" \
    MNI_THRESHOLD_TABLE="${THRESHOLD_TABLE}" \
        bash "${SCRIPT_DIR}/submit_cohort_manuscript_analysis.sh" \
        final_132 "${EXTRA_ARGS[@]}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "personalized" ]]; then
    COMPARISON_ROOT="${COMPARISON_ROOT}" \
    COMPARISON_THRESHOLDS_COLON="${THRESHOLDS}" \
    PUBLICATION_PACKAGER="${PACKAGER}" \
    PUBLICATION_RENDERER="${RENDERER}" \
    MNI_THRESHOLD_TABLE="${THRESHOLD_TABLE}" \
        bash "${SCRIPT_DIR}/submit_personalized_vs_generic_analysis.sh" \
        "${EXTRA_ARGS[@]}"
fi
