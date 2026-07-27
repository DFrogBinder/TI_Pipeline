#!/bin/bash
#SBATCH --job-name=cohort_post_manuscript_collect
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR to the repository Scripts directory.}"
: "${STUDY_ROOT:?Set STUDY_ROOT.}"
: "${SUBJECTS_FILE:?Set SUBJECTS_FILE.}"
: "${POST_CAMPAIGN_ROOT:?Set POST_CAMPAIGN_ROOT.}"
: "${COHORT_ID:?Set COHORT_ID.}"
: "${MNI_FIXED_ATLAS_PATH:?Set MNI_FIXED_ATLAS_PATH.}"
: "${MNI_BASELINE_PARENT:?Set MNI_BASELINE_PARENT.}"
: "${PYTHON:?Set PYTHON.}"

THRESHOLDS="${MANUSCRIPT_THRESHOLDS_COLON:-0.18:0.15}"
THRESHOLDS="${THRESHOLDS//:/,}"
OUTPUT_DIR="${POST_CAMPAIGN_ROOT}/manuscript_analysis/results"

printf '%s\n' \
    "[INFO] CamCan manuscript analysis collector" \
    "[INFO] Study root:           ${STUDY_ROOT}" \
    "[INFO] Thresholds (V/m):     ${THRESHOLDS}" \
    "[INFO] Repeat aggregation:   arithmetic mean of per-repeat metrics" \
    "[INFO] MNI atlas:            ${MNI_FIXED_ATLAS_PATH}" \
    "[INFO] MNI baselines:        ${MNI_BASELINE_PARENT}" \
    "[INFO] Output directory:     ${OUTPUT_DIR}"

"${PYTHON}" \
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_manuscript_analysis.py" \
    collect \
    --study-root "${STUDY_ROOT}" \
    --subjects-file "${SUBJECTS_FILE}" \
    --mni-atlas "${MNI_FIXED_ATLAS_PATH}" \
    --mni-baseline-parent "${MNI_BASELINE_PARENT}" \
    --out-dir "${OUTPUT_DIR}" \
    --thresholds "${THRESHOLDS}" \
    --top-percentile "${MANUSCRIPT_TOP_PERCENTILE:-95.0}" \
    --robust-max-percentile "${MANUSCRIPT_ROBUST_MAX_PERCENTILE:-99.9}" \
    --upper-tail-fraction "${MANUSCRIPT_UPPER_TAIL_FRACTION:-0.01}"

test -s "${OUTPUT_DIR}/analysis_manifest.json"
ARCHIVE="${POST_CAMPAIGN_ROOT}/manuscript_analysis/${COHORT_ID}_manuscript_analysis_results.tar.gz"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
tar -C "${OUTPUT_DIR}" -czf "${ARCHIVE}" .
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
printf '%s\n' \
    "[INFO] Compact result archive: ${ARCHIVE}" \
    "[INFO] Archive checksum:       ${ARCHIVE}.sha256"
