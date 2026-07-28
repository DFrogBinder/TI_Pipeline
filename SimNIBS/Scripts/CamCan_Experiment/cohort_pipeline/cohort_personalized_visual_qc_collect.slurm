#!/bin/bash
#SBATCH --job-name=camcan_personalized_visual_qc_collect
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${COMPARISON_ROOT:?Set COMPARISON_ROOT.}"
: "${VISUAL_QC_ROOT:?Set VISUAL_QC_ROOT.}"
: "${PYTHON:?Set PYTHON.}"

printf '%s\n' \
    "[INFO] Selected personalized-vs-generic full-post visual-QC collector" \
    "[INFO] Required selected pairs:          8" \
    "[INFO] Required full-post records:       160" \
    "[INFO] Required paired PNGs:             160" \
    "[INFO] Required multipage pair reports:  16" \
    "[INFO] Out-of-scope personalized fields: 200 (excluded)" \
    "[INFO] Output:                           ${VISUAL_QC_ROOT}"

"${PYTHON}" \
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_personalized_visual_qc.py" \
    collect \
    --allowlist "${COMPARISON_ROOT}/selection_allowlist.csv" \
    --output-root "${VISUAL_QC_ROOT}"

MANIFEST="${VISUAL_QC_ROOT}/results/analysis_manifest.json"
test -s "${MANIFEST}"
"${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="complete" and p["selected_subject_roi_pairs"]==8 and p["full_post_records"]==160 and p["paired_visualization_pngs"]==160 and p["multipage_pair_reports"]==16 and p["excluded_out_of_scope_personalized_simulations"]==200' \
    "${MANIFEST}"

ARCHIVE="${VISUAL_QC_ROOT}/optimized_best_worst_7_full_post_visual_qc.tar.gz"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
tar \
    -C "${VISUAL_QC_ROOT}" \
    --exclude='*.nii' \
    --exclude='*.nii.gz' \
    --exclude='logs' \
    --exclude="$(basename "${ARCHIVE}")" \
    --exclude="$(basename "${ARCHIVE}").sha256" \
    -czf "${ARCHIVE}" \
    results \
    pair_summaries \
    post_records \
    post_products \
    paired_fields \
    pair_reports
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

printf '%s\n' \
    "[INFO] Visual QC completed successfully." \
    "[INFO] Compact archive (derived NIfTI excluded): ${ARCHIVE}" \
    "[INFO] Archive checksum:                       ${ARCHIVE}.sha256"
