#!/bin/bash
#SBATCH --job-name=camcan_personalized_comparison_collect
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${COMPARISON_ROOT:?Set COMPARISON_ROOT.}"
: "${PYTHON:?Set PYTHON.}"

THRESHOLDS="${COMPARISON_THRESHOLDS_COLON:-0.20:0.18:0.15}"
THRESHOLDS="${THRESHOLDS//:/,}"

printf '%s\n' \
    "[INFO] Selected personalized-vs-generic collector" \
    "[INFO] Required pair summaries: 8" \
    "[INFO] Required repeat records:  160" \
    "[INFO] Thresholds (V/m):         ${THRESHOLDS}" \
    "[INFO] Repeat aggregation:       per-condition arithmetic mean after metric calculation" \
    "[INFO] Repeat pairing:           none across conditions" \
    "[INFO] Output:                   ${COMPARISON_ROOT}/results"

"${PYTHON}" \
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_personalized_comparison.py" \
    collect \
    --allowlist "${COMPARISON_ROOT}/selection_allowlist.csv" \
    --output-root "${COMPARISON_ROOT}" \
    --thresholds "${THRESHOLDS}" \
    --top-percentile "${COMPARISON_TOP_PERCENTILE:-95.0}" \
    --robust-max-percentile "${COMPARISON_ROBUST_MAX_PERCENTILE:-99.9}" \
    --upper-tail-fraction "${COMPARISON_UPPER_TAIL_FRACTION:-0.01}"

MANIFEST="${COMPARISON_ROOT}/results/analysis_manifest.json"
test -s "${MANIFEST}"
"${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="complete" and p["selected_subject_roi_pairs"]==8 and p["repeat_level_records"]==160 and p["condition_repeat_mean_records"]==16' \
    "${MANIFEST}"

ARCHIVE="${COMPARISON_ROOT}/optimized_best_worst_7_personalized_vs_generic_results.tar.gz"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
tar -C "${COMPARISON_ROOT}/results" -czf "${ARCHIVE}" .
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

printf '%s\n' \
    "[INFO] Comparison completed successfully." \
    "[INFO] Compact result archive: ${ARCHIVE}" \
    "[INFO] Archive checksum:       ${ARCHIVE}.sha256"
