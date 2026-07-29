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
PUBLICATION_PACKAGER="${PUBLICATION_PACKAGER:-${PIPELINE_DIR}/CamCan_Experiment/post/package_camcan_publication_inputs.py}"
PUBLICATION_RENDERER="${PUBLICATION_RENDERER:-${PIPELINE_DIR}/CamCan_Experiment/post/build_camcan_supervisor_revision_figures.py}"
MNI_THRESHOLD_TABLE="${MNI_THRESHOLD_TABLE:-${PIPELINE_DIR}/CamCan_Experiment/post/mni152_simnibs401_roi_thresholds.csv}"

printf '%s\n' \
    "[INFO] All-configuration personalized-vs-generic collector" \
    "[INFO] Required pair summaries: 28" \
    "[INFO] Required repeat records:  560" \
    "[INFO] Thresholds (V/m):         ${THRESHOLDS}" \
    "[INFO] Repeat aggregation:       per-condition arithmetic mean after metric calculation" \
    "[INFO] Repeat pairing:           none across conditions" \
    "[INFO] Output:                   ${COMPARISON_ROOT}/results"

COLLECT_COMMAND=(
    "${PYTHON}"
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_personalized_comparison.py"
    collect
    --allowlist "${COMPARISON_ROOT}/selection_allowlist.csv"
    --output-root "${COMPARISON_ROOT}"
    --thresholds "${THRESHOLDS}"
    --top-percentile "${COMPARISON_TOP_PERCENTILE:-95.0}"
    --robust-max-percentile "${COMPARISON_ROBUST_MAX_PERCENTILE:-99.9}"
    --upper-tail-fraction "${COMPARISON_UPPER_TAIL_FRACTION:-0.01}"
)
if [ "$(basename "${PUBLICATION_PACKAGER}")" = "package_camcan_publication_inputs_v5.py" ]; then
    COLLECT_COMMAND+=(--skip-legacy-figures)
fi
"${COLLECT_COMMAND[@]}"

MANIFEST="${COMPARISON_ROOT}/results/analysis_manifest.json"
test -s "${MANIFEST}"
"${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="complete" and p["subject_roi_configurations"]==28 and p["repeat_level_records"]==560 and p["condition_repeat_mean_records"]==56 and p["excluded_personalized_simulations"]==0' \
    "${MANIFEST}"

PACKAGER_ARGS=(
    --role personalized
    --results-dir "${COMPARISON_ROOT}/results"
    --renderer "${PUBLICATION_RENDERER}"
)
if [ "$(basename "${PUBLICATION_PACKAGER}")" = "package_camcan_publication_inputs_v5.py" ]; then
    PACKAGER_ARGS+=(--mni-threshold-table "${MNI_THRESHOLD_TABLE}")
fi
"${PYTHON}" "${PUBLICATION_PACKAGER}" "${PACKAGER_ARGS[@]}"
test -s "${COMPARISON_ROOT}/results/publication_figure_input_manifest.json"
test -s "${COMPARISON_ROOT}/results/publication_inputs.sha256"
(
    cd "${COMPARISON_ROOT}/results"
    sha256sum -c publication_inputs.sha256
)

ARCHIVE="${COMPARISON_ROOT}/optimized_best_worst_7_all_roi_optimizer_matched_results.tar.gz"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
tar -C "${COMPARISON_ROOT}/results" -czf "${ARCHIVE}" .
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

printf '%s\n' \
    "[INFO] Comparison completed successfully." \
    "[INFO] Figure artifact contract: exact source tables + checksums + versioned renderer" \
    "[INFO] Compact result archive: ${ARCHIVE}" \
    "[INFO] Archive checksum:       ${ARCHIVE}.sha256"
