#!/bin/bash
#SBATCH --job-name=camcan_personalized_comparison_pair
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${GENERIC_STUDY_ROOT:?Set GENERIC_STUDY_ROOT.}"
: "${PERSONALIZED_STUDY_ROOT:?Set PERSONALIZED_STUDY_ROOT.}"
: "${COMPARISON_ROOT:?Set COMPARISON_ROOT.}"
: "${FASTSURFER_ROOT:?Set FASTSURFER_ROOT.}"
: "${PYTHON:?Set PYTHON.}"
: "${GENERIC_TARGETS_SHA256:?Set GENERIC_TARGETS_SHA256.}"
: "${INDIVIDUALIZED_TARGETS_SHA256:?Set INDIVIDUALIZED_TARGETS_SHA256.}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}"

if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
   [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || \
   [ "${SLURM_ARRAY_TASK_ID}" -gt 27 ]
then
    echo "[ERROR] Comparison pair index must be in 0-27." >&2
    exit 2
fi

THRESHOLDS="${COMPARISON_THRESHOLDS_COLON:-0.20:0.18:0.15}"
THRESHOLDS="${THRESHOLDS//:/,}"
COMMAND=(
    "${PYTHON}"
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_personalized_comparison.py"
    extract-pair
    --pair-index "${SLURM_ARRAY_TASK_ID}"
    --allowlist "${COMPARISON_ROOT}/selection_allowlist.csv"
    --generic-study-root "${GENERIC_STUDY_ROOT}"
    --personalized-study-root "${PERSONALIZED_STUDY_ROOT}"
    --atlas-root "${FASTSURFER_ROOT}"
    --output-root "${COMPARISON_ROOT}"
    --targets-sha256 "${GENERIC_TARGETS_SHA256}"
    --individualized-targets-sha256 "${INDIVIDUALIZED_TARGETS_SHA256}"
    --workers "${COMPARISON_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
    --thresholds "${THRESHOLDS}"
    --top-percentile "${COMPARISON_TOP_PERCENTILE:-95.0}"
    --robust-max-percentile "${COMPARISON_ROBUST_MAX_PERCENTILE:-99.9}"
    --upper-tail-fraction "${COMPARISON_UPPER_TAIL_FRACTION:-0.01}"
)
if [ "${COMPARISON_FORCE:-0}" = "1" ]; then
    COMMAND+=(--force)
fi

printf '%s\n' \
    "[INFO] All-configuration personalized-vs-generic comparison" \
    "[INFO] Pair index:             ${SLURM_ARRAY_TASK_ID}" \
    "[INFO] Inputs:                 2 conditions x 10 repeats = 20" \
    "[INFO] Thresholds (V/m):       ${THRESHOLDS}" \
    "[INFO] Workers:                ${COMPARISON_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" \
    "[INFO] Generic study:          ${GENERIC_STUDY_ROOT}" \
    "[INFO] Personalized study:     ${PERSONALIZED_STUDY_ROOT}" \
    "[INFO] Comparison output:      ${COMPARISON_ROOT}"

"${COMMAND[@]}"
