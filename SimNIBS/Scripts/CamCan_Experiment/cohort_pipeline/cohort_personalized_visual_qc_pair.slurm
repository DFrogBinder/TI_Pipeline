#!/bin/bash
#SBATCH --job-name=camcan_personalized_visual_qc_pair
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
: "${VISUAL_QC_ROOT:?Set VISUAL_QC_ROOT.}"
: "${FASTSURFER_ROOT:?Set FASTSURFER_ROOT.}"
: "${MNI_FIXED_ATLAS_PATH:?Set MNI_FIXED_ATLAS_PATH.}"
: "${MNI_BASELINE_PARENT:?Set MNI_BASELINE_PARENT.}"
: "${PYTHON:?Set PYTHON.}"
: "${GENERIC_TARGETS_SHA256:?Set GENERIC_TARGETS_SHA256.}"
: "${INDIVIDUALIZED_TARGETS_SHA256:?Set INDIVIDUALIZED_TARGETS_SHA256.}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}"

if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-7]$ ]]; then
    echo "[ERROR] Visual-QC pair index must be in 0-7." >&2
    exit 2
fi

COMMAND=(
    "${PYTHON}"
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_personalized_visual_qc.py"
    run-pair
    --pair-index "${SLURM_ARRAY_TASK_ID}"
    --allowlist "${COMPARISON_ROOT}/selection_allowlist.csv"
    --generic-study-root "${GENERIC_STUDY_ROOT}"
    --personalized-study-root "${PERSONALIZED_STUDY_ROOT}"
    --atlas-root "${FASTSURFER_ROOT}"
    --mni-fixed-atlas "${MNI_FIXED_ATLAS_PATH}"
    --mni-baseline-parent "${MNI_BASELINE_PARENT}"
    --output-root "${VISUAL_QC_ROOT}"
    --targets-sha256 "${GENERIC_TARGETS_SHA256}"
    --individualized-targets-sha256 "${INDIVIDUALIZED_TARGETS_SHA256}"
    --workers "${VISUAL_QC_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
)
if [ "${VISUAL_QC_FORCE:-0}" = "1" ]; then
    COMMAND+=(--force)
fi

printf '%s\n' \
    "[INFO] Selected personalized-vs-generic full-post visual QC" \
    "[INFO] Pair index:               ${SLURM_ARRAY_TASK_ID}" \
    "[INFO] Full-post inputs:         2 conditions x 10 repeats = 20" \
    "[INFO] Paired PNGs expected:     20" \
    "[INFO] Multipage PDFs expected:  2" \
    "[INFO] Workers:                  ${VISUAL_QC_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" \
    "[INFO] Source simulations:       read-only" \
    "[INFO] Source post directories:  read-only" \
    "[INFO] Isolated QC output:       ${VISUAL_QC_ROOT}"

"${COMMAND[@]}"
