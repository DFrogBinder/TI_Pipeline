#!/bin/bash
#SBATCH --job-name=cohort_post_manuscript_subjects
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
: "${FASTSURFER_ROOT:?Set FASTSURFER_ROOT.}"
: "${PYTHON:?Set PYTHON.}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}"

COMMON_SH="${PIPELINE_DIR}/CamCan_Experiment/cohort_pipeline/cohort_post_common.sh"
if [ ! -f "${COMMON_SH}" ]; then
    echo "[ERROR] Cohort post helper is missing: ${COMMON_SH}" >&2
    exit 2
fi
# shellcheck source=cohort_post_common.sh
source "${COMMON_SH}"

if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
   [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || \
   [ "${SLURM_ARRAY_TASK_ID}" -gt 39 ]
then
    echo "[ERROR] Manuscript subject-stage array index must be in 0-39." >&2
    exit 2
fi

ROI_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
REPEAT_NUMBER=$((SLURM_ARRAY_TASK_ID % 10 + 1))
ROI="$(post_roi_for_index "${ROI_INDEX}")"
REPEAT_ID="$(printf '%02d' "${REPEAT_NUMBER}")"
DATASET_ROOT="${STUDY_ROOT}/runs/${ROI}_Runs/${ROI}_Data_${REPEAT_ID}"
SUMMARY="${POST_CAMPAIGN_ROOT}/manuscript_analysis/dataset_summaries/${ROI}_repeat_${REPEAT_ID}.json"
THRESHOLDS="${MANUSCRIPT_THRESHOLDS_COLON:-0.18:0.15}"
THRESHOLDS="${THRESHOLDS//:/,}"

COMMAND=(
    "${PYTHON}"
    "${PIPELINE_DIR}/CamCan_Experiment/post/camcan_manuscript_analysis.py"
    extract-dataset
    --dataset-root "${DATASET_ROOT}"
    --roi "${ROI}"
    --repeat "${REPEAT_ID}"
    --subjects-file "${SUBJECTS_FILE}"
    --atlas-root "${FASTSURFER_ROOT}"
    --summary "${SUMMARY}"
    --workers "${MANUSCRIPT_WORKERS:-${SLURM_CPUS_PER_TASK:-12}}"
    --thresholds "${THRESHOLDS}"
    --top-percentile "${MANUSCRIPT_TOP_PERCENTILE:-95.0}"
    --robust-max-percentile "${MANUSCRIPT_ROBUST_MAX_PERCENTILE:-99.9}"
    --upper-tail-fraction "${MANUSCRIPT_UPPER_TAIL_FRACTION:-0.01}"
)
if [ "${MANUSCRIPT_FORCE:-0}" = "1" ]; then
    COMMAND+=(--force)
fi

printf '%s\n' \
    "[INFO] CamCan manuscript metric extraction" \
    "[INFO] Array index:          ${SLURM_ARRAY_TASK_ID}" \
    "[INFO] ROI:                  ${ROI}" \
    "[INFO] Repeat:               ${REPEAT_ID}" \
    "[INFO] Dataset root:         ${DATASET_ROOT}" \
    "[INFO] Thresholds (V/m):     ${THRESHOLDS}" \
    "[INFO] Robust maximum:       P${MANUSCRIPT_ROBUST_MAX_PERCENTILE:-99.9}" \
    "[INFO] Upper-tail sensitivity:${MANUSCRIPT_UPPER_TAIL_FRACTION:-0.01}" \
    "[INFO] Workers:              ${MANUSCRIPT_WORKERS:-${SLURM_CPUS_PER_TASK:-12}}" \
    "[INFO] Summary:              ${SUMMARY}"

"${COMMAND[@]}"
