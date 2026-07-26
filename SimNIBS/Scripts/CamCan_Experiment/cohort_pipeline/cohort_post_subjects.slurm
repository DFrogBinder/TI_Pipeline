#!/bin/bash
#SBATCH --job-name=cohort_post_subjects
#SBATCH --partition=sheffield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR to the repository Scripts directory.}"
COMMON_SH="${PIPELINE_DIR}/CamCan_Experiment/cohort_pipeline/cohort_post_common.sh"
if [ ! -f "${COMMON_SH}" ]; then
    echo "[ERROR] Cohort post helper is missing: ${COMMON_SH}" >&2
    exit 2
fi
# shellcheck source=cohort_post_common.sh
source "${COMMON_SH}"

: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task.}"
if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || \
   [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || \
   [ "${SLURM_ARRAY_TASK_ID}" -gt 39 ]
then
    echo "[ERROR] Subject-stage array index must be in 0-39." >&2
    exit 2
fi

ROI_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
REPEAT_NUMBER=$((SLURM_ARRAY_TASK_ID % 10 + 1))
ROI="$(post_roi_for_index "${ROI_INDEX}")"
REPEAT_ID="$(printf '%02d' "${REPEAT_NUMBER}")"

configure_cohort_post_environment "${ROI}"

export BATCH_REPEATS="${REPEAT_ID}"
export BATCH_SUMMARY_FILENAME="${POST_CAMPAIGN_ROOT}/subject_summaries/${ROI}/repeat_${REPEAT_ID}.json"
export BATCH_STOP_ON_ERROR="1"
export PIPELINE_POPULATION_ENABLED="0"
export PIPELINE_POPULATION_OUT_DIR=""
export PIPELINE_REPEATABILITY_ENABLED="0"
export PIPELINE_REPEATABILITY_OUTPUT_DIR=""
export PIPELINE_REPEATABILITY_LOGS_ROOT=""
export PIPELINE_FIGURE_GENERATION_ENABLED="0"
export PIPELINE_FIGURE_OUTPUT_DIR=""

mkdir -p "$(dirname "${BATCH_SUMMARY_FILENAME}")"

printf '%s\n' \
    "[INFO] Cohort post subject stage" \
    "[INFO] Array index: ${SLURM_ARRAY_TASK_ID}" \
    "[INFO] ROI:         ${ROI}" \
    "[INFO] Repeat:      ${REPEAT_ID}" \
    "[INFO] Subjects:    $(wc -w <<< "${PIPELINE_SUBJECTS}")" \
    "[INFO] Batch root:  ${BATCH_ROOT}"

source "${REPO_DIR}/HPC_scripts/run_post_processing_batch_common.sh"
