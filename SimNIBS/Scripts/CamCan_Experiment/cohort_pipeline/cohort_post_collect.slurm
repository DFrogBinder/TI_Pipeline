#!/bin/bash
#SBATCH --job-name=cohort_post_collect
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
   [ "${SLURM_ARRAY_TASK_ID}" -gt 3 ]
then
    echo "[ERROR] Collector array index must be in 0-3." >&2
    exit 2
fi

ROI="$(post_roi_for_index "${SLURM_ARRAY_TASK_ID}")"
configure_cohort_post_environment "${ROI}"

export BATCH_REPEATS="01 02 03 04 05 06 07 08 09 10"
export BATCH_SUMMARY_FILENAME="${BATCH_ROOT}/post_processing_batch_summary.json"
export BATCH_STOP_ON_ERROR="1"
export PIPELINE_POPULATION_ENABLED="1"
export PIPELINE_POPULATION_OUT_DIR=""
export PIPELINE_REPEATABILITY_ENABLED="1"
export PIPELINE_REPEATABILITY_OUTPUT_DIR=""
export PIPELINE_REPEATABILITY_LOGS_ROOT=""
export PIPELINE_FIGURE_GENERATION_ENABLED="1"
export PIPELINE_FIGURE_OUTPUT_DIR=""

printf '%s\n' \
    "[INFO] Cohort post collector stage" \
    "[INFO] Array index: ${SLURM_ARRAY_TASK_ID}" \
    "[INFO] ROI:         ${ROI}" \
    "[INFO] Repeats:     01-10" \
    "[INFO] Subjects:    $(wc -w <<< "${PIPELINE_SUBJECTS}")" \
    "[INFO] Batch root:  ${BATCH_ROOT}"

# Completed subject metrics are fingerprint-checked and skipped. This pass
# builds complete-case per-repeat population outputs, cross-repeat
# repeatability results, and static figures for this ROI.
source "${REPO_DIR}/HPC_scripts/run_post_processing_batch_common.sh"
