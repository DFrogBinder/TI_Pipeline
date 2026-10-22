#!/bin/bash
#SBATCH --job-name=ti326h_refs
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail
module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a
export PYTHONNOUSERSITE=1
: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"
LOG_DIR="${LOG_DIR:?Set LOG_DIR.}"
mkdir -p "$LOG_DIR/retry_state"
cd "$(dirname "$PIPELINE_DIR")"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing array task ID.}"
JOB_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="$LOG_DIR/retry_state/${JOB_KEY}.retry"
MAX_RETRIES="${SIMNIBS326H_MAX_RETRIES:-1}"
RETRY_COUNT=0
[ ! -f "$RETRY_FILE" ] || RETRY_COUNT="$(tr -d '[:space:]' < "$RETRY_FILE")"
[[ "$RETRY_COUNT" =~ ^[0-9]+$ ]] || RETRY_COUNT=0
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid retry cap." >&2; exit 2; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ATTEMPT_LOG="$LOG_DIR/reference_${JOB_KEY}_retry_${RETRY_COUNT}_${STAMP}.log"

set +e
python -u -m simnibs_326_corrected_label_repeatability stage-reference \
    --subject-index "$TASK_ID" 2>&1 | tee "$ATTEMPT_LOG"
RUN_EXIT=${PIPESTATUS[0]}
set -e
if [ "$RUN_EXIT" -eq 0 ]; then rm -f "$RETRY_FILE"; exit 0; fi
if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then exit "$RUN_EXIT"; fi
printf '%s\n' "$((RETRY_COUNT + 1))" > "$RETRY_FILE"
scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exit 0
