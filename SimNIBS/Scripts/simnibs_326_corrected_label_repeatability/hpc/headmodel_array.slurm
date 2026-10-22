#!/bin/bash
#SBATCH --job-name=ti326h_head
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail
module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${PREFLIGHT_RECEIPT:?Set PREFLIGHT_RECEIPT.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"
LOG_DIR="${LOG_DIR:?Set LOG_DIR.}"
mkdir -p "$LOG_DIR/retry_state"
cd "$(dirname "$PIPELINE_DIR")"
python -m simnibs_326_corrected_label_repeatability validate-module-receipt \
    --receipt "$PREFLIGHT_RECEIPT" >/dev/null

TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing array task ID.}"
JOB_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RETRY_FILE="$LOG_DIR/retry_state/${JOB_KEY}.retry"
MAX_RETRIES="${SIMNIBS326H_MAX_RETRIES:-1}"
RETRY_COUNT=0
[ ! -f "$RETRY_FILE" ] || RETRY_COUNT="$(tr -d '[:space:]' < "$RETRY_FILE")"
[[ "$RETRY_COUNT" =~ ^[0-9]+$ ]] || RETRY_COUNT=0
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid retry cap." >&2; exit 2; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ATTEMPT_LOG="$LOG_DIR/headmodel_${JOB_KEY}_retry_${RETRY_COUNT}_${STAMP}.log"

set +e
timeout --kill-after=30s 7h python -u -m simnibs_326_corrected_label_repeatability \
    build-headmodel --subject-index "$TASK_ID" 2>&1 | tee "$ATTEMPT_LOG"
RUN_EXIT=${PIPESTATUS[0]}
set -e
if [ "$RUN_EXIT" -eq 0 ]; then rm -f "$RETRY_FILE"; exit 0; fi
if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then exit "$RUN_EXIT"; fi
printf '%s\n' "$((RETRY_COUNT + 1))" > "$RETRY_FILE"
scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
exit 0
