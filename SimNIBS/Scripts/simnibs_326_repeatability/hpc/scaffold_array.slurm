#!/bin/bash
#SBATCH --job-name=ti326_headreco
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a.out
#SBATCH --requeue

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR to simnibs_326_repeatability.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"
LOG_DIR="${LOG_DIR:-$PIPELINE_DIR/logs/scaffold}"
mkdir -p "$LOG_DIR/retry_state"
cd "$PIPELINE_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID.}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}"
RETRY_FILE="$LOG_DIR/retry_state/${JOB_KEY}.retry"
LOG_FILE="$LOG_DIR/scaffold_${SLURM_JOB_ID}_${TASK_ID}.log"
RETRY_COUNT=0
if [ -f "$RETRY_FILE" ]; then
    RETRY_COUNT="$(tr -d '[:space:]' < "$RETRY_FILE")"
fi
if ! [[ "$RETRY_COUNT" =~ ^[0-9]+$ ]]; then
    RETRY_COUNT=0
fi

echo "[INFO] Module: SimNIBS/3.2.6-foss-2023a"
echo "[INFO] Scaffold task: $TASK_ID/9"
echo "[INFO] Retry count: $RETRY_COUNT (unlimited until validation passes)"

set +e
timeout --kill-after=30s 7h python -u scaffold_runner.py build \
    --subject-index "$TASK_ID" 2>&1 | tee "$LOG_FILE"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "$RUN_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE"
    exit 0
fi

NEXT_RETRY=$((RETRY_COUNT + 1))
printf '%s\n' "$NEXT_RETRY" > "$RETRY_FILE"
echo "[WARN] Scaffold task failed with exit $RUN_EXIT; requeueing retry $NEXT_RETRY."
scontrol requeue "$SLURM_JOB_ID"
