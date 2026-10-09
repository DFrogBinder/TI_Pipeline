#!/bin/bash
#SBATCH --job-name=ti326_repeat
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

module purge
module use "$HOME/modules"
SIMNIBS326_MATLAB_MODULE="${SIMNIBS326_MATLAB_MODULE:-MATLAB/2023b}"
module load "${SIMNIBS326_MATLAB_MODULE}"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR to simnibs_326_repeatability.}"
: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"

LOG_DIR="${LOG_DIR:-$PIPELINE_DIR/logs/simulation}"
TASK_OFFSET="${TASK_OFFSET:-0}"
TIMEOUT_HOURS="${TI_TASK_TIMEOUT_HOURS:-4}"
mkdir -p "$LOG_DIR/retry_state"
cd "$PIPELINE_DIR"

if ! [[ "$TASK_OFFSET" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TASK_OFFSET must be a non-negative integer." >&2
    exit 2
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID.}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?Missing SLURM_ARRAY_JOB_ID.}"
TASK_INDEX=$((TASK_OFFSET + TASK_ID))
JOB_KEY="${ARRAY_JOB_ID}_${TASK_ID}"
REQUEUE_JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"
RETRY_FILE="$LOG_DIR/retry_state/${JOB_KEY}.retry"
SKIPPED_INPUTS="$LOG_DIR/tasks_skipped_missing_inputs.tsv"
MAX_RETRIES="${SIMNIBS326_MAX_RETRIES:-1}"
RETRY_COUNT=0
if [ -f "$RETRY_FILE" ]; then
    RETRY_COUNT="$(tr -d '[:space:]' < "$RETRY_FILE")"
fi
if ! [[ "$RETRY_COUNT" =~ ^[0-9]+$ ]]; then
    RETRY_COUNT=0
fi
if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SIMNIBS326_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi
ATTEMPT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/task_${JOB_KEY}_retry_${RETRY_COUNT}_${ATTEMPT_STAMP}.log"

EXTRA_ARGS=()
if [ "$RETRY_COUNT" -gt 0 ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "[INFO] Module: SimNIBS/3.2.6-foss-2023a"
echo "[INFO] headreco MATLAB dependency: ${SIMNIBS326_MATLAB_MODULE}"
echo "[INFO] Config: $EXPERIMENT_CONFIG"
echo "[INFO] Global task index: $TASK_INDEX"
echo "[INFO] Attempt log: $LOG_FILE"
echo "[INFO] Retry count: $RETRY_COUNT/$MAX_RETRIES"

set +e
timeout --kill-after=30s "${TIMEOUT_HOURS}h" \
    python -u simulation_runner.py run-task \
        --config "$EXPERIMENT_CONFIG" \
        --task-index "$TASK_INDEX" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "$RUN_EXIT" -eq 126 ]; then
    printf '%s\t%s\t%s\n' "$(date -Is)" "$TASK_INDEX" "$RUN_EXIT" >> "$SKIPPED_INPUTS"
    rm -f "$RETRY_FILE"
    echo "[WARN] Missing task inputs recorded without requeue."
    exit 0
fi

set +e
python -u "$PIPELINE_DIR/../ti_current_repair/validate_repeatability_task.py" \
    --config "$EXPERIMENT_CONFIG" \
    --task-index "$TASK_INDEX" 2>&1 | tee -a "$LOG_FILE"
VALIDATE_EXIT=${PIPESTATUS[0]}
set -e

if [ "$RUN_EXIT" -eq 0 ] && [ "$VALIDATE_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE"
    exit 0
fi

FINAL_EXIT="$RUN_EXIT"
if [ "$FINAL_EXIT" -eq 0 ]; then
    FINAL_EXIT="$VALIDATE_EXIT"
fi
if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
    echo "[ERROR] Incomplete task (runner=$RUN_EXIT validator=$VALIDATE_EXIT); retry limit $MAX_RETRIES exhausted." | tee -a "$LOG_FILE"
    exit "$FINAL_EXIT"
fi

NEXT_RETRY=$((RETRY_COUNT + 1))
printf '%s\n' "$NEXT_RETRY" > "$RETRY_FILE"
echo "[WARN] Incomplete task (runner=$RUN_EXIT validator=$VALIDATE_EXIT); requeueing $REQUEUE_JOB_ID for retry $NEXT_RETRY/$MAX_RETRIES." | tee -a "$LOG_FILE"

set +e
REQUEUE_OUTPUT="$(scontrol requeue "$REQUEUE_JOB_ID" 2>&1)"
REQUEUE_EXIT=$?
set -e
if [ -n "$REQUEUE_OUTPUT" ]; then
    printf '%s\n' "$REQUEUE_OUTPUT" | tee -a "$LOG_FILE"
fi
if [ "$REQUEUE_EXIT" -ne 0 ]; then
    echo "[ERROR] Failed to requeue $REQUEUE_JOB_ID." | tee -a "$LOG_FILE"
    exit "$REQUEUE_EXIT"
fi
exit 0
