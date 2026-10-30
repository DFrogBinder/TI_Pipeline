#!/bin/bash
#SBATCH --job-name=tdcs_rmi_replay
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${TDCS_REPLAY_PLAN:?Set TDCS_REPLAY_PLAN.}"
: "${TDCS_REPLAY_AUDIT:?Set TDCS_REPLAY_AUDIT.}"
: "${TDCS_REPLAY_VERSION:?Set TDCS_REPLAY_VERSION.}"
: "${TDCS_REPLAY_MODULE_RECEIPT:?Set TDCS_REPLAY_MODULE_RECEIPT.}"

# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc_scripts/tdcs_replay_activate.sh"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID.}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?Missing SLURM_ARRAY_JOB_ID.}"
JOB_KEY="${ARRAY_JOB_ID}_${TASK_ID}"
REQUEUE_JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"
LOG_DIR="${TDCS_REPLAY_LOG_DIR:-$PIPELINE_DIR/logs/tdcs_replay}"
STATE_DIR="$LOG_DIR/retry_state"
MAX_RETRIES="${TDCS_REPLAY_MAX_RETRIES:-1}"
TIMEOUT_HOURS="${TDCS_REPLAY_TIMEOUT_HOURS:-4}"
RETRY_FILE="$STATE_DIR/${JOB_KEY}.retry"
mkdir -p "$STATE_DIR"

if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TDCS_REPLAY_MAX_RETRIES must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "$TIMEOUT_HOURS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] TDCS_REPLAY_TIMEOUT_HOURS must be a positive integer." >&2
    exit 2
fi

RETRY_COUNT=0
if [ -f "$RETRY_FILE" ]; then
    RETRY_COUNT="$(tr -d '[:space:]' < "$RETRY_FILE")"
fi
if ! [[ "$RETRY_COUNT" =~ ^[0-9]+$ ]]; then
    RETRY_COUNT=0
fi

ATTEMPT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/task_${JOB_KEY}_retry_${RETRY_COUNT}_${ATTEMPT_STAMP}.log"
RUNNER="$PIPELINE_DIR/simulation_runners/tdcs_replay.py"
WORKFLOW="$PIPELINE_DIR/pipeline/tdcs_replay_workflow.py"

cd "$PIPELINE_DIR"
python -u "$WORKFLOW" validate-module-receipt \
    --plan "$TDCS_REPLAY_PLAN" \
    --audit "$TDCS_REPLAY_AUDIT" \
    --version "$TDCS_REPLAY_VERSION" \
    --receipt "$TDCS_REPLAY_MODULE_RECEIPT" >/dev/null

EXTRA_ARGS=()
if [ "$RETRY_COUNT" -gt 0 ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "[INFO] Version: $TDCS_REPLAY_VERSION"
echo "[INFO] Task index: $TASK_ID"
echo "[INFO] Attempt log: $LOG_FILE"
echo "[INFO] Retry count: $RETRY_COUNT/$MAX_RETRIES"
echo "[INFO] Source policy: read-only"

set +e
timeout --kill-after=30s "${TIMEOUT_HOURS}h" \
    python -u "$RUNNER" run-task \
        --plan "$TDCS_REPLAY_PLAN" \
        --audit "$TDCS_REPLAY_AUDIT" \
        --version "$TDCS_REPLAY_VERSION" \
        --task-index "$TASK_ID" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "$RUN_EXIT" -eq 126 ]; then
    echo "[ERROR] Audited source input is unavailable or has drifted; not requeueing." | tee -a "$LOG_FILE"
    exit 126
fi

set +e
python -u "$WORKFLOW" validate-task \
    --plan "$TDCS_REPLAY_PLAN" \
    --audit "$TDCS_REPLAY_AUDIT" \
    --version "$TDCS_REPLAY_VERSION" \
    --task-index "$TASK_ID" 2>&1 | tee -a "$LOG_FILE"
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
    echo "[ERROR] Incomplete task (runner=$RUN_EXIT validator=$VALIDATE_EXIT); retry limit exhausted." | tee -a "$LOG_FILE"
    exit "$FINAL_EXIT"
fi

NEXT_RETRY=$((RETRY_COUNT + 1))
printf '%s\n' "$NEXT_RETRY" > "$RETRY_FILE"
echo "[WARN] Incomplete task; requeueing exact element $REQUEUE_JOB_ID for retry $NEXT_RETRY/$MAX_RETRIES." | tee -a "$LOG_FILE"

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
