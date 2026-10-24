#!/bin/bash
#SBATCH --job-name=ti326_subject_scaffold
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0%1
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

: "${SCRIPTS_ROOT:?Set SCRIPTS_ROOT to SimNIBS/Scripts.}"
: "${REPEATABILITY_DIR:?Set REPEATABILITY_DIR to simnibs_326_repeatability.}"
: "${PREFLIGHT_RECEIPT:?Set PREFLIGHT_RECEIPT to the module receipt.}"
: "${SUBJECT:?Set SUBJECT to a canonical final-132 subject.}"
: "${CORRECTED_SOURCE_ROOT:?Set CORRECTED_SOURCE_ROOT.}"
: "${CREATOR_OUTPUT_ROOT:?Set CREATOR_OUTPUT_ROOT.}"
: "${FINAL132_SUBJECTS:?Set FINAL132_SUBJECTS.}"
: "${LOG_DIR:?Set LOG_DIR.}"

# The shared SimNIBS 3.2.6 compatibility shim names its package-root
# interface PIPELINE_DIR. This worker exposes the same directory under the
# more specific REPEATABILITY_DIR name, so map it explicitly at the boundary.
export PIPELINE_DIR="$REPEATABILITY_DIR"
# shellcheck source=/dev/null
source "$REPEATABILITY_DIR/hpc/activate_compat.sh"
cd "$REPEATABILITY_DIR"
python -u pipeline.py validate-module-receipt \
    --receipt "$PREFLIGHT_RECEIPT" >/dev/null

TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID.}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?Missing SLURM_ARRAY_JOB_ID.}"
if [ "$TASK_ID" != "0" ]; then
    echo "[ERROR] Singleton scaffold task must have array index 0." >&2
    exit 2
fi
JOB_KEY="${ARRAY_JOB_ID}_${TASK_ID}"
REQUEUE_JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"
mkdir -p "$LOG_DIR/retry_state"
RETRY_FILE="$LOG_DIR/retry_state/${JOB_KEY}.retry"
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
LOG_FILE="$LOG_DIR/scaffold_${JOB_KEY}_retry_${RETRY_COUNT}_${ATTEMPT_STAMP}.log"

echo "[INFO] Subject: $SUBJECT"
echo "[INFO] Module: SimNIBS/3.2.6-foss-2023a"
echo "[INFO] headreco MATLAB dependency: ${SIMNIBS326_MATLAB_MODULE}"
echo "[INFO] Retention: complete headreco all --noclean output"
echo "[INFO] Attempt log: $LOG_FILE"
echo "[INFO] Retry count: $RETRY_COUNT/$MAX_RETRIES"

cd "$SCRIPTS_ROOT"
set +e
timeout --kill-after=30s 7h python -u \
    -m simnibs_326_subject_scaffold_creator build \
    --subject="$SUBJECT" \
    --corrected-source-root="$CORRECTED_SOURCE_ROOT" \
    --output-root="$CREATOR_OUTPUT_ROOT" \
    --final132-subjects="$FINAL132_SUBJECTS" \
    --confirm-output-root="$CREATOR_OUTPUT_ROOT" 2>&1 | tee "$LOG_FILE"
RUN_EXIT=${PIPESTATUS[0]}
set -e

if [ "$RUN_EXIT" -eq 0 ]; then
    rm -f "$RETRY_FILE"
    exit 0
fi
if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
    echo "[ERROR] Scaffold task failed with exit $RUN_EXIT; retry limit $MAX_RETRIES exhausted." | tee -a "$LOG_FILE"
    exit "$RUN_EXIT"
fi

NEXT_RETRY=$((RETRY_COUNT + 1))
printf '%s\n' "$NEXT_RETRY" > "$RETRY_FILE"
echo "[WARN] Scaffold task failed with exit $RUN_EXIT; requeueing $REQUEUE_JOB_ID for retry $NEXT_RETRY/$MAX_RETRIES." | tee -a "$LOG_FILE"
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
