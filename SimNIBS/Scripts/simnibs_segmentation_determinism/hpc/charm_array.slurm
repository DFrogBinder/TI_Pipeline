#!/bin/bash
#SBATCH --job-name=segdet401
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%10

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR}"
: "${SEGDET_OUTPUT_ROOT:?Set SEGDET_OUTPUT_ROOT}"
: "${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}"
: "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
MAX_RETRIES="${SEGDET_MAX_RETRIES:-1}"
if ! [[ "$MAX_RETRIES" =~ ^[01]$ ]]; then
    echo "SEGDET_MAX_RETRIES must be 0 or 1." >&2
    exit 2
fi

module --force purge
module load SimNIBS/4.0.1-foss-2023a
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"
CHARM_BIN="$(command -v charm)"
export CHARM_BIN

retry_dir="$SEGDET_OUTPUT_ROOT/_pipeline/retry_state/simnibs401_charm"
mkdir -p "$retry_dir"
attempt_file="$retry_dir/task_${SLURM_ARRAY_TASK_ID}.failed_attempts"
failed_attempts=0
if [[ -f "$attempt_file" ]]; then
    read -r failed_attempts < "$attempt_file"
fi
if ! [[ "$failed_attempts" =~ ^[0-9]+$ ]]; then
    echo "Invalid retry state: $attempt_file" >&2
    exit 2
fi
attempt=$((failed_attempts + 1))

if python3 -m simnibs_segmentation_determinism run-repeat \
    --arm simnibs401_charm \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --attempt "$attempt"; then
    exit 0
else
    status=$?
fi

if (( failed_attempts < MAX_RETRIES )); then
    printf '%s\n' "$attempt" > "$attempt_file"
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$status"
