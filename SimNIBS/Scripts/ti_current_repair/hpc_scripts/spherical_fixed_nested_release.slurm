#!/bin/bash
#SBATCH --job-name=ti_nested_release
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${NESTED_OUTPUT_ROOT:?Set NESTED_OUTPUT_ROOT.}"
: "${CONTINUE_OFFSET:?Set CONTINUE_OFFSET.}"
: "${PREVIOUS_JOB:?Set PREVIOUS_JOB.}"

SUBMITTER="$PIPELINE_DIR/hpc_scripts/submit_spherical_fixed_nested_experiment.sh"
if [ ! -f "$SUBMITTER" ]; then
    echo "[ERROR] Missing nested submitter: $SUBMITTER" >&2
    exit 2
fi

echo "[INFO] Nested QOS-safe release controller"
echo "[INFO] Controller job:       ${SLURM_JOB_ID:-manual}"
echo "[INFO] Completed predecessor: $PREVIOUS_JOB"
echo "[INFO] Continue global offset: $CONTINUE_OFFSET"
echo "[INFO] Nested output root:    $NESTED_OUTPUT_ROOT"

bash "$SUBMITTER" \
    nested \
    --internal-release \
    "--continue-offset=$CONTINUE_OFFSET" \
    "--previous-job=$PREVIOUS_JOB"
