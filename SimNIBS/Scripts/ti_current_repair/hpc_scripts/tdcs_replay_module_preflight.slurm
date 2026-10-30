#!/bin/bash
#SBATCH --job-name=tdcs_rmi_check
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${TDCS_REPLAY_PLAN:?Set TDCS_REPLAY_PLAN.}"
: "${TDCS_REPLAY_AUDIT:?Set TDCS_REPLAY_AUDIT.}"
: "${TDCS_REPLAY_VERSION:?Set TDCS_REPLAY_VERSION.}"
: "${TDCS_REPLAY_MODULE_RECEIPT:?Set TDCS_REPLAY_MODULE_RECEIPT.}"

# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc_scripts/tdcs_replay_activate.sh"
cd "$PIPELINE_DIR"

python -u pipeline/tdcs_replay_workflow.py module-preflight \
    --plan "$TDCS_REPLAY_PLAN" \
    --audit "$TDCS_REPLAY_AUDIT" \
    --version "$TDCS_REPLAY_VERSION" \
    --receipt "$TDCS_REPLAY_MODULE_RECEIPT"
