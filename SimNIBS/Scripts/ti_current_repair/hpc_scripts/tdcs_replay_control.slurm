#!/bin/bash
#SBATCH --job-name=tdcs_rmi_control
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${TDCS_REPLAY_PLAN:?Set TDCS_REPLAY_PLAN.}"
: "${TDCS_REPLAY_AUDIT:?Set TDCS_REPLAY_AUDIT.}"
: "${TDCS_REPLAY_VERSION:?Set TDCS_REPLAY_VERSION.}"
: "${TDCS_REPLAY_ACTION:?Set TDCS_REPLAY_ACTION.}"

# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc_scripts/tdcs_replay_activate.sh"
cd "$PIPELINE_DIR"

case "$TDCS_REPLAY_ACTION" in
    finalize-version)
        python -u pipeline/tdcs_replay_workflow.py finalize-version \
            --plan "$TDCS_REPLAY_PLAN" \
            --audit "$TDCS_REPLAY_AUDIT" \
            --version "$TDCS_REPLAY_VERSION"
        ;;
    finalize-all)
        python -u pipeline/tdcs_replay_workflow.py finalize-all \
            --plan "$TDCS_REPLAY_PLAN" \
            --audit "$TDCS_REPLAY_AUDIT"
        ;;
    *)
        echo "[ERROR] Unsupported TDCS_REPLAY_ACTION: $TDCS_REPLAY_ACTION" >&2
        exit 2
        ;;
esac
