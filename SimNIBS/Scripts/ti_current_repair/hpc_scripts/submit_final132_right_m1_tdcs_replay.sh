#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python -u "$PIPELINE_DIR/pipeline/tdcs_replay_workflow.py" submit \
    "$@"
