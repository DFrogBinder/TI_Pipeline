#!/bin/bash
set -euo pipefail

PACKAGE_DIR="${PACKAGE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS_ROOT="$(dirname "$PACKAGE_DIR")"
export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${1:-preflight}"
[ "$#" -eq 0 ] || shift

case "$MODE" in
    preflight|prepare|copy|verify|checksum|status|archive-preflight|archive-copy|archive-verify|archive-checksum|retire-plan|retire-submit|retire-status) ;;
    *) echo "[ERROR] Unsupported mode: $MODE" >&2; exit 2 ;;
esac

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[ERROR] This workflow must run on the login node, not inside a Slurm job." >&2
    exit 2
fi

exec "$PYTHON_BIN" -m repeatability_dataset_consolidation "$MODE" "$@"
