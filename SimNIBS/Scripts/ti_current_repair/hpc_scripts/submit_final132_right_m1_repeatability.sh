#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_REPAIR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" \
    "${CURRENT_REPAIR_DIR}/pipeline/setup_final132_right_m1_repeatability.py" \
    "$@"
