#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--preflight" ]; }; then
    echo "Usage: bash $0 [--preflight]" >&2
    exit 2
fi

export STUDY_CONFIG="${STUDY_CONFIG:-${SCRIPT_DIR}/studies/corrected_v4_individualized_optimized.json}"

if [ "$#" -eq 1 ]; then
    exec bash "${SCRIPT_DIR}/submit_cohort_pipeline.sh" \
        optimized_best_worst_7 \
        --preflight
fi
exec bash "${SCRIPT_DIR}/submit_cohort_pipeline.sh" \
    optimized_best_worst_7
