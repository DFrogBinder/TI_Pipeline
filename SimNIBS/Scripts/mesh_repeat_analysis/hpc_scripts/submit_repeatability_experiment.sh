#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <experiment-config.json> [max-concurrent-tasks]"
    exit 1
fi

CONFIG_PATH="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1")"

MAX_CONCURRENT="${2:-16}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_COUNT="$(python "${SCRIPT_DIR}/simulation_runners/repeatability_experiment.py" \
    show-plan \
    --config "${CONFIG_PATH}" \
    --count-only)"

if ! [[ "${TASK_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Could not resolve task count from config: ${CONFIG_PATH}"
    exit 1
fi

if [ "${TASK_COUNT}" -lt 1 ]; then
    echo "[ERROR] Config produced zero tasks: ${CONFIG_PATH}"
    exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"

echo "[INFO] Experiment config: ${CONFIG_PATH}"
echo "[INFO] Task count:        ${TASK_COUNT}"
echo "[INFO] Array spec:        ${ARRAY_SPEC}"

sbatch \
    --array="${ARRAY_SPEC}" \
    --export=ALL,EXPERIMENT_CONFIG="${CONFIG_PATH}",PIPELINE_DIR="${SCRIPT_DIR}" \
    "${SCRIPT_DIR}/hpc_scripts/repeatability_experiment_array.slurm"
