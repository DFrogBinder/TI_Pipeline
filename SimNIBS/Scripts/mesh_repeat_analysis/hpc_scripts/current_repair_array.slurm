#!/bin/bash
#SBATCH --job-name=ti_current_repair
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=current-repair-%A_%a.out

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] Submit this script as a Slurm array."
    exit 1
fi

: "${REPAIR_CLI:?Missing REPAIR_CLI.}"
: "${EXPERIMENT_CONFIG:?Missing EXPERIMENT_CONFIG.}"
: "${OUTPUT_ROOT:?Missing OUTPUT_ROOT.}"
: "${MODE:?Missing MODE.}"
: "${LOG_DIR:?Missing LOG_DIR.}"

if [ "${MODE}" = "compare" ]; then
    : "${SCALED_ROOT:?Missing SCALED_ROOT for compare mode.}"
    : "${PAIR2_RERUN_ROOT:?Missing PAIR2_RERUN_ROOT for compare mode.}"
else
    : "${ORIGINAL_ROOT:?Missing ORIGINAL_ROOT for repair mode.}"
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/current_repair_${MODE}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}.log"

PYTHON_ARGS=(
    "${REPAIR_CLI}"
    --mode "${MODE}"
    --config "${EXPERIMENT_CONFIG}"
    --output-root "${OUTPUT_ROOT}"
    --task-index "${SLURM_ARRAY_TASK_ID}"
)

if [ "${OVERWRITE_OUTPUT:-0}" = "1" ]; then
    PYTHON_ARGS+=(--overwrite)
fi

if [ "${MODE}" = "compare" ]; then
    PYTHON_ARGS+=(--scaled-root "${SCALED_ROOT}" --pair2-rerun-root "${PAIR2_RERUN_ROOT}")
else
    PYTHON_ARGS+=(--original-root "${ORIGINAL_ROOT}")
fi

echo "[INFO] Mode:        ${MODE}"
echo "[INFO] Task index:  ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Log file:    ${LOG_FILE}"

python -u "${PYTHON_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
