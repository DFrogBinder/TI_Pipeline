#!/bin/bash
#SBATCH --job-name=ti_repeat_flow
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

set -euo pipefail

: "${PIPELINE_DIR:?Missing absolute PIPELINE_DIR export.}"
: "${EXPERIMENT_ROOT:?Missing EXPERIMENT_ROOT export.}"
: "${WORKFLOW_STEP:?Missing WORKFLOW_STEP export.}"
: "${MAX_CONCURRENT_TASKS:?Missing MAX_CONCURRENT_TASKS export.}"
: "${ANALYSIS_MAX_CONCURRENT_TASKS:?Missing ANALYSIS_MAX_CONCURRENT_TASKS export.}"

WORKFLOW_PY="${PIPELINE_DIR}/pipeline/staged_median_fixed_experiment.py"
if [ ! -f "${WORKFLOW_PY}" ]; then
    echo "[ERROR] Workflow entry point does not exist: ${WORKFLOW_PY}" >&2
    exit 2
fi
if [ ! -d "${EXPERIMENT_ROOT}/_pipeline/configs" ]; then
    echo "[ERROR] Experiment is not initialized: ${EXPERIMENT_ROOT}" >&2
    exit 2
fi

cd "${PIPELINE_DIR}"

echo "[INFO] Pipeline root:             ${PIPELINE_DIR}"
echo "[INFO] Experiment root:           ${EXPERIMENT_ROOT}"
echo "[INFO] Workflow step:             ${WORKFLOW_STEP}"
echo "[INFO] Simulation concurrency:    ${MAX_CONCURRENT_TASKS}"
echo "[INFO] Analysis concurrency:      ${ANALYSIS_MAX_CONCURRENT_TASKS}"
echo "[INFO] Controller Slurm job:      ${SLURM_JOB_ID:-manual}"

python -u "${WORKFLOW_PY}" advance-workflow \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --step "${WORKFLOW_STEP}" \
    --max-concurrent "${MAX_CONCURRENT_TASKS}" \
    --analysis-max-concurrent "${ANALYSIS_MAX_CONCURRENT_TASKS}"
