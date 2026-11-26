#!/bin/bash
#SBATCH --job-name=hist_native_extract
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-19

set -euo pipefail
module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${DATASET_ROOT:?Set DATASET_ROOT.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
: "${V4_ATLAS_DIR:?Set V4_ATLAS_DIR.}"
: "${HISTORICAL_VERSION:?Set HISTORICAL_VERSION.}"
: "${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID.}"

cd "$PIPELINE_DIR"
python -u -m ti_current_repair.repeatability_v2.historical_cli extract-task \
    --dataset-root "$DATASET_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --v4-atlas-dir "$V4_ATLAS_DIR" \
    --version "$HISTORICAL_VERSION" \
    --task-index "$SLURM_ARRAY_TASK_ID"
