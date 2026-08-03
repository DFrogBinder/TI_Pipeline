#!/bin/bash
#SBATCH --job-name=ti_roi_collect
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
: "${EXTRACTOR:?Set EXTRACTOR.}"
: "${ARCHIVE_PATH:?Set ARCHIVE_PATH.}"

cd "$PIPELINE_DIR"

echo "[INFO] Stage:             collect optimizer-matched ROI metrics"
echo "[INFO] Pipeline root:     $PIPELINE_DIR"
echo "[INFO] Experiment config: $EXPERIMENT_CONFIG"
echo "[INFO] Output root:       $OUTPUT_ROOT"
echo "[INFO] Archive:           $ARCHIVE_PATH"
echo "[INFO] Source policy:     read-only"
echo "[INFO] Meshing/FEM tasks: 0"

python -u "$EXTRACTOR" collect \
    --config "$EXPERIMENT_CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --archive "$ARCHIVE_PATH"
