#!/bin/bash
#SBATCH --job-name=ti326_roi
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG.}"
: "${METRICS_ROOT:?Set METRICS_ROOT.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"

EXTRACTOR="$PIPELINE_DIR/../ti_current_repair/post/extract_repeatability_optimizer_roi_metrics.py"
cd "$PIPELINE_DIR/.."
python -u "$EXTRACTOR" extract-subject \
    --config "$EXPERIMENT_CONFIG" \
    --subject-index "$SLURM_ARRAY_TASK_ID" \
    --output-root "$METRICS_ROOT"
