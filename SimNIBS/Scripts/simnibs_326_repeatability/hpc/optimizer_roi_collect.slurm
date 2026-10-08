#!/bin/bash
#SBATCH --job-name=ti326_roi_collect
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG.}"
: "${METRICS_ROOT:?Set METRICS_ROOT.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"

EXTRACTOR="$PIPELINE_DIR/../ti_current_repair/post/extract_repeatability_optimizer_roi_metrics.py"
ARCHIVE="$METRICS_ROOT/optimizer_roi_metrics_simnibs326.tar.gz"
cd "$PIPELINE_DIR/.."
python -u "$EXTRACTOR" collect \
    --config "$EXPERIMENT_CONFIG" \
    --output-root "$METRICS_ROOT" \
    --archive "$ARCHIVE"
