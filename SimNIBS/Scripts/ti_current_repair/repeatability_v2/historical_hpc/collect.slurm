#!/bin/bash
#SBATCH --job-name=hist_native_collect
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a
export PYTHONNOUSERSITE=1

: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT.}"
: "${HISTORICAL_VERSION:?Set HISTORICAL_VERSION.}"

cd "$PIPELINE_DIR"
python -u -m ti_current_repair.repeatability_v2.historical_cli collect-version \
    --output-root "$OUTPUT_ROOT" \
    --version "$HISTORICAL_VERSION"
