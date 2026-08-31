#!/bin/bash
#SBATCH --job-name=ti_nested_analysis
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1
set -euo pipefail

PIPELINE_DIR_DEFAULT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/ti_current_repair"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
: "${METRICS_CSV:?Set METRICS_CSV.}"
: "${SELECTION_MANIFEST:?Set SELECTION_MANIFEST.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR.}"

ANALYZER="$PIPELINE_DIR/post/analyze_nested_repeatability.py"
if [ ! -f "$ANALYZER" ]; then
    echo "[ERROR] Missing analyzer: $ANALYZER" >&2
    exit 2
fi

python -u "$ANALYZER" \
    --metrics-csv "$METRICS_CSV" \
    --selection-manifest "$SELECTION_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --metric roi_median_v_per_m \
    --expected-meshes 40 \
    --expected-repeats-per-mesh 40
