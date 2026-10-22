#!/bin/bash
#SBATCH --job-name=ti326h_module
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a
export PYTHONNOUSERSITE=1
: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${PREFLIGHT_RECEIPT:?Set PREFLIGHT_RECEIPT.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"
cd "$(dirname "$PIPELINE_DIR")"
python -u -m simnibs_326_corrected_label_repeatability module-preflight \
    --receipt "$PREFLIGHT_RECEIPT"
