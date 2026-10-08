#!/bin/bash
#SBATCH --job-name=ti326_module_check
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge
module use "$HOME/modules"
SIMNIBS326_MATLAB_MODULE="${SIMNIBS326_MATLAB_MODULE:-MATLAB/2023b}"
module load "${SIMNIBS326_MATLAB_MODULE}"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1

: "${PIPELINE_DIR:?Set PIPELINE_DIR to simnibs_326_repeatability.}"
: "${PREFLIGHT_RECEIPT:?Set PREFLIGHT_RECEIPT.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"

cd "$PIPELINE_DIR"
echo "[INFO] MATLAB dependency: ${SIMNIBS326_MATLAB_MODULE}"
echo "[INFO] SimNIBS module: SimNIBS/3.2.6-foss-2023a"
python -u pipeline.py module-preflight --receipt "$PREFLIGHT_RECEIPT"
