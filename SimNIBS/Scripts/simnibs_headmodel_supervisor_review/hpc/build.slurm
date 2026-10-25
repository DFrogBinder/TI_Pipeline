#!/bin/bash
#SBATCH --job-name=ti_headmodel_review
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

: "${SCRIPTS_ROOT:?Set SCRIPTS_ROOT to SimNIBS/Scripts.}"
: "${SIMNIBS_REVIEW_OUTPUT_ROOT:?Set the dedicated review-package output root.}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PIPELINE_DIR="$SCRIPTS_ROOT/simnibs_326_repeatability"
export PYTHONPATH="$SCRIPTS_ROOT:${PYTHONPATH:-}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"

cd "$SCRIPTS_ROOT"
echo "[INFO] Building receipt-bound supervisor review package"
echo "[INFO] SimNIBS module: SimNIBS/3.2.6-foss-2023a"
echo "[INFO] Output root: $SIMNIBS_REVIEW_OUTPUT_ROOT"

timeout --kill-after=30s 7h python -u \
    -m simnibs_headmodel_supervisor_review build \
    --confirm-output-root="$SIMNIBS_REVIEW_OUTPUT_ROOT"

python -u -m simnibs_headmodel_supervisor_review validate
