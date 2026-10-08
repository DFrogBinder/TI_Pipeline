#!/bin/bash
#SBATCH --job-name=ti326_control
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

: "${PIPELINE_DIR:?Set PIPELINE_DIR to simnibs_326_repeatability.}"
: "${CONTROL_ACTION:?Set CONTROL_ACTION.}"
: "${TARGET:?Set TARGET.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"

cd "$PIPELINE_DIR"
case "$CONTROL_ACTION" in
    complete-remesh)
        python -u pipeline.py complete-remesh --target "$TARGET"
        ;;
    prepare-fixed)
        : "${METRICS_CSV:?Set METRICS_CSV for prepare-fixed.}"
        python -u pipeline.py prepare-fixed \
            --target "$TARGET" \
            --metrics-csv "$METRICS_CSV"
        ;;
    complete-final)
        python -u pipeline.py complete-final --target "$TARGET"
        ;;
    *)
        echo "[ERROR] Unsupported CONTROL_ACTION: $CONTROL_ACTION" >&2
        exit 2
        ;;
esac
