#!/bin/bash
#SBATCH --job-name=ti326h_control
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
: "${PIPELINE_DIR:?Set PIPELINE_DIR.}"
: "${CONTROL_ACTION:?Set CONTROL_ACTION.}"
: "${TARGET:?Set TARGET.}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"
cd "$(dirname "$PIPELINE_DIR")"
case "$CONTROL_ACTION" in
    collect-references)
        python -u -m simnibs_326_corrected_label_repeatability collect-references
        ;;
    complete-remesh)
        python -u -m simnibs_326_corrected_label_repeatability complete-remesh --target "$TARGET"
        ;;
    prepare-fixed)
        : "${METRICS_CSV:?Set METRICS_CSV.}"
        python -u -m simnibs_326_corrected_label_repeatability prepare-fixed \
            --target "$TARGET" --metrics-csv "$METRICS_CSV"
        ;;
    complete-final)
        python -u -m simnibs_326_corrected_label_repeatability complete-final --target "$TARGET"
        ;;
    *) echo "[ERROR] Unsupported action: $CONTROL_ACTION" >&2; exit 2 ;;
esac
