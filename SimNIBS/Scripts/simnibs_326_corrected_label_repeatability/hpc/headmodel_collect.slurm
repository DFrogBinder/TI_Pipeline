#!/bin/bash
#SBATCH --job-name=ti326h_collect
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
: "${SIMNIBS326H_RENDER_PYTHON:?Set the rendering interpreter.}"
[ -x "$SIMNIBS326H_RENDER_PYTHON" ] || { echo "[ERROR] Renderer is not executable." >&2; exit 2; }
# shellcheck source=/dev/null
source "$PIPELINE_DIR/hpc/activate_compat.sh"
cd "$(dirname "$PIPELINE_DIR")"
python -u -m simnibs_326_corrected_label_repeatability collect-headmodels \
    --render-python "$SIMNIBS326H_RENDER_PYTHON"
