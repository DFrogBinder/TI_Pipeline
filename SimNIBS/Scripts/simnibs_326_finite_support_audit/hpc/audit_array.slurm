#!/bin/bash
#SBATCH --job-name=ti326_support_audit
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%10
#SBATCH --output=slurm-%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail

module purge
module use "$HOME/modules"
module load SimNIBS/3.2.6-foss-2023a

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
: "${PIPELINE_DIR:?Set PIPELINE_DIR to simnibs_326_finite_support_audit.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the isolated audit output root.}"
: "${SLURM_ARRAY_TASK_ID:?Missing SLURM_ARRAY_TASK_ID.}"
FINITE_SUPPORT_RENDER_PYTHON="${FINITE_SUPPORT_RENDER_PYTHON:-$HOME/.conda/envs/ti-post/bin/python}"
export FINITE_SUPPORT_RENDER_PYTHON

if [ ! -x "$FINITE_SUPPORT_RENDER_PYTHON" ]; then
    echo "[ERROR] Rendering Python is missing or not executable: $FINITE_SUPPORT_RENDER_PYTHON" >&2
    exit 2
fi

PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="$OUTPUT_ROOT/.matplotlib/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

python -c 'import nibabel, numpy, simnibs; assert str(simnibs.__version__).startswith("3.2.6"), simnibs.__version__'
"$FINITE_SUPPORT_RENDER_PYTHON" -c 'import matplotlib, nibabel, numpy'

python -u -m simnibs_326_finite_support_audit audit-cell \
    --cell-index "$SLURM_ARRAY_TASK_ID" \
    --output-root "$OUTPUT_ROOT"
