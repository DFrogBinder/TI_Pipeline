#!/bin/bash
#SBATCH --job-name=segdet_collect
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR}"
module --force purge
module load SimNIBS/4.0.1-foss-2023a
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"
python3 -m simnibs_segmentation_determinism collect
