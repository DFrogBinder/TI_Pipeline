#!/bin/bash
#SBATCH --job-name=ti_spherical_finalize
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PIPELINE_DIR_DEFAULT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/ti_current_repair"
PIPELINE_DIR="${PIPELINE_DIR:-$PIPELINE_DIR_DEFAULT}"
: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG.}"

WORKFLOW="$PIPELINE_DIR/pipeline/spherical_fixed_nested_experiment.py"
if [ ! -f "$WORKFLOW" ]; then
    echo "[ERROR] Missing workflow: $WORKFLOW" >&2
    exit 2
fi

python3 -u "$WORKFLOW" finalize --config "$EXPERIMENT_CONFIG"
