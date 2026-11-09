#!/bin/bash
#SBATCH --job-name=segdet401_preflight
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR}"
: "${SEGDET_OUTPUT_ROOT:?Set SEGDET_OUTPUT_ROOT}"

module --force purge
module load SimNIBS/4.0.1-foss-2023a
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"
command -v charm >/dev/null
charm --help >/dev/null

receipt_dir="$SEGDET_OUTPUT_ROOT/_pipeline/module_preflight"
mkdir -p "$receipt_dir"
receipt_tmp="$receipt_dir/.simnibs401.json.tmp-${SLURM_JOB_ID}"
python3 -m simnibs_segmentation_determinism preflight > "$receipt_tmp"
mv "$receipt_tmp" "$receipt_dir/simnibs401.json"
