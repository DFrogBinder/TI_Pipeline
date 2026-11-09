#!/bin/bash
#SBATCH --job-name=segdet326_preflight
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR}"
: "${SEGDET_OUTPUT_ROOT:?Set SEGDET_OUTPUT_ROOT}"

module --force purge
module load SimNIBS/3.2.6-foss-2023a
module load MATLAB/2023b
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"
# shellcheck source=/dev/null
source "$PIPELINE_DIR/simnibs_326_repeatability/hpc/activate_compat.sh"

receipt_dir="$SEGDET_OUTPUT_ROOT/_pipeline/module_preflight"
mkdir -p "$receipt_dir"
probe_tmp="$receipt_dir/.simnibs326_cat12_probe.json.tmp-${SLURM_JOB_ID}"
receipt_tmp="$receipt_dir/.simnibs326.json.tmp-${SLURM_JOB_ID}"

python3 "$PIPELINE_DIR/simnibs_326_repeatability/cat12_compat.py" \
    probe --matlab matlab > "$probe_tmp"
mv "$probe_tmp" "$receipt_dir/simnibs326_cat12_probe.json"
python3 -m simnibs_segmentation_determinism preflight > "$receipt_tmp"
mv "$receipt_tmp" "$receipt_dir/simnibs326.json"
