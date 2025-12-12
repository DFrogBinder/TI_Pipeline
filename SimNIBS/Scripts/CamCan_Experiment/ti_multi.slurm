#!/bin/bash
#SBATCH --job-name=ti_multi
#SBATCH --partition=compute           # adjust for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # ~4 cores per subject ×3, plus headroom
#SBATCH --mem=36G                     # ~12 GB per subject; bump if needed
#SBATCH --time=12:00:00               # adjust as needed
#SBATCH --output=logs/ti_multi.%j.out
#SBATCH --error=logs/ti_multi.%j.err

# Cap internal threading for CHARM/SimNIBS/OpenMP-backed libs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Limit concurrent subjects
export MAX_WORKERS=3

# Optional: use node-local scratch if available
SCRATCH=${TMPDIR:-/tmp}/ti_${SLURM_JOB_ID}
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

# Activate your SimNIBS environment if needed
# source /path/to/env/bin/activate
# or: module load simnibs/4.5

cd /home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment
PYTHON=/home/boyan/SimNIBS-4.5/bin/simnibs_python

# If TI_runner_multi-core.py is updated to read MAX_WORKERS, this will cap workers at 3.
# Otherwise set max_workers directly in the script before submitting.
$PYTHON simulation/TI_runner_multi-core.py
