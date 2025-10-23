#!/usr/bin/env bash
#SBATCH --job-name=ti_array
#SBATCH --array=0-49%10           # adjust number of subjects & max concurrency
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail
mkdir -p logs

# ---- Load environment ----
# module load anaconda/3
# source activate simnibs_post
# OR:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate simnibs_post

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOTDIR=/home/boyan/sandbox/Jake_Data/camcan_test_run
SUBJ=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" subjects.txt)

python analyze_one.py --subject "$SUBJ" --rootDIR "$ROOTDIR"
