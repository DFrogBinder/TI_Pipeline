#!/bin/bash
#SBATCH --job-name=ti_array
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8             # per-subject threads (tune if needed)
#SBATCH --mem=32G                     # per-subject RAM (tune if needed)
#SBATCH --time=08:00:00
#SBATCH --array=0-175%16              # 176 subjects, max 16 running at once
#SBATCH --output=slurm-%A_%a.out

# -------------------------
# 1) Environment
# -------------------------

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

# Avoid user site-packages overriding the cluster env
export PYTHONNOUSERSITE=1

# If you still need the numpy bool shim, add it here.
# Example:
# mkdir -p fix_numpy_bool
# cat > fix_numpy_bool/sitecustomize.py << 'EOF'
# import numpy as _np
# if not hasattr(_np, "bool"):
#     _np.bool = _np.bool_
# EOF
# export PYTHONPATH="$PWD/fix_numpy_bool"

# Make SimNIBS/OpenMP respect the Slurm allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------
# 2) Select subject from list
# -------------------------

cd /users/cop23bi/scripts

SUBJECTS_FILE="subjects.txt"

# SLURM_ARRAY_TASK_ID is 0-based; sed is 1-based
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")

if [ -z "$SUBJECT" ]; then
    echo "[ERROR] No subject found for array index ${SLURM_ARRAY_TASK_ID} in ${SUBJECTS_FILE}."
    exit 1
fi

echo "[INFO] Job ID:      ${SLURM_JOB_ID}"
echo "[INFO] Array index: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Subject ID:  ${SUBJECT}"
echo "[INFO] CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Mem per task:${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_TASK}} MB"

# -------------------------
# 3) Run TI pipeline for ONE subject
# -------------------------

python /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/TI_runner_multi-core.py \
    --subject "$SUBJECT"
