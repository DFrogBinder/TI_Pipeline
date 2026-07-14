#!/bin/bash
#SBATCH --job-name=charm_seg
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/parscratch/users/cop23bi/charm_segmentations/logs/slurm-%A_%a.out
#SBATCH --requeue

set -euo pipefail

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERROR] This launcher must run as a Slurm array." >&2
    exit 1
fi

CAMCAN_DIR="${CAMCAN_DIR:-/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment}"
SOURCE_ROOT="${TI_CHARM_SOURCE_ROOT:-/mnt/parscratch/users/cop23bi/ti_dataset}"
OUT_ROOT="${TI_CHARM_OUTPUT_ROOT:-/mnt/parscratch/users/cop23bi/charm_segmentations}"
SUBJECTS_FILE="${TI_CHARM_SUBJECTS_FILE:-${OUT_ROOT}/submission/subjects.txt}"
RUNNER_PY="${TI_CHARM_RUNNER_PY:-${CAMCAN_DIR}/charm_segmentation_batch/run_charm_segmentation.py}"
TIMEOUT_HOURS="${TI_CHARM_TIMEOUT_HOURS:-7.5}"

if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "[ERROR] Subjects file not found: $SUBJECTS_FILE" >&2
    exit 1
fi
if [ ! -f "$RUNNER_PY" ]; then
    echo "[ERROR] Runner not found: $RUNNER_PY" >&2
    exit 1
fi

SUBJECT=$(awk -v target="$((SLURM_ARRAY_TASK_ID + 1))" 'NF > 0 && $1 !~ /^#/ { count++; if (count == target) { print $1; exit } }' "$SUBJECTS_FILE")
if [ -z "$SUBJECT" ]; then
    echo "[ERROR] No subject for array index ${SLURM_ARRAY_TASK_ID}." >&2
    exit 1
fi

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1

# Preserve the NumPy compatibility shim used by the existing CamCan CHARM
# launchers on this SimNIBS 4.0.1 module. The Gmsh/session shim is unnecessary
# here because this workflow never requests a mesh or simulation session.
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_charm_seg_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHIM_DIR"
cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass
EOF
export PYTHONPATH="${SHIM_DIR}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "[INFO] Job ID:        ${SLURM_JOB_ID}"
echo "[INFO] Array index:   ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Subject:       ${SUBJECT}"
echo "[INFO] Source root:   ${SOURCE_ROOT}"
echo "[INFO] Output root:   ${OUT_ROOT}"
echo "[INFO] CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Timeout hours: ${TIMEOUT_HOURS}"
echo "[INFO] CHARM mode:    segmentation only; internal prerequisites run in temporary scratch"

python -u "$RUNNER_PY" \
    --source-root "$SOURCE_ROOT" \
    --out-root "$OUT_ROOT" \
    --subject "$SUBJECT" \
    --work-root "${SLURM_TMPDIR:-${OUT_ROOT}/.tmp}" \
    --timeout-hours "$TIMEOUT_HOURS"
