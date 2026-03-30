#!/bin/bash
#SBATCH --job-name=ti_array
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8             # per-subject threads (tune if needed)
#SBATCH --mem=32G                     # per-subject RAM (tune if needed)
#SBATCH --time=08:00:00
#SBATCH --array=0-175%8              # 176 subjects, max 8 running at once
#SBATCH --output=slurm-%A_%a.out

set -o pipefail

# -------------------------
# 1) Environment
# -------------------------

DEFAULT_SCRIPT_PATH="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/HPC_scripts/my_jobArray.slurm"
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_PATH="${TI_JOB_ARRAY_SCRIPT:-$DEFAULT_SCRIPT_PATH}"
if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$(readlink -f "$SCRIPT_SOURCE" 2>/dev/null || realpath "$SCRIPT_SOURCE" 2>/dev/null || echo "$SCRIPT_SOURCE")"
fi
MESH_TIMEOUT_HOURS="${TI_MESH_TIMEOUT_HOURS:-4}"
MESH_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-1}"
MESH_RETRY_COUNT="${TI_MESH_RETRY_COUNT:-0}"
MESH_TIMEOUT_EXIT_CODE=124

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

# Avoid user site-packages overriding the cluster env
export PYTHONNOUSERSITE=1

# --- FIX: runtime shims (NumPy alias + disable SimNIBS Gmsh visualization) ---
# This avoids:
#   1) SimNIBS 4.0.1 crash with NumPy >= 1.24 (np.bool removal)
#   2) bundled gmsh crash on older glibc nodes (GLIBC_2.23 not found)
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs_shim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SHIM_DIR"

cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
# Auto-imported by Python at startup if on PYTHONPATH.

# (1) NumPy compatibility: restore np.bool if missing (NumPy >= 1.24)
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass

# (2) Disable SimNIBS auto-opening/visualizations in Gmsh (avoids bundled gmsh binary)
import builtins
_real_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)

    try:
        if name == "simnibs" or name.startswith("simnibs."):
            from simnibs import sim_struct as _sim_struct

            if not getattr(_sim_struct.SESSION, "_gmsh_disabled", False):
                _orig_init = _sim_struct.SESSION.__init__

                def _new_init(self, *a, **k):
                    _orig_init(self, *a, **k)
                    try:
                        self.open_in_gmsh = False
                    except Exception:
                        pass

                _sim_struct.SESSION.__init__ = _new_init
                _sim_struct.SESSION._gmsh_disabled = True
    except Exception:
        pass

    return mod

builtins.__import__ = _patched_import
EOF

export PYTHONPATH="${SHIM_DIR}:${PYTHONPATH:-}"
# --- END FIX ---

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
echo "[INFO] Mesh timeout: ${MESH_TIMEOUT_HOURS} hour(s) per CHARM step"
echo "[INFO] Mesh retry:   ${MESH_RETRY_COUNT}/${MESH_MAX_RETRIES}"

# -------------------------
# 3) Run TI pipeline for ONE subject
# -------------------------

LOG_DIR="/users/cop23bi/scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SUBJECT}.log"

python -u /users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/simulation/TI_runner_multi-core.py \
    --subject "$SUBJECT" \
    --mesh-timeout-hours "$MESH_TIMEOUT_HOURS" 2>&1 | tee -a "$LOG_FILE"
PY_EXIT=$?

if [ "$PY_EXIT" -eq "$MESH_TIMEOUT_EXIT_CODE" ]; then
    if [ "$MESH_RETRY_COUNT" -lt "$MESH_MAX_RETRIES" ]; then
        NEXT_RETRY_COUNT=$((MESH_RETRY_COUNT + 1))
        echo "[WARN] Mesh timeout detected for ${SUBJECT}; resubmitting array index ${SLURM_ARRAY_TASK_ID} (retry ${NEXT_RETRY_COUNT}/${MESH_MAX_RETRIES})."

        RESUBMIT_OUTPUT=$(sbatch \
            --array="${SLURM_ARRAY_TASK_ID}" \
            --export=ALL,TI_MESH_TIMEOUT_HOURS="${MESH_TIMEOUT_HOURS}",TI_MESH_MAX_RETRIES="${MESH_MAX_RETRIES}",TI_MESH_RETRY_COUNT="${NEXT_RETRY_COUNT}" \
            "$SCRIPT_PATH" 2>&1)
        RESUBMIT_EXIT=$?
        echo "$RESUBMIT_OUTPUT"

        if [ "$RESUBMIT_EXIT" -ne 0 ]; then
            echo "[ERROR] Failed to resubmit timed-out subject ${SUBJECT}."
            exit "$RESUBMIT_EXIT"
        fi

        exit 0
    fi

    echo "[ERROR] Mesh timeout retries exhausted for ${SUBJECT} after ${MESH_RETRY_COUNT} retry/retries."
fi

exit "$PY_EXIT"
