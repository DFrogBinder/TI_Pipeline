#!/bin/bash
#SBATCH --job-name=ti_resolution_repeat_array   # any short name, e.g., "ti_res_repeat"
#SBATCH --partition=sheffield                   # e.g., "sheffield"
#SBATCH --cpus-per-task=8                       # e.g., 8
#SBATCH --mem=32G                               # e.g., 32G
#SBATCH --time=08:00:00                         # e.g., 08:00:00
#SBATCH --array=0-29%8                          # e.g., 0-(REPEAT_COUNT-1)%MAX_CONCURRENT
#SBATCH --output=slurm-%A_%a.out                # e.g., slurm-%A_%a.out

# -------------------------
# Control Panel (edit these)
# -------------------------

SUBJECT_ID="sub-CC110056"      # e.g., "sub-CC110056"
REPEAT_COUNT=30                 # e.g., 30 (must match #SBATCH --array range)

# Optional meshing controls (leave empty to disable)
MESH_TARGET_SIZE_MM=0.8         # e.g., 0.8 (mm). Leave empty to disable.
MESH_SCOPE="brain"              # supported: "brain" or "all"

# Optional overrides
PIPELINE_DIR_DEFAULT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis"  # e.g., "/users/<you>/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis"
LOG_DIR_OVERRIDE=""             # e.g., "/path/to/logs" (empty = use <script>/logs)

# -------------------------
# 1) Environment
# -------------------------

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
# 2) Select subject + repeat index from array ID
# -------------------------

set -euo pipefail

# Prefer an explicit pipeline path; fall back to submit dir or script dir.
SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [ -d "${PIPELINE_DIR_DEFAULT}" ]; then
    SCRIPT_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_DEFAULT}}"
fi
cd "$SCRIPT_DIR"

if [ -z "${SUBJECT_ID}" ]; then
    echo "[ERROR] SUBJECT_ID is required for single-subject repeats."
    exit 1
fi

# Map array index -> repeat index
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID % REPEAT_COUNT + 1))

TOTAL_TASKS=$((REPEAT_COUNT))

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
    echo "[INFO] Array index ${SLURM_ARRAY_TASK_ID} >= total tasks ${TOTAL_TASKS}; exiting."
    exit 0
fi
SUBJECT="${SUBJECT_ID}"

echo "[INFO] Job ID:      ${SLURM_JOB_ID}"
echo "[INFO] Array index: ${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Subject ID:  ${SUBJECT}"
echo "[INFO] Repeat idx:  ${REPEAT_INDEX}/${REPEAT_COUNT}"
echo "[INFO] CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Mem per task:${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_TASK}} MB"

# Optional meshing controls
EXTRA_ARGS=()
if [ -n "${MESH_TARGET_SIZE_MM:-}" ]; then
    EXTRA_ARGS+=(--mesh-target-size-mm "${MESH_TARGET_SIZE_MM}")
fi
if [ -n "${MESH_SCOPE:-}" ]; then
    EXTRA_ARGS+=(--mesh-scope "${MESH_SCOPE}")
fi

# -------------------------
# 3) Run TI pipeline for ONE subject + ONE repeat
# -------------------------

LOG_DIR="${LOG_DIR_OVERRIDE:-${LOG_DIR:-${SCRIPT_DIR}/logs}}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SUBJECT}_repeat_${REPEAT_INDEX}_res.log"

python -u "${SCRIPT_DIR}/TI_runner_multi-core_resolution-repeat.py" \
    --subject "$SUBJECT" \
    --repeat-index "${REPEAT_INDEX}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
