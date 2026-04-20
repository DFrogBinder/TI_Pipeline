#!/bin/bash
#SBATCH --job-name=ti_reuse_mesh_batch
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

module purge
module use "$HOME/modules"
module load SimNIBS/4.0.1-foss-2023a

export PYTHONNOUSERSITE=1

# Runtime shims:
# 1) restore np.bool for NumPy >= 1.24
# 2) disable SimNIBS auto-opening Gmsh windows
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs_shim_${SLURM_JOB_ID}"
mkdir -p "$SHIM_DIR"

cat > "${SHIM_DIR}/sitecustomize.py" << 'EOF'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

set -euo pipefail

PIPELINE_DIR_DEFAULT="/users/cop23bi/Repos/TI_Pipeline/SimNIBS/Scripts/mesh_repeat_analysis"
SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ -d "${PIPELINE_DIR_DEFAULT}" ]; then
    SCRIPT_DIR="${PIPELINE_DIR:-${PIPELINE_DIR_DEFAULT}}"
fi
cd "$SCRIPT_DIR"

DATASET_ROOT="${DATASET_ROOT:-/mnt/parscratch/users/cop23bi/repeatability-ti-dataset}"
REPEAT_COUNT="${REPEAT_COUNT:-40}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/batch_reuse_mesh_${SLURM_JOB_ID}.log"

echo "[INFO] Pipeline root: ${SCRIPT_DIR}"
echo "[INFO] Dataset root:  ${DATASET_ROOT}"
echo "[INFO] Repeats:       ${REPEAT_COUNT}"
echo "[INFO] CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "[INFO] Log file:      ${LOG_FILE}"

python -u "${SCRIPT_DIR}/simulation_runners/TI_runner_batch_reuse_mesh.py" \
    --rootdir "${DATASET_ROOT}" \
    --repeats "${REPEAT_COUNT}" 2>&1 | tee "${LOG_FILE}"
