#!/bin/bash
# Runtime-only compatibility aliases required by legacy SimNIBS 3.x code when
# the surrounding EasyBuild stack provides a newer NumPy.

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR before sourcing activate_compat.sh.}"

TASK_SUFFIX="${SLURM_ARRAY_TASK_ID:-single}"
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs326_shim_${SLURM_JOB_ID:-manual}_${TASK_SUFFIX}"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/sitecustomize.py" <<'PY'
try:
    import numpy as _np
    if not hasattr(_np, "bool"):
        _np.bool = _np.bool_
except Exception:
    pass
PY
export PYTHONPATH="$SHIM_DIR:$PIPELINE_DIR:${PYTHONPATH:-}"

