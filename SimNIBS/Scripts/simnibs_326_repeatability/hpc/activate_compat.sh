#!/bin/bash
# Runtime-only compatibility aliases required by legacy SimNIBS 3.x code when
# the surrounding EasyBuild stack provides newer NumPy and NiBabel releases.

set -euo pipefail

: "${PIPELINE_DIR:?Set PIPELINE_DIR before sourcing activate_compat.sh.}"

TASK_SUFFIX="${SLURM_ARRAY_TASK_ID:-single}"
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs326_shim_${SLURM_JOB_ID:-manual}_${TASK_SUFFIX}"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/sitecustomize.py" <<'PY'
try:
    import numpy as _np
    if "bool" not in _np.__dict__:
        _np.bool = _np.bool_
except Exception:
    pass

try:
    import nibabel as _nib
    from nibabel.dataobj_images import DataobjImage as _DataobjImage

    if int(_nib.__version__.split(".", 1)[0]) >= 5:
        def _legacy_get_data(self, caching="fill"):
            if caching not in ("fill", "unchanged"):
                raise ValueError("caching value should be 'fill' or 'unchanged'")
            data = _np.asanyarray(self.dataobj)
            if caching == "fill":
                self._data_cache = data
            return data

        _DataobjImage.get_data = _legacy_get_data
except Exception:
    pass
PY
export PYTHONPATH="$SHIM_DIR:$PIPELINE_DIR:${PYTHONPATH:-}"
