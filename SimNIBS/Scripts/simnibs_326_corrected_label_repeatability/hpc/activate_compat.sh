#!/bin/bash
# Runtime-only aliases required by the SimNIBS 3.2.6 Python stack.

: "${PIPELINE_DIR:?Set PIPELINE_DIR before sourcing activate_compat.sh.}"

TASK_SUFFIX="${SLURM_ARRAY_TASK_ID:-single}"
SHIM_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_simnibs326h_shim_${SLURM_JOB_ID:-manual}_${TASK_SUFFIX}"
mkdir -p "$SHIM_DIR"
cat > "$SHIM_DIR/sitecustomize.py" <<'PY'
try:
    import numpy as _np
    for _name, _value in {"bool": bool, "int": int, "float": float}.items():
        if _name not in _np.__dict__:
            setattr(_np, _name, _value)
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
PIPELINE_PARENT="$(dirname "$PIPELINE_DIR")"
export PYTHONPATH="$SHIM_DIR:$PIPELINE_PARENT:${PYTHONPATH:-}"
