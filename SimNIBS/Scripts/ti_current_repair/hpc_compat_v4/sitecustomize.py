"""Runtime compatibility aliases for the pinned SimNIBS 4.0.1 module."""

try:
    import numpy as _np

    if "bool" not in _np.__dict__:
        _np.bool = _np.bool_
except Exception:
    pass
