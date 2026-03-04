#!/usr/bin/env python3
"""
Export MATLAB .mat 'all_results' (ROI fields -> per-ROI result struct) to CSV.

Supports:
- all_results as scipy.io.matlab.mat_struct
- all_results as numpy structured dtype (np.void / structured ndarray)
- nested 1x1 wrappers common in MATLAB files

Usage:
  python mat2csv.py input.mat output.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io


def is_mat_struct(x: Any) -> bool:
    """Detect scipy mat_struct across scipy versions."""
    return hasattr(x, "_fieldnames") and isinstance(getattr(x, "_fieldnames"), list)


def unwrap_singleton(x: Any) -> Any:
    """Unwrap repeated 1-element numpy arrays."""
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    return x


def to_python_scalar(x: Any) -> Any:
    """
    Convert MATLAB-loaded scalars to Python scalars/strings.
    Keeps non-scalar arrays as compact strings.
    """
    x = unwrap_singleton(x)

    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)

    if isinstance(x, (np.str_, str)):
        return str(x)

    if isinstance(x, np.generic):
        return x.item()

    # If it's a non-scalar ndarray, keep a short representation
    if isinstance(x, np.ndarray):
        return np.array2string(x, threshold=50)

    return x


def struct_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a MATLAB struct (mat_struct or np.void structured record) to dict,
    unwrapping scalar fields.
    """
    obj = unwrap_singleton(obj)

    # scipy mat_struct
    if is_mat_struct(obj):
        out: Dict[str, Any] = {}
        for f in obj._fieldnames:
            out[f] = to_python_scalar(getattr(obj, f))
        return out

    # numpy structured record
    if isinstance(obj, np.void) and obj.dtype.names:
        out = {}
        for name in obj.dtype.names:
            out[name] = to_python_scalar(obj[name])
        return out

    # structured ndarray (possibly >1 element)
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        if obj.size == 1:
            return struct_to_dict(obj.item())
        # If multiple entries, represent as string unless you want "long format"
        return {"_structured_array": np.array2string(obj, threshold=100)}

    # fallback
    return {"_value": to_python_scalar(obj)}


def list_struct_fields(obj: Any) -> List[str]:
    """Get field names from a MATLAB struct-like object."""
    obj = unwrap_singleton(obj)
    if is_mat_struct(obj):
        return list(obj._fieldnames)
    if isinstance(obj, np.void) and obj.dtype.names:
        return list(obj.dtype.names)
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        # For 1x1 structured arrays
        if obj.size == 1:
            return list(obj.item().dtype.names)  # type: ignore[union-attr]
        return list(obj.dtype.names)
    raise TypeError(f"Object has no struct fields: {type(obj)}")


def get_struct_field(obj: Any, field: str) -> Any:
    """Access a field from mat_struct or structured record/array."""
    obj = unwrap_singleton(obj)
    if is_mat_struct(obj):
        return getattr(obj, field)
    if isinstance(obj, np.void) and obj.dtype.names:
        return obj[field]
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        if obj.size == 1:
            return obj.item()[field]
        # If multiple entries, user likely has per-subject etc. Not handled here.
        raise ValueError("Structured ndarray has multiple entries; need a 'long format' exporter.")
    raise TypeError(f"Cannot access field '{field}' on type {type(obj)}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python mat2csv.py input.mat output.csv")
        return 2

    in_path = Path(sys.argv[1]).expanduser()
    out_path = Path(sys.argv[2]).expanduser()

    if not in_path.exists():
        print(f"File not found: {in_path}")
        return 1

    mat = scipy.io.loadmat(
        str(in_path),
        struct_as_record=False,
        squeeze_me=True,
    )

    if "all_results" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"'all_results' not found. Available keys: {keys}")

    all_results = mat["all_results"]
    all_results = unwrap_singleton(all_results)

    # all_results should be a struct-like container with ROI fields
    roi_fields = list_struct_fields(all_results)

    rows: List[Dict[str, Any]] = []
    for roi in roi_fields:
        roi_blob = get_struct_field(all_results, roi)
        roi_blob = unwrap_singleton(roi_blob)

        # Each ROI field is itself a struct with E_target, stimulated_volume, etc.
        roi_dict = struct_to_dict(roi_blob)

        row = {"roi": roi}
        row.update(roi_dict)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Prefer a stable column order
    preferred = [
        "roi",
        "E_target",
        "stimulated_volume",
        "configuration",
        "pair1",
        "pair2",
        "current1",
        "current2",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
