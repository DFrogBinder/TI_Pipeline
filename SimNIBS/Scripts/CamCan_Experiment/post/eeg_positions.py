"""Readers for SimNIBS and conventional EEG-position text files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np


def _xyz(values: list[str]) -> np.ndarray | None:
    try:
        return np.asarray([float(values[0]), float(values[1]), float(values[2])], dtype=float)
    except (ValueError, IndexError):
        return None


def read_eeg_positions(path: str | Path) -> Dict[str, np.ndarray]:
    """Read SimNIBS ``Electrode,x,y,z,name`` or name/x/y/z tables."""
    source = Path(path)
    if not source.is_file():
        return {}

    text = source.read_text(encoding="utf-8-sig")
    rows = [row for row in csv.reader(text.splitlines()) if row]
    out: Dict[str, np.ndarray] = {}

    # Native SimNIBS cap rows: point type, x, y, z, displayed EEG name.
    for row in rows:
        if len(row) < 5:
            continue
        point = _xyz([cell.strip() for cell in row[1:4]])
        name = row[4].strip()
        if point is not None and name:
            out[name] = point
    if out:
        return out

    if rows:
        header = {value.strip().lower(): index for index, value in enumerate(rows[0])}
        if {"name", "x", "y", "z"}.issubset(header):
            for row in rows[1:]:
                try:
                    name = row[header["name"]].strip()
                    point = _xyz([row[header[axis]].strip() for axis in ("x", "y", "z")])
                except IndexError:
                    continue
                if name and point is not None:
                    out[name] = point
            if out:
                return out

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        point = _xyz(parts[1:4])
        if point is not None:
            out[parts[0]] = point
    return out
