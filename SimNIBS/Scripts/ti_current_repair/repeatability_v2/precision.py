"""End-to-end float64 and provenance contract for scientific metric tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


FLOAT_FORMAT = ".17g"


class PrecisionContractError(ValueError):
    """Raised when an analytical value or source violates the v2 contract."""


def require_float64(values: object, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64):
        raise PrecisionContractError(
            f"{label} must be stored and analysed as float64, observed {array.dtype}"
        )
    return array


def assert_authoritative_metric_row(row: Mapping[str, object]) -> None:
    source = str(row.get("source_msh_path", ""))
    lowered = source.lower()
    if lowered.endswith((".nii", ".nii.gz")):
        raise PrecisionContractError("NIfTI cannot be an authoritative metric source")
    if not lowered.endswith(".msh"):
        raise PrecisionContractError("Authoritative metric source must be a native .msh")
    for field in ("stored_field_dtype", "analysis_dtype"):
        if str(row.get(field, "")) != "float64":
            raise PrecisionContractError(f"{field} must be float64")


def _serialize(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise PrecisionContractError("Scientific metric tables require finite floats")
        return format(number, FLOAT_FORMAT)
    return value


def write_metric_table(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise PrecisionContractError("Cannot write an empty scientific metric table")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _serialize(row.get(key, "")) for key in fieldnames})


def read_metric_table(
    path: Path,
    *,
    float_fields: Iterable[str],
    float_precision: str,
) -> list[dict[str, object]]:
    if float_precision != "round_trip":
        raise PrecisionContractError(
            "Scientific metric tables must be read with float_precision='round_trip'"
        )
    float_names = set(float_fields)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in float_names:
                if field not in row or row[field] == "":
                    raise PrecisionContractError(f"Missing float field {field}")
                value = np.float64(row[field])
                if not np.isfinite(value):
                    raise PrecisionContractError(f"Non-finite float field {field}")
                row[field] = value
            rows.append(row)
    return rows

