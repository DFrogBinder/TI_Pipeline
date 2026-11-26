"""Native-mesh outcome extraction with common ROI and float64 provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .inventory import sha256_file
from .precision import assert_authoritative_metric_row, require_float64
from .precision import write_metric_table
from .roi import volume_weighted_metrics


EXTRACTION_IMPLEMENTATION = "repeatability_v2.native_mesh_extraction"
EXTRACTION_VERSION = 2


@dataclass(frozen=True)
class NativeMeshField:
    path: Path
    field_name: str
    values: np.ndarray
    tetra_centres: np.ndarray
    tetra_volumes: np.ndarray
    geometry_sha256: str
    stored_dtype: str
    tetra_tags: np.ndarray | None = None


def _provenance_sha256(provenance: Mapping[str, Any]) -> str:
    payload = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scalar_values(field: NativeMeshField) -> np.ndarray:
    values = require_float64(field.values, f"{field.field_name} field")
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 3:
        return np.sqrt(
            np.sum(values * values, axis=1, dtype=np.float64), dtype=np.float64
        )
    raise ValueError(f"Unexpected field shape for {field.field_name}: {values.shape}")


def _row(
    *,
    field: NativeMeshField,
    scalar_values: np.ndarray,
    outcome_mode: str,
    roi_mask: np.ndarray,
    roi_provenance: Mapping[str, Any],
    common: Mapping[str, object],
    source_msh_sha256: str | None = None,
) -> dict[str, object]:
    metrics = volume_weighted_metrics(
        values=scalar_values,
        volumes=require_float64(field.tetra_volumes, "tetrahedron volumes"),
        roi_mask=roi_mask,
    )
    row: dict[str, object] = {
        **common,
        "schema_version": 2,
        "outcome_mode": outcome_mode,
        "source_msh_path": str(field.path.resolve()),
        "source_msh_sha256": source_msh_sha256 or sha256_file(field.path),
        "source_field_name": field.field_name,
        "stored_field_dtype": field.stored_dtype,
        "analysis_dtype": "float64",
        "geometry_sha256": field.geometry_sha256,
        "roi_provenance_sha256": _provenance_sha256(roi_provenance),
        "roi_provenance_json": json.dumps(
            roi_provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "roi_mean_v_per_m": metrics["mean"],
        "roi_median_v_per_m": metrics["median"],
        "roi_p95_v_per_m": metrics["p95"],
        "achieved_roi_volume_mm3": metrics["achieved_roi_volume_mm3"],
        "finite_volume_mm3": metrics["finite_volume_mm3"],
        "finite_volume_fraction": metrics["finite_volume_fraction"],
        "finite_tetrahedra": metrics["finite_tetrahedra"],
        "nonfinite_tetrahedra": metrics["nonfinite_tetrahedra"],
        "extraction_implementation": EXTRACTION_IMPLEMENTATION,
        "extraction_version": EXTRACTION_VERSION,
    }
    row["electrode_name"] = str(roi_provenance.get("electrode_name", ""))
    assert_authoritative_metric_row(row)
    return row


def extract_three_outcomes(
    *,
    ti_field: NativeMeshField,
    pair1_field: NativeMeshField,
    matched_target_mask: object,
    pair1_matched_target_mask: object | None = None,
    matched_roi_provenance: Mapping[str, Any],
    under_anode_mask: object,
    under_anode_provenance: Mapping[str, Any],
    common: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if (
        pair1_matched_target_mask is None
        and ti_field.geometry_sha256 != pair1_field.geometry_sha256
    ):
        raise ValueError("TI and pair-1 native meshes have different geometry")
    ti_values = _scalar_values(ti_field)
    pair_values = _scalar_values(pair1_field)
    if pair1_matched_target_mask is None and ti_values.shape != pair_values.shape:
        raise ValueError("TI and pair-1 fields have different tetrahedral support")
    matched = np.asarray(matched_target_mask, dtype=bool)
    pair_matched = (
        matched
        if pair1_matched_target_mask is None
        else np.asarray(pair1_matched_target_mask, dtype=bool)
    )
    under = np.asarray(under_anode_mask, dtype=bool)
    if (
        matched.shape != ti_values.shape
        or pair_matched.shape != pair_values.shape
        or under.shape != pair_values.shape
    ):
        raise ValueError("ROI mask and native field shapes differ")
    ti_msh_sha256 = sha256_file(ti_field.path)
    pair1_msh_sha256 = sha256_file(pair1_field.path)
    return (
        _row(
            field=ti_field,
            scalar_values=ti_values,
            outcome_mode="ti_target",
            roi_mask=matched,
            roi_provenance=matched_roi_provenance,
            common=common,
            source_msh_sha256=ti_msh_sha256,
        ),
        _row(
            field=pair1_field,
            scalar_values=pair_values,
            outcome_mode="tdcs_pair1_matched_target",
            roi_mask=pair_matched,
            roi_provenance=matched_roi_provenance,
            common=common,
            source_msh_sha256=pair1_msh_sha256,
        ),
        _row(
            field=pair1_field,
            scalar_values=pair_values,
            outcome_mode="tdcs_pair1_under_anode",
            roi_mask=under,
            roi_provenance=under_anode_provenance,
            common=common,
            source_msh_sha256=pair1_msh_sha256,
        ),
    )


TABLE_BY_OUTCOME = {
    "ti_target": "ti_native_mesh_metrics.csv",
    "tdcs_pair1_matched_target": "tdcs_pair1_matched_target_metrics.csv",
    "tdcs_pair1_under_anode": "tdcs_pair1_under_anode_metrics.csv",
}


def write_extraction_tables(
    output_root: Path,
    rows: object,
    geometry_rows: object,
) -> dict[str, Path]:
    """Write the three outcome tables and a separate geometry audit."""

    materialized = [dict(row) for row in rows]
    geometry = [dict(row) for row in geometry_rows]
    unknown = sorted(
        {str(row.get("outcome_mode")) for row in materialized}.difference(
            TABLE_BY_OUTCOME
        )
    )
    if unknown:
        raise ValueError(f"Unknown v2 outcome modes: {unknown}")
    written: dict[str, Path] = {}
    for outcome, filename in TABLE_BY_OUTCOME.items():
        selected = [row for row in materialized if row.get("outcome_mode") == outcome]
        if not selected:
            raise ValueError(f"No rows supplied for {outcome}")
        destination = output_root / filename
        write_metric_table(destination, selected)
        written[filename] = destination
    geometry_path = output_root / "geometry_audit.csv"
    write_metric_table(geometry_path, geometry)
    written[geometry_path.name] = geometry_path
    return written
