"""Acceptance gates for the mandatory local four-simulation pilot."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import math
import re


PILOT_SCOPE = {
    "subject": "sub-CC110174",
    "target": "left_hippocampus",
    "remesh_simulations": 2,
    "fixed_simulations": 2,
    "outcome_rows": 12,
}
PILOT_IDENTITY_KEYS = (
    "campaign_identity_sha256",
    "operational_profile_sha256",
    "code_commit",
    "simnibs_version",
)


def _validated_pilot_identity(provenance: Mapping[str, object]) -> dict[str, str]:
    identity = {key: str(provenance.get(key, "")) for key in PILOT_IDENTITY_KEYS}
    for key in ("campaign_identity_sha256", "operational_profile_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None:
            raise ValueError(f"Pilot provenance {key} must bind a SHA-256 identity")
    if re.fullmatch(r"[0-9a-f]{40,64}", identity["code_commit"]) is None:
        raise ValueError("Pilot provenance code_commit must bind a Git commit")
    if identity["simnibs_version"] != "4.6.0":
        raise ValueError("Pilot provenance must bind SimNIBS 4.6.0")
    return identity


def _simulation_key(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row["subject"]),
        str(row["target"]),
        str(row["condition"]),
        int(row["repeat"]),
    )


def validate_local_pilot(
    *,
    manifests: Iterable[Mapping[str, object]],
    cache_manifest: Mapping[str, object],
    metric_rows: Iterable[Mapping[str, object]],
    qc_comparisons: Iterable[Mapping[str, object]],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    identity = _validated_pilot_identity(provenance)
    simulations = [dict(value) for value in manifests]
    rows = [dict(value) for value in metric_rows]
    comparisons = [dict(value) for value in qc_comparisons]
    if len(simulations) != 4:
        raise ValueError("Local pilot requires exactly four simulation manifests")
    if any(value.get("status") != "complete" for value in simulations):
        raise ValueError("Every local pilot simulation must be complete")
    subjects = {str(value.get("subject")) for value in simulations}
    targets = {str(value.get("target")) for value in simulations}
    if subjects != {"sub-CC110174"} or targets != {"left_hippocampus"}:
        raise ValueError("Local pilot must use sub-CC110174 left hippocampus")
    counts = Counter(str(value.get("condition")) for value in simulations)
    if counts != {"remesh": 2, "fixed_mesh": 2}:
        raise ValueError("Local pilot requires two remesh and two fixed simulations")
    for manifest in simulations:
        operations = [str(event.get("operation")) for event in manifest.get("events", [])]
        if manifest["condition"] == "remesh":
            if operations.count("head_mesh_rebuilt") != 1:
                raise ValueError("Each remesh pilot must rebuild one head mesh")
            electrodes = [
                event for event in manifest["events"]
                if event.get("operation") == "electrode_geometry_constructed"
            ]
            if [event.get("pair") for event in electrodes] != [1, 2] or any(
                event.get("seed") is not None for event in electrodes
            ):
                raise ValueError(
                    "Each remesh pilot must construct both electrode pairs independently and unseeded"
                )
        else:
            if manifest.get("electrode_placement_invoked") is not False or any(
                "electrode_geometry" in operation for operation in operations
            ):
                raise ValueError("Fixed pilot invoked electrode placement")
            if operations.count("existing_mesh_component_solved") != 2:
                raise ValueError("Fixed pilot must solve both existing component meshes")
    if cache_manifest.get("status") != "ready":
        raise ValueError("Pilot fixed cache is not ready")
    expected_geometry = cache_manifest.get("geometry_sha256")
    for manifest in simulations:
        if manifest["condition"] == "fixed_mesh" and manifest.get(
            "geometry_sha256"
        ) != expected_geometry:
            raise ValueError("Fixed pilot geometry hashes changed across repetitions")
    if len(rows) != 12:
        raise ValueError("Local pilot requires three outcome rows for four simulations")
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_simulation_key(row), []).append(row)
        if row.get("stored_field_dtype") != "float64" or row.get(
            "analysis_dtype"
        ) != "float64":
            raise ValueError("Local pilot analytical values must remain float64")
        finite_fraction = float(row.get("finite_volume_fraction", float("nan")))
        if not math.isfinite(finite_fraction) or finite_fraction <= 0:
            raise ValueError("Local pilot outcome has empty finite-volume support")
    if set(grouped) != {_simulation_key(value) for value in simulations}:
        raise ValueError("Local pilot metric rows do not match its simulations")
    expected_modes = {
        "ti_target",
        "tdcs_pair1_matched_target",
        "tdcs_pair1_under_anode",
    }
    for key, group in grouped.items():
        by_mode = {str(row["outcome_mode"]): row for row in group}
        if set(by_mode) != expected_modes:
            raise ValueError(f"Pilot simulation {key} lacks all three outcomes")
        if by_mode["ti_target"]["roi_provenance_sha256"] != by_mode[
            "tdcs_pair1_matched_target"
        ]["roi_provenance_sha256"]:
            raise ValueError("TI and matched-target rows use different ROI provenance")
        if by_mode["tdcs_pair1_under_anode"].get("electrode_name") != "F8":
            raise ValueError("Left-hippocampus under-anode ROI must use F8")
    if not comparisons:
        raise ValueError("Native-versus-NIfTI QC comparisons are required")
    for comparison in comparisons:
        native = float(comparison["native"])
        nifti = float(comparison["nifti"])
        tolerance = float(comparison["atol"])
        if not all(math.isfinite(value) for value in (native, nifti, tolerance)):
            raise ValueError("Native-versus-NIfTI comparison must be finite")
        if abs(native - nifti) > tolerance:
            raise ValueError("Native and derived NIfTI values exceed float32 tolerance")
    return {
        "schema_version": 2,
        "status": "ready",
        **identity,
        "scope": {**PILOT_SCOPE, "outcome_rows": len(rows)},
        "fixed_geometry_sha256": expected_geometry,
        "native_nifti_qc_comparisons": len(comparisons),
        "source_outputs_modified": False,
    }

