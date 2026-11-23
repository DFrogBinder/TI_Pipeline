"""Geometry audit and conditional replacement decision for SimNIBS 3.2.6."""

from __future__ import annotations

from collections import defaultdict
import inspect
from pathlib import Path
from typing import Iterable, Mapping


def probe_existing_mesh_solver(fem_module: object) -> dict[str, object]:
    tdcs = getattr(fem_module, "tdcs", None)
    if not callable(tdcs):
        raise ValueError("SimNIBS v3 lacks an existing-mesh fem.tdcs API")
    parameters = set(inspect.signature(tdcs).parameters)
    required = {"mesh", "cond", "currents", "electrode_surface_tags"}
    if not required.issubset(parameters):
        raise ValueError(
            "SimNIBS v3 fem.tdcs is not an existing-mesh solver API; "
            f"missing {sorted(required.difference(parameters))}"
        )
    return {
        "status": "ready",
        "api": "fem.tdcs_existing_mesh",
        "required_parameters": sorted(required),
        "electrode_placement_required": False,
    }


def audit_v3_fixed_geometry(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    materialized = [dict(row) for row in rows]
    if len(materialized) != 1600:
        raise ValueError(
            f"Expected 1,600 pair-geometry rows for 800 fixed simulations, "
            f"observed {len(materialized)}"
        )
    keys: set[tuple[str, str, int, int]] = set()
    hashes: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    subjects: set[str] = set()
    for row in materialized:
        if str(row.get("condition")) != "fixed_mesh":
            raise ValueError("V3 fixed geometry audit received a non-fixed row")
        subject = str(row["subject"])
        target = str(row["target"])
        repeat = int(row["repeat"])
        pair = int(row["pair"])
        if target not in {"left_hippocampus", "right_m1"} or not 1 <= repeat <= 40 or pair not in {1, 2}:
            raise ValueError("V3 geometry row has invalid target, repeat, or pair")
        key = subject, target, repeat, pair
        if key in keys:
            raise ValueError(f"Duplicate V3 geometry row: {key}")
        keys.add(key)
        subjects.add(subject)
        hashes[(subject, target, pair)].add(str(row["geometry_sha256"]))
    if len(subjects) != 10 or len(hashes) != 40:
        raise ValueError("V3 geometry audit does not cover 10 subjects x 2 targets x 2 pairs")
    affected = sorted(
        {
            (subject, target)
            for (subject, target, _pair), values in hashes.items()
            if len(values) != 1
        }
    )
    replacement = bool(affected)
    return {
        "schema_version": 2,
        "status": (
            "full_fixed_replacement_required"
            if replacement
            else "retain_existing_fixed_results"
        ),
        "fixed_simulations_audited": 800,
        "pair_geometry_rows": len(materialized),
        "participant_target_pair_cases": len(hashes),
        "affected_cases": len(affected),
        "affected_subject_targets": [
            {"subject": subject, "target": target}
            for subject, target in affected
        ],
        "replacement_simulations": 800 if replacement else 0,
        "replacement_component_solves": 1600 if replacement else 0,
        "replacement_condition": "fixed_mesh" if replacement else None,
        "replacement_output_policy": "isolated_v2_roots" if replacement else None,
        "original_data_policy": "preserve",
        "remesh_interpretation": "repeated_pipeline_control_not_mesh_variability",
    }


def build_v3_replacement_plan(
    *, audit: Mapping[str, object], output_root: Path
) -> dict[str, object]:
    if audit.get("status") != "full_fixed_replacement_required":
        raise ValueError("V3 replacement plan requires a failed fixed-geometry audit")
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("V3 replacement output root must be new and isolated")
    return {
        "schema_version": 2,
        "status": "planned_not_submitted",
        "simnibs_version": "3.2.6",
        "condition": "fixed_mesh",
        "subjects": 10,
        "targets": 2,
        "repeats": 40,
        "simulations": 800,
        "component_solves": 1600,
        "array": "0-799%50",
        "output_root": str(output_root.resolve(strict=False)),
        "cache_selection": "nearest_native_ti_target_roi_median_then_lowest_repeat",
        "pair_specific_caches": True,
        "electrode_placement_allowed": False,
        "existing_mesh_capability_probe_required": True,
        "original_data_policy": "preserve",
    }

