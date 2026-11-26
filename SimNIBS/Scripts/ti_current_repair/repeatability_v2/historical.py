"""Read-only native extraction from the consolidated v3.2.6/v4.0.1 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from .config import CANONICAL_SUBJECTS, CONDITIONS, TARGETS
from .extraction import TABLE_BY_OUTCOME, extract_three_outcomes, write_extraction_tables
from .inventory import sha256_file
from .mesh_io import load_native_element_field, load_tagged_surface
from .native_pipeline import (
    CORTICAL_GM_TAG,
    TARGET_LABEL_IDS,
    map_ti_target_mask_to_pair1,
    parcel_mask_from_atlas,
    under_anode_name,
)
from .precision import read_metric_table, write_metric_table
from .roi import build_parcel_clipped_roi, build_under_anode_roi


HISTORICAL_EXTRACTION_VERSION = 1
PAIR1_ANODE_SURFACE_TAG = 2101
REPEAT_COUNT = 40
SIMNIBS326_HARMONIZED_RUNTIME_KEY = "simnibs_326_harmonized_runtime"
SIMNIBS326_HARMONIZED_RUNTIME = {
    "simnibs_version": "3.2.6",
    "simnibs_module": "SimNIBS/3.2.6-foss-2023a",
    "backend": "simnibs-3.2.6-harmonized-corrected-label-v2",
}
VERSION_SPECS = {
    "simnibs326": "3.2.6",
    "simnibs4": "4.0.1",
}
ANALYSIS_VERSION_LABELS = {
    "simnibs326": "simnibs326",
    "simnibs4": "simnibs401",
}
METRIC_FLOAT_FIELDS = (
    "roi_mean_v_per_m",
    "roi_median_v_per_m",
    "roi_p95_v_per_m",
    "achieved_roi_volume_mm3",
    "finite_volume_mm3",
    "finite_volume_fraction",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe {label}: {path}")
    return path


@dataclass(frozen=True)
class HistoricalTask:
    version: str
    target: str
    subject: str


@dataclass(frozen=True)
class HistoricalSourceRun:
    version: str
    simnibs_version: str
    target: str
    subject: str
    condition: str
    repeat: int
    repeat_tag: str
    ti_msh: Path
    pair1_msh: Path
    pair2_msh: Path
    manifest_path: Path
    atlas_path: Path
    anode_name: str


def task_matrix(
    version: str, *, subjects: tuple[str, ...] = CANONICAL_SUBJECTS
) -> tuple[HistoricalTask, ...]:
    if version not in VERSION_SPECS:
        raise ValueError(f"Unsupported historical version: {version}")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Historical task subjects contain duplicates")
    return tuple(
        HistoricalTask(version=version, target=target, subject=subject)
        for target in TARGETS
        for subject in subjects
    )


def validate_consolidation_receipt(
    dataset_root: Path,
    *,
    subject_count: int = len(CANONICAL_SUBJECTS),
    repeat_count: int = REPEAT_COUNT,
) -> dict[str, object]:
    root = dataset_root.expanduser().resolve()
    if subject_count == len(CANONICAL_SUBJECTS) and repeat_count == REPEAT_COUNT:
        from repeatability_dataset_consolidation.settings import production_config
        from repeatability_dataset_consolidation.workflow import (
            validate_dataset_completion,
        )

        production = production_config()
        if root == production.output_root.resolve():
            # This validates the owner, plan, checksum-verification, inventory,
            # canonical-config, and completion bindings without rehashing all
            # data bodies.
            return validate_dataset_completion(production)
    receipt_path = root / "_consolidation/complete.json"
    payload = _read_json(receipt_path, "consolidation completion receipt")
    expected_cells = subject_count * len(VERSION_SPECS) * len(TARGETS) * len(CONDITIONS)
    expected_scope = {
        "subjects": subject_count,
        "cells": expected_cells,
        "simulations": expected_cells * repeat_count,
        "v3_headmodels": subject_count,
        "v4_scaffolds": subject_count,
    }
    if (
        payload.get("status") != "complete"
        or payload.get("workflow") != "repeatability_dataset_consolidation"
        or payload.get("output_root") != str(root)
        or payload.get("scope") != expected_scope
    ):
        raise ValueError(
            "Consolidation completion receipt does not bind the exact requested scope"
        )
    return payload


def _atlas_path(
    *,
    dataset_root: Path,
    version: str,
    subject: str,
    v4_atlas_dir: Path,
) -> Path:
    if version == "simnibs326":
        path = (
            dataset_root
            / "provenance/headmodels/simnibs326"
            / subject
            / "harmonized_label.nii.gz"
        )
    else:
        path = v4_atlas_dir / f"{subject}.nii.gz"
        if not path.is_file():
            alternate = v4_atlas_dir / f"{subject}.nii"
            if alternate.is_file():
                path = alternate
    return _safe_file(path, f"{version} subject atlas")


def _manifest_anode(
    manifest: dict[str, object],
    *,
    version: str,
    target: str,
    subject: str,
    condition: str,
    repeat: int,
    manifest_path: Path,
) -> str:
    expected_repeat_tag = f"repeat_{repeat:03d}"
    expected = {
        "subject": subject,
        "condition": condition,
        "repeat_index": repeat,
        "repeat_tag": expected_repeat_tag,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Historical task manifest {key} differs at {manifest_path}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    if version == "simnibs326":
        runtime = manifest.get(SIMNIBS326_HARMONIZED_RUNTIME_KEY)
        if not isinstance(runtime, dict) or any(
            runtime.get(key) != value
            for key, value in SIMNIBS326_HARMONIZED_RUNTIME.items()
        ):
            raise ValueError(
                "Historical v3 task lacks exact harmonized runtime provenance: "
                f"{manifest_path}"
            )
    stimulation = manifest.get("stimulation")
    pair1 = stimulation.get("pair1") if isinstance(stimulation, dict) else None
    if not isinstance(pair1, dict):
        raise ValueError(
            f"Historical task lacks recorded pair-1 stimulation provenance: {manifest_path}"
        )
    anode = str(pair1.get("anode", "")).strip()
    expected_anode = under_anode_name(target)
    if anode.casefold() != expected_anode.casefold():
        raise ValueError(
            f"Historical pair-1 anode is {anode or '<missing>'}, expected "
            f"{expected_anode}: {manifest_path}"
        )
    cathode = str(pair1.get("cathode", "")).strip()
    try:
        current = float(pair1.get("current_a"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Historical pair-1 current is invalid: {manifest_path}") from error
    if (
        not cathode
        or cathode.casefold() == anode.casefold()
        or not np.isfinite(current)
        or current <= 0
    ):
        raise ValueError(f"Historical pair-1 stimulation is invalid: {manifest_path}")
    return expected_anode


def discover_task_runs(
    *,
    dataset_root: Path,
    version: str,
    target: str,
    subject: str,
    v4_atlas_dir: Path,
    repeat_count: int = REPEAT_COUNT,
) -> tuple[HistoricalSourceRun, ...]:
    root = dataset_root.expanduser().resolve()
    if version not in VERSION_SPECS:
        raise ValueError(f"Unsupported historical version: {version}")
    if target not in TARGETS:
        raise ValueError(f"Unsupported historical target: {target}")
    atlas = _atlas_path(
        dataset_root=root,
        version=version,
        subject=subject,
        v4_atlas_dir=v4_atlas_dir.expanduser().resolve(),
    )
    subject_root = root / "data" / version / target / f"{subject}_repeatability"
    runs: list[HistoricalSourceRun] = []
    for condition in CONDITIONS:
        for repeat in range(1, repeat_count + 1):
            repeat_tag = f"repeat_{repeat:03d}"
            repeat_subject_root = (
                subject_root
                / condition
                / "repeats"
                / repeat_tag
                / subject
            )
            workspace = repeat_subject_root / "anat/SimNIBS"
            output = workspace / "Output" / subject
            ti = _safe_file(output / "TI.msh", "native TI mesh")
            pair1 = _safe_file(
                output / f"{subject}_TDCS_1_scalar.msh", "native pair-1 mesh"
            )
            pair2 = _safe_file(
                output / f"{subject}_TDCS_2_scalar.msh", "native pair-2 mesh"
            )
            manifest_path = repeat_subject_root / "task_manifest.json"
            manifest = _read_json(manifest_path, "historical task manifest")
            anode = _manifest_anode(
                manifest,
                version=version,
                target=target,
                subject=subject,
                condition=condition,
                repeat=repeat,
                manifest_path=manifest_path,
            )
            runs.append(
                HistoricalSourceRun(
                    version=version,
                    simnibs_version=VERSION_SPECS[version],
                    target=target,
                    subject=subject,
                    condition=condition,
                    repeat=repeat,
                    repeat_tag=repeat_tag,
                    ti_msh=ti,
                    pair1_msh=pair1,
                    pair2_msh=pair2,
                    manifest_path=manifest_path,
                    atlas_path=atlas,
                    anode_name=anode,
                )
            )
    return tuple(runs)


def _validate_output_root(
    output_root: Path, *, dataset_root: Path | None = None
) -> None:
    if output_root.is_symlink():
        raise ValueError(f"Historical analysis output root cannot be a symlink: {output_root}")
    if not output_root.exists():
        return
    owner_path = output_root / "_historical_component_analysis/owner.json"
    owner = _read_json(owner_path, "historical output owner receipt")
    if (
        owner.get("status") != "owned"
        or owner.get("workflow") != "historical_component_analysis"
        or owner.get("output_root") != str(output_root.resolve())
        or owner.get("source_outputs_modified") is not False
    ):
        raise ValueError(f"Historical output root ownership differs: {output_root}")
    if dataset_root is not None:
        completion = dataset_root / "_consolidation/complete.json"
        if (
            owner.get("dataset_root") != str(dataset_root.resolve())
            or owner.get("dataset_completion_receipt_sha256")
            != sha256_file(completion)
        ):
            raise ValueError(
                f"Historical output root dataset binding differs: {output_root}"
            )


def initialize_output(*, dataset_root: Path, output_root: Path) -> dict[str, object]:
    validate_consolidation_receipt(dataset_root)
    _validate_output_root(output_root, dataset_root=dataset_root)
    owner_path = output_root / "_historical_component_analysis/owner.json"
    if owner_path.is_file():
        return _read_json(owner_path, "historical output owner receipt")
    output_root.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "owned",
        "workflow": "historical_component_analysis",
        "created_utc": _utc_now(),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_completion_receipt_sha256": sha256_file(
            dataset_root / "_consolidation/complete.json"
        ),
        "output_root": str(output_root.resolve()),
        "source_outputs_modified": False,
    }
    _write_json(owner_path, receipt)
    return receipt


def _probe_run(run: HistoricalSourceRun) -> None:
    ti = load_native_element_field(run.ti_msh, "TImax")
    pair1 = load_native_element_field(run.pair1_msh, "E")
    if ti.stored_dtype != "float64" or pair1.stored_dtype != "float64":
        raise ValueError("Historical native fields are not stored as float64")
    load_tagged_surface(run.pair1_msh, PAIR1_ANODE_SURFACE_TAG)
    parcel = parcel_mask_from_atlas(
        ti.tetra_centres,
        run.atlas_path,
        label_ids=TARGET_LABEL_IDS[run.target],
    )
    if not np.any(parcel):
        raise ValueError(
            f"Historical atlas has no target support on the native mesh: {run.atlas_path}"
        )


def preflight_historical(
    *,
    dataset_root: Path,
    output_root: Path,
    v4_atlas_dir: Path,
    subjects: tuple[str, ...] = CANONICAL_SUBJECTS,
    repeat_count: int = REPEAT_COUNT,
    probe_native: bool = True,
) -> dict[str, object]:
    receipt = validate_consolidation_receipt(
        dataset_root, subject_count=len(subjects), repeat_count=repeat_count
    )
    _validate_output_root(output_root, dataset_root=dataset_root)
    task_count = 0
    run_count = 0
    probes: dict[str, str] = {}
    for version in VERSION_SPECS:
        first_by_target: dict[str, HistoricalSourceRun] = {}
        for task in task_matrix(version, subjects=subjects):
            runs = discover_task_runs(
                dataset_root=dataset_root,
                version=version,
                target=task.target,
                subject=task.subject,
                v4_atlas_dir=v4_atlas_dir,
                repeat_count=repeat_count,
            )
            first_by_target.setdefault(task.target, runs[0])
            task_count += 1
            run_count += len(runs)
        if probe_native:
            for target, first_run in first_by_target.items():
                _probe_run(first_run)
                probes[f"{version}/{target}"] = str(first_run.ti_msh)
    expected_runs = (
        len(subjects)
        * len(VERSION_SPECS)
        * len(TARGETS)
        * len(CONDITIONS)
        * repeat_count
    )
    if run_count != expected_runs:
        raise AssertionError("Historical preflight run count changed unexpectedly")
    return {
        "schema_version": 1,
        "status": "ready",
        "mode": "read_only_historical_native_component_extraction",
        "dataset_root": str(dataset_root.resolve()),
        "dataset_completion_receipt": str(
            dataset_root.resolve() / "_consolidation/complete.json"
        ),
        "dataset_completion_receipt_sha256": sha256_file(
            dataset_root / "_consolidation/complete.json"
        ),
        "consolidation_scope": receipt["scope"],
        "output_root": str(output_root.resolve()),
        "versions": VERSION_SPECS,
        "subject_tasks": task_count,
        "existing_simulations": run_count,
        "expected_metric_rows": run_count * 3,
        "native_probe_paths": probes,
        "pair1_anode_surface_tag": PAIR1_ANODE_SURFACE_TAG,
        "meshing_tasks": 0,
        "fem_tasks": 0,
        "source_outputs_modified": False,
    }


def _extract_run(
    run: HistoricalSourceRun,
    *,
    atlas_sha256: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    ti = load_native_element_field(run.ti_msh, "TImax")
    pair1 = load_native_element_field(run.pair1_msh, "E")
    pair2 = load_native_element_field(run.pair2_msh, "E")
    parcel = parcel_mask_from_atlas(
        ti.tetra_centres,
        run.atlas_path,
        label_ids=TARGET_LABEL_IDS[run.target],
    )
    target_roi = build_parcel_clipped_roi(
        centres=ti.tetra_centres,
        volumes=ti.tetra_volumes,
        parcel_mask=parcel,
        target=run.target,
    )
    pair1_target = map_ti_target_mask_to_pair1(
        ti=ti, pair1=pair1, ti_target_mask=target_roi.mask
    )
    surface_nodes, surface_triangles = load_tagged_surface(
        run.pair1_msh, PAIR1_ANODE_SURFACE_TAG
    )
    if pair1.tetra_tags is None:
        raise ValueError(f"Pair-1 mesh lacks tetrahedral tags: {run.pair1_msh}")
    under = build_under_anode_roi(
        electrode_name=run.anode_name,
        surface_nodes=surface_nodes,
        surface_triangles=surface_triangles,
        tetra_centres=pair1.tetra_centres,
        tetra_volumes=pair1.tetra_volumes,
        cortical_gm_mask=np.asarray(pair1.tetra_tags) == CORTICAL_GM_TAG,
        target_volume_mm3=100.0,
    )
    manifest_sha256 = sha256_file(run.manifest_path)
    matched_provenance = {
        **target_roi.provenance,
        "atlas_path": str(run.atlas_path.resolve()),
        "atlas_sha256": atlas_sha256,
        "atlas_label_ids": list(TARGET_LABEL_IDS[run.target]),
        "field_sampling": "native_tetrahedron_centres",
    }
    under_provenance = {
        **under.provenance,
        "electrode_channel": 1,
        "electrode_surface_tag": PAIR1_ANODE_SURFACE_TAG,
        "surface_source_msh": str(run.pair1_msh.resolve()),
        "task_manifest": str(run.manifest_path.resolve()),
        "task_manifest_sha256": manifest_sha256,
    }
    common = {
        "subject": run.subject,
        "version": ANALYSIS_VERSION_LABELS[run.version],
        "target": run.target,
        "condition": run.condition,
        "repeat": run.repeat,
        "repeat_tag": run.repeat_tag,
        "simnibs_version": run.simnibs_version,
        "preliminary": True,
    }
    rows = extract_three_outcomes(
        ti_field=ti,
        pair1_field=pair1,
        matched_target_mask=target_roi.mask,
        pair1_matched_target_mask=pair1_target,
        matched_roi_provenance=matched_provenance,
        under_anode_mask=under.mask,
        under_anode_provenance=under_provenance,
        common=common,
    )
    geometry_rows = (
        {
            **common,
            "pair": 1,
            "geometry_sha256": pair1.geometry_sha256,
            "source_msh_path": str(run.pair1_msh.resolve()),
            "source_msh_sha256": rows[1]["source_msh_sha256"],
            "task_manifest": str(run.manifest_path.resolve()),
            "task_manifest_sha256": manifest_sha256,
            "electrode_surface_tag": PAIR1_ANODE_SURFACE_TAG,
            "electrode_name": run.anode_name,
            "historical_electrode_placement_invocation": "not_recorded_per_task",
        },
        {
            **common,
            "pair": 2,
            "geometry_sha256": pair2.geometry_sha256,
            "source_msh_path": str(run.pair2_msh.resolve()),
            "source_msh_sha256": sha256_file(run.pair2_msh),
            "task_manifest": str(run.manifest_path.resolve()),
            "task_manifest_sha256": manifest_sha256,
            "electrode_surface_tag": "",
            "electrode_name": "",
            "historical_electrode_placement_invocation": "not_recorded_per_task",
        },
    )
    return tuple(rows), geometry_rows


def _task_root(output_root: Path, task: HistoricalTask) -> Path:
    return output_root / "tasks" / task.version / task.target / task.subject


def _table_record(path: Path, *, root: Path, rows: int) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "rows": rows,
        "sha256": sha256_file(path),
    }


def _bound_table(
    *, root: Path, record: object, expected_rows: int, label: str
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Missing {label} table receipt")
    raw = record.get("path")
    if not isinstance(raw, str):
        raise ValueError(f"Invalid {label} table path")
    relative = Path(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe {label} table path")
    path = root / relative
    resolved_root = root.resolve()
    resolved = path.resolve(strict=False)
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"Escaping {label} table path")
    _safe_file(path, label)
    if record.get("rows") != expected_rows or record.get("sha256") != sha256_file(path):
        raise ValueError(f"Stale {label} table receipt")
    return path


def validate_task_receipt(
    task_root: Path,
    *,
    expected_runs: int = len(CONDITIONS) * REPEAT_COUNT,
    expected_task: HistoricalTask | None = None,
) -> dict[str, object]:
    receipt_path = task_root / "task_receipt.json"
    receipt = _read_json(receipt_path, "historical task receipt")
    if (
        receipt.get("status") != "complete"
        or receipt.get("workflow") != "historical_component_analysis"
        or receipt.get("extraction_version") != HISTORICAL_EXTRACTION_VERSION
        or receipt.get("source_runs") != expected_runs
        or receipt.get("outcome_rows") != expected_runs * 3
        or receipt.get("geometry_rows") != expected_runs * 2
    ):
        raise ValueError(f"Historical task receipt scope differs: {receipt_path}")
    if expected_task is not None:
        expected_identity = {
            "version": expected_task.version,
            "simnibs_version": VERSION_SPECS[expected_task.version],
            "target": expected_task.target,
            "subject": expected_task.subject,
        }
        if receipt.get("task") != expected_identity:
            raise ValueError(f"Historical task receipt identity differs: {receipt_path}")
    tables = receipt.get("tables")
    if not isinstance(tables, dict):
        raise ValueError(f"Historical task receipt lacks tables: {receipt_path}")
    for filename in TABLE_BY_OUTCOME.values():
        _bound_table(
            root=task_root,
            record=tables.get(filename),
            expected_rows=expected_runs,
            label=filename,
        )
    _bound_table(
        root=task_root,
        record=tables.get("geometry_audit.csv"),
        expected_rows=expected_runs * 2,
        label="geometry_audit.csv",
    )
    return receipt


def extract_task(
    *,
    dataset_root: Path,
    output_root: Path,
    v4_atlas_dir: Path,
    task: HistoricalTask,
    repeat_count: int = REPEAT_COUNT,
) -> dict[str, object]:
    validate_consolidation_receipt(
        dataset_root,
        subject_count=len(CANONICAL_SUBJECTS),
        repeat_count=repeat_count,
    )
    _validate_output_root(output_root, dataset_root=dataset_root)
    runs = discover_task_runs(
        dataset_root=dataset_root,
        version=task.version,
        target=task.target,
        subject=task.subject,
        v4_atlas_dir=v4_atlas_dir,
        repeat_count=repeat_count,
    )
    destination = _task_root(output_root, task)
    if destination.exists() or destination.is_symlink():
        return validate_task_receipt(
            destination,
            expected_runs=len(CONDITIONS) * repeat_count,
            expected_task=task,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise ValueError(f"Historical task staging path already exists: {staging}")
    staging.mkdir()
    try:
        rows: list[dict[str, object]] = []
        geometry: list[dict[str, object]] = []
        atlas_hash = sha256_file(runs[0].atlas_path)
        for run in runs:
            run_rows, run_geometry = _extract_run(run, atlas_sha256=atlas_hash)
            rows.extend(run_rows)
            geometry.extend(run_geometry)
        written = write_extraction_tables(staging, rows, geometry)
        expected_runs = len(runs)
        tables = {
            filename: _table_record(
                path,
                root=staging,
                rows=expected_runs * 2 if filename == "geometry_audit.csv" else expected_runs,
            )
            for filename, path in written.items()
        }
        receipt: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "workflow": "historical_component_analysis",
            "extraction_version": HISTORICAL_EXTRACTION_VERSION,
            "created_utc": _utc_now(),
            "task": {
                "version": task.version,
                "simnibs_version": VERSION_SPECS[task.version],
                "target": task.target,
                "subject": task.subject,
            },
            "source_runs": expected_runs,
            "outcome_rows": len(rows),
            "geometry_rows": len(geometry),
            "atlas_path": str(runs[0].atlas_path.resolve()),
            "atlas_sha256": atlas_hash,
            "dataset_completion_receipt_sha256": sha256_file(
                dataset_root / "_consolidation/complete.json"
            ),
            "tables": tables,
            "meshing_tasks": 0,
            "fem_tasks": 0,
            "source_outputs_modified": False,
        }
        _write_json(staging / "task_receipt.json", receipt)
        staging.replace(destination)
        return receipt
    except Exception:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _read_task_table(
    task_root: Path,
    filename: str,
    expected_rows: int,
    *,
    expected_task: HistoricalTask,
) -> list[dict[str, object]]:
    receipt = validate_task_receipt(
        task_root, expected_runs=expected_rows, expected_task=expected_task
    )
    tables = receipt["tables"]
    assert isinstance(tables, dict)
    path = _bound_table(
        root=task_root,
        record=tables[filename],
        expected_rows=expected_rows * 2 if filename == "geometry_audit.csv" else expected_rows,
        label=filename,
    )
    float_fields = () if filename == "geometry_audit.csv" else METRIC_FLOAT_FIELDS
    return read_metric_table(path, float_fields=float_fields, float_precision="round_trip")


def collect_version(
    *,
    output_root: Path,
    version: str,
    subjects: tuple[str, ...] = CANONICAL_SUBJECTS,
    repeat_count: int = REPEAT_COUNT,
) -> dict[str, object]:
    _validate_output_root(output_root)
    if version not in VERSION_SPECS:
        raise ValueError(f"Unsupported historical version: {version}")
    destination = output_root / "collected" / version
    if destination.exists() or destination.is_symlink():
        return validate_collection_receipt(
            destination,
            version=version,
            subjects=subjects,
            repeat_count=repeat_count,
        )
    rows_by_key: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    geometry: list[dict[str, object]] = []
    expected_task_runs = len(CONDITIONS) * repeat_count
    for task in task_matrix(version, subjects=subjects):
        task_root = _task_root(output_root, task)
        for outcome, filename in TABLE_BY_OUTCOME.items():
            rows = _read_task_table(
                task_root,
                filename,
                expected_task_runs,
                expected_task=task,
            )
            for row in rows:
                key = (str(row["target"]), str(row["condition"]), outcome)
                rows_by_key.setdefault(key, []).append(row)
        geometry.extend(
            _read_task_table(
                task_root,
                "geometry_audit.csv",
                expected_task_runs,
                expected_task=task,
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{version}.tmp-{os.getpid()}"
    if staging.exists():
        raise ValueError(f"Historical collection staging path exists: {staging}")
    staging.mkdir()
    try:
        tables: dict[str, dict[str, object]] = {}
        expected_rows = len(subjects) * repeat_count
        for target in TARGETS:
            for condition in CONDITIONS:
                for outcome, filename in TABLE_BY_OUTCOME.items():
                    key = (target, condition, outcome)
                    rows = rows_by_key.get(key, [])
                    if len(rows) != expected_rows:
                        raise ValueError(
                            f"Historical collection {key} has {len(rows)} rows; "
                            f"expected {expected_rows}"
                        )
                    path = staging / target / condition / filename
                    write_metric_table(path, rows)
                    tables[f"{target}/{condition}/{outcome}"] = _table_record(
                        path, root=staging, rows=len(rows)
                    )
        expected_geometry = len(subjects) * len(TARGETS) * len(CONDITIONS) * repeat_count * 2
        if len(geometry) != expected_geometry:
            raise ValueError(
                f"Historical geometry collection has {len(geometry)} rows; "
                f"expected {expected_geometry}"
            )
        geometry_path = staging / "geometry_audit.csv"
        write_metric_table(geometry_path, geometry)
        receipt: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "workflow": "historical_component_analysis_collection",
            "created_utc": _utc_now(),
            "version": version,
            "analysis_version": ANALYSIS_VERSION_LABELS[version],
            "simnibs_version": VERSION_SPECS[version],
            "subject_tasks": len(task_matrix(version, subjects=subjects)),
            "source_runs": len(subjects) * len(TARGETS) * len(CONDITIONS) * repeat_count,
            "outcome_rows": sum(int(record["rows"]) for record in tables.values()),
            "geometry_rows": len(geometry),
            "tables": tables,
            "geometry_audit": _table_record(
                geometry_path, root=staging, rows=len(geometry)
            ),
            "preliminary": True,
            "source_outputs_modified": False,
        }
        _write_json(staging / "collection_receipt.json", receipt)
        staging.replace(destination)
        return receipt
    except Exception:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def validate_collection_receipt(
    collection_root: Path,
    *,
    version: str,
    subjects: tuple[str, ...] = CANONICAL_SUBJECTS,
    repeat_count: int = REPEAT_COUNT,
) -> dict[str, object]:
    receipt = _read_json(
        collection_root / "collection_receipt.json", "collection receipt"
    )
    expected_source_runs = len(subjects) * len(TARGETS) * len(CONDITIONS) * repeat_count
    expected_identity = {
        "status": "complete",
        "workflow": "historical_component_analysis_collection",
        "version": version,
        "analysis_version": ANALYSIS_VERSION_LABELS[version],
        "simnibs_version": VERSION_SPECS[version],
        "subject_tasks": len(task_matrix(version, subjects=subjects)),
        "source_runs": expected_source_runs,
        "outcome_rows": expected_source_runs * len(TABLE_BY_OUTCOME),
        "geometry_rows": expected_source_runs * 2,
        "preliminary": True,
        "source_outputs_modified": False,
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"Historical collection receipt {key} differs: {collection_root}"
            )
    tables = receipt.get("tables")
    if not isinstance(tables, dict) or len(tables) != 12:
        raise ValueError(f"Historical collection table scope differs: {collection_root}")
    expected_rows = len(subjects) * repeat_count
    for target in TARGETS:
        for condition in CONDITIONS:
            for outcome in TABLE_BY_OUTCOME:
                key = f"{target}/{condition}/{outcome}"
                _bound_table(
                    root=collection_root,
                    record=tables.get(key),
                    expected_rows=expected_rows,
                    label=key,
                )
    _bound_table(
        root=collection_root,
        record=receipt.get("geometry_audit"),
        expected_rows=expected_source_runs * 2,
        label="geometry_audit.csv",
    )
    return receipt


def preliminary_metric_sources(output_root: Path):
    """Return exact schema-compatible sources from both completed collectors."""

    from repeatability_combined_analysis_v2.assembly import MetricSource

    _validate_output_root(output_root)
    sources = []
    for version in VERSION_SPECS:
        root = output_root / "collected" / version
        receipt = validate_collection_receipt(root, version=version)
        tables = receipt.get("tables")
        if not isinstance(tables, dict):
            raise ValueError(f"Historical collection has no tables: {root}")
        for target in TARGETS:
            for condition in CONDITIONS:
                for outcome in TABLE_BY_OUTCOME:
                    key = f"{target}/{condition}/{outcome}"
                    path = _bound_table(
                        root=root,
                        record=tables.get(key),
                        expected_rows=len(CANONICAL_SUBJECTS) * REPEAT_COUNT,
                        label=key,
                    )
                    sources.append(
                        MetricSource(
                            version=ANALYSIS_VERSION_LABELS[version],
                            target=target,
                            condition=condition,
                            outcome_mode=outcome,
                            path=path,
                            preliminary=True,
                        )
                    )
    if len(sources) != 24:
        raise AssertionError("Historical preliminary source count changed")
    return tuple(sources)
