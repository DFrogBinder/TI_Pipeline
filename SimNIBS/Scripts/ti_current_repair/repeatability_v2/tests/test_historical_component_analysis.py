from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ti_current_repair.repeatability_v2.historical as historical_module
from ti_current_repair.repeatability_v2.extraction import NativeMeshField
from ti_current_repair.repeatability_v2.historical import (
    ANALYSIS_VERSION_LABELS,
    HISTORICAL_EXTRACTION_VERSION,
    HistoricalSourceRun,
    PAIR1_ANODE_SURFACE_TAG,
    VERSION_SPECS,
    collect_version,
    discover_task_runs,
    preflight_historical,
    task_matrix,
    validate_consolidation_receipt,
)
from ti_current_repair.repeatability_v2.inventory import sha256_file
from ti_current_repair.repeatability_v2.precision import write_metric_table


def _write(path: Path, content: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _completion(root: Path, *, subjects: int, repeats: int) -> Path:
    cells = subjects * 2 * 2 * 2
    payload = {
        "status": "complete",
        "workflow": "repeatability_dataset_consolidation",
        "output_root": str(root.resolve()),
        "scope": {
            "subjects": subjects,
            "cells": cells,
            "simulations": cells * repeats,
            "v3_headmodels": subjects,
            "v4_scaffolds": subjects,
        },
    }
    return _write(
        root / "_consolidation/complete.json",
        json.dumps(payload),
    )


def _seed_run(
    root: Path,
    *,
    version: str,
    target: str,
    subject: str,
    condition: str,
    repeat: int,
    anode: str,
) -> Path:
    repeat_subject_root = (
        root
        / "data"
        / version
        / target
        / f"{subject}_repeatability"
        / condition
        / "repeats"
        / f"repeat_{repeat:03d}"
        / subject
    )
    output = repeat_subject_root / "anat/SimNIBS/Output" / subject
    for filename in (
        "TI.msh",
        f"{subject}_TDCS_1_scalar.msh",
        f"{subject}_TDCS_2_scalar.msh",
    ):
        _write(output / filename)
    stimulation = {
        "montage_preset": target.replace("_", "-"),
        "pair1": {"anode": anode, "cathode": "P8", "current_a": 0.001},
        "pair2": {"anode": "T7", "cathode": "P7", "current_a": 0.001},
    }
    manifest = {
        "subject": subject,
        "condition": condition,
        "repeat_index": repeat,
        "repeat_tag": f"repeat_{repeat:03d}",
        "stimulation": stimulation,
    }
    if version == "simnibs326":
        manifest["simnibs_326_harmonized_runtime"] = {
            "backend": "simnibs-3.2.6-harmonized-corrected-label-v2",
            "config": "/retired/source/_pipeline/configs/remesh_only.json",
            "harmonized_map_sha256": "a" * 64,
            "headmodel_manifest_sha256": "b" * 64,
            "headmodel_marker_sha256": "c" * 64,
            "headmodel_promotion_sha256": "d" * 64,
            "headmodel_root": "/retired/headmodels",
            "simnibs_module": "SimNIBS/3.2.6-foss-2023a",
            "simnibs_version": "3.2.6",
        }
    manifest_path = repeat_subject_root / "task_manifest.json"
    _write(manifest_path, json.dumps(manifest))
    return manifest_path


def test_consolidation_receipt_and_task_matrix_are_exact(tmp_path: Path) -> None:
    root = tmp_path / "combined"
    _completion(root, subjects=10, repeats=40)

    receipt = validate_consolidation_receipt(root)
    tasks = task_matrix("simnibs326")

    assert receipt["scope"]["simulations"] == 3200
    assert len(tasks) == 20
    assert len({(task.target, task.subject) for task in tasks}) == 20
    assert PAIR1_ANODE_SURFACE_TAG == 2101


def test_discovery_resolves_native_meshes_and_requires_recorded_anode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "combined"
    subject = "sub-CC110174"
    for condition in ("remesh", "fixed_mesh"):
        for repeat in (1, 2):
            _seed_run(
                root,
                version="simnibs4",
                target="right_m1",
                subject=subject,
                condition=condition,
                repeat=repeat,
                anode="Fp2",
            )
    atlas = _write(tmp_path / "atlases" / f"{subject}.nii.gz")

    runs = discover_task_runs(
        dataset_root=root,
        version="simnibs4",
        target="right_m1",
        subject=subject,
        repeat_count=2,
        v4_atlas_dir=atlas.parent,
    )

    assert len(runs) == 4
    assert {run.condition for run in runs} == {"remesh", "fixed_mesh"}
    assert all(run.pair1_msh.name.endswith("_TDCS_1_scalar.msh") for run in runs)
    assert all(run.atlas_path == atlas for run in runs)
    assert all(run.anode_name == "Fp2" for run in runs)

    manifest = runs[0].manifest_path
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["stimulation"]["pair1"]["anode"] = "C3"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected Fp2"):
        discover_task_runs(
            dataset_root=root,
            version="simnibs4",
            target="right_m1",
            subject=subject,
            repeat_count=2,
            v4_atlas_dir=atlas.parent,
        )


def test_v3_discovery_uses_consolidated_harmonized_atlas(tmp_path: Path) -> None:
    root = tmp_path / "combined"
    subject = "sub-CC110174"
    for condition in ("remesh", "fixed_mesh"):
        _seed_run(
            root,
            version="simnibs326",
            target="left_hippocampus",
            subject=subject,
            condition=condition,
            repeat=1,
            anode="F8",
        )
    atlas = _write(
        root
        / "provenance/headmodels/simnibs326"
        / subject
        / "harmonized_label.nii.gz"
    )

    runs = discover_task_runs(
        dataset_root=root,
        version="simnibs326",
        target="left_hippocampus",
        subject=subject,
        repeat_count=1,
        v4_atlas_dir=tmp_path / "unused",
    )

    assert {run.atlas_path for run in runs} == {atlas}
    assert {run.simnibs_version for run in runs} == {"3.2.6"}


def test_v3_discovery_rejects_legacy_only_runtime_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "combined"
    subject = "sub-CC110174"
    manifest_paths = []
    for condition in ("remesh", "fixed_mesh"):
        manifest_paths.append(
            _seed_run(
                root,
                version="simnibs326",
                target="left_hippocampus",
                subject=subject,
                condition=condition,
                repeat=1,
                anode="F8",
            )
        )
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.pop("simnibs_326_harmonized_runtime")
        payload["simnibs_326_runtime"] = {"simnibs_version": "3.2.6"}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(
        root
        / "provenance/headmodels/simnibs326"
        / subject
        / "harmonized_label.nii.gz"
    )

    with pytest.raises(ValueError, match="harmonized runtime provenance"):
        discover_task_runs(
            dataset_root=root,
            version="simnibs326",
            target="left_hippocampus",
            subject=subject,
            repeat_count=1,
            v4_atlas_dir=tmp_path / "unused",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("simnibs_version", "3.2.5"),
        ("simnibs_module", "SimNIBS/4.0.1-foss-2023a"),
        ("backend", "simnibs-3.2.6-headreco-volumemesh"),
    ),
)
def test_v3_discovery_rejects_mismatched_harmonized_runtime_provenance(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    root = tmp_path / "combined"
    subject = "sub-CC110174"
    manifest_paths = []
    for condition in ("remesh", "fixed_mesh"):
        manifest_paths.append(
            _seed_run(
                root,
                version="simnibs326",
                target="left_hippocampus",
                subject=subject,
                condition=condition,
                repeat=1,
                anode="F8",
            )
        )
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["simnibs_326_runtime"] = {"simnibs_version": "3.2.6"}
        payload["simnibs_326_harmonized_runtime"][field] = value
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    _write(
        root
        / "provenance/headmodels/simnibs326"
        / subject
        / "harmonized_label.nii.gz"
    )

    with pytest.raises(ValueError, match="harmonized runtime provenance"):
        discover_task_runs(
            dataset_root=root,
            version="simnibs326",
            target="left_hippocampus",
            subject=subject,
            repeat_count=1,
            v4_atlas_dir=tmp_path / "unused",
        )


def test_preflight_is_read_only_and_reports_zero_simulation_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "combined"
    output = tmp_path / "new-output"
    subjects = ("sub-CC110174",)
    _completion(root, subjects=1, repeats=1)
    for version in ("simnibs326", "simnibs4"):
        for target, anode in (("left_hippocampus", "F8"), ("right_m1", "Fp2")):
            for condition in ("remesh", "fixed_mesh"):
                _seed_run(
                    root,
                    version=version,
                    target=target,
                    subject=subjects[0],
                    condition=condition,
                    repeat=1,
                    anode=anode,
                )
    _write(
        root
        / "provenance/headmodels/simnibs326"
        / subjects[0]
        / "harmonized_label.nii.gz"
    )
    v4_atlas = _write(tmp_path / "atlases" / f"{subjects[0]}.nii.gz")

    result = preflight_historical(
        dataset_root=root,
        output_root=output,
        v4_atlas_dir=v4_atlas.parent,
        subjects=subjects,
        repeat_count=1,
        probe_native=False,
    )

    assert result["status"] == "ready"
    assert result["existing_simulations"] == 8
    assert result["expected_metric_rows"] == 24
    assert result["meshing_tasks"] == 0
    assert result["fem_tasks"] == 0
    assert result["source_outputs_modified"] is False
    assert not output.exists()


def test_collect_version_requires_complete_task_receipts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="task receipt"):
        collect_version(
            output_root=output,
            version="simnibs326",
            subjects=("sub-CC110174",),
        )


def _seed_task_receipt(
    output: Path, *, version: str, target: str, subject: str
) -> None:
    task_root = output / "tasks" / version / target / subject
    common = {
        "subject": subject,
        "version": ANALYSIS_VERSION_LABELS[version],
        "target": target,
        "repeat": 1,
        "repeat_tag": "repeat_001",
        "simnibs_version": VERSION_SPECS[version],
        "preliminary": True,
        "roi_mean_v_per_m": 0.1,
        "roi_median_v_per_m": 0.1,
        "roi_p95_v_per_m": 0.2,
        "achieved_roi_volume_mm3": 100.0,
        "finite_volume_mm3": 100.0,
        "finite_volume_fraction": 1.0,
    }
    filenames = {
        "ti_native_mesh_metrics.csv": "ti_target",
        "tdcs_pair1_matched_target_metrics.csv": "tdcs_pair1_matched_target",
        "tdcs_pair1_under_anode_metrics.csv": "tdcs_pair1_under_anode",
    }
    tables = {}
    for filename, outcome in filenames.items():
        rows = [
            {**common, "condition": condition, "outcome_mode": outcome}
            for condition in ("remesh", "fixed_mesh")
        ]
        path = task_root / filename
        write_metric_table(path, rows)
        tables[filename] = {
            "path": filename,
            "rows": 2,
            "sha256": sha256_file(path),
        }
    geometry_path = task_root / "geometry_audit.csv"
    write_metric_table(
        geometry_path,
        [
            {
                **common,
                "condition": condition,
                "pair": pair,
                "geometry_sha256": str(pair) * 64,
            }
            for condition in ("remesh", "fixed_mesh")
            for pair in (1, 2)
        ],
    )
    tables["geometry_audit.csv"] = {
        "path": "geometry_audit.csv",
        "rows": 4,
        "sha256": sha256_file(geometry_path),
    }
    _write(
        task_root / "task_receipt.json",
        json.dumps(
            {
                "status": "complete",
                "workflow": "historical_component_analysis",
                "extraction_version": HISTORICAL_EXTRACTION_VERSION,
                "source_runs": 2,
                "outcome_rows": 6,
                "geometry_rows": 4,
                "task": {
                    "version": version,
                    "simnibs_version": VERSION_SPECS[version],
                    "target": target,
                    "subject": subject,
                },
                "tables": tables,
            }
        ),
    )


def test_version_collection_is_exact_and_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _write(
        output / "_historical_component_analysis/owner.json",
        json.dumps(
            {
                "status": "owned",
                "workflow": "historical_component_analysis",
                "output_root": str(output.resolve()),
                "source_outputs_modified": False,
            }
        ),
    )
    subject = "sub-CC110174"
    for target in ("left_hippocampus", "right_m1"):
        _seed_task_receipt(
            output,
            version="simnibs326",
            target=target,
            subject=subject,
        )

    receipt = collect_version(
        output_root=output,
        version="simnibs326",
        subjects=(subject,),
        repeat_count=1,
    )

    assert receipt["source_runs"] == 4
    assert receipt["outcome_rows"] == 12
    assert receipt["geometry_rows"] == 8
    assert len(receipt["tables"]) == 12

    first = next(iter(receipt["tables"].values()))
    table = output / "collected/simnibs326" / first["path"]
    table.write_text(table.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale"):
        collect_version(
            output_root=output,
            version="simnibs326",
            subjects=(subject,),
            repeat_count=1,
        )


def test_historical_run_extracts_all_three_native_outcomes_and_tag_2101(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject = "sub-CC110174"
    ti_path = _write(tmp_path / "TI.msh")
    pair1_path = _write(tmp_path / f"{subject}_TDCS_1_scalar.msh")
    pair2_path = _write(tmp_path / f"{subject}_TDCS_2_scalar.msh")
    manifest = _write(tmp_path / "task_manifest.json", "{}")
    atlas = _write(tmp_path / "atlas.nii.gz")
    centres = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    volumes = np.asarray([60.0, 60.0], dtype=np.float64)
    tags = np.asarray([2, 2], dtype=np.int64)

    def load_field(path: Path, field_name: str) -> NativeMeshField:
        values = (
            np.asarray([0.1, 0.2], dtype=np.float64)
            if field_name == "TImax"
            else np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
        )
        return NativeMeshField(
            path=path,
            field_name=field_name,
            values=values,
            tetra_centres=centres,
            tetra_volumes=volumes,
            geometry_sha256="a" * 64,
            stored_dtype="float64",
            tetra_tags=tags,
        )

    observed_surface_tags = []

    def load_surface(path: Path, surface_tag: int):
        observed_surface_tags.append((path, surface_tag))
        return (
            np.asarray(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                dtype=np.float64,
            ),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )

    monkeypatch.setattr(historical_module, "load_native_element_field", load_field)
    monkeypatch.setattr(historical_module, "load_tagged_surface", load_surface)
    monkeypatch.setattr(
        historical_module,
        "parcel_mask_from_atlas",
        lambda *args, **kwargs: np.asarray([True, True]),
    )
    run = HistoricalSourceRun(
        version="simnibs4",
        simnibs_version="4.0.1",
        target="left_hippocampus",
        subject=subject,
        condition="remesh",
        repeat=1,
        repeat_tag="repeat_001",
        ti_msh=ti_path,
        pair1_msh=pair1_path,
        pair2_msh=pair2_path,
        manifest_path=manifest,
        atlas_path=atlas,
        anode_name="F8",
    )

    rows, geometry = historical_module._extract_run(run, atlas_sha256="b" * 64)

    assert {row["outcome_mode"] for row in rows} == {
        "ti_target",
        "tdcs_pair1_matched_target",
        "tdcs_pair1_under_anode",
    }
    assert observed_surface_tags == [(pair1_path, 2101)]
    assert len(geometry) == 2
    under = next(row for row in rows if row["outcome_mode"].endswith("under_anode"))
    provenance = json.loads(under["roi_provenance_json"])
    assert provenance["electrode_name"] == "F8"
    assert provenance["electrode_surface_tag"] == 2101


def test_historical_hpc_launcher_is_guarded() -> None:
    launcher = Path(__file__).parents[1] / "historical_hpc/submit.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "--submit" in text
    assert "PREFLIGHT_ONLY=1" in text
    assert "existing simulations: 3200" in text
    assert "expected metric rows: 9600" in text
    assert "meshing tasks: 0" in text
    assert "FEM tasks: 0" in text
    assert "SimNIBS/4.0.1-foss-2023a" in text
