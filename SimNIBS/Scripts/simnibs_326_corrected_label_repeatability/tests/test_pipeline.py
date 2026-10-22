from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import simnibs_326_corrected_label_repeatability.pipeline as pipeline
from simnibs_326_corrected_label_repeatability.settings import (
    SUBJECTS,
    campaign_paths_from_environment,
)


def _paths(tmp_path: Path):
    base = campaign_paths_from_environment({})
    return replace(
        base,
        corrected_root=tmp_path / "corrected",
        legacy_reference_root=tmp_path / "legacy",
        atlas_root=tmp_path / "atlases",
        headmodel_root=tmp_path / "new-headmodels",
        left_root=tmp_path / "new-left",
        right_root=tmp_path / "new-right",
        completed_v3_left_root=tmp_path / "old-v3-left",
        completed_v3_right_root=tmp_path / "old-v3-right",
        completed_v4_left_root=tmp_path / "old-v4-left",
        completed_v4_right_root=tmp_path / "old-v4-right",
    )


def _promotion(paths):
    return {
        "manifest_sha256": "a" * 64,
        "promotion_sha256": "b" * 64,
        "promotion_path": str(paths.headmodel_root / "promoted.json"),
    }


def _stim(target: str, _targets_csv: Path | None = None):
    return SimpleNamespace(to_dict=lambda: {"montage_preset": target})


def test_preflight_is_read_only_and_checks_all_ten_subjects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    inspected: list[str] = []
    targets = tmp_path / "targets.csv"
    targets.write_text("target\n", encoding="utf-8")
    paths.atlas_root.mkdir()
    for subject in SUBJECTS:
        (paths.atlas_root / f"{subject}.nii.gz").write_bytes(b"atlas")
    monkeypatch.setattr(
        pipeline,
        "_inspect_subject_inputs",
        lambda _paths, subject: inspected.append(subject) or {"subject": subject},
    )
    monkeypatch.setattr(pipeline, "_resolve_stimulation", _stim)

    result = pipeline.preflight(paths=paths, targets_csv=targets)

    assert result["status"] == "ready"
    assert inspected == list(SUBJECTS)
    assert result["source_outputs_modified"] is False
    assert not paths.headmodel_root.exists()
    assert not paths.left_root.exists()
    assert not paths.right_root.exists()


def test_subject_preflight_records_all_required_file_hashes_sizes_and_grids(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    subject = SUBJECTS[0]
    resolved = pipeline.subject_paths(paths, subject)
    resolved.corrected_label.parent.mkdir(parents=True)
    resolved.legacy_m2m.mkdir(parents=True)
    data = np.ones((3, 4, 5), dtype=np.uint8)
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    for path in (resolved.corrected_label, resolved.corrected_t1, resolved.corrected_t2):
        nib.save(nib.Nifti1Image(data, affine), path)
    for path in (
        resolved.legacy_m2m / "T1fs_conform.nii.gz",
        resolved.legacy_m2m / "T1fs_nu_conform.nii.gz",
    ):
        nib.save(nib.Nifti1Image(data, affine), path)
    resolved.legacy_ready_marker.write_text("{}", encoding="utf-8")
    (resolved.legacy_m2m / "eeg_positions").mkdir()
    (resolved.legacy_m2m / "toMNI").mkdir()

    result = pipeline._inspect_subject_inputs(paths, subject)

    assert set(result["files"]) == {
        "corrected_label", "corrected_t1", "corrected_t2", "legacy_ready_marker",
        "t1fs_conform", "t1fs_nu_conform",
    }
    assert all(record["size_bytes"] > 0 for record in result["files"].values())
    assert all(len(record["sha256"]) == 64 for record in result["files"].values())
    assert result["source_grid"]["shape"] == [3, 4, 5]
    assert result["source_grid"]["affine"] == affine.tolist()
    assert result["reference_grid"]["voxel_sizes_mm"] == [1.0, 2.0, 3.0]


def test_initialize_writes_only_two_exact_400_task_remesh_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "_validated_promotion", lambda _paths: _promotion(paths))
    monkeypatch.setattr(pipeline, "_resolve_stimulation", _stim)

    result = pipeline.initialize(paths=paths, dry_run=False)

    assert result["status"] == "initialized"
    for target, root in (
        ("left-hippocampus", paths.left_root),
        ("right-m1", paths.right_root),
    ):
        config = json.loads(
            (root / "_pipeline/configs/remesh_only.json").read_text(encoding="utf-8")
        )
        assert config["subjects"] == list(SUBJECTS)
        assert config["conditions"] == [
            {
                "description": pipeline.REMESH_DESCRIPTION,
                "mesh_mode": "remesh",
                "name": "remesh",
                "repeat_count": 40,
            }
        ]
        assert len(config["subjects"]) * config["conditions"][0]["repeat_count"] == 400
        assert config["analysis"]["roi_preset"] == target
        assert config[pipeline.CONFIG_RUNTIME_KEY]["headmodel_manifest_sha256"] == "a" * 64
        manifest = json.loads(
            (root / "_simnibs326/campaign.json").read_text(encoding="utf-8")
        )
        assert len(manifest["git_commit"]) >= 7
        assert not (root / "_pipeline/configs/fixed_mesh_spherical_median.json").exists()


def test_initialize_fails_closed_on_missing_or_stale_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "_validated_promotion",
        lambda _paths: (_ for _ in ()).throw(RuntimeError("stale promotion")),
    )
    with pytest.raises(RuntimeError, match="stale promotion"):
        pipeline.initialize(paths=paths, dry_run=False)
    assert not paths.left_root.exists()
    assert not paths.right_root.exists()


def test_module_preflight_probes_meshing_and_fem_without_matlab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "module.json"
    monkeypatch.setattr(pipeline, "load_v3_meshing_api", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "probe_v3_meshing_api",
        lambda _api: {"status": "ready", "simnibs_version": "3.2.6"},
    )
    monkeypatch.setattr(
        pipeline,
        "_load_simnibs_fem_api",
        lambda: {
            "simnibs_version": "3.2.6",
            "session_api": True,
            "element_data_api": True,
            "conductivities": ["WM", "GM", "CSF", "Bone", "Scalp"],
        },
    )

    result = pipeline.module_preflight(receipt=receipt)

    assert result["status"] == "ready"
    assert result["matlab_required"] is False
    assert result["meshing_api"]["status"] == "ready"
    assert result["fem_api"]["session_api"] is True
    assert len(result["git_commit"]) >= 7
    assert "matlab" not in json.dumps(result).lower().replace('"matlab_required": false', "")
    assert pipeline.validate_module_preflight_receipt(receipt)["status"] == "ready"


def test_completion_gates_require_400_then_800_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    root = paths.left_root
    config_dir = root / "_pipeline/configs"
    config_dir.mkdir(parents=True)
    remesh = {"subjects": list(SUBJECTS), "conditions": [{"name": "remesh"}]}
    fixed = {"subjects": list(SUBJECTS), "conditions": [{"name": "fixed_mesh"}]}
    (config_dir / pipeline.REMESH_CONFIG).write_text(json.dumps(remesh), encoding="utf-8")
    (config_dir / pipeline.PAIRED_CONFIG).write_text(json.dumps(remesh), encoding="utf-8")
    (config_dir / pipeline.FIXED_CONFIG).write_text(json.dumps(fixed), encoding="utf-8")
    validations = {
        pipeline.REMESH_CONFIG: {
            "status": "complete", "expected_tasks": 400, "complete_tasks": 400,
            "failure_count": 0, "failures": [],
        },
        pipeline.FIXED_CONFIG: {
            "status": "complete", "expected_tasks": 400, "complete_tasks": 400,
            "failure_count": 0, "failures": [],
        },
    }
    monkeypatch.setattr(
        pipeline,
        "_validate_config_outputs",
        lambda path: {"config": str(path), **validations[path.name]},
    )
    monkeypatch.setattr(pipeline, "_receipt_scope", lambda *args, **kwargs: {"expected": 400})

    remesh_receipt = pipeline.complete_remesh("left-hippocampus", paths=paths)
    final_receipt = pipeline.complete_final("left-hippocampus", paths=paths)

    assert remesh_receipt["validation"]["complete_tasks"] == 400
    assert len(remesh_receipt["git_commit"]) >= 7
    assert final_receipt["validation"]["total_tasks"] == 800
    assert len(json.loads((config_dir / pipeline.PAIRED_CONFIG).read_text())["conditions"]) == 2


def test_fixed_preparation_uses_established_median_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    receipt = paths.left_root / "_pipeline/workflow/complete.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "complete", "stage": "remesh"}), encoding="utf-8")
    observed = {}
    monkeypatch.setattr(
        pipeline,
        "prepare_fixed",
        lambda **kwargs: observed.update(kwargs) or {"status": "planned"},
    )

    result = pipeline.prepare_fixed_stage(
        "left-hippocampus", metrics_csv=tmp_path / "metrics.csv", dry_run=True, paths=paths
    )

    assert result["status"] == "planned"
    assert observed["repeat_count"] == 40
    assert observed["source_experiment_root"] == paths.left_root


def test_status_reports_ordered_campaign_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "_target_counts", lambda _root: {"remesh": 400, "fixed_mesh": 0})
    (paths.headmodel_root / "_pipeline/headmodels").mkdir(parents=True)
    (paths.headmodel_root / "_pipeline/reference_stage_collection.json").write_text("{}")
    (paths.headmodel_root / "_pipeline/headmodels/manifest.json").write_text("{}")
    (paths.headmodel_root / "_pipeline/headmodels/promoted.json").write_text("{}")
    for root in (paths.left_root, paths.right_root):
        (root / "_simnibs326").mkdir(parents=True)
        (root / "_simnibs326/campaign.json").write_text("{}")

    result = pipeline.status(paths=paths)

    assert result["state"] == "remesh-complete"
    assert result["gates"] == {
        "initialized": True,
        "references_staged": True,
        "headmodels_built": True,
        "promoted": True,
        "remesh_complete": True,
        "final_complete": False,
    }


def test_cli_exposes_the_explicit_command_surface() -> None:
    parser = pipeline.build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert commands == {
        "preflight", "init", "module-preflight", "validate-module-receipt",
        "stage-reference", "collect-references", "build-headmodel",
        "validate-headmodel", "render-qc", "collect-headmodels",
        "promote-headmodels", "complete-remesh", "prepare-fixed",
        "complete-final", "validate-final", "show-plan", "status",
    }
