from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import simnibs_326_corrected_label_repeatability.simulation_runner as simulation_runner
from simnibs_326_corrected_label_repeatability.settings import (
    SUBJECTS,
    campaign_paths_from_environment,
    subject_paths,
)
from simnibs_326_corrected_label_repeatability.v3_meshing import MeshValidation


def test_internal_max_ti_matches_validated_legacy_geometries() -> None:
    first = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    second = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    observed = simulation_runner.max_ti(first, second)

    assert np.allclose(observed, [1.0, np.sqrt(2.0), 0.0])


def test_v3_label_export_preserves_element_tag_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeMesh:
        elm = SimpleNamespace(tag1=np.array([1, 2, 3, 5], dtype=np.int32))

        def crop_mesh(self, *, elm_type: int):
            calls["elm_type"] = elm_type
            return self

    class FakeElementData:
        def __init__(self, values) -> None:
            calls["values"] = np.asarray(values).copy()
            self.mesh = None

        def to_nifti(self, dimensions, affine, *, fn, qform, method) -> None:
            calls["dimensions"] = tuple(int(value) for value in dimensions)
            calls["method"] = method
            calls["mesh"] = self.mesh
            Path(fn).write_bytes(b"label-nifti")

    fake_io = SimpleNamespace(
        read_msh=lambda path: calls.update(mesh_path=path) or FakeMesh(),
        ElementData=FakeElementData,
    )
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.eye(4)), reference)
    mesh = tmp_path / "TI.msh"
    mesh.write_bytes(b"mesh")
    monkeypatch.setattr(simulation_runner.runner, "SIM_MODULE", object())
    monkeypatch.setattr(simulation_runner.runner, "SIM_MESH_IO", fake_io)

    output = simulation_runner._write_label_volume_326(
        mesh, reference, tmp_path / "TI_Volumetric_Labels"
    )

    assert output == tmp_path / "TI_Volumetric_Labels.nii.gz"
    assert output.read_bytes() == b"label-nifti"
    assert calls["mesh_path"] == str(mesh)
    assert calls["elm_type"] == 4
    assert np.array_equal(calls["values"], [1, 2, 3, 5])
    assert calls["dimensions"] == (3, 4, 5)
    assert calls["method"] == "assign"
    assert isinstance(calls["mesh"], FakeMesh)


def test_remesh_workspace_copies_only_promoted_references_and_new_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved, promotion = _promoted_subject_layout(tmp_path)
    workspace = _workspace(tmp_path / "repeat", resolved.subject)
    calls: dict[str, object] = {}
    monkeypatch.setattr(simulation_runner, "_validated_promotion", lambda _paths: promotion)
    monkeypatch.setattr(simulation_runner, "load_v3_meshing_api", lambda: SimpleNamespace())

    def fake_build(
        api,
        subject,
        surfaces,
        relabel_surfaces,
        m2m_dir,
        output_mesh,
        keep_air=False,
    ):
        calls.update(
            api=api,
            subject=subject,
            surfaces=tuple(surfaces),
            relabel=tuple(relabel_surfaces),
            m2m_dir=m2m_dir,
            keep_air=keep_air,
        )
        output_mesh.write_bytes(b"repeat-mesh")
        return MeshValidation(str(output_mesh), 6, (1, 2, 3, 4, 5, 6), 0.5)

    monkeypatch.setattr(simulation_runner, "build_volume_mesh", fake_build)

    mesh = simulation_runner._build_remesh_workspace(
        paths,
        resolved.subject,
        workspace,
        force_mesh=False,
    )

    copied = sorted(
        path.relative_to(workspace.mesh_dir).as_posix()
        for path in workspace.mesh_dir.rglob("*")
        if path.is_file()
    )
    assert set(copied) == {
        "T1fs_conform.nii.gz",
        "T1fs_nu_conform.nii.gz",
        f"{resolved.subject}.msh",
        "bone.stl",
        "csf.stl",
        "eeg_positions/cap.csv",
        "eyes.stl",
        "gm.stl",
        "skin.stl",
        "toMNI/warp.nii.gz",
        "wm.stl",
    }
    assert not any("MASK_" in entry or "segment" in entry for entry in copied)
    assert mesh == workspace.mesh_path
    assert calls["subject"] == resolved.subject
    assert calls["keep_air"] is False
    assert {path.name for path in calls["surfaces"]} == {
        "wm.stl",
        "gm.stl",
        "csf.stl",
        "bone.stl",
        "skin.stl",
        "eyes.stl",
    }
    ready = json.loads((workspace.anat_dir / ".mesh_ready.json").read_text())
    assert ready["backend"] == simulation_runner.BACKEND_NAME
    assert ready["promotion_sha256"] == promotion["promotion_sha256"]
    assert ready["manifest_sha256"] == promotion["manifest_sha256"]
    assert ready["headmodel_marker_sha256"] == _sha256(resolved.headmodel_receipt)
    assert ready["harmonized_map_sha256"] == _sha256(resolved.harmonized_label)
    assert ready["mesh_sha256"] == _sha256(workspace.mesh_path)


def test_fixed_workspace_consumes_selected_immutable_mesh_without_meshing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved, promotion = _promoted_subject_layout(tmp_path)
    workspace = _workspace(tmp_path / "fixed-cache", resolved.subject)
    workspace.mesh_dir.mkdir(parents=True)
    workspace.mesh_path.write_bytes(b"selected-mesh")
    source_mesh = tmp_path / "new-left" / "repeat_017" / resolved.subject / "anat" / f"m2m_{resolved.subject}" / f"{resolved.subject}.msh"
    source_mesh.parent.mkdir(parents=True)
    source_mesh.write_bytes(b"selected-mesh")
    checksum = _sha256(workspace.mesh_path)
    (workspace.anat_dir / ".mesh_ready.json").write_text(
        json.dumps(
            {
                "status": "mesh_ready",
                "subject": resolved.subject,
                "seed_mode": "physical_cache_copy",
                "source_mesh_path": str(source_mesh),
                "mesh_checksum": checksum,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(simulation_runner, "_validated_promotion", lambda _paths: promotion)
    monkeypatch.setattr(
        simulation_runner,
        "build_volume_mesh",
        lambda *args, **kwargs: pytest.fail("fixed task must not remesh"),
    )

    result = simulation_runner._mesh_workspace_harmonized(
        workspace,
        subject=resolved.subject,
        force_mesh=False,
        paths=paths,
    )

    assert result == workspace.mesh_path
    assert _sha256(result) == checksum


@pytest.mark.parametrize(
    "bad_root_field",
    [
        "corrected_root",
        "legacy_reference_root",
        "completed_v3_left_root",
        "completed_v4_left_root",
    ],
)
def test_config_rejects_protected_or_completed_experiment_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_root_field: str,
) -> None:
    paths, _, promotion = _promoted_subject_layout(tmp_path)
    monkeypatch.setattr(simulation_runner, "_validated_promotion", lambda _paths: promotion)
    config = _valid_config(paths, promotion)
    config["experiment_root"] = str(getattr(paths, bad_root_field))
    config_path = tmp_path / f"bad-{bad_root_field}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SystemExit, match="new harmonized result root"):
        simulation_runner._validate_campaign_config(config_path, paths=paths)


def test_config_and_task_reject_unpromoted_or_stale_headmodels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved, promotion = _promoted_subject_layout(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_valid_config(paths, promotion)), encoding="utf-8")
    monkeypatch.setattr(
        simulation_runner,
        "_validated_promotion",
        lambda _paths: (_ for _ in ()).throw(simulation_runner.SimulationBoundaryError("not promoted")),
    )

    with pytest.raises(SystemExit, match="not promoted"):
        simulation_runner._validate_campaign_config(config_path, paths=paths)

    workspace = _workspace(tmp_path / "repeat", resolved.subject)
    with pytest.raises(simulation_runner.SimulationBoundaryError, match="not promoted"):
        simulation_runner._mesh_workspace_harmonized(
            workspace,
            subject=resolved.subject,
            force_mesh=False,
            paths=paths,
        )


def test_seeded_fixed_config_uses_the_initialized_campaign_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, promotion = _promoted_subject_layout(tmp_path)
    monkeypatch.setattr(simulation_runner, "_validated_promotion", lambda _paths: promotion)
    config = _valid_config(paths, promotion)
    config["conditions"] = [
        {"name": "fixed_mesh", "mesh_mode": "fixed_mesh", "repeat_count": 40}
    ]
    runtime = config.pop(simulation_runner.CONFIG_RUNTIME_KEY)
    config_path = paths.left_root / "_pipeline/configs/fixed_mesh_spherical_median.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    campaign = paths.left_root / "_simnibs326/campaign.json"
    campaign.parent.mkdir(parents=True)
    campaign.write_text(json.dumps({"target": "left-hippocampus", "runtime": runtime}))

    simulation_runner._validate_campaign_config(config_path, paths=paths)

    runtime["headmodel_manifest_sha256"] = "f" * 64
    campaign.write_text(json.dumps({"target": "left-hippocampus", "runtime": runtime}))
    with pytest.raises(SystemExit, match="promotion/runtime provenance differs"):
        simulation_runner._validate_campaign_config(config_path, paths=paths)


@pytest.mark.parametrize(
    "conditions",
    [
        [{"name": "remesh", "mesh_mode": "remesh", "repeat_count": 39}],
        [{"name": "fixed_mesh", "mesh_mode": "remesh", "repeat_count": 40}],
        [
            {"name": "remesh", "mesh_mode": "remesh", "repeat_count": 40},
            {"name": "fixed_mesh", "mesh_mode": "fixed_mesh", "repeat_count": 40},
        ],
    ],
)
def test_task_config_rejects_changed_condition_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conditions: list[dict[str, object]],
) -> None:
    paths, _, promotion = _promoted_subject_layout(tmp_path)
    monkeypatch.setattr(simulation_runner, "_validated_promotion", lambda _paths: promotion)
    config = _valid_config(paths, promotion)
    config["conditions"] = conditions
    path = tmp_path / "changed-scope.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SystemExit, match="condition scope"):
        simulation_runner._validate_campaign_config(path, paths=paths)


def _promoted_subject_layout(tmp_path: Path):
    defaults = campaign_paths_from_environment({})
    paths = replace(
        defaults,
        corrected_root=(tmp_path / "corrected").resolve(),
        legacy_reference_root=(tmp_path / "legacy").resolve(),
        atlas_root=(tmp_path / "atlas").resolve(),
        headmodel_root=(tmp_path / "headmodels").resolve(),
        left_root=(tmp_path / "new-left").resolve(),
        right_root=(tmp_path / "new-right").resolve(),
        completed_v3_left_root=(tmp_path / "old-v3-left").resolve(),
        completed_v3_right_root=(tmp_path / "old-v3-right").resolve(),
        completed_v4_left_root=(tmp_path / "old-v4-left").resolve(),
        completed_v4_right_root=(tmp_path / "old-v4-right").resolve(),
    )
    resolved = subject_paths(paths, SUBJECTS[0])
    resolved.staged_m2m.mkdir(parents=True)
    for name, content in (
        ("T1fs_conform.nii.gz", b"t1"),
        ("T1fs_nu_conform.nii.gz", b"nu"),
    ):
        (resolved.staged_m2m / name).write_bytes(content)
    (resolved.staged_m2m / "eeg_positions").mkdir()
    (resolved.staged_m2m / "eeg_positions" / "cap.csv").write_bytes(b"cap")
    (resolved.staged_m2m / "toMNI").mkdir()
    (resolved.staged_m2m / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    surfaces = []
    for name in ("wm", "gm", "csf", "bone", "skin", "eyes"):
        path = resolved.staged_m2m / f"{name}.stl"
        path.write_bytes(f"new-{name}".encode())
        surfaces.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    resolved.harmonized_label.parent.mkdir(parents=True)
    resolved.harmonized_label.write_bytes(b"harmonized")
    resolved.head_mesh.parent.mkdir(parents=True, exist_ok=True)
    resolved.head_mesh.write_bytes(b"head-mesh")
    resolved.headmodel_receipt.parent.mkdir(parents=True)
    resolved.headmodel_receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject": resolved.subject,
                "harmonized_map": {
                    "path": str(resolved.harmonized_label.resolve()),
                    "sha256": _sha256(resolved.harmonized_label),
                },
                "mesh": {
                    "path": str(resolved.head_mesh.resolve()),
                    "sha256": _sha256(resolved.head_mesh),
                },
                "surfaces": surfaces,
            }
        ),
        encoding="utf-8",
    )
    promotion_file = paths.headmodel_root / "_pipeline" / "headmodels" / "promoted.json"
    promotion_file.parent.mkdir(parents=True, exist_ok=True)
    promotion_file.write_text(b"promoted".decode(), encoding="utf-8")
    promotion = {
        "manifest_sha256": "1" * 64,
        "promotion_sha256": _sha256(promotion_file),
        "promotion_path": str(promotion_file.resolve()),
    }
    return paths, resolved, promotion


def _workspace(root: Path, subject: str):
    anat = root / subject / "anat"
    return SimpleNamespace(
        root=root / subject,
        anat_dir=anat,
        output_root=anat / "SimNIBS",
        mesh_dir=anat / f"m2m_{subject}",
        mesh_path=anat / f"m2m_{subject}" / f"{subject}.msh",
    )


def _valid_config(paths, promotion):
    return {
        "source_root": str(paths.corrected_root),
        "experiment_root": str(paths.left_root),
        "subjects": list(SUBJECTS),
        "conditions": [
            {"name": "remesh", "mesh_mode": "remesh", "repeat_count": 40}
        ],
        simulation_runner.CONFIG_RUNTIME_KEY: {
            "simnibs_module": "SimNIBS/3.2.6-foss-2023a",
            "backend": simulation_runner.BACKEND_NAME,
            "headmodel_root": str(paths.headmodel_root),
            "headmodel_manifest_sha256": promotion["manifest_sha256"],
            "headmodel_promotion_sha256": promotion["promotion_sha256"],
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
