from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from simnibs_326_repeatability import scaffold_core


SUBJECT = "sub-CC510415"
MODULE = "SimNIBS/3.2.6-foss-2023a"
PATCH_ID = "test-cat12-patch"


def _request(tmp_path: Path) -> scaffold_core.ScaffoldRequest:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    t1 = inputs / f"{SUBJECT}_T1w.nii"
    t2 = inputs / f"{SUBJECT}_T2w.nii.gz"
    labels = inputs / f"{SUBJECT}_labels.nii.gz"
    t1.write_bytes(b"t1")
    t2.write_bytes(b"t2")
    labels.write_bytes(b"labels")
    compat = tmp_path / "cat12_compat.py"
    compat.write_text("# compatibility entry point\n", encoding="utf-8")
    return scaffold_core.ScaffoldRequest(
        subject=SUBJECT,
        t1=t1,
        t2=t2,
        corrected_labels=labels,
        scaffold_root=tmp_path / "scaffolds",
        simnibs_module=MODULE,
        cat12_compat_script=compat,
        cat12_patch_id=PATCH_ID,
        head_model_strategy="headreco-all-scaffold-then-fresh-volumemesh",
        segmentation_provenance="test provenance",
        workflow="test-subject-scaffold",
    )


def _complete_fake_headreco(request: scaffold_core.ScaffoldRequest) -> None:
    paths = scaffold_core.scaffold_paths(request)
    paths.m2m.mkdir(parents=True, exist_ok=True)
    paths.native_mesh.write_bytes(b"mesh")
    (paths.m2m / "T1fs_conform.nii.gz").write_bytes(b"conform")
    (paths.m2m / "T1fs_nu_conform.nii.gz").write_bytes(b"nu")
    (paths.m2m / "toMNI").mkdir()
    (paths.m2m / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    (paths.m2m / "eeg_positions").mkdir()
    (paths.m2m / "eeg_positions" / "positions.csv").write_text(
        "C3,1,2,3\n", encoding="utf-8"
    )
    for index in range(5):
        (paths.m2m / f"surface_{index}.stl").write_bytes(b"solid surface")
    (paths.m2m / "diagnostic_intermediate.nii.gz").write_bytes(b"keep me")


def test_build_runs_exact_headreco_command_and_reuses_ready_scaffold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command, *, cwd, check):
        calls.append((command, Path(cwd), check))
        _complete_fake_headreco(request)

    monkeypatch.setattr(scaffold_core.subprocess, "run", fake_run)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    monkeypatch.setenv("LOADEDMODULES", "MATLAB/2023b:SimNIBS/3.2.6-foss-2023a")

    result = scaffold_core.build_scaffold(request)
    paths = scaffold_core.scaffold_paths(request)
    expected_command = [
        sys.executable,
        str(request.cat12_compat_script),
        "headreco",
        "all",
        "--noclean",
        SUBJECT,
        str(request.t1),
        str(request.t2),
    ]
    assert calls == [(expected_command, paths.anat, True)]
    assert result["status"] == "ready"
    assert result["action"] == "built"
    assert (paths.m2m / "diagnostic_intermediate.nii.gz").read_bytes() == b"keep me"
    assert paths.native_mesh.read_bytes() == paths.compatibility_mesh.read_bytes()

    marker = json.loads(paths.ready_marker.read_text(encoding="utf-8"))
    assert marker["command"] == expected_command
    assert marker["simnibs_module"] == MODULE
    assert marker["simnibs_version"] == "3.2.6"
    assert marker["cat12_compatibility_patch"] == PATCH_ID
    assert marker["source_t1_sha256"] == scaffold_core.sha256_file(request.t1)
    assert marker["source_t2_sha256"] == scaffold_core.sha256_file(request.t2)
    assert marker["corrected_v4_labels_sha256"] == scaffold_core.sha256_file(
        request.corrected_labels
    )
    assert marker["native_mesh_sha256"] == scaffold_core.sha256_file(paths.native_mesh)
    assert marker["compatibility_mesh_sha256"] == scaffold_core.sha256_file(
        paths.compatibility_mesh
    )

    reused = scaffold_core.build_scaffold(request)
    assert reused["status"] == "ready"
    assert reused["action"] == "reused"
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["t1", "t2", "corrected_labels"])
@pytest.mark.parametrize("failure", ["missing", "empty"])
def test_build_rejects_missing_or_empty_inputs(
    tmp_path: Path, field: str, failure: str
) -> None:
    request = _request(tmp_path)
    path = getattr(request, field)
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"")
    with pytest.raises((FileNotFoundError, ValueError), match="input"):
        scaffold_core.build_scaffold(request)


def test_validate_requires_complete_nonempty_scaffold(tmp_path: Path) -> None:
    request = _request(tmp_path)
    _complete_fake_headreco(request)
    paths = scaffold_core.scaffold_paths(request)
    paths.compatibility_mesh.write_bytes(paths.native_mesh.read_bytes())
    paths.ready_marker.write_text("{}\n", encoding="utf-8")

    result = scaffold_core.validate_scaffold(request)
    assert result["status"] == "incomplete"
    assert result["checks"]["ready_marker_binding"] is False

    (paths.m2m / "surface_4.stl").unlink()
    assert scaffold_core.validate_scaffold(request)["checks"]["surface_meshes"] is False

    (paths.m2m / "surface_4.stl").write_bytes(b"solid")
    (paths.m2m / "toMNI" / "warp.nii.gz").unlink()
    assert scaffold_core.validate_scaffold(request)["checks"]["to_mni"] is False

    (paths.m2m / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    (paths.m2m / "eeg_positions" / "positions.csv").unlink()
    assert scaffold_core.validate_scaffold(request)["checks"]["eeg_positions"] is False


def test_validation_detects_stale_inputs_and_different_meshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)

    def fake_run(command, *, cwd, check):
        _complete_fake_headreco(request)

    monkeypatch.setattr(scaffold_core.subprocess, "run", fake_run)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    scaffold_core.build_scaffold(request)
    paths = scaffold_core.scaffold_paths(request)

    request.t1.write_bytes(b"changed t1")
    stale = scaffold_core.validate_scaffold(request)
    assert stale["status"] == "incomplete"
    assert stale["checks"]["ready_marker_binding"] is False

    request.t1.write_bytes(b"t1")
    paths.compatibility_mesh.write_bytes(b"different mesh")
    mismatched = scaffold_core.validate_scaffold(request)
    assert mismatched["status"] == "incomplete"
    assert mismatched["checks"]["mesh_copies_match"] is False


def test_validation_allows_a_different_python_interpreter_but_not_command_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)

    def fake_run(command, *, cwd, check):
        _complete_fake_headreco(request)

    monkeypatch.setattr(scaffold_core.subprocess, "run", fake_run)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    scaffold_core.build_scaffold(request)
    marker_path = scaffold_core.scaffold_paths(request).ready_marker
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    marker["command"][0] = "/different/module/python"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert scaffold_core.validate_scaffold(request)["status"] == "ready"

    marker["command"][2] = "not-headreco"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    drifted = scaffold_core.validate_scaffold(request)
    assert drifted["status"] == "incomplete"
    assert drifted["checks"]["ready_marker_binding"] is False


def test_build_fails_when_headreco_does_not_create_native_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)

    def fake_run(command, *, cwd, check):
        paths = scaffold_core.scaffold_paths(request)
        paths.m2m.mkdir(parents=True)

    monkeypatch.setattr(scaffold_core.subprocess, "run", fake_run)
    with pytest.raises(FileNotFoundError, match="native mesh"):
        scaffold_core.build_scaffold(request)
