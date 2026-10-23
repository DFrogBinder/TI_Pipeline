from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

import simnibs_326_corrected_label_repeatability.headmodels as headmodel_module
from simnibs_326_corrected_label_repeatability.headmodels import (
    HeadModelError,
    build_headmodel,
    collect_headmodels,
    headmodel_output_dir,
    promote_headmodels,
    validate_headmodel_manifest,
    write_qc_request,
)
from simnibs_326_corrected_label_repeatability.reference_stage import StageReceipt
from simnibs_326_corrected_label_repeatability.settings import (
    BASE9_SUBJECTS,
    SUBJECTS,
    campaign_paths_from_environment,
    subject_paths,
)
from simnibs_326_corrected_label_repeatability.v3_meshing import (
    MeshValidation,
    SurfaceResult,
)


_REAL_HARMONIZE = headmodel_module.harmonize_subject_labels
_REAL_MASKS = headmodel_module.temporary_surface_masks
_TI_POST_PYTHON = Path("/home/boyan/anaconda3/envs/simnibs_post/bin/python3.12")


def test_build_lifecycle_orders_inputs_meshing_validation_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved = _headmodel_layout(tmp_path)
    events: list[str] = []
    masks_seen: list[Path] = []
    _install_fake_lifecycle(monkeypatch, resolved, events, masks_seen)

    result = build_headmodel(paths, resolved.subject, SimpleNamespace())

    assert result["action"] == "built"
    assert _ordered(events, [
        "stage",
        "harmonize",
        "masks_enter",
        "probe",
        "surfaces",
        "volume",
        "validate_mesh",
        "finite_nodes",
        "marker",
        "masks_exit",
    ])
    assert all(not path.exists() for path in masks_seen)
    assert list((paths.headmodel_root / "_pipeline" / "scratch" / resolved.subject).iterdir()) == []
    assert resolved.headmodel_receipt.is_file()


def test_marker_binds_every_scientific_and_software_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved = _headmodel_layout(tmp_path)
    _install_fake_lifecycle(monkeypatch, resolved, [], [])

    build_headmodel(paths, resolved.subject, SimpleNamespace())
    marker = json.loads(resolved.headmodel_receipt.read_text(encoding="utf-8"))

    assert marker["status"] == "complete"
    assert marker["subject"] == resolved.subject
    assert marker["source"]["path"] == str(resolved.corrected_label.resolve())
    assert marker["source"]["sha256"] == _sha256(resolved.corrected_label)
    assert marker["harmonized_map"]["path"] == str(resolved.harmonized_label.resolve())
    assert marker["harmonized_map"]["sha256"] == _sha256(resolved.harmonized_label)
    assert marker["staged_reference"]["receipt_path"] == str(
        resolved.reference_receipt.resolve()
    )
    assert marker["staged_reference"]["receipt_sha256"] == _sha256(
        resolved.reference_receipt
    )
    assert {record["name"] for record in marker["surfaces"]} >= {
        "wm.stl",
        "gm.stl",
        "csf.stl",
        "bone.stl",
        "skin.stl",
    }
    assert all(record["sha256"] for record in marker["surfaces"])
    assert marker["mesh"]["path"] == str(resolved.head_mesh.resolve())
    assert marker["mesh"]["sha256"] == _sha256(resolved.head_mesh)
    assert marker["api_probe"]["simnibs_version"] == "3.2.6"
    assert marker["api_probe_sha256"] == _json_sha256(marker["api_probe"])
    assert len(marker["git_commit"]) >= 7
    assert marker["legacy_hash_comparison"]["mesh_reused"] is False
    assert marker["legacy_hash_comparison"]["surface_hashes_reused"] == []


def test_retry_reuses_only_a_fully_hash_valid_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved = _headmodel_layout(tmp_path)
    events: list[str] = []
    _install_fake_lifecycle(monkeypatch, resolved, events, [])
    build_headmodel(paths, resolved.subject, SimpleNamespace())
    events.clear()

    result = build_headmodel(paths, resolved.subject, SimpleNamespace())

    assert result["action"] == "reused"
    assert "harmonize" not in events
    assert "surfaces" not in events
    assert "volume" not in events
    assert "validate_mesh" in events

    surface = resolved.staged_m2m / "wm.stl"
    surface.write_bytes(surface.read_bytes() + b"tampered")
    with pytest.raises(HeadModelError, match="existing completion marker"):
        build_headmodel(paths, resolved.subject, SimpleNamespace())
    assert events.count("surfaces") == 0


def test_retry_rebuilds_owned_partial_outputs_without_touching_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved = _headmodel_layout(tmp_path)
    partial = resolved.staged_m2m / "wm.stl"
    partial.write_bytes(b"partial")
    resolved.head_mesh.parent.mkdir(parents=True, exist_ok=True)
    resolved.head_mesh.write_bytes(b"partial mesh")
    reference_before = (resolved.staged_m2m / "T1fs_conform.nii.gz").read_bytes()
    events: list[str] = []
    _install_fake_lifecycle(monkeypatch, resolved, events, [])

    result = build_headmodel(paths, resolved.subject, SimpleNamespace())

    assert result["action"] == "built"
    assert partial.read_bytes() == b"new-wm"
    assert resolved.head_mesh.read_bytes() == b"new-mesh"
    assert (resolved.staged_m2m / "T1fs_conform.nii.gz").read_bytes() == reference_before


def test_build_rejects_a_surface_or_mesh_reused_from_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, resolved = _headmodel_layout(tmp_path)
    legacy_mesh = resolved.legacy_anat / f"{resolved.subject}.msh"
    legacy_mesh.write_bytes(b"new-mesh")
    _install_fake_lifecycle(monkeypatch, resolved, [], [])

    with pytest.raises(HeadModelError, match="legacy mesh"):
        build_headmodel(paths, resolved.subject, SimpleNamespace())

    assert not resolved.headmodel_receipt.exists()


def test_renderer_is_lazy_hash_bound_and_draws_three_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _TI_POST_PYTHON.is_file():
        pytest.skip("local ti-post rendering interpreter is unavailable")
    paths, resolved = _headmodel_layout(tmp_path)
    _install_fake_lifecycle(monkeypatch, resolved, [], [])
    build_headmodel(paths, resolved.subject, SimpleNamespace())
    request = write_qc_request(paths, resolved.subject)
    payload = json.loads(request.read_text(encoding="utf-8"))
    request_hash = payload.pop("request_sha256")
    assert request_hash == _json_sha256(payload)
    assert payload["views"] == ["sagittal", "coronal", "axial"]
    assert payload["t1"]["sha256"] == _sha256(resolved.staged_m2m / "T1fs_conform.nii.gz")
    assert payload["labels"]["sha256"] == _sha256(resolved.harmonized_label)

    code = (
        "import sys; "
        "assert 'matplotlib' not in sys.modules; "
        "from simnibs_326_corrected_label_repeatability.render_qc import render_qc_request; "
        "assert 'matplotlib' not in sys.modules; "
        f"render_qc_request({str(request)!r})"
    )
    scripts_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [str(_TI_POST_PYTHON), "-c", code],
        cwd=scripts_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert resolved.harmonization_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    width, height = _png_dimensions(resolved.harmonization_figure)
    assert width > height * 2


def test_tampered_render_request_fails_without_writing_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _TI_POST_PYTHON.is_file():
        pytest.skip("local ti-post rendering interpreter is unavailable")
    paths, resolved = _headmodel_layout(tmp_path)
    _install_fake_lifecycle(monkeypatch, resolved, [], [])
    build_headmodel(paths, resolved.subject, SimpleNamespace())
    request = write_qc_request(paths, resolved.subject)
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["slices"]["axial"] += 1
    request.write_text(json.dumps(payload), encoding="utf-8")
    scripts_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            str(_TI_POST_PYTHON),
            "-m",
            "simnibs_326_corrected_label_repeatability.render_qc",
            str(request),
        ],
        cwd=scripts_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "request hash" in completed.stderr.lower()
    assert not resolved.harmonization_figure.exists()


def test_collector_requires_ten_markers_and_panels_then_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, subjects = _build_fake_cohort(tmp_path, monkeypatch)
    monkeypatch.setattr(headmodel_module, "load_v3_meshing_api", lambda: SimpleNamespace())
    last_subject = SUBJECTS[-1]

    def incomplete_renderer(_python: str, request: Path) -> None:
        payload = json.loads(request.read_text(encoding="utf-8"))
        if payload["subject"] != last_subject:
            _write_fake_png(Path(payload["output_png"]))

    monkeypatch.setattr(headmodel_module, "_run_renderer", incomplete_renderer)
    manifest_path = paths.headmodel_root / "_pipeline" / "headmodels" / "manifest.json"
    with pytest.raises(HeadModelError, match=last_subject):
        collect_headmodels(paths, "/fake/ti-post/python")
    assert not manifest_path.exists()

    def complete_renderer(_python: str, request: Path) -> None:
        payload = json.loads(request.read_text(encoding="utf-8"))
        _write_fake_png(Path(payload["output_png"]))

    monkeypatch.setattr(headmodel_module, "_run_renderer", complete_renderer)
    result = collect_headmodels(paths, "/fake/ti-post/python")

    assert result["status"] == "complete"
    assert result["subject_count"] == 10
    assert result["manifest_sha256"] == _sha256(manifest_path)
    assert (manifest_path.parent / "headmodels.tsv").read_text(encoding="utf-8").count("\n") == 11
    assert (manifest_path.parent / "checksums.sha256").is_file()
    assert not list(paths.headmodel_root.rglob("*_T1w_ras_1mm_T1andT2_masks.nii*"))
    validated = validate_headmodel_manifest(paths)
    assert validated["manifest_sha256"] == result["manifest_sha256"]
    assert [entry["subject"] for entry in validated["subjects"]] == list(SUBJECTS)
    assert all(subject.harmonization_figure.is_file() for subject in subjects)


def test_promotion_requires_exact_manifest_hash_and_explicit_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _build_fake_cohort(tmp_path, monkeypatch)
    monkeypatch.setattr(headmodel_module, "load_v3_meshing_api", lambda: SimpleNamespace())

    def renderer(_python: str, request: Path) -> None:
        payload = json.loads(request.read_text(encoding="utf-8"))
        _write_fake_png(Path(payload["output_png"]))

    monkeypatch.setattr(headmodel_module, "_run_renderer", renderer)
    collected = collect_headmodels(paths, "/fake/ti-post/python")
    promotion = paths.headmodel_root / "_pipeline" / "headmodels" / "promoted.json"

    with pytest.raises(HeadModelError, match="explicit"):
        promote_headmodels(paths, collected["manifest_sha256"], False)
    assert not promotion.exists()
    with pytest.raises(HeadModelError, match="manifest hash"):
        promote_headmodels(paths, "0" * 64, True)
    assert not promotion.exists()

    result = promote_headmodels(paths, collected["manifest_sha256"], True)
    assert result["status"] == "promoted"
    assert result["manifest_sha256"] == collected["manifest_sha256"]
    assert result["panels_reviewed"] == 10
    assert promotion.is_file()


def test_base9_collection_and_promotion_use_separate_immutable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _build_fake_cohort(tmp_path, monkeypatch)
    monkeypatch.setattr(headmodel_module, "load_v3_meshing_api", lambda: SimpleNamespace())

    def renderer(_python: str, request: Path) -> None:
        payload = json.loads(request.read_text(encoding="utf-8"))
        _write_fake_png(Path(payload["output_png"]))

    monkeypatch.setattr(headmodel_module, "_run_renderer", renderer)

    collected = collect_headmodels(
        paths,
        "/fake/ti-post/python",
        subjects=BASE9_SUBJECTS,
    )
    output_dir = paths.headmodel_root / "_pipeline" / "headmodels" / "base9"
    manifest = output_dir / "manifest.json"

    assert headmodel_output_dir(paths, BASE9_SUBJECTS) == output_dir
    assert collected["subject_count"] == 9
    assert collected["subject_order"] == list(BASE9_SUBJECTS)
    assert manifest.is_file()
    assert not (paths.headmodel_root / "_pipeline/headmodels/manifest.json").exists()
    assert all(
        Path(entry["artifacts"][-1]["path"]).parent == output_dir / "qc_panels"
        for entry in collected["subjects"]
    )
    validated = validate_headmodel_manifest(paths, subjects=BASE9_SUBJECTS)
    assert validated["manifest_sha256"] == collected["manifest_sha256"]

    promoted = promote_headmodels(
        paths,
        collected["manifest_sha256"],
        True,
        subjects=BASE9_SUBJECTS,
    )
    assert promoted["subjects"] == list(BASE9_SUBJECTS)
    assert promoted["panels_reviewed"] == 9
    assert (output_dir / "promoted.json").is_file()
    assert not (paths.headmodel_root / "_pipeline/headmodels/promoted.json").exists()


def test_tampered_panel_invalidates_manifest_and_blocks_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, subjects = _build_fake_cohort(tmp_path, monkeypatch)
    monkeypatch.setattr(headmodel_module, "load_v3_meshing_api", lambda: SimpleNamespace())

    def renderer(_python: str, request: Path) -> None:
        payload = json.loads(request.read_text(encoding="utf-8"))
        _write_fake_png(Path(payload["output_png"]))

    monkeypatch.setattr(headmodel_module, "_run_renderer", renderer)
    collected = collect_headmodels(paths, "/fake/ti-post/python")
    subjects[0].harmonization_figure.write_bytes(b"tampered")
    promotion = paths.headmodel_root / "_pipeline" / "headmodels" / "promoted.json"

    with pytest.raises(HeadModelError, match="hash mismatch"):
        validate_headmodel_manifest(paths)
    with pytest.raises(HeadModelError, match="hash mismatch"):
        promote_headmodels(paths, collected["manifest_sha256"], True)
    assert not promotion.exists()


def _headmodel_layout(tmp_path: Path, subject: str = SUBJECTS[0]):
    defaults = campaign_paths_from_environment({})
    paths = replace(
        defaults,
        corrected_root=(tmp_path / "corrected").resolve(),
        legacy_reference_root=(tmp_path / "legacy").resolve(),
        atlas_root=(tmp_path / "atlas").resolve(),
        headmodel_root=(tmp_path / "headmodels").resolve(),
        left_root=(tmp_path / "left").resolve(),
        right_root=(tmp_path / "right").resolve(),
        completed_v3_left_root=(tmp_path / "old-v3-left").resolve(),
        completed_v3_right_root=(tmp_path / "old-v3-right").resolve(),
        completed_v4_left_root=(tmp_path / "old-v4-left").resolve(),
        completed_v4_right_root=(tmp_path / "old-v4-right").resolve(),
    )
    resolved = subject_paths(paths, subject)
    resolved.corrected_label.parent.mkdir(parents=True)
    resolved.legacy_m2m.mkdir(parents=True)
    resolved.staged_m2m.mkdir(parents=True)
    source = np.zeros((6, 6, 6), dtype=np.uint8)
    for coordinate, label in zip(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)],
        [1, 2, 3, 7, 5, 6],
        strict=True,
    ):
        source[coordinate] = label
    source[1, 1, 0] = 9
    nib.save(nib.Nifti1Image(source, np.eye(4)), resolved.corrected_label)
    nib.save(
        nib.Nifti1Image(np.ones(source.shape, dtype=np.float32), np.eye(4)),
        resolved.corrected_t1,
    )
    for name in ("T1fs_conform.nii.gz", "T1fs_nu_conform.nii.gz"):
        nib.save(
            nib.Nifti1Image(np.ones(source.shape, dtype=np.float32), np.eye(4)),
            resolved.staged_m2m / name,
        )
    (resolved.staged_m2m / "eeg_positions").mkdir()
    (resolved.staged_m2m / "toMNI").mkdir()
    resolved.reference_receipt.parent.mkdir(parents=True)
    resolved.reference_receipt.write_text(
        json.dumps({"status": "complete", "subject": subject}) + "\n",
        encoding="utf-8",
    )
    (resolved.legacy_m2m / "wm.stl").write_bytes(b"old-wm")
    (resolved.legacy_anat / f"{subject}.msh").write_bytes(b"old-mesh")
    return paths, resolved


def _install_fake_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    resolved,
    events: list[str],
    masks_seen: list[Path],
) -> None:
    def fake_stage(_resolved):
        events.append("stage")
        return StageReceipt(
            subject=resolved.subject,
            source_root=str(resolved.legacy_m2m),
            destination_root=str(resolved.staged_m2m),
            receipt_path=str(resolved.reference_receipt),
            file_count=4,
            byte_count=100,
        )

    def recording_harmonize(_resolved):
        events.append("harmonize")
        return _REAL_HARMONIZE(_resolved)

    @contextmanager
    def recording_masks(path, scratch):
        events.append("masks_enter")
        with _REAL_MASKS(path, scratch) as masks:
            masks_seen.extend(masks.values())
            yield masks
        events.append("masks_exit")

    def fake_probe(_api):
        events.append("probe")
        return {
            "status": "ready",
            "simnibs_version": "3.2.6",
            "module_files": {"headreco": "/fake/headreco.py"},
        }

    def fake_surfaces(_api, _masks, m2m, _scratch, vertex_density=0.5):
        events.append("surfaces")
        surfaces = []
        for name in ("wm", "gm", "csf", "bone", "skin", "eyes"):
            path = m2m / f"{name}.stl"
            path.write_bytes(f"new-{name}".encode())
            surfaces.append(path)
        return SurfaceResult(tuple(surfaces), (), vertex_density)

    def fake_volume(
        _api,
        _subject,
        _surfaces,
        _relabel,
        _m2m,
        output,
        keep_air=False,
    ):
        del keep_air
        events.append("volume")
        output.write_bytes(b"new-mesh")
        return _validation(output)

    def fake_validate(_api, output):
        events.append("validate_mesh")
        return _validation(output)

    def fake_nodes(_api, _output):
        events.append("finite_nodes")
        return 12

    def marker_hook(_path, _payload):
        events.append("marker")

    monkeypatch.setattr(headmodel_module, "validate_staged_reference", fake_stage)
    monkeypatch.setattr(headmodel_module, "harmonize_subject_labels", recording_harmonize)
    monkeypatch.setattr(headmodel_module, "temporary_surface_masks", recording_masks)
    monkeypatch.setattr(headmodel_module, "probe_v3_meshing_api", fake_probe)
    monkeypatch.setattr(headmodel_module, "build_surfaces", fake_surfaces)
    monkeypatch.setattr(headmodel_module, "build_volume_mesh", fake_volume)
    monkeypatch.setattr(headmodel_module, "validate_mesh", fake_validate)
    monkeypatch.setattr(headmodel_module, "_validate_finite_mesh_nodes", fake_nodes)
    monkeypatch.setattr(headmodel_module, "_after_marker_write", marker_hook)


def _validation(path: Path) -> MeshValidation:
    return MeshValidation(
        mesh_path=str(path.resolve()),
        tetrahedra=6,
        tissue_tags=(1, 2, 3, 4, 5, 6),
        minimum_tetrahedron_volume=0.5,
    )


def _ordered(events: list[str], expected: list[str]) -> bool:
    positions = [events.index(item) for item in expected]
    return positions == sorted(positions)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_fake_cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = None
    resolved_subjects = []
    for subject in SUBJECTS:
        paths, resolved = _headmodel_layout(tmp_path, subject)
        _install_fake_lifecycle(monkeypatch, resolved, [], [])
        build_headmodel(paths, subject, SimpleNamespace())
        resolved_subjects.append(resolved)
    assert paths is not None
    return paths, resolved_subjects


def _write_fake_png(path: Path) -> None:
    # Complete 1x1 RGBA PNG, sufficient for collector signature validation.
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63606060f80f0001040100f31c0d3b0000000049454e44"
        "ae426082"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
