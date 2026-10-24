from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

import simnibs_326_corrected_label_repeatability.reference_stage as stage_module
from simnibs_326_corrected_label_repeatability.reference_stage import (
    ReferenceStageError,
    collect_reference_stage,
    stage_subject_reference,
    validate_collected_reference,
    validate_staged_reference,
)
from simnibs_326_corrected_label_repeatability.settings import (
    SUBJECTS,
    campaign_paths_from_environment,
    subject_paths,
)


def test_allowlist_stages_only_coordinate_reference_files(tmp_path: Path) -> None:
    _, resolved = _reference_layout(tmp_path)

    receipt = stage_subject_reference(resolved)

    copied = sorted(
        path.relative_to(resolved.staged_m2m).as_posix()
        for path in resolved.staged_m2m.rglob("*")
        if path.is_file()
    )
    assert copied == [
        "T1fs_conform.nii.gz",
        "T1fs_nu_conform.nii.gz",
        "eeg_positions/EEG10-10_UI_Jurak_2007.csv",
        "eeg_positions/nested/notes.txt",
        "legacy_scaffold_ready.provenance.json",
        "toMNI/MNI2conform_6DOF.txt",
        "toMNI/warp.nii.gz",
    ]
    forbidden_tokens = ("segment", "mask_prep", ".msh", ".stl", ".off", "control")
    assert not any(token in entry for entry in copied for token in forbidden_tokens)
    assert receipt.subject == resolved.subject
    assert receipt.source_root == str(resolved.legacy_m2m.resolve())
    assert receipt.destination_root == str(resolved.staged_m2m.resolve())
    assert receipt.file_count == len(copied)
    payload = json.loads(resolved.reference_receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["subject"] == resolved.subject
    assert all(item["source_sha256"] == item["destination_sha256"] for item in payload["files"])


def test_symlink_anywhere_in_allowlist_is_rejected(tmp_path: Path) -> None:
    _, resolved = _reference_layout(tmp_path)
    target = resolved.legacy_m2m / "outside.txt"
    target.write_text("target", encoding="utf-8")
    (resolved.legacy_m2m / "eeg_positions" / "unsafe-link").symlink_to(target)

    with pytest.raises(ReferenceStageError, match="Symlink"):
        stage_subject_reference(resolved)

    assert not resolved.staged_m2m.exists()


def test_escape_from_legacy_subject_root_is_rejected(tmp_path: Path) -> None:
    _, resolved = _reference_layout(tmp_path)
    outside = tmp_path / "outside-m2m"
    outside.mkdir()
    escaped = replace(resolved, legacy_m2m=outside)

    with pytest.raises(ReferenceStageError, match="escapes"):
        stage_subject_reference(escaped)


def test_source_changed_after_first_hash_is_not_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, resolved = _reference_layout(tmp_path)
    original_copy = stage_module._copy_file
    changed = False

    def mutating_copy(source: Path, destination: Path) -> None:
        nonlocal changed
        original_copy(source, destination)
        if not changed:
            source.write_bytes(source.read_bytes() + b"changed")
            changed = True

    monkeypatch.setattr(stage_module, "_copy_file", mutating_copy)

    with pytest.raises(ReferenceStageError, match="changed"):
        stage_subject_reference(resolved)

    assert not resolved.staged_m2m.exists()
    assert not resolved.reference_receipt.exists()


def test_incomplete_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    _, resolved = _reference_layout(tmp_path)
    resolved.staged_m2m.mkdir(parents=True)
    sentinel = resolved.staged_m2m / "keep-me.txt"
    sentinel.write_text("partial", encoding="utf-8")

    with pytest.raises(ReferenceStageError, match="incomplete"):
        stage_subject_reference(resolved)

    assert sentinel.read_text(encoding="utf-8") == "partial"
    assert not resolved.reference_receipt.exists()


def test_forged_receipt_prevents_validation_and_reuse(tmp_path: Path) -> None:
    _, resolved = _reference_layout(tmp_path)
    stage_subject_reference(resolved)
    payload = json.loads(resolved.reference_receipt.read_text(encoding="utf-8"))
    payload["files"][0]["destination_sha256"] = "0" * 64
    resolved.reference_receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReferenceStageError, match="hash mismatch"):
        validate_staged_reference(resolved)
    with pytest.raises(ReferenceStageError, match="incomplete or invalid"):
        stage_subject_reference(resolved)


def test_cohort_collection_requires_all_ten_valid_receipts(tmp_path: Path) -> None:
    paths = None
    for subject in SUBJECTS[:-1]:
        paths, resolved = _reference_layout(tmp_path, subject)
        stage_subject_reference(resolved)
    assert paths is not None
    collection = paths.headmodel_root / "_pipeline" / "reference_stage_collection.json"

    with pytest.raises(ReferenceStageError, match=SUBJECTS[-1]):
        collect_reference_stage(paths)
    assert not collection.exists()

    paths, resolved = _reference_layout(tmp_path, SUBJECTS[-1])
    stage_subject_reference(resolved)
    result = collect_reference_stage(paths)

    assert result["status"] == "complete"
    assert result["subject_count"] == 10
    assert [entry["subject"] for entry in result["subjects"]] == list(SUBJECTS)
    assert collection.is_file()


def test_collected_stage_remains_valid_after_legacy_root_is_removed(tmp_path: Path) -> None:
    paths = None
    for subject in SUBJECTS:
        paths, resolved = _reference_layout(tmp_path, subject)
        stage_subject_reference(resolved)
    assert paths is not None
    collect_reference_stage(paths)
    shutil.rmtree(paths.legacy_reference_root)

    receipt = validate_collected_reference(subject_paths(paths, SUBJECTS[0]))

    assert receipt.subject == SUBJECTS[0]
    assert receipt.file_count == 7

    staged_receipt = subject_paths(paths, SUBJECTS[0]).reference_receipt
    staged_receipt.write_text(staged_receipt.read_text() + "tampered", encoding="utf-8")
    with pytest.raises(ReferenceStageError, match="receipt hash"):
        validate_collected_reference(subject_paths(paths, SUBJECTS[0]))


def test_external_collection_contains_only_manifest_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = "sub-CC999999"
    manifest = tmp_path / "replacement_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "simnibs_subject_replacement",
                "profile": "external1",
                "status": "prepared",
                "subject": subject,
                "subjects": [subject],
                    "paths": {"manifest": str(manifest.resolve())},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SIMNIBS326H_COHORT", "external1")
    monkeypatch.setenv("SIMNIBS326H_EXTERNAL_MANIFEST", str(manifest))
    paths, resolved = _reference_layout(tmp_path, subject)
    stage_subject_reference(resolved)

    collected = collect_reference_stage(paths)
    validated = validate_collected_reference(resolved)

    assert collected["subject_count"] == 1
    assert [entry["subject"] for entry in collected["subjects"]] == [subject]
    assert validated.subject == subject


def _reference_layout(tmp_path: Path, subject: str = SUBJECTS[0]):
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
    legacy = resolved.legacy_m2m
    (legacy / "eeg_positions" / "nested").mkdir(parents=True)
    (legacy / "toMNI").mkdir()
    (legacy / "segment").mkdir()
    (legacy / "mask_prep").mkdir()
    (legacy / "T1fs_conform.nii.gz").write_bytes(b"conform")
    (legacy / "T1fs_nu_conform.nii.gz").write_bytes(b"nu-conform")
    (legacy / "eeg_positions" / "EEG10-10_UI_Jurak_2007.csv").write_text(
        "name,x,y,z\nC3,1,2,3\n", encoding="utf-8"
    )
    (legacy / "eeg_positions" / "nested" / "notes.txt").write_text(
        "reference", encoding="utf-8"
    )
    (legacy / "toMNI" / "MNI2conform_6DOF.txt").write_text(
        "matrix", encoding="utf-8"
    )
    (legacy / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    (legacy / "segment" / "MASK_WM.nii.gz").write_bytes(b"forbidden")
    (legacy / "mask_prep" / "MASK_GM.nii.gz").write_bytes(b"forbidden")
    (legacy / f"{subject}.msh").write_bytes(b"old mesh")
    (legacy / "wm.stl").write_bytes(b"old surface")
    (legacy / "gm.off").write_bytes(b"old surface")
    (legacy / "mesh_control.nii.gz").write_bytes(b"old control")
    resolved.legacy_ready_marker.write_text(
        json.dumps({"status": "ready", "subject": subject}) + "\n",
        encoding="utf-8",
    )
    return paths, resolved
