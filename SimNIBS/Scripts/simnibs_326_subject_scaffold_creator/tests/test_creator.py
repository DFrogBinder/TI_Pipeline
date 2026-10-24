from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from simnibs_326_repeatability import scaffold_core
from simnibs_326_subject_scaffold_creator import creator
from simnibs_326_subject_scaffold_creator.settings import (
    DEFAULT_OUTPUT_ROOT,
    CreatorConfig,
    assert_output_root_isolation,
    creator_paths,
    production_paths,
    validate_subject,
)


def _config(layout: dict[str, Path | str]) -> CreatorConfig:
    return CreatorConfig(
        subject=str(layout["subject"]),
        corrected_source_root=Path(layout["source"]),
        output_root=Path(layout["output"]),
        final132_subjects_file=Path(layout["subjects"]),
    )


def _write_module_receipt(config: CreatorConfig) -> None:
    paths = creator_paths(config)
    paths.module_receipt.parent.mkdir(parents=True, exist_ok=True)
    paths.module_receipt.write_text(
        json.dumps(
            {
                "status": "ready",
                "expected_module": "SimNIBS/3.2.6-foss-2023a",
                "expected_matlab_module": "MATLAB/2023b",
                "cat12_compatibility": {
                    "status": "ready",
                    "patch_id": creator.PATCH_ID,
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_headreco(command, *, cwd, check) -> None:
    subject = command[-3]
    anat = Path(cwd)
    m2m = anat / f"m2m_{subject}"
    m2m.mkdir(parents=True, exist_ok=True)
    (anat / f"{subject}.msh").write_bytes(b"mesh")
    (m2m / "T1fs_conform.nii.gz").write_bytes(b"conform")
    (m2m / "T1fs_nu_conform.nii.gz").write_bytes(b"nu")
    (m2m / "toMNI").mkdir()
    (m2m / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    (m2m / "eeg_positions").mkdir()
    (m2m / "eeg_positions" / "positions.csv").write_text("C3,1,2,3\n")
    for index in range(5):
        (m2m / f"surface_{index}.stl").write_bytes(b"surface")
    (m2m / "retained_diagnostic.nii.gz").write_bytes(b"diagnostic")


def test_subject_validation_is_runtime_driven_and_exact(
    creator_layout: dict[str, Path | str],
) -> None:
    config = _config(creator_layout)
    assert validate_subject(config) == ("sub-CC110056", "sub-CC510415")
    assert config.subject == "sub-CC510415"
    assert CreatorConfig(subject=config.subject).output_root == DEFAULT_OUTPUT_ROOT

    with pytest.raises(ValueError, match="canonical"):
        validate_subject(replace(config, subject="../sub-CC510415"))
    with pytest.raises(ValueError, match="exactly once"):
        validate_subject(replace(config, subject="sub-CC999999"))

    config.final132_subjects_file.write_text(
        f"{config.subject}\n{config.subject}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_subject(config)
    config.final132_subjects_file.write_text(
        f"{config.subject}\nnot-a-subject\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-canonical"):
        validate_subject(config)


def test_preflight_resolves_copy_and_gzip_receipt_without_writing(
    creator_layout: dict[str, Path | str],
) -> None:
    config = _config(creator_layout)
    paths = creator_paths(config)

    result = creator.preflight(config)

    assert result["status"] == "ready"
    assert result["subject"] == config.subject
    assert [item["kind"] for item in result["sources"]] == [
        "segmentation",
        "t1",
        "t2",
    ]
    by_kind = {item["kind"]: item for item in result["sources"]}
    assert by_kind["t1"]["transform"] == "copy"
    assert by_kind["t2"]["transform"] == "gunzip"
    assert by_kind["segmentation"]["transform"] == "gunzip"
    assert not config.output_root.exists()
    assert not paths.owner_receipt.exists()


def test_preflight_rejects_source_hash_mismatch_and_root_overlap(
    creator_layout: dict[str, Path | str],
) -> None:
    config = _config(creator_layout)
    Path(creator_layout["t1"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        creator.preflight(config)
    assert not config.output_root.exists()

    config = replace(config, output_root=config.corrected_source_root)
    with pytest.raises(ValueError, match="overlap"):
        assert_output_root_isolation(config)


def test_preflight_rejects_protected_production_root(
    creator_layout: dict[str, Path | str],
) -> None:
    config = _config(creator_layout)
    production = production_paths().headmodel_root
    with pytest.raises(ValueError, match="overlap"):
        assert_output_root_isolation(replace(config, output_root=production))


def test_unowned_or_mismatched_subject_output_fails_closed(
    creator_layout: dict[str, Path | str],
) -> None:
    config = _config(creator_layout)
    paths = creator_paths(config)
    paths.subject_anat.mkdir(parents=True)
    (paths.subject_anat / "partial.txt").write_text("do not adopt")
    with pytest.raises(RuntimeError, match="unowned"):
        creator.preflight(config)
    assert (paths.subject_anat / "partial.txt").is_file()

    paths.owner_receipt.parent.mkdir(parents=True)
    paths.owner_receipt.write_text(
        json.dumps({"workflow": creator.WORKFLOW, "subject": "sub-CC999999"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="mismatch|bound"):
        creator.preflight(config)


def test_owned_partial_build_retries_and_writes_bound_receipts(
    creator_layout: dict[str, Path | str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(creator_layout)
    paths = creator_paths(config)
    _write_module_receipt(config)
    real_build = creator.build_scaffold

    def fail_after_partial(request):
        target = scaffold_core.scaffold_paths(request)
        target.anat.mkdir(parents=True, exist_ok=True)
        (target.anat / "partial.txt").write_text("diagnostic partial")
        raise RuntimeError("simulated headreco failure")

    monkeypatch.setattr(creator, "build_scaffold", fail_after_partial)
    with pytest.raises(RuntimeError, match="simulated"):
        creator.build(config)
    assert paths.owner_receipt.is_file()
    assert (paths.subject_anat / "partial.txt").is_file()

    monkeypatch.setattr(creator, "build_scaffold", real_build)
    monkeypatch.setattr(scaffold_core.subprocess, "run", _fake_headreco)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    result = creator.build(config)
    validated = creator.validate(config)

    assert result["status"] == "ready"
    assert validated["status"] == "ready"
    assert paths.completion_receipt.is_file()
    assert not (paths.subject_anat / "partial.txt").exists()
    assert (
        paths.subject_anat / f"m2m_{config.subject}" / "retained_diagnostic.nii.gz"
    ).is_file()
    owner = json.loads(paths.owner_receipt.read_text(encoding="utf-8"))
    completion = json.loads(paths.completion_receipt.read_text(encoding="utf-8"))
    assert owner["subject"] == config.subject
    assert completion["owner_receipt_sha256"] == creator.sha256_file(
        paths.owner_receipt
    )
    assert completion["ready_marker_sha256"] == creator.sha256_file(
        paths.subject_anat / scaffold_core.READY_MARKER
    )
    assert completion["module_receipt_sha256"] == creator.sha256_file(
        paths.module_receipt
    )
    assert completion["module_receipt_evidence_sha256"] == creator._json_sha256(
        completion["module_receipt_evidence"]
    )

    reused = creator.build(config)
    assert reused["status"] == "ready"
    assert reused["action"] == "reused"


def test_validation_accepts_equivalent_replacement_of_shared_module_receipt(
    creator_layout: dict[str, Path | str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(creator_layout)
    paths = creator_paths(config)
    _write_module_receipt(config)
    monkeypatch.setattr(scaffold_core.subprocess, "run", _fake_headreco)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    creator.build(config)
    original_completion = json.loads(
        paths.completion_receipt.read_text(encoding="utf-8")
    )

    current_module = json.loads(paths.module_receipt.read_text(encoding="utf-8"))
    current_module["created_utc"] = "2026-10-24T12:00:00+00:00"
    paths.module_receipt.write_text(
        json.dumps(current_module, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert (
        creator.sha256_file(paths.module_receipt)
        != original_completion["module_receipt_sha256"]
    )
    assert creator.validate(config)["status"] == "ready"


def test_validation_binds_source_receipt_and_authoritative_subject_list(
    creator_layout: dict[str, Path | str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(creator_layout)
    _write_module_receipt(config)
    monkeypatch.setattr(scaffold_core.subprocess, "run", _fake_headreco)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    creator.build(config)

    source_receipt = Path(creator_layout["record"])
    original_source_receipt = source_receipt.read_text(encoding="utf-8")
    receipt_payload = json.loads(original_source_receipt)
    receipt_payload["review_note"] = "changed provenance"
    source_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="owner receipt.*mismatch|different binding"):
        creator.validate(config)

    source_receipt.write_text(original_source_receipt, encoding="utf-8")
    config.final132_subjects_file.write_text(
        config.final132_subjects_file.read_text(encoding="utf-8") + "sub-CC220088\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="owner receipt.*mismatch|different binding"):
        creator.validate(config)


def test_validation_detects_stale_source_marker_and_mesh(
    creator_layout: dict[str, Path | str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(creator_layout)
    paths = creator_paths(config)
    _write_module_receipt(config)
    monkeypatch.setattr(scaffold_core.subprocess, "run", _fake_headreco)
    monkeypatch.setattr(scaffold_core, "_simnibs_version", lambda: "3.2.6")
    creator.build(config)

    Path(creator_layout["t1"]).write_bytes(b"stale")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        creator.validate(config)
    Path(creator_layout["t1"]).write_bytes(b"t1")

    marker = paths.subject_anat / scaffold_core.READY_MARKER
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["source_t1_sha256"] = "0" * 64
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scaffold"):
        creator.validate(config)

    creator.build(config)
    compatibility = (
        paths.subject_anat / f"m2m_{config.subject}" / f"{config.subject}.msh"
    )
    compatibility.write_bytes(b"changed mesh")
    with pytest.raises(RuntimeError, match="scaffold"):
        creator.validate(config)


def test_status_is_read_only_and_reports_incomplete(
    creator_layout: dict[str, Path | str],
) -> None:
    config = _config(creator_layout)
    before = set(config.output_root.parent.iterdir())
    result = creator.status(config)
    after = set(config.output_root.parent.iterdir())
    assert result["status"] == "not-built"
    assert before == after
