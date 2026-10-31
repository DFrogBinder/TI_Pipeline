from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.workflow import (
    WorkflowError,
    copy_dataset,
    prepare,
    verify_dataset,
)


def _source_cell(config) -> Path:
    return (
        config.sources.v3_left
        / "sub-CC110174_repeatability"
        / "remesh"
    )


def _destination_cell(config) -> Path:
    return (
        config.output_root
        / "data/simnibs326/left_hippocampus"
        / "sub-CC110174_repeatability/remesh"
    )


def test_copy_is_resumable_physical_and_never_deletes(miniature_config) -> None:
    source = _source_cell(miniature_config)
    link_target = source / "link-target.txt"
    link_target.write_text("physical bytes\n", encoding="utf-8")
    (source / "materialized-link.txt").symlink_to(link_target.name)
    source_probe = source / "repeats/repeat_001/sub-CC110174/anat/SimNIBS/Output/sub-CC110174/TI.msh"
    before = (source_probe.read_bytes(), source_probe.stat().st_mtime_ns)
    prepare(miniature_config)

    first = copy_dataset(miniature_config)
    receipts = miniature_config.output_root / "_consolidation/copy_items"
    first_payloads = [
        json.loads(path.read_text()) for path in receipts.glob("*.json")
    ]
    second = copy_dataset(miniature_config)

    assert first["status"] == "copied"
    assert first["copied_items"] > 0
    assert second["reused_items"] == second["planned_items"]
    copied_link = _destination_cell(miniature_config) / "materialized-link.txt"
    assert copied_link.is_file() and not copied_link.is_symlink()
    assert copied_link.read_bytes() == b"physical bytes\n"
    assert (source_probe.read_bytes(), source_probe.stat().st_mtime_ns) == before

    payloads = [json.loads(path.read_text()) for path in receipts.glob("*.json")]
    assert len(payloads) == first["planned_items"]
    assert all("--copy-links" in payload["command"] for payload in payloads)
    assert all("--delete" not in payload["command"] for payload in first_payloads)
    assert all("--dry-run" in payload["command"] for payload in payloads)


def test_verify_distinguishes_quick_and_checksum_and_detects_extras(
    miniature_config,
) -> None:
    prepare(miniature_config)
    copy_dataset(miniature_config)
    destination = _destination_cell(miniature_config)
    field = destination / "repeats/repeat_001/sub-CC110174/anat/SimNIBS/Output/sub-CC110174/TI.msh"
    original_mtime = field.stat().st_mtime_ns
    field.write_bytes(b"other 1\n")
    os.utime(field, ns=(original_mtime, original_mtime))

    quick = verify_dataset(miniature_config, checksum=False)
    checksum = verify_dataset(miniature_config, checksum=True)

    assert quick["status"] == "matched"
    assert quick["mode"] == "size_mtime"
    assert checksum["status"] == "mismatch"
    assert checksum["mode"] == "checksum"
    assert checksum["mismatched_items"] >= 1

    field.write_text("field 1\n", encoding="utf-8")
    extra = destination / "unplanned.txt"
    extra.write_text("extra\n", encoding="utf-8")
    extra_result = verify_dataset(miniature_config, checksum=False)
    assert extra_result["status"] == "mismatch"
    assert extra_result["mismatched_items"] >= 1
    assert any("*deleting" in change for change in extra_result["changes"])

    extra.unlink()
    field.unlink()
    missing_result = verify_dataset(miniature_config, checksum=True)
    assert missing_result["status"] == "mismatch"


def test_copy_refuses_symlinked_destination_component(miniature_config) -> None:
    prepare(miniature_config)
    headmodels = miniature_config.output_root / "headmodels/simnibs326"
    headmodels.rmdir()
    outside = miniature_config.output_root.parent / "outside"
    outside.mkdir()
    headmodels.symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkflowError, match="symlinked destination component"):
        copy_dataset(miniature_config)

