from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.inventory import (
    build_file_inventory,
    sha256_file,
)
from repeatability_dataset_consolidation.planning import build_copy_plan
from repeatability_dataset_consolidation.workflow import (
    WorkflowError,
    complete_dataset,
    copy_dataset,
    prepare,
    verify_dataset,
)


def _ready(config) -> None:
    prepare(config)
    copy_dataset(config)
    result = verify_dataset(config, checksum=True)
    assert result["status"] == "matched"


def test_completion_binds_checksum_inventory_and_exact_scope(miniature_config) -> None:
    _ready(miniature_config)

    result = complete_dataset(miniature_config)

    assert result["status"] == "complete"
    assert result["scope"] == {
        "subjects": 2,
        "cells": 16,
        "simulations": 32,
        "v3_headmodels": 2,
        "v4_scaffolds": 2,
    }
    receipt = json.loads(
        (miniature_config.output_root / "_consolidation/complete.json").read_text()
    )
    inventory_path = miniature_config.output_root / receipt["inventory"]["path"]
    inventory = json.loads(inventory_path.read_text())
    paths = [entry["path"] for entry in inventory["entries"]]
    assert paths == sorted(paths)
    assert receipt["inventory"]["sha256"] == inventory["inventory_sha256"]
    assert receipt["verification"]["mode"] == "checksum"
    source_mapping = miniature_config.output_root / "provenance/source_mapping.tsv"
    mapping_rows = source_mapping.read_text(encoding="utf-8").splitlines()
    assert len(mapping_rows) == len(build_copy_plan(miniature_config).items) + 1
    assert all(row.endswith("\tchecksum_matched") for row in mapping_rows[1:])
    assert receipt["source_mapping"]["sha256"] == sha256_file(source_mapping)
    assert len(receipt["source_mapping"]["sha256"]) == 64
    assert len(receipt["dataset_manifest"]["sha256"]) == 64

    historical = (
        miniature_config.output_root
        / "provenance/configs/historical/v3_left/paired_analysis.json"
    )
    source_config = (
        miniature_config.sources.v3_left
        / "_pipeline/configs/paired_analysis.json"
    )
    assert historical.read_bytes() == source_config.read_bytes()
    canonical = (
        miniature_config.output_root
        / "provenance/configs/canonical/v3_left/paired_analysis.json"
    )
    canonical_payload = json.loads(canonical.read_text(encoding="utf-8"))
    assert canonical_payload["experiment_root"] == str(
        miniature_config.output_root / "data/simnibs326/left_hippocampus"
    )
    assert canonical_payload["unchanged_parameter"] == "median_roi"
    canonical_manifest = json.loads(
        (
            miniature_config.output_root
            / "provenance/configs/canonical/manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert canonical_manifest["configs"][0]["source"] == str(source_config)
    assert len(canonical_manifest["configs"][0]["source_sha256"]) == 64
    canonical_manifest_path = (
        miniature_config.output_root
        / "provenance/configs/canonical/manifest.json"
    )
    assert receipt["canonical_configs"]["sha256"] == sha256_file(
        canonical_manifest_path
    )
    assert len(canonical_manifest["configs_sha256"]) == 64
    assert not any(path.is_symlink() for path in miniature_config.output_root.rglob("*"))


def test_completion_requires_untampered_checksum_receipts(miniature_config) -> None:
    prepare(miniature_config)
    copy_dataset(miniature_config)
    with pytest.raises(WorkflowError, match="checksum verification"):
        complete_dataset(miniature_config)

    verify_dataset(miniature_config, checksum=True)
    item_receipt = next(
        (miniature_config.output_root / "_consolidation/verification/checksum/items").glob("*.json")
    )
    payload = json.loads(item_receipt.read_text())
    payload["status"] = "mismatch"
    item_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkflowError, match="verification item receipt"):
        complete_dataset(miniature_config)


def test_completion_requires_untampered_copy_receipts(miniature_config) -> None:
    _ready(miniature_config)
    copy_receipt = next(
        (miniature_config.output_root / "_consolidation/copy_items").glob("*.json")
    )
    payload = json.loads(copy_receipt.read_text())
    payload["status"] = "partial"
    copy_receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WorkflowError, match="copy item receipt"):
        complete_dataset(miniature_config)


def test_completion_rejects_unplanned_destination_file(miniature_config) -> None:
    _ready(miniature_config)
    (miniature_config.output_root / "unplanned.txt").write_text(
        "not in the copy plan\n", encoding="utf-8"
    )

    with pytest.raises(WorkflowError, match="unplanned destination path"):
        complete_dataset(miniature_config)


def test_completion_rejects_changed_source_or_destination(miniature_config) -> None:
    _ready(miniature_config)
    source_file = miniature_config.sources.paper_analysis / "analysis.json"
    source_file.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(WorkflowError, match="source inventory changed"):
        complete_dataset(miniature_config)

    copy_dataset(miniature_config)
    verify_dataset(miniature_config, checksum=True)
    replacement = (
        miniature_config.output_root
        / "data/simnibs4/right_m1/sub-CC510415_repeatability/remesh"
        / "repeats/repeat_001/sub-CC510415/anat/SimNIBS/Output/sub-CC510415/TI.msh"
    )
    replacement.unlink()
    with pytest.raises(WorkflowError, match="destination inventory changed"):
        complete_dataset(miniature_config)


def test_completion_rejects_excluded_subject_and_inventory_mismatch(
    miniature_config,
) -> None:
    _ready(miniature_config)
    injected = miniature_config.output_root / "data/simnibs4/right_m1/sub-CC520083_repeatability"
    injected.mkdir()
    with pytest.raises(WorkflowError, match="excluded subject"):
        complete_dataset(miniature_config)
    injected.rmdir()

    complete_dataset(miniature_config)
    data_file = next(
        path
        for path in miniature_config.output_root.rglob("TI.msh")
        if path.is_file()
    )
    data_file.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(WorkflowError, match="existing completion receipt"):
        complete_dataset(miniature_config)


def test_file_inventory_is_sorted_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "inventory"
    root.mkdir()
    (root / "b.txt").write_text("b\n", encoding="utf-8")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    inventory = build_file_inventory(root)
    assert [entry.path for entry in inventory.entries] == ["a.txt", "b.txt"]

    (root / "link.txt").symlink_to("a.txt")
    with pytest.raises(ValueError, match="symlink"):
        build_file_inventory(root)
