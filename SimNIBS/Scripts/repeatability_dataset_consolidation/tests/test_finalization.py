from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.disposition import (
    build_retirement_disposition,
    disposition_path,
)
from repeatability_dataset_consolidation.finalization import (
    finalization_status,
    finalize_dataset,
)
from repeatability_dataset_consolidation.planning import build_copy_plan
from repeatability_dataset_consolidation.workflow import (
    WorkflowError,
    copy_dataset,
    prepare,
    validate_dataset_completion,
    verify_dataset,
)


def _checksum_ready(config) -> None:
    prepare(config)
    copy_dataset(config)
    result = verify_dataset(config, checksum=True)
    assert result["status"] == "matched"


def test_recovers_complete_manifest_without_completion_receipt(
    miniature_config,
    monkeypatch,
) -> None:
    _checksum_ready(miniature_config)
    plan = build_copy_plan(miniature_config)
    manifest_path = miniature_config.output_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["completed_utc"] = "2026-11-12T03:18:38+00:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    disposition_path(miniature_config).parent.mkdir(parents=True, exist_ok=True)
    disposition_path(miniature_config).write_text(
        json.dumps(build_retirement_disposition(miniature_config, plan)),
        encoding="utf-8",
    )

    observed_manifest_states: list[str] = []
    from repeatability_dataset_consolidation import finalization

    real_writer = finalization.write_inventory_stream

    def observe_manifest(*args, **kwargs):
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_manifest_states.append(current["status"])
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(finalization, "write_inventory_stream", observe_manifest)

    result = finalize_dataset(miniature_config)

    assert result["status"] == "complete"
    assert observed_manifest_states
    assert set(observed_manifest_states) == {"finalizing"}
    checkpoints = list(
        (
            miniature_config.output_root
            / "_consolidation/finalization/items"
        ).glob("*.json")
    )
    streams = list(
        (
            miniature_config.output_root
            / "_consolidation/finalization/items"
        ).glob("*.jsonl")
    )
    assert len(checkpoints) == len(plan.items)
    assert len(streams) == len(plan.items)
    assert (
        miniature_config.output_root
        / "_consolidation/verification/file_inventory.json"
    ).is_file()
    assert (miniature_config.output_root / "_consolidation/complete.json").is_file()
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert completed_manifest["status"] == "complete"
    assert completed_manifest["completed_utc"] != "2026-11-12T03:18:38+00:00"


def test_finalization_resumes_completed_items_after_interruption(
    miniature_config,
    monkeypatch,
) -> None:
    _checksum_ready(miniature_config)
    plan = build_copy_plan(miniature_config)
    from repeatability_dataset_consolidation import finalization

    real_writer = finalization.write_inventory_stream
    calls = 0

    def interrupt_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(finalization, "write_inventory_stream", interrupt_third)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        finalize_dataset(miniature_config)

    receipt_root = (
        miniature_config.output_root / "_consolidation/finalization/items"
    )
    receipts = sorted(receipt_root.glob("*.json"))
    assert len(receipts) == 2
    mtimes = {path.name: path.stat().st_mtime_ns for path in receipts}
    status = finalization_status(miniature_config)
    assert status["completed_items"] == 2
    assert status["expected_items"] == len(plan.items)
    assert status["last_failure"]["message"] == "simulated interruption"

    resumed_calls = 0

    def count_resumed(*args, **kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(finalization, "write_inventory_stream", count_resumed)
    result = finalize_dataset(miniature_config)

    assert result["status"] == "complete"
    assert result["finalization"]["reused_items"] == 2
    assert resumed_calls == len(plan.items) - 2
    assert {path.name: path.stat().st_mtime_ns for path in receipts} == mtimes
    assert finalization_status(miniature_config)["last_failure"] is None


def test_changed_destination_with_restored_mtime_fails_closed_and_rehashes_only_it(
    miniature_config,
    monkeypatch,
) -> None:
    _checksum_ready(miniature_config)
    finalize_dataset(miniature_config)
    plan = build_copy_plan(miniature_config)
    target_item = next(item for item in plan.items if item.source_type == "directory")
    destination = miniature_config.output_root / target_item.destination
    changed = next(path for path in destination.rglob("*") if path.is_file())
    original = changed.stat()
    data = changed.read_bytes()
    changed.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(changed, ns=(original.st_atime_ns, original.st_mtime_ns))
    (miniature_config.output_root / "_consolidation/complete.json").unlink()
    (
        miniature_config.output_root
        / "_consolidation/verification/file_inventory.json"
    ).unlink()

    from repeatability_dataset_consolidation import finalization

    real_writer = finalization.write_inventory_stream
    rehashed: list[Path] = []

    def record_rehash(root, *args, **kwargs):
        rehashed.append(root)
        return real_writer(root, *args, **kwargs)

    monkeypatch.setattr(finalization, "write_inventory_stream", record_rehash)

    with pytest.raises(WorkflowError, match="destination inventory changed"):
        finalize_dataset(miniature_config)

    assert rehashed == [destination]


def test_replaced_inode_with_identical_content_invalidates_only_one_checkpoint(
    miniature_config,
    monkeypatch,
) -> None:
    _checksum_ready(miniature_config)
    finalize_dataset(miniature_config)
    plan = build_copy_plan(miniature_config)
    target_item = next(item for item in plan.items if item.source_type == "directory")
    destination = miniature_config.output_root / target_item.destination
    changed = next(path for path in destination.rglob("*") if path.is_file())
    original = changed.stat()
    replacement = changed.with_name(changed.name + ".replacement")
    replacement.write_bytes(changed.read_bytes())
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, changed)
    (miniature_config.output_root / "_consolidation/complete.json").unlink()
    (
        miniature_config.output_root
        / "_consolidation/verification/file_inventory.json"
    ).unlink()

    from repeatability_dataset_consolidation import finalization

    real_writer = finalization.write_inventory_stream
    rehashed: list[Path] = []

    def record_rehash(root, *args, **kwargs):
        rehashed.append(root)
        return real_writer(root, *args, **kwargs)

    monkeypatch.setattr(finalization, "write_inventory_stream", record_rehash)
    result = finalize_dataset(miniature_config)

    assert result["status"] == "complete"
    assert rehashed == [destination]
    assert result["finalization"]["reused_items"] == len(plan.items) - 1


def test_final_inventory_excludes_operational_checkpoints_and_includes_controls(
    miniature_config,
) -> None:
    _checksum_ready(miniature_config)

    finalize_dataset(miniature_config)

    inventory_path = (
        miniature_config.output_root
        / "_consolidation/verification/file_inventory.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in inventory["entries"]}
    assert "README.md" in paths
    assert "dataset_manifest.json" in paths
    assert "provenance/source_mapping.tsv" in paths
    assert not any(
        path.startswith("_consolidation/finalization/") for path in paths
    )


def test_completion_receipt_is_absent_when_final_inventory_merge_fails(
    miniature_config,
    monkeypatch,
) -> None:
    _checksum_ready(miniature_config)
    from repeatability_dataset_consolidation import finalization

    def reject_duplicate(*args, **kwargs):
        raise ValueError("duplicate inventory path: duplicate.txt")

    monkeypatch.setattr(finalization, "merge_inventory_streams", reject_duplicate)

    with pytest.raises(ValueError, match="duplicate inventory path"):
        finalize_dataset(miniature_config)

    assert not (
        miniature_config.output_root / "_consolidation/complete.json"
    ).exists()
    manifest = json.loads(
        (miniature_config.output_root / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "finalizing"


def test_completion_validation_uses_bound_checkpoints_not_full_tree_rehash(
    miniature_config,
    monkeypatch,
) -> None:
    _checksum_ready(miniature_config)
    receipt = finalize_dataset(miniature_config)

    from repeatability_dataset_consolidation import workflow

    def forbid_full_tree_rehash(*args, **kwargs):
        raise AssertionError("normal completion validation rehashed the full dataset")

    monkeypatch.setattr(workflow, "build_file_inventory", forbid_full_tree_rehash)

    assert validate_dataset_completion(miniature_config) == {
        key: value for key, value in receipt.items() if key != "action"
    }


def test_completion_validation_rejects_changed_checkpoint_metadata(
    miniature_config,
) -> None:
    _checksum_ready(miniature_config)
    finalize_dataset(miniature_config)
    plan = build_copy_plan(miniature_config)
    item = next(candidate for candidate in plan.items if candidate.source_type == "directory")
    destination = miniature_config.output_root / item.destination
    changed = next(path for path in destination.rglob("*") if path.is_file())
    replacement = changed.with_name(changed.name + ".replacement")
    replacement.write_bytes(changed.read_bytes())
    os.replace(replacement, changed)

    with pytest.raises(WorkflowError, match="metadata fingerprint"):
        validate_dataset_completion(miniature_config)


def test_finalization_rejects_a_removed_bound_checksum_item_receipt(
    miniature_config,
) -> None:
    _checksum_ready(miniature_config)
    item_receipt = next(
        (
            miniature_config.output_root
            / "_consolidation/verification/checksum/items"
        ).glob("*.json")
    )
    item_receipt.unlink()

    with pytest.raises(WorkflowError, match="changed or removed"):
        finalize_dataset(miniature_config)
