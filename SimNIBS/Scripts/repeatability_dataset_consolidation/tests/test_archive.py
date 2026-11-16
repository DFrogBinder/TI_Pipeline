from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pytest

from repeatability_dataset_consolidation.archive import (
    ArchiveError,
    archive_copy,
    archive_preflight,
    archive_verify,
    validate_archive_completion,
)


def _seed_archive_sources(config) -> dict[str, tuple[bytes, int]]:
    before: dict[str, tuple[bytes, int]] = {}
    for index, (key, root) in enumerate(config.archive_sources.items(), start=1):
        data = root / "nested/data.bin"
        data.parent.mkdir()
        data.write_bytes(f"source-{index}\n".encode())
        hard = root / "nested/data-hard.bin"
        os.link(data, hard)
        (root / "data-link.bin").symlink_to("nested/data.bin")
        before[key] = (data.read_bytes(), data.stat().st_mtime_ns)
    return before


def test_archive_preflight_is_exact_and_read_only(miniature_config) -> None:
    _seed_archive_sources(miniature_config)

    result = archive_preflight(miniature_config)

    assert result["status"] == "ready"
    assert result["archive_root"] == str(miniature_config.shared_archive_root)
    assert {mapping["key"] for mapping in result["mappings"]} == set(
        miniature_config.archive_sources
    )
    assert {
        Path(mapping["destination"]).name for mapping in result["mappings"]
    } == {path.name for path in miniature_config.archive_sources.values()}
    assert result["required_bytes"] > 0
    assert result["capacity_sufficient"] is True
    assert result["capacity_gate_enforced"] is True
    assert not miniature_config.shared_archive_root.exists()
    assert not miniature_config.shared_receipt_root.exists()


def test_archive_capacity_fails_before_initial_root_creation(
    miniature_config, monkeypatch
) -> None:
    _seed_archive_sources(miniature_config)
    usage_type = type(shutil.disk_usage(miniature_config.shared_archive_root.parent))
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.archive.shutil.disk_usage",
        lambda _path: usage_type(total=100, used=99, free=1),
    )

    with pytest.raises(ArchiveError, match="Insufficient capacity"):
        archive_preflight(miniature_config)
    with pytest.raises(ArchiveError, match="Insufficient capacity"):
        archive_copy(miniature_config)
    assert not miniature_config.shared_archive_root.exists()
    assert not miniature_config.shared_receipt_root.exists()


def test_archive_copy_resumes_and_materializes_links(miniature_config) -> None:
    before = _seed_archive_sources(miniature_config)

    first = archive_copy(miniature_config)
    second = archive_copy(miniature_config)

    assert first["status"] == "copied"
    assert first["copied_roots"] == 4
    assert second["reused_roots"] == 4
    for key, source in miniature_config.archive_sources.items():
        destination = miniature_config.shared_archive_root / source.name
        regular = destination / "nested/data.bin"
        hard = destination / "nested/data-hard.bin"
        link = destination / "data-link.bin"
        assert link.is_file()
        assert not link.is_symlink()
        assert link.read_bytes() == regular.read_bytes()
        assert regular.stat().st_ino == hard.stat().st_ino
        assert (regular.read_bytes(), regular.stat().st_mtime_ns) == before[key]
        source_data = source / "nested/data.bin"
        assert (source_data.read_bytes(), source_data.stat().st_mtime_ns) == before[key]

    payloads = [
        json.loads(path.read_text())
        for path in (miniature_config.shared_archive_root / "_archive/copy_items").glob("*.json")
    ]
    assert all("--copy-links" in payload["command"] for payload in payloads)
    assert all("--no-owner" in payload["command"] for payload in payloads)
    assert all("--delete" not in payload["command"] for payload in payloads if payload["action"] == "copied")


def test_archive_materializes_absolute_links_between_archived_roots(
    miniature_config,
) -> None:
    _seed_archive_sources(miniature_config)
    roots = list(miniature_config.archive_sources.values())
    target = roots[1] / "nested/data.bin"
    link = roots[0] / "cross-root-absolute-link.bin"
    link.symlink_to(target.resolve())

    archive_copy(miniature_config)
    assert archive_verify(miniature_config, checksum=True)["status"] == "complete"

    archived = miniature_config.shared_archive_root / roots[0].name / link.name
    assert archived.is_file()
    assert not archived.is_symlink()
    assert archived.read_bytes() == target.read_bytes()


def test_archive_materializes_links_from_protected_root(miniature_config) -> None:
    _seed_archive_sources(miniature_config)
    protected_root = miniature_config.protected_materialization_roots[0]
    target = protected_root / "sub-CC520127_T1w.nii"
    target.write_bytes(b"protected corrected input\n")
    source = next(iter(miniature_config.archive_sources.values()))
    link = source / "protected-input-link.nii"
    link.symlink_to(target.resolve())

    preflight = archive_preflight(miniature_config)
    assert preflight["protected_materialization_roots"] == [
        str(protected_root.resolve())
    ]

    archive_copy(miniature_config)
    assert archive_verify(miniature_config, checksum=True)["status"] == "complete"

    archived = miniature_config.shared_archive_root / source.name / link.name
    assert archived.is_file()
    assert not archived.is_symlink()
    assert archived.read_bytes() == target.read_bytes()


def test_archive_plan_binds_protected_materialization_roots(
    miniature_config,
) -> None:
    _seed_archive_sources(miniature_config)
    without_protected_roots = replace(
        miniature_config,
        protected_materialization_roots=(),
    )

    with_hash = archive_preflight(miniature_config)["plan_sha256"]
    without_hash = archive_preflight(without_protected_roots)["plan_sha256"]

    assert with_hash != without_hash


def test_archive_rejects_protected_root_overlapping_retirement_target(
    miniature_config,
) -> None:
    _seed_archive_sources(miniature_config)
    archive_source = next(iter(miniature_config.archive_sources.values()))
    unsafe = replace(
        miniature_config,
        protected_materialization_roots=(archive_source,),
    )

    with pytest.raises(ArchiveError, match="Protected materialization root overlaps"):
        archive_preflight(unsafe)


def test_archive_rejects_links_outside_archive_sources(miniature_config) -> None:
    _seed_archive_sources(miniature_config)
    outside = miniature_config.shared_archive_root.parent / "not-an-archive-source.bin"
    outside.write_bytes(b"outside\n")
    root = next(iter(miniature_config.archive_sources.values()))
    (root / "external-link.bin").symlink_to(outside.resolve())

    with pytest.raises(ArchiveError, match="outside archive sources"):
        archive_preflight(miniature_config)


def test_archive_checksum_writes_external_immutable_receipt(miniature_config) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)

    quick = archive_verify(miniature_config, checksum=False)
    checksum = archive_verify(miniature_config, checksum=True)

    assert quick["status"] == "matched"
    assert checksum["status"] == "complete"
    receipt = miniature_config.shared_receipt_root / "legacy_v3_archive_v1.complete.json"
    assert receipt.is_file()
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "complete"
    assert len(payload["mappings"]) == 4
    assert all(mapping["source_inventory_sha256"] == mapping["destination_inventory_sha256"] for mapping in payload["mappings"])

    receipt.write_text('{"status":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ArchiveError, match="existing archive completion receipt"):
        archive_verify(miniature_config, checksum=True)


def test_archive_verify_detects_extra_and_changed_content(miniature_config) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)
    destination = miniature_config.shared_archive_root / next(
        iter(miniature_config.archive_sources.values())
    ).name
    extra = destination / "extra.txt"
    extra.write_text("extra\n", encoding="utf-8")
    assert archive_verify(miniature_config, checksum=False)["status"] == "mismatch"
    extra.unlink()

    data = destination / "nested/data.bin"
    original_mtime = data.stat().st_mtime_ns
    data.write_bytes(b"changed!\n")
    os.utime(data, ns=(original_mtime, original_mtime))
    with pytest.raises(ArchiveError, match="no longer matches"):
        archive_verify(miniature_config, checksum=True)


def test_archive_rejects_missing_unowned_and_different_roots(miniature_config) -> None:
    _seed_archive_sources(miniature_config)
    missing = next(iter(miniature_config.archive_sources.values()))
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()
    with pytest.raises(ArchiveError, match="missing archive source"):
        archive_preflight(miniature_config)

    missing.mkdir()
    miniature_config.shared_archive_root.mkdir()
    (miniature_config.shared_archive_root / "foreign.txt").write_text("no\n")
    with pytest.raises(ArchiveError, match="unowned archive root"):
        archive_copy(miniature_config)

    alternate = replace(
        miniature_config,
        shared_archive_root=miniature_config.shared_archive_root.with_name("alternate"),
    )
    alternate.shared_archive_root.symlink_to(miniature_config.shared_archive_root)
    with pytest.raises(ArchiveError, match="symbolic-link archive root"):
        archive_copy(alternate)


def test_archive_refuses_symlinked_destination(miniature_config) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)
    source = next(iter(miniature_config.archive_sources.values()))
    destination = miniature_config.shared_archive_root / source.name
    shutil.rmtree(destination)
    outside = miniature_config.shared_archive_root.parent / "outside-archive"
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArchiveError, match="symbolic-link archive destination"):
        archive_copy(miniature_config)


def test_archive_checksum_resumes_after_one_completed_root(
    miniature_config,
    monkeypatch,
) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)
    from repeatability_dataset_consolidation import archive

    real_writer = archive.write_inventory_stream
    calls = 0

    def interrupt_second_root(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated archive interruption")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(archive, "write_inventory_stream", interrupt_second_root)
    with pytest.raises(RuntimeError, match="simulated archive interruption"):
        archive_verify(miniature_config, checksum=True)

    checkpoint_root = (
        miniature_config.shared_archive_root
        / "_archive/verification/checksum/items"
    )
    receipts = list(checkpoint_root.glob("*/receipt.json"))
    assert len(receipts) == 1
    preserved_mtime = receipts[0].stat().st_mtime_ns
    assert not (
        miniature_config.shared_receipt_root
        / "legacy_v3_archive_v1.complete.json"
    ).exists()

    resumed_calls = 0

    def count_resumed(*args, **kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(archive, "write_inventory_stream", count_resumed)
    result = archive_verify(miniature_config, checksum=True)

    assert result["status"] == "complete"
    assert result["reused_roots"] == 1
    assert resumed_calls == 6
    assert receipts[0].stat().st_mtime_ns == preserved_mtime


def test_archive_completion_validation_uses_metadata_checkpoints(
    miniature_config,
    monkeypatch,
) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)
    archive_verify(miniature_config, checksum=True)
    from repeatability_dataset_consolidation import archive

    def forbid_deep_rehash(*args, **kwargs):
        raise AssertionError("normal archive validation performed a deep rehash")

    monkeypatch.setattr(archive, "write_inventory_stream", forbid_deep_rehash)

    assert validate_archive_completion(miniature_config)["status"] == "complete"


def test_archive_stale_stream_is_recomputed_but_same_size_content_change_fails(
    miniature_config,
    monkeypatch,
) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)
    archive_verify(miniature_config, checksum=True)
    from repeatability_dataset_consolidation import archive

    checkpoint_root = (
        miniature_config.shared_archive_root
        / "_archive/verification/checksum/items"
    )
    stream = next(checkpoint_root.glob("*/destination.jsonl"))
    stream.write_text("tampered\n", encoding="utf-8")
    real_writer = archive.write_inventory_stream
    rehashed: list[Path] = []

    def record_rehash(root, *args, **kwargs):
        rehashed.append(root)
        return real_writer(root, *args, **kwargs)

    monkeypatch.setattr(archive, "write_inventory_stream", record_rehash)
    assert archive_verify(miniature_config, checksum=True)["status"] == "complete"
    assert len(rehashed) == 2

    source = next(iter(miniature_config.archive_sources.values()))
    source_file = source / "nested/data.bin"
    original = source_file.stat()
    data = source_file.read_bytes()
    source_file.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    os.utime(source_file, ns=(original.st_atime_ns, original.st_mtime_ns))

    with pytest.raises(ArchiveError, match="no longer matches"):
        archive_verify(miniature_config, checksum=True)


def test_archive_completion_rejects_metadata_only_destination_change(
    miniature_config,
) -> None:
    _seed_archive_sources(miniature_config)
    archive_copy(miniature_config)
    archive_verify(miniature_config, checksum=True)
    destination = miniature_config.shared_archive_root / next(
        iter(miniature_config.archive_sources.values())
    ).name
    changed = destination / "nested/data.bin"
    current = changed.stat()
    os.utime(changed, ns=(current.st_atime_ns, current.st_mtime_ns + 1))

    with pytest.raises(ArchiveError, match="metadata fingerprint"):
        validate_archive_completion(miniature_config)
