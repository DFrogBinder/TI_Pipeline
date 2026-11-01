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
    result = archive_verify(miniature_config, checksum=True)
    assert result["status"] == "mismatch"
    assert result["mismatched_roots"] == 1


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
