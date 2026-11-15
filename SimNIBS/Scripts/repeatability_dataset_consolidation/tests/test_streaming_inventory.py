from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.inventory import (
    InventoryEntry,
    build_file_inventory,
    build_materialized_inventory,
)
from repeatability_dataset_consolidation.streaming_inventory import (
    InventoryStreamSummary,
    merge_inventory_streams,
    metadata_fingerprint,
    validate_inventory_stream,
    write_inventory_stream,
)


def test_write_inventory_stream_matches_legacy_inventory_without_returning_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    (root / "z").mkdir(parents=True)
    (root / "z/last.txt").write_text("last\n", encoding="utf-8")
    (root / "first.bin").write_bytes(b"first")
    entries_path = tmp_path / "inventory.jsonl"

    summary = write_inventory_stream(root, entries_path)
    legacy = build_file_inventory(root)

    assert isinstance(summary, InventoryStreamSummary)
    assert not hasattr(summary, "entries")
    assert summary.root == str(root.resolve())
    assert summary.entry_count == 2
    assert summary.total_bytes == 10
    assert summary.inventory_sha256 == legacy.inventory_sha256

    entries = [
        json.loads(line)
        for line in entries_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["path"] for entry in entries] == ["first.bin", "z/last.txt"]
    assert entries == [entry.as_dict() for entry in legacy.entries]
    assert len(summary.entries_sha256) == 64
    assert len(summary.metadata_sha256) == 64


def test_destination_stream_matches_legacy_order_when_directory_prefixes_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    (root / "a").mkdir(parents=True)
    (root / "a/inside.bin").write_bytes(b"inside")
    (root / "a.txt").write_text("sibling\n", encoding="utf-8")
    entries_path = tmp_path / "inventory.jsonl"

    summary = write_inventory_stream(root, entries_path)
    legacy = build_file_inventory(root)
    entries = [
        json.loads(line)
        for line in entries_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [entry["path"] for entry in entries] == [
        "a.txt",
        "a/inside.bin",
    ]
    assert entries == [entry.as_dict() for entry in legacy.entries]
    assert summary.inventory_sha256 == legacy.inventory_sha256


def test_materialized_stream_matches_legacy_order_when_directory_prefixes_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "inside.bin").write_bytes(b"inside")
    (root / "a").symlink_to(external, target_is_directory=True)
    (root / "a.txt").write_text("sibling\n", encoding="utf-8")
    entries_path = tmp_path / "materialized.jsonl"

    summary = write_inventory_stream(
        root,
        entries_path,
        materialize_links=True,
    )
    legacy = build_materialized_inventory(root)
    entries = [
        json.loads(line)
        for line in entries_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [entry["path"] for entry in entries] == [
        "a.txt",
        "a/inside.bin",
    ]
    assert entries == [entry.as_dict() for entry in legacy.entries]
    assert summary.inventory_sha256 == legacy.inventory_sha256


def test_write_inventory_stream_atomically_replaces_an_existing_stream(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "file.txt").write_text("payload\n", encoding="utf-8")
    entries_path = tmp_path / "inventory.jsonl"
    entries_path.write_text("stale data\n", encoding="utf-8")

    write_inventory_stream(root, entries_path)

    assert not list(tmp_path.glob(f".{entries_path.name}.*.tmp"))
    assert json.loads(entries_path.read_text(encoding="utf-8"))["path"] == "file.txt"


def test_metadata_fingerprint_invalidates_restored_mtime_and_changed_inode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    file_path = root / "file.bin"
    file_path.write_bytes(b"aaaa")
    entries_path = tmp_path / "inventory.jsonl"
    summary = write_inventory_stream(root, entries_path)
    original = file_path.stat()

    assert metadata_fingerprint(root) == summary.metadata_sha256
    validate_inventory_stream(entries_path, summary)

    file_path.write_bytes(b"bbbb")
    os.utime(file_path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert metadata_fingerprint(root) != summary.metadata_sha256

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"aaaa")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, file_path)
    assert metadata_fingerprint(root) != summary.metadata_sha256


def test_validate_inventory_stream_rejects_tampered_or_noncanonical_lines(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "file.txt").write_text("content\n", encoding="utf-8")
    entries_path = tmp_path / "inventory.jsonl"
    summary = write_inventory_stream(root, entries_path)

    entry = json.loads(entries_path.read_text(encoding="utf-8"))
    entry["size_bytes"] += 1
    entries_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory stream"):
        validate_inventory_stream(entries_path, summary)


def test_materialized_stream_expands_internal_links_and_rejects_broken_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    target = root / "target"
    target.mkdir(parents=True)
    (target / "data.txt").write_text("data\n", encoding="utf-8")
    (root / "alias").symlink_to(target, target_is_directory=True)
    entries_path = tmp_path / "materialized.jsonl"

    summary = write_inventory_stream(
        root,
        entries_path,
        materialize_links=True,
    )

    entries = [
        json.loads(line)
        for line in entries_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["path"] for entry in entries] == [
        "alias/data.txt",
        "target/data.txt",
    ]
    assert summary.entry_count == 2
    assert metadata_fingerprint(root, materialize_links=True) == summary.metadata_sha256

    (root / "broken").symlink_to(root / "missing")
    with pytest.raises(ValueError, match="broken link"):
        write_inventory_stream(root, entries_path, materialize_links=True)


def test_streamed_inventory_rejects_unsupported_special_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    os.mkfifo(root / "named-pipe")

    with pytest.raises(ValueError, match="special file"):
        write_inventory_stream(root, tmp_path / "destination.jsonl")
    with pytest.raises(ValueError, match="special file"):
        write_inventory_stream(
            root,
            tmp_path / "materialized.jsonl",
            materialize_links=True,
        )


def test_merge_inventory_streams_orders_prefixes_and_control_entries(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "z.txt").write_text("z\n", encoding="utf-8")
    (second / "a.txt").write_text("a\n", encoding="utf-8")
    first_stream = tmp_path / "first.jsonl"
    second_stream = tmp_path / "second.jsonl"
    write_inventory_stream(first, first_stream)
    write_inventory_stream(second, second_stream)
    output = tmp_path / "file_inventory.json"
    control = InventoryEntry(path="README.md", size_bytes=7, sha256="a" * 64)

    summary = merge_inventory_streams(
        [
            (Path("data/z-cell"), first_stream),
            (Path("data/a-cell"), second_stream),
        ],
        [control],
        output,
        root=tmp_path / "dataset",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["root"] == str((tmp_path / "dataset").resolve())
    assert payload["file_count"] == 3
    assert payload["total_bytes"] == 11
    assert [entry["path"] for entry in payload["entries"]] == [
        "README.md",
        "data/a-cell/a.txt",
        "data/z-cell/z.txt",
    ]
    assert payload["inventory_sha256"] == summary.inventory_sha256
    assert summary.entry_count == 3
    assert summary.total_bytes == 11


def test_merge_inventory_streams_rejects_duplicates_without_replacing_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "same.txt").write_text("one\n", encoding="utf-8")
    stream_path = tmp_path / "stream.jsonl"
    write_inventory_stream(root, stream_path)
    output = tmp_path / "inventory.json"
    output.write_text("previous output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate inventory path"):
        merge_inventory_streams(
            [(Path(), stream_path)],
            [InventoryEntry(path="same.txt", size_bytes=4, sha256="b" * 64)],
            output,
            root=tmp_path,
        )

    assert output.read_text(encoding="utf-8") == "previous output\n"


def test_merge_inventory_streams_rejects_malformed_input_stream(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text('{"path":"b"}\n{"path":"a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="inventory stream"):
        merge_inventory_streams(
            [(Path("data"), malformed)],
            [],
            tmp_path / "inventory.json",
            root=tmp_path,
        )
