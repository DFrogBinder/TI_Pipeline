"""Bounded-memory file inventories and checkpoint validation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterable, Iterator, Optional, Sequence

from .inventory import InventoryEntry, sha256_file


@dataclass(frozen=True)
class InventoryStreamSummary:
    root: str
    entry_count: int
    total_bytes: int
    inventory_sha256: str
    entries_sha256: str
    metadata_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _destination_records(
    root: Path,
    excluded: frozenset[Path],
) -> Iterator[tuple[str, Path]]:
    if root.is_symlink():
        raise ValueError(f"Inventory root is a symlink: {root}")
    if root.is_file():
        yield ".", root
        return
    if not root.is_dir():
        raise ValueError(f"Inventory root is missing: {root}")

    def visit(directory: Path, relative: Path) -> Iterator[tuple[str, Path]]:
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if child.resolve(strict=False) in excluded:
                continue
            if child.is_symlink():
                raise ValueError(f"Destination inventory contains symlink: {child}")
            mode = child.lstat().st_mode
            logical = relative / child.name
            if stat.S_ISDIR(mode):
                yield from visit(child, logical)
            elif stat.S_ISREG(mode):
                yield logical.as_posix(), child
            else:
                raise ValueError(
                    f"Destination inventory contains special file: {child}"
                )

    yield from visit(root, Path())


def _materialized_records(root: Path) -> Iterator[tuple[str, Path]]:
    try:
        physical_root = root.resolve(strict=True) if root.is_symlink() else root
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"Materialized inventory has a broken link: {root}") from exc
    if physical_root.is_file():
        yield ".", physical_root
        return
    if not physical_root.is_dir():
        raise ValueError(f"Materialized inventory root is missing: {root}")

    def visit(
        physical: Path,
        logical: Path,
        ancestors: frozenset[tuple[int, int]],
    ) -> Iterator[tuple[str, Path]]:
        info = physical.stat()
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in ancestors:
            raise ValueError(f"Source link materialization cycle: {logical}")
        next_ancestors = ancestors | {identity}
        for child in sorted(physical.iterdir(), key=lambda path: path.name):
            logical_child = logical / child.name
            try:
                target = child.resolve(strict=True) if child.is_symlink() else child
            except (FileNotFoundError, RuntimeError) as exc:
                raise ValueError(
                    f"Materialized inventory has a broken link: {logical_child}"
                ) from exc
            if target.is_dir():
                yield from visit(target, logical_child, next_ancestors)
            elif target.is_file():
                yield logical_child.as_posix(), target
            else:
                raise ValueError(
                    f"Source contains unsupported special file: {logical_child}"
                )

    yield from visit(physical_root, Path(), frozenset())


def _records(
    root: Path,
    *,
    materialize_links: bool,
    exclude: Iterable[Path],
) -> Iterator[tuple[str, Path]]:
    if materialize_links:
        if tuple(exclude):
            raise ValueError("Materialized inventory exclusions are unsupported")
        yield from _materialized_records(root)
        return
    excluded = frozenset(path.resolve(strict=False) for path in exclude)
    yield from _destination_records(root.resolve(strict=False), excluded)


def _metadata_record(relative: str, path: Path) -> dict[str, object]:
    info = path.stat()
    return {
        "path": relative,
        "type": "file",
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size_bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def metadata_fingerprint(
    root: Path,
    *,
    materialize_links: bool = False,
    exclude: Iterable[Path] = (),
) -> str:
    """Fingerprint path identity and metadata without reading file bodies."""
    digest = hashlib.sha256()
    for relative, path in _records(
        root,
        materialize_links=materialize_links,
        exclude=exclude,
    ):
        digest.update(_canonical(_metadata_record(relative, path)))
    return digest.hexdigest()


def write_inventory_stream(
    root: Path,
    entries_path: Path,
    *,
    materialize_links: bool = False,
    exclude: Iterable[Path] = (),
) -> InventoryStreamSummary:
    """Hash *root* into sorted canonical JSONL without retaining all entries."""
    logical_root = root.resolve(strict=False)
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    list_digest = hashlib.sha256()
    entries_digest = hashlib.sha256()
    metadata_digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0

    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=entries_path.parent,
            prefix=f".{entries_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            list_digest.update(b"[")
            for relative, path in _records(
                root,
                materialize_links=materialize_links,
                exclude=exclude,
            ):
                info = path.stat()
                entry = InventoryEntry(
                    path=relative,
                    size_bytes=int(info.st_size),
                    sha256=sha256_file(path),
                )
                encoded = _canonical(entry.as_dict())
                if entry_count:
                    list_digest.update(b",")
                list_digest.update(encoded)
                line = encoded + b"\n"
                stream.write(line)
                entries_digest.update(line)
                metadata_digest.update(_canonical(_metadata_record(relative, path)))
                entry_count += 1
                total_bytes += entry.size_bytes
            list_digest.update(b"]")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, entries_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return InventoryStreamSummary(
        root=str(logical_root),
        entry_count=entry_count,
        total_bytes=total_bytes,
        inventory_sha256=list_digest.hexdigest(),
        entries_sha256=entries_digest.hexdigest(),
        metadata_sha256=metadata_digest.hexdigest(),
    )


def validate_inventory_stream(
    entries_path: Path,
    summary: InventoryStreamSummary,
) -> None:
    """Validate a canonical JSONL inventory stream against its summary."""
    entries_digest = hashlib.sha256()
    list_digest = hashlib.sha256()
    list_digest.update(b"[")
    entry_count = 0
    total_bytes = 0
    previous_path: Optional[str] = None
    try:
        with entries_path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n") or line == b"\n":
                    raise ValueError("inventory stream contains a malformed line")
                try:
                    payload = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("inventory stream contains invalid JSON") from exc
                if not isinstance(payload, dict) or set(payload) != {
                    "path",
                    "size_bytes",
                    "sha256",
                }:
                    raise ValueError("inventory stream entry schema differs")
                path = payload["path"]
                size = payload["size_bytes"]
                sha256 = payload["sha256"]
                if (
                    not isinstance(path, str)
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or size < 0
                    or not isinstance(sha256, str)
                    or len(sha256) != 64
                ):
                    raise ValueError("inventory stream entry values are invalid")
                encoded = _canonical(payload)
                if line != encoded + b"\n":
                    raise ValueError("inventory stream line is not canonical")
                if previous_path is not None and path <= previous_path:
                    raise ValueError("inventory stream paths are not strictly ordered")
                previous_path = path
                if entry_count:
                    list_digest.update(b",")
                list_digest.update(encoded)
                entries_digest.update(line)
                entry_count += 1
                total_bytes += size
    except OSError as exc:
        raise ValueError(f"inventory stream cannot be read: {entries_path}") from exc
    list_digest.update(b"]")

    if (
        entry_count != summary.entry_count
        or total_bytes != summary.total_bytes
        or list_digest.hexdigest() != summary.inventory_sha256
        or entries_digest.hexdigest() != summary.entries_sha256
    ):
        raise ValueError("inventory stream does not match its summary")


def _read_inventory_entries(entries_path: Path) -> Iterator[InventoryEntry]:
    previous: Optional[str] = None
    try:
        with entries_path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n") or line == b"\n":
                    raise ValueError("inventory stream contains a malformed line")
                try:
                    payload = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("inventory stream contains invalid JSON") from exc
                if not isinstance(payload, dict) or set(payload) != {
                    "path",
                    "size_bytes",
                    "sha256",
                }:
                    raise ValueError("inventory stream entry schema differs")
                if line != _canonical(payload) + b"\n":
                    raise ValueError("inventory stream line is not canonical")
                path = payload["path"]
                size = payload["size_bytes"]
                sha256 = payload["sha256"]
                if (
                    not isinstance(path, str)
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or size < 0
                    or not isinstance(sha256, str)
                    or len(sha256) != 64
                ):
                    raise ValueError("inventory stream entry values are invalid")
                if previous is not None and path <= previous:
                    raise ValueError("inventory stream paths are not strictly ordered")
                previous = path
                yield InventoryEntry(path=path, size_bytes=size, sha256=sha256)
    except OSError as exc:
        raise ValueError(f"inventory stream cannot be read: {entries_path}") from exc


def _prefixed_entries(prefix: Path, entries_path: Path) -> Iterator[InventoryEntry]:
    if prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError(f"inventory stream prefix must be relative: {prefix}")
    for entry in _read_inventory_entries(entries_path):
        if entry.path == ".":
            logical = prefix.as_posix()
        else:
            logical = (prefix / entry.path).as_posix()
        yield InventoryEntry(
            path=logical,
            size_bytes=entry.size_bytes,
            sha256=entry.sha256,
        )


def merge_inventory_streams(
    streams: Sequence[tuple[Path, Path]],
    control_entries: Iterable[InventoryEntry],
    output_path: Path,
    *,
    root: Path,
) -> InventoryStreamSummary:
    """Merge sorted JSONL streams into a legacy-compatible inventory JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterables: list[Iterator[InventoryEntry]] = [
        _prefixed_entries(prefix, entries_path)
        for prefix, entries_path in streams
    ]
    controls = sorted(control_entries, key=lambda entry: entry.path)
    iterables.append(iter(controls))
    merged = heapq.merge(*iterables, key=lambda entry: entry.path)

    entries_temporary: Optional[Path] = None
    output_temporary: Optional[Path] = None
    list_digest = hashlib.sha256()
    entries_digest = hashlib.sha256()
    entry_count = 0
    total_bytes = 0
    previous: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.entries.",
            suffix=".tmp",
            delete=False,
        ) as entry_stream:
            entries_temporary = Path(entry_stream.name)
            list_digest.update(b"[")
            for entry in merged:
                if previous is not None and entry.path == previous:
                    raise ValueError(f"duplicate inventory path: {entry.path}")
                if previous is not None and entry.path < previous:
                    raise ValueError("inventory stream merge is not ordered")
                previous = entry.path
                encoded = _canonical(entry.as_dict())
                if entry_count:
                    list_digest.update(b",")
                list_digest.update(encoded)
                line = encoded + b"\n"
                entry_stream.write(line)
                entries_digest.update(line)
                entry_count += 1
                total_bytes += entry.size_bytes
            list_digest.update(b"]")
            entry_stream.flush()
            os.fsync(entry_stream.fileno())

        inventory_sha256 = list_digest.hexdigest()
        header = {
            "schema_version": 1,
            "root": str(root.resolve(strict=False)),
            "file_count": entry_count,
            "total_bytes": total_bytes,
            "inventory_sha256": inventory_sha256,
        }
        header_bytes = _canonical(header)
        assert header_bytes.endswith(b"}")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_stream:
            output_temporary = Path(output_stream.name)
            output_stream.write(header_bytes[:-1])
            output_stream.write(b',"entries":[')
            first = True
            with entries_temporary.open("rb") as entry_stream:
                for line in entry_stream:
                    if not first:
                        output_stream.write(b",")
                    output_stream.write(line.rstrip(b"\n"))
                    first = False
            output_stream.write(b"]}\n")
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(output_temporary, output_path)
        output_temporary = None
    finally:
        if entries_temporary is not None:
            entries_temporary.unlink(missing_ok=True)
        if output_temporary is not None:
            output_temporary.unlink(missing_ok=True)

    return InventoryStreamSummary(
        root=str(root.resolve(strict=False)),
        entry_count=entry_count,
        total_bytes=total_bytes,
        inventory_sha256=inventory_sha256,
        entries_sha256=entries_digest.hexdigest(),
        metadata_sha256=entries_digest.hexdigest(),
    )
