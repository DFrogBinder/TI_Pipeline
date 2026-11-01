"""Deterministic file inventories for copied and materialized trees."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Iterable


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FileInventory:
    root: str
    entries: tuple[InventoryEntry, ...]
    total_bytes: int
    inventory_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "root": self.root,
            "file_count": len(self.entries),
            "total_bytes": self.total_bytes,
            "inventory_sha256": self.inventory_sha256,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def estimate_materialized_bytes(root: Path) -> int:
    """Count logical file bytes after link materialization without hashing data."""
    try:
        target = root.resolve(strict=True) if root.is_symlink() else root
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"Materialized byte source has a broken link: {root}") from exc
    if target.is_file():
        return int(target.stat().st_size)
    if not target.is_dir():
        raise ValueError(f"Materialized byte source is missing: {root}")

    total = 0

    def visit(directory: Path, ancestors: frozenset[tuple[int, int]]) -> None:
        nonlocal total
        info = directory.stat()
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in ancestors:
            raise ValueError(f"Source link materialization cycle: {directory}")
        next_ancestors = ancestors | {identity}
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            try:
                materialized = child.resolve(strict=True) if child.is_symlink() else child
            except (FileNotFoundError, RuntimeError) as exc:
                raise ValueError(
                    f"Materialized byte source has a broken link: {child}"
                ) from exc
            if materialized.is_dir():
                visit(materialized, next_ancestors)
            elif materialized.is_file():
                total += int(materialized.stat().st_size)
            else:
                raise ValueError(
                    f"Source contains unsupported special file: {child}"
                )

    visit(target, frozenset())
    return total


def _finish(root: Path, records: Iterable[tuple[str, Path]]) -> FileInventory:
    entries = tuple(
        InventoryEntry(path=relative, size_bytes=path.stat().st_size, sha256=sha256_file(path))
        for relative, path in sorted(records, key=lambda record: record[0])
    )
    encoded = json.dumps(
        [entry.as_dict() for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return FileInventory(
        root=str(root.resolve(strict=False)),
        entries=entries,
        total_bytes=sum(entry.size_bytes for entry in entries),
        inventory_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def build_file_inventory(
    root: Path,
    *,
    exclude: Iterable[Path] = (),
) -> FileInventory:
    """Hash a destination tree and reject every symbolic or special file."""
    if root.is_symlink():
        raise ValueError(f"Inventory root is a symlink: {root}")
    root = root.resolve(strict=False)
    excluded = {path.resolve(strict=False) for path in exclude}
    if root.is_file():
        return _finish(root, ((".", root),))
    if not root.is_dir():
        raise ValueError(f"Inventory root is missing: {root}")
    records: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.resolve(strict=False) in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"Destination inventory contains symlink: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Destination inventory contains special file: {path}")
        records.append((path.relative_to(root).as_posix(), path))
    return _finish(root, records)


def build_materialized_inventory(root: Path) -> FileInventory:
    """Hash a source as it will appear after rsync ``--copy-links``."""
    logical_root = root.resolve(strict=False)
    if root.is_file() or (root.is_symlink() and root.resolve(strict=True).is_file()):
        return _finish(logical_root, ((".", root),))
    if not root.is_dir():
        raise ValueError(f"Materialized inventory root is missing: {root}")
    records: list[tuple[str, Path]] = []

    def visit(
        physical: Path,
        logical: Path,
        ancestors: frozenset[tuple[int, int]],
    ) -> None:
        info = physical.stat()
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in ancestors:
            raise ValueError(f"Source link materialization cycle: {logical}")
        next_ancestors = ancestors | {identity}
        for child in sorted(physical.iterdir(), key=lambda path: path.name):
            target = child.resolve(strict=True) if child.is_symlink() else child
            logical_child = logical / child.name
            if target.is_dir():
                visit(target, logical_child, next_ancestors)
            elif target.is_file():
                records.append((logical_child.as_posix(), target))
            else:
                raise ValueError(
                    f"Source contains unsupported special file: {logical_child}"
                )

    visit(root.resolve(strict=True), Path(), frozenset())
    return _finish(logical_root, records)
