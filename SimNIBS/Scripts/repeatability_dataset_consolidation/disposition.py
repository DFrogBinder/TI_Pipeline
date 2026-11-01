"""Exhaustive, checksum-bound disposition of every retirement-root file."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from .inventory import sha256_file
from .planning import ConsolidationPlan, CopyItem
from .settings import DatasetConfig, retirement_allowlist


class DispositionError(RuntimeError):
    """Raised when retirement-root content cannot be classified safely."""


def disposition_path(config: DatasetConfig) -> Path:
    return config.output_root / "provenance/retirement/source_disposition.json"


def _destinations_for(
    config: DatasetConfig,
    item: CopyItem,
    source_path: Path,
) -> tuple[str, ...]:
    source = item.source.absolute()
    candidate = source_path.absolute()
    if item.source_type == "directory":
        if candidate != source and source not in candidate.parents:
            return ()
        relative = candidate.relative_to(source)
        return (str((config.output_root / item.destination / relative).absolute()),)
    if candidate == source:
        return (str((config.output_root / item.destination).absolute()),)
    return ()


def _entry(
    *,
    root_key: str,
    root: Path,
    path: Path,
    disposition: str,
    destinations: tuple[str, ...],
) -> dict[str, object]:
    info = path.lstat()
    relative = path.relative_to(root).as_posix()
    if stat.S_ISREG(info.st_mode):
        entry_type = "file"
        size = int(info.st_size)
        digest = sha256_file(path)
        link_target = None
    elif stat.S_ISLNK(info.st_mode):
        entry_type = "symlink"
        link_target = os.readlink(path)
        encoded = link_target.encode("utf-8", errors="surrogateescape")
        size = len(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
    else:
        raise DispositionError(f"Retirement root contains unsupported entry: {path}")
    return {
        "root_key": root_key,
        "root": str(root),
        "relative_path": relative,
        "type": entry_type,
        "size_bytes": size,
        "sha256": digest,
        "link_target": link_target,
        "disposition": disposition,
        "destinations": list(destinations),
    }


def build_retirement_disposition(
    config: DatasetConfig,
    plan: ConsolidationPlan,
) -> dict[str, object]:
    """Classify and hash every regular file or symlink below every deletion root."""
    archive_keys = set(config.archive_sources)
    entries: list[dict[str, object]] = []
    summary: dict[str, dict[str, int]] = {}
    for root_key, configured_root in retirement_allowlist(config).items():
        if not configured_root.is_dir() or configured_root.is_symlink():
            raise DispositionError(
                f"Retirement disposition root is missing or unsafe: {configured_root}"
            )
        root = configured_root.resolve()
        counts = {
            "dataset_copy_files": 0,
            "dataset_copy_bytes": 0,
            "shared_archive_files": 0,
            "shared_archive_bytes": 0,
            "intentional_retirement_files": 0,
            "intentional_retirement_bytes": 0,
        }
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if root_key in archive_keys:
                disposition = "shared_archive"
                destinations = (
                    str(
                        config.shared_archive_root
                        / root.name
                        / path.relative_to(root)
                    ),
                )
            else:
                mapped = tuple(
                    destination
                    for item in plan.items
                    for destination in _destinations_for(config, item, path)
                )
                destinations = tuple(sorted(set(mapped)))
                disposition = "dataset_copy" if destinations else "intentional_retirement"
            record = _entry(
                root_key=root_key,
                root=root,
                path=path,
                disposition=disposition,
                destinations=destinations,
            )
            entries.append(record)
            counts[f"{disposition}_files"] += 1
            counts[f"{disposition}_bytes"] += int(record["size_bytes"])
        summary[root_key] = counts
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "status": "complete",
        "entries_sha256": hashlib.sha256(encoded).hexdigest(),
        "entries": entries,
        "summary": summary,
    }
