"""Exhaustive, checksum-bound disposition of every retirement-root file."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterator, Optional, TextIO

from .inventory import sha256_file
from .planning import ConsolidationPlan, CopyItem
from .settings import DatasetConfig, retirement_allowlist
from .streaming_inventory import metadata_fingerprint


class DispositionError(RuntimeError):
    """Raised when retirement-root content cannot be classified safely."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def disposition_path(config: DatasetConfig) -> Path:
    return config.output_root / "provenance/retirement/source_disposition.json"


def retirement_root_metadata(config: DatasetConfig) -> dict[str, dict[str, object]]:
    """Fingerprint live retirement roots without reading scientific file bodies."""
    records: dict[str, dict[str, object]] = {}
    for key, configured_root in retirement_allowlist(config).items():
        if not configured_root.is_dir() or configured_root.is_symlink():
            raise DispositionError(
                f"Retirement metadata root is missing or unsafe: {configured_root}"
            )
        root = configured_root.resolve()
        before = root.lstat()
        try:
            fingerprint = metadata_fingerprint(root, materialize_links=True)
        except ValueError as exc:
            raise DispositionError(
                f"Retirement metadata cannot be fingerprinted: {root}"
            ) from exc
        after = root.lstat()
        before_identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        after_identity = (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        if before_identity != after_identity:
            raise DispositionError(
                f"Retirement metadata root changed while fingerprinting: {root}"
            )
        records[key] = {
            "path": str(root),
            "device": before_identity[0],
            "inode": before_identity[1],
            "size_bytes": before_identity[2],
            "mtime_ns": before_identity[3],
            "ctime_ns": before_identity[4],
            "metadata_sha256": fingerprint,
        }
    return records


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


def _root_paths(root: Path) -> Iterator[Path]:
    """Yield a stable tree order while retaining only one directory in memory."""

    def visit(directory: Path) -> Iterator[Path]:
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise DispositionError(
                f"Retirement root cannot be enumerated: {directory}"
            ) from exc
        for child in children:
            mode = child.lstat().st_mode
            if stat.S_ISDIR(mode):
                yield from visit(child)
            else:
                yield child

    yield from visit(root)


def _empty_counts() -> dict[str, int]:
    return {
        "dataset_copy_files": 0,
        "dataset_copy_bytes": 0,
        "shared_archive_files": 0,
        "shared_archive_bytes": 0,
        "intentional_retirement_files": 0,
        "intentional_retirement_bytes": 0,
    }


def _disposition_entries(
    config: DatasetConfig,
    plan: ConsolidationPlan,
    summary: dict[str, dict[str, int]],
) -> Iterator[dict[str, object]]:
    archive_keys = set(config.archive_sources)
    for root_key, configured_root in retirement_allowlist(config).items():
        if not configured_root.is_dir() or configured_root.is_symlink():
            raise DispositionError(
                f"Retirement disposition root is missing or unsafe: {configured_root}"
            )
        root = configured_root.resolve()
        counts = _empty_counts()
        for path in _root_paths(root):
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
                disposition = (
                    "dataset_copy" if destinations else "intentional_retirement"
                )
            record = _entry(
                root_key=root_key,
                root=root,
                path=path,
                disposition=disposition,
                destinations=destinations,
            )
            counts[f"{disposition}_files"] += 1
            counts[f"{disposition}_bytes"] += int(record["size_bytes"])
            yield record
        summary[root_key] = counts


def build_retirement_disposition(
    config: DatasetConfig,
    plan: ConsolidationPlan,
) -> dict[str, object]:
    """Classify and hash every regular file or symlink below every deletion root."""
    entries: list[dict[str, object]] = []
    summary: dict[str, dict[str, int]] = {}
    entries.extend(_disposition_entries(config, plan, summary))
    encoded = _canonical(entries)
    return {
        "schema_version": 1,
        "status": "complete",
        "entries_sha256": hashlib.sha256(encoded).hexdigest(),
        "entries": entries,
        "summary": summary,
    }


def write_retirement_disposition_stream(
    config: DatasetConfig,
    plan: ConsolidationPlan,
    path: Path,
) -> dict[str, object]:
    """Write the exhaustive disposition atomically with bounded memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    digest = hashlib.sha256()
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(b'{"schema_version":1,"status":"complete","entries":[')
            digest.update(b"[")
            count = 0
            for record in _disposition_entries(config, plan, summary):
                encoded = _canonical(record)
                if count:
                    stream.write(b",")
                    digest.update(b",")
                stream.write(encoded)
                digest.update(encoded)
                count += 1
            stream.write(b"]")
            digest.update(b"]")
            entries_sha256 = digest.hexdigest()
            stream.write(b',"entries_sha256":')
            stream.write(_canonical(entries_sha256))
            stream.write(b',"summary":')
            stream.write(_canonical(summary))
            stream.write(b"}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "status": "complete",
        "entries_sha256": entries_sha256,
        "summary": summary,
    }


class _JSONStream:
    """Small incremental JSON reader for one disposition object."""

    _MAX_VALUE_BYTES = 16 * 1024 * 1024

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _compact_and_fill(self) -> None:
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        block = self.stream.read(1024 * 1024)
        if block:
            self.buffer += block
        else:
            self.eof = True

    def _skip_space(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[
                self.position
            ].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._compact_and_fill()

    def peek(self) -> str:
        self._skip_space()
        if self.position >= len(self.buffer):
            raise DispositionError("Retirement disposition ended unexpectedly")
        return self.buffer[self.position]

    def consume(self, expected: str) -> None:
        if self.peek() != expected:
            raise DispositionError(
                f"Retirement disposition expected {expected!r}"
            )
        self.position += 1

    def value(self) -> object:
        self._skip_space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise DispositionError(
                        "Retirement disposition contains invalid JSON"
                    ) from exc
                if len(self.buffer) - self.position > self._MAX_VALUE_BYTES:
                    raise DispositionError(
                        "Retirement disposition contains an oversized value"
                    ) from exc
                self._compact_and_fill()
            else:
                self.position = end
                return value

    def finish(self) -> None:
        self._skip_space()
        if self.position < len(self.buffer):
            raise DispositionError("Retirement disposition has trailing content")
        if not self.eof:
            self._compact_and_fill()
            self._skip_space()
            if self.position < len(self.buffer):
                raise DispositionError("Retirement disposition has trailing content")


def _parse_entries(
    parser: _JSONStream,
    config: DatasetConfig,
) -> tuple[str, dict[str, dict[str, int]]]:
    parser.consume("[")
    digest = hashlib.sha256()
    digest.update(b"[")
    summary = {
        key: _empty_counts() for key in retirement_allowlist(config)
    }
    count = 0
    while parser.peek() != "]":
        value = parser.value()
        if not isinstance(value, dict):
            raise DispositionError("Retirement disposition entry is invalid")
        required = {
            "root_key",
            "root",
            "relative_path",
            "type",
            "size_bytes",
            "sha256",
            "link_target",
            "disposition",
            "destinations",
        }
        if set(value) != required:
            raise DispositionError("Retirement disposition entry schema differs")
        root_key = value.get("root_key")
        disposition = value.get("disposition")
        size = value.get("size_bytes")
        if (
            not isinstance(root_key, str)
            or disposition
            not in {"dataset_copy", "shared_archive", "intentional_retirement"}
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or root_key not in summary
        ):
            raise DispositionError("Retirement disposition entry values are invalid")
        counts = summary[root_key]
        counts[f"{disposition}_files"] += 1
        counts[f"{disposition}_bytes"] += size
        encoded = _canonical(value)
        if count:
            digest.update(b",")
        digest.update(encoded)
        count += 1
        if parser.peek() == ",":
            parser.consume(",")
        elif parser.peek() != "]":
            raise DispositionError("Retirement disposition entries are malformed")
    parser.consume("]")
    digest.update(b"]")
    return digest.hexdigest(), summary


def validate_retirement_disposition_stream(
    config: DatasetConfig,
    path: Path,
) -> dict[str, object]:
    """Validate an existing disposition without materializing its entries."""
    if not path.is_file() or path.is_symlink():
        raise DispositionError(f"Retirement disposition is missing or unsafe: {path}")
    fields: dict[str, object] = {}
    calculated_sha: Optional[str] = None
    calculated_summary: Optional[dict[str, dict[str, int]]] = None
    try:
        with path.open("r", encoding="utf-8") as stream:
            parser = _JSONStream(stream)
            parser.consume("{")
            while parser.peek() != "}":
                key = parser.value()
                if not isinstance(key, str) or key in fields or key == "entries":
                    if key == "entries" and calculated_sha is None:
                        pass
                    else:
                        raise DispositionError(
                            "Retirement disposition has duplicate or invalid keys"
                        )
                parser.consume(":")
                if key == "entries":
                    calculated_sha, calculated_summary = _parse_entries(parser, config)
                else:
                    fields[key] = parser.value()
                if parser.peek() == ",":
                    parser.consume(",")
                elif parser.peek() != "}":
                    raise DispositionError("Retirement disposition is malformed")
            parser.consume("}")
            parser.finish()
    except OSError as exc:
        raise DispositionError("Retirement disposition cannot be read") from exc
    expected_keys = {"schema_version", "status", "entries_sha256", "summary"}
    if (
        set(fields) != expected_keys
        or fields.get("schema_version") != 1
        or fields.get("status") != "complete"
        or calculated_sha is None
        or calculated_summary is None
        or fields.get("entries_sha256") != calculated_sha
        or fields.get("summary") != calculated_summary
        or set(calculated_summary) != set(retirement_allowlist(config))
    ):
        raise DispositionError("Retirement disposition bindings are invalid")
    return {
        "schema_version": 1,
        "status": "complete",
        "entries_sha256": calculated_sha,
        "summary": calculated_summary,
    }
