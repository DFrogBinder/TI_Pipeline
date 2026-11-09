from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class ProvenanceError(RuntimeError):
    """Raised when pinned provenance cannot be proved."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    """Write stable JSON atomically and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def write_sha256_sidecar(path: Path) -> Path:
    sidecar = path.with_name(path.name + ".sha256")
    line = f"{sha256_file(path)}  {path.name}\n".encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(sidecar))
    finally:
        if temporary.exists():
            temporary.unlink()
    return sidecar


def path_is_within(path: Path, root: Path) -> bool:
    candidate = path.expanduser().resolve(strict=False)
    boundary = root.expanduser().resolve(strict=False)
    try:
        candidate.relative_to(boundary)
    except ValueError:
        return False
    return True


def require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ProvenanceError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ProvenanceError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}: {path}"
        )
