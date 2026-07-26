#!/usr/bin/env python3
"""Provenance and filesystem validation helpers for staged HPC runs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SBATCH_JOB_RE = re.compile(r"\bSubmitted batch job\s+(\d+)\b")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_event(path: Path, event: str, **fields: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": utc_timestamp(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
    return payload


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    return path


def parse_sbatch_job_id(output: str) -> str | None:
    match = SBATCH_JOB_RE.search(output)
    return match.group(1) if match else None


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def find_symlinks(root: Path) -> list[Path]:
    if root.is_symlink():
        return [root]
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_symlink()]


def has_no_symlinks(root: Path) -> bool:
    return not find_symlinks(root)


def write_stage_status(experiment_root: Path, status: dict[str, Any]) -> Path:
    return write_json(experiment_root / "_pipeline" / "stage_status.json", status)


def write_submitted_job_record(
    experiment_root: Path,
    *,
    stage: str,
    command: list[str],
    env: dict[str, str],
    stdout: str,
    stderr: str,
    returncode: int,
    job_id: str | None,
    expected_outputs: dict[str, Any],
) -> Path:
    current_path = (
        experiment_root / "_pipeline" / "submitted_jobs" / f"{stage}.json"
    )
    if current_path.is_file():
        existing = read_json(current_path)
        timestamp = str(existing.get("timestamp_utc", "unknown"))
        safe_timestamp = re.sub(r"[^0-9A-Za-z]+", "-", timestamp).strip("-")
        attempts_dir = current_path.parent / "attempts" / stage
        attempts_dir.mkdir(parents=True, exist_ok=True)
        archive_path = attempts_dir / f"{safe_timestamp or 'unknown'}.json"
        suffix = 1
        while archive_path.exists():
            archive_path = attempts_dir / (
                f"{safe_timestamp or 'unknown'}-{suffix}.json"
            )
            suffix += 1
        current_path.replace(archive_path)

    payload = {
        "stage": stage,
        "timestamp_utc": utc_timestamp(),
        "command": command,
        "env": env,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "job_id": job_id,
        "expected_outputs": expected_outputs,
    }
    return write_json(current_path, payload)
