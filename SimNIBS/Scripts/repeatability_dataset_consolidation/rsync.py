"""Argument-array rsync execution for safe per-item copy and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


class RsyncError(RuntimeError):
    """Raised when rsync cannot copy or compare a planned item."""


@dataclass(frozen=True)
class RsyncResult:
    command: tuple[str, ...]
    changes: tuple[str, ...]

    @property
    def matched(self) -> bool:
        return not self.changes


def _endpoints(source: Path, destination: Path) -> tuple[str, str]:
    if source.is_dir() and not source.is_symlink():
        return f"{source}/", f"{destination}/"
    return str(source), str(destination)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RsyncError(
            f"rsync exited {completed.returncode}: {detail or 'no diagnostic'}"
        )
    return completed


def compare_item(
    source: Path,
    destination: Path,
    *,
    checksum: bool,
) -> RsyncResult:
    """Compare one item without changing either tree."""
    source_arg, destination_arg = _endpoints(source, destination)
    command = [
        "rsync",
        "-rltH",
        "--copy-links",
        "--protect-args",
        "--dry-run",
        "--itemize-changes",
        "--out-format=%i|%n%L",
        "--delete",
    ]
    if checksum:
        command.append("--checksum")
    command.extend((source_arg, destination_arg))
    if not destination.exists() and not destination.is_symlink():
        return RsyncResult(tuple(command), (f"missing|{destination}",))
    if not destination.parent.is_dir():
        return RsyncResult(tuple(command), (f"missing-parent|{destination.parent}",))
    completed = _run(command)
    changes = tuple(line for line in completed.stdout.splitlines() if line.strip())
    return RsyncResult(tuple(command), changes)

def copy_item(source: Path, destination: Path) -> RsyncResult:
    """Physically copy one item without deleting anything from the destination."""
    if source.is_dir() and not source.is_symlink():
        destination.mkdir(parents=True, exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    source_arg, destination_arg = _endpoints(source, destination)
    command = [
        "rsync",
        "-rltH",
        "--copy-links",
        "--partial",
        "--protect-args",
        "--checksum",
        "--itemize-changes",
        "--out-format=%i|%n%L",
        source_arg,
        destination_arg,
    ]
    completed = _run(command)
    changes = tuple(line for line in completed.stdout.splitlines() if line.strip())
    return RsyncResult(tuple(command), changes)

