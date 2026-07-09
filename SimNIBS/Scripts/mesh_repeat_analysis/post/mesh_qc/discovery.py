from __future__ import annotations

import re
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class MeshRecord:
    path: Path
    roi: str
    subject: str
    repeat: str
    mesh_id: str


@dataclass(frozen=True)
class DiscoveryStats:
    dirs_scanned: int
    files_seen: int
    matches: int
    current_dir: Path


ROI_HINTS = ("roi", "hippocampus", "pallidum", "m1", "motor", "target", "dlpc", "dlpfc", "thalamus")
SUBJECT_RE = re.compile(r"^(sub-[A-Za-z0-9_-]+|CC\d+)$", re.IGNORECASE)
REPEAT_RE = re.compile(r"^(repeat[-_ ]?\d+|rep[-_ ]?\d+|seed[-_ ]?\d+)$", re.IGNORECASE)
DATA_REPEAT_RE = re.compile(r"(?:^|[-_ ])(?:data|run|runs|repeat|rep|seed)[-_ ]?(\d+)$", re.IGNORECASE)


def _relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    return rel.parts


def _first_regex(parts: Iterable[str], pattern: str | None) -> str | None:
    if not pattern:
        return None
    regex = re.compile(pattern)
    for part in parts:
        match = regex.search(part)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return None


def infer_record(
    path: Path,
    *,
    root: Path,
    roi_regex: str | None = None,
    subject_regex: str | None = None,
    repeat_regex: str | None = None,
) -> MeshRecord:
    parts = _relative_parts(path, root)
    dir_parts = parts[:-1]

    roi = _first_regex(dir_parts, roi_regex)
    if roi is None:
        roi = next(
            (
                part
                for part in dir_parts
                if any(hint in part.lower() for hint in ROI_HINTS)
            ),
            "unknown_roi",
        )

    subject = _first_regex(dir_parts, subject_regex)
    if subject is None:
        subject = next((part for part in dir_parts if SUBJECT_RE.match(part)), "unknown_subject")

    repeat = _first_regex(dir_parts, repeat_regex)
    if repeat is None:
        repeat = next((part for part in dir_parts if REPEAT_RE.match(part)), "unknown_repeat")
    if repeat == "unknown_repeat":
        for part in dir_parts:
            match = DATA_REPEAT_RE.search(part)
            if match:
                repeat = match.group(1)
                break
    if repeat == "unknown_repeat":
        for idx, part in enumerate(dir_parts[:-1]):
            if "repeat" in part.lower() and dir_parts[idx + 1].isdigit():
                repeat = dir_parts[idx + 1]
                break

    mesh_id = next((part for part in reversed(dir_parts) if part.lower().startswith("m2m")), path.stem)

    return MeshRecord(path=path, roi=roi, subject=subject, repeat=repeat, mesh_id=mesh_id)


def _path_has_m2m_dir(path: Path) -> bool:
    return any(part.lower().startswith("m2m") for part in path.parts[:-1])


def discover_meshes(
    root: Path,
    *,
    mesh_glob: str | None = None,
    roi_regex: str | None = None,
    subject_regex: str | None = None,
    repeat_regex: str | None = None,
    progress_callback: Callable[[DiscoveryStats], None] | None = None,
    progress_interval_sec: float = 5.0,
) -> list[MeshRecord]:
    root = Path(root).expanduser().resolve()
    paths: list[Path] = []
    dirs_scanned = 0
    files_seen = 0
    last_progress = time.monotonic()

    for dirpath, _, filenames in os.walk(root):
        dirs_scanned += 1
        current_dir = Path(dirpath)
        files_seen += len(filenames)
        for filename in filenames:
            path = current_dir / filename
            if mesh_glob is None:
                if filename.endswith(".msh") and _path_has_m2m_dir(path):
                    paths.append(path)
            elif path.match(mesh_glob):
                paths.append(path)

        now = time.monotonic()
        if progress_callback and now - last_progress >= progress_interval_sec:
            progress_callback(
                DiscoveryStats(
                    dirs_scanned=dirs_scanned,
                    files_seen=files_seen,
                    matches=len(paths),
                    current_dir=current_dir,
                )
            )
            last_progress = now

    if progress_callback:
        progress_callback(
            DiscoveryStats(
                dirs_scanned=dirs_scanned,
                files_seen=files_seen,
                matches=len(paths),
                current_dir=root,
            )
        )

    paths = sorted(paths)
    return [
        infer_record(
            path,
            root=root,
            roi_regex=roi_regex,
            subject_regex=subject_regex,
            repeat_regex=repeat_regex,
        )
        for path in paths
    ]
