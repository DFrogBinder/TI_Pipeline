from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MeshRecord:
    path: Path
    roi: str
    subject: str
    repeat: str


ROI_HINTS = ("roi", "hippocampus", "pallidum", "m1", "motor", "target")
SUBJECT_RE = re.compile(r"^(sub-[A-Za-z0-9_-]+|CC\d+)$", re.IGNORECASE)
REPEAT_RE = re.compile(r"^(repeat[-_ ]?\d+|rep[-_ ]?\d+|seed[-_ ]?\d+)$", re.IGNORECASE)


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
        for idx, part in enumerate(dir_parts[:-1]):
            if "repeat" in part.lower() and dir_parts[idx + 1].isdigit():
                repeat = dir_parts[idx + 1]
                break

    return MeshRecord(path=path, roi=roi, subject=subject, repeat=repeat)


def discover_meshes(
    root: Path,
    *,
    mesh_glob: str = "*.msh",
    roi_regex: str | None = None,
    subject_regex: str | None = None,
    repeat_regex: str | None = None,
) -> list[MeshRecord]:
    root = Path(root).expanduser().resolve()
    paths = sorted(p for p in root.rglob(mesh_glob) if p.is_file())
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

