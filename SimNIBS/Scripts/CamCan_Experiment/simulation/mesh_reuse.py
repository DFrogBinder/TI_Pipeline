from __future__ import annotations

from pathlib import Path


def subject_suffix(subject: str) -> str:
    return subject.split("-")[-1].upper()


def candidate_mesh_paths(anat_dir: str | Path, subject: str) -> tuple[Path, ...]:
    anat = Path(anat_dir)
    suffix = subject_suffix(subject)
    candidates = (
        anat / f"m2m_{subject}" / f"{subject}.msh",
        anat / f"m2m_sub-{suffix}" / f"{subject}.msh",
        anat / f"m2m_sub-{suffix}" / f"sub-{suffix}.msh",
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return tuple(unique)


def resolve_existing_mesh(anat_dir: str | Path, subject: str) -> Path | None:
    for path in candidate_mesh_paths(anat_dir, subject):
        if path.is_file():
            return path
    return None
