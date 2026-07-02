from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class SubjectInputPaths:
    anat_dir: Path
    subject: str
    t1: Path
    t2: Path
    custom_segmentation: Path | None


def _candidate_paths(anat_dir: str | Path, stem: str) -> tuple[Path, ...]:
    anat = Path(anat_dir)
    return tuple(anat / f"{stem}{suffix}" for suffix in NIFTI_SUFFIXES)


def _resolve_existing_path(
    anat_dir: str | Path,
    stem: str,
    *,
    required: bool,
) -> Path | None:
    candidates = _candidate_paths(anat_dir, stem)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Required NIfTI input not found. Checked: {joined}")
    return None


def resolve_subject_t1_path(anat_dir: str | Path, subject: str) -> Path:
    return _resolve_existing_path(anat_dir, f"{subject}_T1w", required=True)


def resolve_subject_t2_path(anat_dir: str | Path, subject: str) -> Path:
    return _resolve_existing_path(anat_dir, f"{subject}_T2w", required=True)


def resolve_subject_custom_segmentation_path(anat_dir: str | Path, subject: str) -> Path | None:
    return _resolve_existing_path(
        anat_dir,
        f"{subject}_T1w_ras_1mm_T1andT2_masks",
        required=False,
    )


def resolve_subject_input_paths(anat_dir: str | Path, subject: str) -> SubjectInputPaths:
    anat = Path(anat_dir)
    return SubjectInputPaths(
        anat_dir=anat,
        subject=subject,
        t1=resolve_subject_t1_path(anat, subject),
        t2=resolve_subject_t2_path(anat, subject),
        custom_segmentation=resolve_subject_custom_segmentation_path(anat, subject),
    )
