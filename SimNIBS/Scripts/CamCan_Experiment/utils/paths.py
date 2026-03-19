"""
Centralized path helpers for TI pipeline outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def subject_root(root: str, subject: str) -> Path:
    return Path(root).expanduser() / subject


def anat_root(root: str, subject: str) -> Path:
    return subject_root(root, subject) / "anat"


def simnibs_root(root: str, subject: str) -> Path:
    return anat_root(root, subject) / "SimNIBS"


def sim_output_dir(root: str, subject: str) -> Path:
    return simnibs_root(root, subject) / "Output" / subject


def ti_brain_path(root: str, subject: str) -> Path:
    return simnibs_root(root, subject) / "ti_brain_only.nii.gz"


def t1_path(root: str, subject: str) -> Path:
    anat_dir = anat_root(root, subject)
    nii_path = anat_dir / f"{subject}_T1w.nii"
    if nii_path.is_file():
        return nii_path

    nii_gz_path = anat_dir / f"{subject}_T1w.nii.gz"
    if nii_gz_path.is_file():
        return nii_gz_path

    return nii_path


def post_root(root: str, subject: str) -> Path:
    return anat_root(root, subject) / "post"


def fastsurfer_atlas_path(root: Optional[str], subject: str, override: Optional[str]) -> Optional[Path]:
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate
        return None
    if root:
        candidate = Path(root).expanduser() / f"{subject}.nii.gz"
        if candidate.is_file():
            return candidate
    return None
