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
    return anat_root(root, subject) / f"{subject}_T1w.nii.gz"


def post_root(root: str, subject: str) -> Path:
    return anat_root(root, subject) / "post"


def fastsurfer_atlas_path(root: Optional[str], subject: str, override: Optional[str]) -> Optional[Path]:
    if override:
        return Path(override)
    if root:
        candidate = Path(root).expanduser() / subject / "mri" / "aparc.DKTatlas+aseg.deep.nii.gz"
        if candidate.is_file():
            return candidate
    return None
