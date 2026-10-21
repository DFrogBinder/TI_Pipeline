"""Fail-closed path resolution for retained repeatability products."""

from __future__ import annotations

from pathlib import Path


def _one_file(parent: Path, pattern: str, *, description: str) -> Path:
    matches = sorted(path for path in parent.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} matching {parent / pattern}; "
            f"observed {len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def _required_file(path: Path, *, description: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {description}: {path}")
    return path


def resolve_run_artifacts(repeat_root: Path, subject: str) -> dict[str, Path]:
    """Resolve the exact mesh, raw volume, mask labels, final field, and T1."""

    repeat_root = Path(repeat_root)
    anat = repeat_root / subject / "anat"
    sim_root = anat / "SimNIBS"
    output = sim_root / "Output" / subject
    return {
        "t1": _required_file(
            anat / f"{subject}_T1w.nii",
            description="subject T1",
        ),
        "masked_field": _required_file(
            sim_root / "ti_brain_only.nii.gz",
            description="brain-only TI field",
        ),
        "mesh": _required_file(output / "TI.msh", description="TI mesh"),
        "raw_field": _one_file(
            output / "Volume_Base",
            "TI_Volumetric_Base*.nii*",
            description="raw field",
        ),
        "tissue_labels": _one_file(
            output / "Volume_Labels",
            "TI_Volumetric_Labels*.nii*",
            description="tissue-label volume",
        ),
    }
