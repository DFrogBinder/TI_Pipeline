from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _paths_module():
    try:
        return importlib.import_module("simnibs_326_finite_support_audit.paths")
    except ModuleNotFoundError:
        pytest.fail("finite-support path resolver has not been implemented")


def test_resolve_run_artifacts_returns_exact_pipeline_products(tmp_path):
    """Catches silently selecting an unrelated NIfTI from an output directory."""

    paths = _paths_module()
    subject = "sub-01"
    repeat_root = tmp_path / "repeat_001"
    anat = repeat_root / subject / "anat"
    sim_root = anat / "SimNIBS"
    output = sim_root / "Output" / subject
    expected = {
        "t1": _touch(anat / f"{subject}_T1w.nii"),
        "masked_field": _touch(sim_root / "ti_brain_only.nii.gz"),
        "mesh": _touch(output / "TI.msh"),
        "raw_field": _touch(
            output / "Volume_Base" / "TI_Volumetric_Base_TImax.nii.gz"
        ),
        "tissue_labels": _touch(
            output / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz"
        ),
    }

    assert paths.resolve_run_artifacts(repeat_root, subject) == expected


def test_resolve_run_artifacts_fails_closed_on_ambiguous_raw_fields(tmp_path):
    """Catches lexicographic first-file selection when multiple field volumes exist."""

    paths = _paths_module()
    subject = "sub-01"
    repeat_root = tmp_path / "repeat_001"
    anat = repeat_root / subject / "anat"
    sim_root = anat / "SimNIBS"
    output = sim_root / "Output" / subject
    _touch(anat / f"{subject}_T1w.nii")
    _touch(sim_root / "ti_brain_only.nii.gz")
    _touch(output / "TI.msh")
    _touch(output / "Volume_Base" / "TI_Volumetric_Base_TImax.nii.gz")
    _touch(output / "Volume_Base" / "TI_Volumetric_Base_other.nii.gz")
    _touch(output / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz")

    with pytest.raises(ValueError, match="exactly one raw field"):
        paths.resolve_run_artifacts(repeat_root, subject)
