from pathlib import Path

import pytest

from utils.subject_inputs import resolve_subject_input_paths


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_resolve_subject_input_paths_prefers_uncompressed_niftis(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat / "sub-01_T1w.nii")
    _touch(anat / "sub-01_T1w.nii.gz")
    _touch(anat / "sub-01_T2w.nii")
    _touch(anat / "sub-01_T2w.nii.gz")

    paths = resolve_subject_input_paths(anat, "sub-01")

    assert paths.t1 == anat / "sub-01_T1w.nii"
    assert paths.t2 == anat / "sub-01_T2w.nii"
    assert paths.custom_segmentation is None


def test_resolve_subject_input_paths_falls_back_to_gzip(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat / "sub-01_T1w.nii.gz")
    _touch(anat / "sub-01_T2w.nii.gz")

    paths = resolve_subject_input_paths(anat, "sub-01")

    assert paths.t1 == anat / "sub-01_T1w.nii.gz"
    assert paths.t2 == anat / "sub-01_T2w.nii.gz"
    assert paths.custom_segmentation is None


def test_resolve_subject_input_paths_detects_optional_custom_segmentation(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat / "sub-01_T1w.nii.gz")
    _touch(anat / "sub-01_T2w.nii.gz")
    _touch(anat / "sub-01_T1w_ras_1mm_T1andT2_masks.nii.gz")

    paths = resolve_subject_input_paths(anat, "sub-01")

    assert paths.custom_segmentation == anat / "sub-01_T1w_ras_1mm_T1andT2_masks.nii.gz"


@pytest.mark.parametrize(
    ("missing_name", "expected_fragment"),
    [
        ("sub-01_T1w.nii.gz", "sub-01_T1w"),
        ("sub-01_T2w.nii.gz", "sub-01_T2w"),
    ],
)
def test_resolve_subject_input_paths_requires_t1_and_t2(tmp_path, missing_name: str, expected_fragment: str):
    anat = tmp_path / "sub-01" / "anat"
    _touch(anat / "sub-01_T1w.nii.gz")
    _touch(anat / "sub-01_T2w.nii.gz")
    (anat / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=expected_fragment):
        resolve_subject_input_paths(anat, "sub-01")
