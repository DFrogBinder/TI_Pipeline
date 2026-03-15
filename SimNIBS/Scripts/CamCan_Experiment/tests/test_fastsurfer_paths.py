from utils.paths import fastsurfer_atlas_path


def test_fastsurfer_atlas_resolution_uses_flat_subject_file_layout(tmp_path):
    atlas_path = tmp_path / "sub-01.nii.gz"
    atlas_path.touch()

    assert fastsurfer_atlas_path(str(tmp_path), "sub-01", None) == atlas_path


def test_fastsurfer_atlas_resolution_ignores_nested_subject_mri_layout(tmp_path):
    nested_dir = tmp_path / "sub-01" / "mri"
    nested_dir.mkdir(parents=True)
    (nested_dir / "aparc.DKTatlas+aseg.deep.nii.gz").touch()

    assert fastsurfer_atlas_path(str(tmp_path), "sub-01", None) is None


def test_fastsurfer_atlas_resolution_requires_existing_override_file(tmp_path):
    atlas_path = tmp_path / "explicit_subject.nii.gz"
    atlas_path.touch()

    assert fastsurfer_atlas_path(None, "sub-01", str(atlas_path)) == atlas_path
    assert fastsurfer_atlas_path(None, "sub-01", str(tmp_path / "missing.nii.gz")) is None
