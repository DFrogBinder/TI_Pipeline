from utils.paths import fastsurfer_atlas_path, t1_path


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


def test_t1_path_falls_back_to_gzip_when_uncompressed_file_is_missing(tmp_path):
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    nii_gz_path = anat_dir / "sub-01_T1w.nii.gz"
    nii_gz_path.touch()

    assert t1_path(str(tmp_path), "sub-01") == nii_gz_path


def test_t1_path_prefers_uncompressed_file_when_both_exist(tmp_path):
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    nii_path = anat_dir / "sub-01_T1w.nii"
    nii_gz_path = anat_dir / "sub-01_T1w.nii.gz"
    nii_path.touch()
    nii_gz_path.touch()

    assert t1_path(str(tmp_path), "sub-01") == nii_path
