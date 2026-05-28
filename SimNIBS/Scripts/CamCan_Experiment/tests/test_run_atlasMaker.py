from pathlib import Path

from atlas.run_atlasMaker import already_processed


def test_already_processed_requires_destrieux_nifti(tmp_path: Path):
    output_dir = tmp_path / "FastSurfer_out"
    mri_dir = output_dir / "sub-01" / "mri"
    mri_dir.mkdir(parents=True)
    (mri_dir / "aparc.DKTatlas+aseg.deep.nii.gz").write_text("dkt", encoding="utf-8")

    assert already_processed(output_dir, "sub-01") is False

    (mri_dir / "aparc.a2009s+aseg.nii.gz").write_text("destrieux", encoding="utf-8")

    assert already_processed(output_dir, "sub-01") is True
