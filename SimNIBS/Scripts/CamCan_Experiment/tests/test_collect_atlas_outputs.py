from pathlib import Path

from atlas.collect_atlas_outputs import (
    DEFAULT_MGZ_SOURCE_CANDIDATES,
    DEFAULT_SOURCE_CANDIDATES,
    DEFAULT_SOURCE_RELATIVE,
    build_atlas_export_plan,
    copy_atlas_outputs,
    default_dest_for,
    output_suffix_for,
    resolve_fastsurfer_out_root,
)


def write_subject_atlas(fastsurfer_out: Path, subject: str, text: str = "atlas") -> Path:
    atlas_path = fastsurfer_out / subject / DEFAULT_SOURCE_RELATIVE
    atlas_path.parent.mkdir(parents=True)
    atlas_path.write_text(text, encoding="utf-8")
    return atlas_path


def test_resolve_fastsurfer_out_root_accepts_parent_data_dir(tmp_path):
    fastsurfer_out = tmp_path / "FastSurfer_out"
    fastsurfer_out.mkdir()

    assert resolve_fastsurfer_out_root(tmp_path) == fastsurfer_out


def test_resolve_fastsurfer_out_root_accepts_output_dir_directly(tmp_path):
    fastsurfer_out = tmp_path / "FastSurfer_out"
    fastsurfer_out.mkdir()

    assert resolve_fastsurfer_out_root(fastsurfer_out) == fastsurfer_out


def test_default_dest_is_atlases_sibling_of_fastsurfer_out(tmp_path):
    fastsurfer_out = tmp_path / "FastSurfer_out"
    fastsurfer_out.mkdir()

    assert default_dest_for(fastsurfer_out) == tmp_path / "atlases"


def test_build_atlas_export_plan_flattens_subject_atlas_names(tmp_path):
    fastsurfer_out = tmp_path / "FastSurfer_out"
    dest = tmp_path / "atlases"
    write_subject_atlas(fastsurfer_out, "sub-01")
    write_subject_atlas(fastsurfer_out, "sub-02")
    (fastsurfer_out / "logs").mkdir()

    items, missing = build_atlas_export_plan(fastsurfer_out=fastsurfer_out, dest=dest)

    assert missing == []
    assert [(item.subject, item.dst.relative_to(dest).as_posix()) for item in items] == [
        ("sub-01", "sub-01.nii.gz"),
        ("sub-02", "sub-02.nii.gz"),
    ]


def test_build_atlas_export_plan_reports_requested_missing_subject(tmp_path):
    fastsurfer_out = tmp_path / "FastSurfer_out"
    dest = tmp_path / "atlases"
    write_subject_atlas(fastsurfer_out, "sub-01")

    items, missing = build_atlas_export_plan(
        fastsurfer_out=fastsurfer_out,
        dest=dest,
        subjects=["sub-01", "sub-02"],
    )

    assert [item.subject for item in items] == ["sub-01"]
    assert [(entry.subject, entry.reason) for entry in missing] == [
        ("sub-02", "subject directory is missing")
    ]


def test_build_atlas_export_plan_tries_default_nifti_candidates(tmp_path):
    fastsurfer_out = tmp_path / "freesurfer"
    dest = tmp_path / "atlases"
    atlas_path = fastsurfer_out / "sub-01" / "mri" / "aparc+aseg.nii.gz"
    atlas_path.parent.mkdir(parents=True)
    atlas_path.write_text("dk atlas", encoding="utf-8")

    items, missing = build_atlas_export_plan(
        fastsurfer_out=fastsurfer_out,
        dest=dest,
        source_relative=DEFAULT_SOURCE_CANDIDATES,
    )

    assert missing == []
    assert [(item.subject, item.src.name, item.dst.name) for item in items] == [
        ("sub-01", "aparc+aseg.nii.gz", "sub-01.nii.gz")
    ]


def test_build_atlas_export_plan_reports_incomplete_recon_mri_contents(tmp_path):
    fastsurfer_out = tmp_path / "freesurfer"
    dest = tmp_path / "atlases"
    mri_dir = fastsurfer_out / "sub-CC110056" / "mri"
    mri_dir.mkdir(parents=True)
    for name in (
        "mri_nu_correct.mni.log",
        "orig.mgz",
        "rawavg.mgz",
    ):
        (mri_dir / name).write_text("not an atlas", encoding="utf-8")

    items, missing = build_atlas_export_plan(
        fastsurfer_out=fastsurfer_out,
        dest=dest,
        source_relative=DEFAULT_SOURCE_CANDIDATES,
    )

    assert items == []
    assert len(missing) == 1
    assert missing[0].subject == "sub-CC110056"
    assert missing[0].reason == "no atlas segmentation file found; recon/segmentation appears incomplete"
    assert "orig.mgz" in missing[0].available
    assert "rawavg.mgz" in missing[0].available


def test_build_atlas_export_plan_converts_default_mgz_atlas_candidates(tmp_path):
    fastsurfer_out = tmp_path / "freesurfer"
    dest = tmp_path / "atlases"
    mri_dir = fastsurfer_out / "sub-01" / "mri"
    mri_dir.mkdir(parents=True)
    (mri_dir / "aparc+aseg.mgz").write_text("mgz atlas", encoding="utf-8")

    items, missing = build_atlas_export_plan(
        fastsurfer_out=fastsurfer_out,
        dest=dest,
        source_relative=DEFAULT_SOURCE_CANDIDATES,
        mgz_source_relative=DEFAULT_MGZ_SOURCE_CANDIDATES,
    )

    assert missing == []
    assert [(item.subject, item.src.name, item.dst.name, item.action) for item in items] == [
        ("sub-01", "aparc+aseg.mgz", "sub-01.nii.gz", "convert")
    ]


def test_copy_atlas_outputs_creates_flat_destination_files(tmp_path):
    fastsurfer_out = tmp_path / "FastSurfer_out"
    dest = tmp_path / "atlases"
    write_subject_atlas(fastsurfer_out, "sub-01", text="subject atlas")
    items, missing = build_atlas_export_plan(fastsurfer_out=fastsurfer_out, dest=dest)

    assert missing == []
    copy_atlas_outputs(items, dest=dest)

    assert (dest / "sub-01.nii.gz").read_text(encoding="utf-8") == "subject atlas"


def test_output_suffix_can_be_overridden():
    assert output_suffix_for(Path("mri/example.mgz"), output_suffix=".nii.gz") == ".nii.gz"
