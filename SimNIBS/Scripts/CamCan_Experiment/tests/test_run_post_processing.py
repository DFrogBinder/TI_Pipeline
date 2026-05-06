import pytest

from post.run_post_processing import (
    PostBatchConfig,
    build_arg_parser,
    resolve_max_workers,
    resolve_subject_fastsurfer_atlas_path,
    run_subject_level_stage,
)


def test_resolve_max_workers_uses_slurm_cpus_per_task(monkeypatch):
    monkeypatch.delenv("POST_MAX_WORKERS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")

    cfg = PostBatchConfig(root="/tmp/example", max_workers=None)

    assert resolve_max_workers(cfg, task_count=10) == 6


def test_resolve_max_workers_prefers_explicit_post_override(monkeypatch):
    monkeypatch.setenv("POST_MAX_WORKERS", "3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")

    cfg = PostBatchConfig(root="/tmp/example", max_workers=None)

    assert resolve_max_workers(cfg, task_count=10) == 3


def test_resolve_max_workers_rejects_invalid_env_values(monkeypatch):
    monkeypatch.setenv("POST_MAX_WORKERS", "0")

    cfg = PostBatchConfig(root="/tmp/example", max_workers=None)

    with pytest.raises(ValueError, match="POST_MAX_WORKERS must be at least 1"):
        resolve_max_workers(cfg, task_count=10)


def test_resolve_subject_fastsurfer_atlas_path_uses_subject_relative_override():
    cfg = PostBatchConfig(
        root="/tmp/example",
        fastsurfer_root="/tmp/atlases",
        fastsurfer_atlas_filename="mri/aparc.DKTatlas+aseg.deep.nii.gz",
    )

    assert resolve_subject_fastsurfer_atlas_path(cfg, "sub-CC110056") == (
        "/tmp/atlases/sub-CC110056/mri/aparc.DKTatlas+aseg.deep.nii.gz"
    )


def test_resolve_subject_fastsurfer_atlas_path_accepts_absolute_override():
    cfg = PostBatchConfig(
        root="/tmp/example",
        fastsurfer_atlas_filename="/tmp/atlases/sub-mni152.nii.gz",
    )

    assert (
        resolve_subject_fastsurfer_atlas_path(cfg, "sub-CC110056")
        == "/tmp/atlases/sub-mni152.nii.gz"
    )


def test_resolve_subject_fastsurfer_atlas_path_requires_fastsurfer_root_for_relative_override():
    cfg = PostBatchConfig(
        root="/tmp/example",
        fastsurfer_atlas_filename="mri/aparc.DKTatlas+aseg.deep.nii.gz",
    )

    with pytest.raises(ValueError, match="relative atlas filename override requires fastsurfer_root"):
        resolve_subject_fastsurfer_atlas_path(cfg, "sub-CC110056")


def test_cli_parser_accepts_atlas_filename_alias():
    args = build_arg_parser().parse_args(
        ["--atlas-filename", "mri/aparc.DKTatlas+aseg.deep.nii.gz"]
    )

    assert args.fastsurfer_atlas_filename == "mri/aparc.DKTatlas+aseg.deep.nii.gz"


def test_run_subject_level_stage_marks_partial_when_some_subjects_are_usable(monkeypatch):
    monkeypatch.setattr(
        "post.run_post_processing.run_batch",
        lambda cfg: {
            "processed": ["sub-01"],
            "skipped": ["sub-02"],
            "failed": [("sub-03", "boom")],
            "incomplete": [],
        },
    )

    result = run_subject_level_stage(PostBatchConfig(root="/tmp/example"))

    assert result["status"] == "partial"
    assert result["usable_subject_count"] == 2


def test_run_subject_level_stage_marks_failed_when_no_subject_outputs_are_usable(monkeypatch):
    monkeypatch.setattr(
        "post.run_post_processing.run_batch",
        lambda cfg: {
            "processed": [],
            "skipped": [],
            "failed": [("sub-03", "boom")],
            "incomplete": [("sub-04", "partial")],
        },
    )

    result = run_subject_level_stage(PostBatchConfig(root="/tmp/example"))

    assert result["status"] == "failed"
    assert result["usable_subject_count"] == 0
