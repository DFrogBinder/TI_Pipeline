import pytest

from post.run_post_processing import (
    PostBatchConfig,
    build_arg_parser,
    resolve_max_workers,
    resolve_subject_fastsurfer_atlas_path,
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
