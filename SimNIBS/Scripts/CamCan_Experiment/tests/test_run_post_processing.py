import pytest

from post.run_post_processing import (
    PostBatchConfig,
    build_arg_parser,
    resolve_max_workers,
    resolve_subject_fastsurfer_atlas_path,
    run_subject_level_stage,
    should_skip_subject,
    validate_post_batch_config,
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


def test_validate_post_batch_config_rejects_electrode_csv_missing_columns(tmp_path):
    electrode_csv = tmp_path / "electrodes.csv"
    electrode_csv.write_text("subject,electrode,x,y\nsub-01,Fp1,1,2\n", encoding="utf-8")
    cfg = PostBatchConfig(root="/tmp/example", electrode_csv=str(electrode_csv))

    with pytest.raises(SystemExit, match="missing required column"):
        validate_post_batch_config(cfg)


def test_validate_post_batch_config_rejects_missing_electrode_dataset_dir(tmp_path):
    cfg = PostBatchConfig(
        root="/tmp/example",
        electrode_dataset_dir=str(tmp_path / "missing-electrodes"),
    )

    with pytest.raises(SystemExit, match="electrode_dataset_dir is not a directory"):
        validate_post_batch_config(cfg)


def test_validate_post_batch_config_requires_fixed_atlas_with_mni_baseline(tmp_path):
    baseline_root = tmp_path / "mni_baseline"
    baseline_root.mkdir()
    cfg = PostBatchConfig(root="/tmp/example", mni_baseline_root=str(baseline_root))

    with pytest.raises(SystemExit, match="without mni_fixed_atlas_path"):
        validate_post_batch_config(cfg)


def test_should_skip_subject_reprocesses_old_metric_schema(monkeypatch, tmp_path):
    out_dir = tmp_path / "sub-01" / "anat" / "post"
    out_dir.mkdir(parents=True)
    (out_dir / "subject_metrics.json").write_text(
        """
        {
          "subject_metrics_meta": {"status": "complete"},
          "extended_metrics_meta": {
            "schema_version": 3,
            "status": "complete",
            "config_fingerprint": "expected"
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "post.post_process.extended_metrics_fingerprint_for_cfg",
        lambda cfg: "expected",
    )

    assert not should_skip_subject(out_dir, object(), force=False)


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
