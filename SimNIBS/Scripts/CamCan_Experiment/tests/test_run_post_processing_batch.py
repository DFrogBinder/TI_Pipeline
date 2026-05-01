from pathlib import Path

from post.run_post_processing_batch import (
    RepeatBatchConfig,
    _resolve_repeatability_output_dir_for_roi,
    discover_repeat_datasets,
    make_default_pipeline_template,
    run_repeat_batch,
)


def test_discover_repeat_datasets_matches_generic_roi_prefixes(tmp_path):
    for name in (
        "Left_Hippocampus_Data_10",
        "Left_Hippocampus_Data_02",
        "Right_Thalamus_Data_01",
        "Left_Hippocampus_Data_test",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    datasets = discover_repeat_datasets(tmp_path)

    assert [(dataset.roi_prefix, dataset.repeat_id) for dataset in datasets] == [
        ("Left_Hippocampus", "02"),
        ("Left_Hippocampus", "10"),
        ("Right_Thalamus", "01"),
    ]


def test_discover_repeat_datasets_filters_repeats_without_hard_coding_padding(tmp_path):
    for name in (
        "Left_Hippocampus_Data_01",
        "Left_Hippocampus_Data_02",
        "Left_Hippocampus_Data_10",
    ):
        (tmp_path / name).mkdir()

    datasets = discover_repeat_datasets(tmp_path, repeats=["1", "10"])

    assert [dataset.repeat_id for dataset in datasets] == ["01", "10"]


def test_run_repeat_batch_uses_fresh_pipeline_template_per_dataset(tmp_path, monkeypatch):
    dataset_names = [
        "Left_Hippocampus_Data_01",
        "Left_Hippocampus_Data_02",
    ]
    for name in dataset_names:
        (tmp_path / name).mkdir()

    template = make_default_pipeline_template()
    template.post.plot_roi = None
    template.population.target_roi = None

    seen_roots = []
    incoming_plot_rois = []
    incoming_target_rois = []

    def fake_run_pipeline(cfg):
        seen_roots.append(Path(cfg.post.root).name)
        incoming_plot_rois.append(cfg.post.plot_roi)
        incoming_target_rois.append(cfg.population.target_roi)
        cfg.post.plot_roi = f"mutated-{Path(cfg.post.root).name}"
        cfg.population.target_roi = f"target-{Path(cfg.post.root).name}"

    monkeypatch.setattr("post.run_post_processing_batch.run_pipeline", fake_run_pipeline)

    summary = run_repeat_batch(
        RepeatBatchConfig(
            batch_root=str(tmp_path),
            repeats=["01", "02"],
            summary_filename=None,
            run_repeatability=False,
        ),
        template,
    )

    assert seen_roots == dataset_names
    assert incoming_plot_rois == [None, None]
    assert incoming_target_rois == [None, None]
    assert template.post.plot_roi is None
    assert template.population.target_roi is None
    assert summary["processed_datasets"] == 2
    assert summary["failed_datasets"] == 0


def test_default_repeatability_output_dir_is_inline_for_single_roi_batch(tmp_path):
    cfg = RepeatBatchConfig(batch_root=str(tmp_path), repeatability_output_dir=None)

    output_dir = _resolve_repeatability_output_dir_for_roi(
        cfg=cfg,
        batch_root=tmp_path,
        roi_name="Left-Hippocampus",
        roi_count=1,
    )

    assert output_dir == tmp_path / "subject_metrics_analysis"


def test_default_repeatability_output_dir_uses_per_roi_subdir_for_multi_roi_batch(tmp_path):
    cfg = RepeatBatchConfig(batch_root=str(tmp_path), repeatability_output_dir=None)

    output_dir = _resolve_repeatability_output_dir_for_roi(
        cfg=cfg,
        batch_root=tmp_path,
        roi_name="Left-Hippocampus",
        roi_count=2,
    )

    assert output_dir == tmp_path / "repeatability_analysis" / "Left_Hippocampus"


def test_explicit_repeatability_output_root_still_creates_per_roi_subdir(tmp_path):
    cfg = RepeatBatchConfig(
        batch_root=str(tmp_path),
        repeatability_output_dir="custom_repeatability_outputs",
    )

    output_dir = _resolve_repeatability_output_dir_for_roi(
        cfg=cfg,
        batch_root=tmp_path,
        roi_name="Left-Hippocampus",
        roi_count=1,
    )

    assert output_dir == tmp_path / "custom_repeatability_outputs" / "Left_Hippocampus"
