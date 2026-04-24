from pathlib import Path

from post.run_post_processing_batch import (
    RepeatBatchConfig,
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
