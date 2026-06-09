import csv
import json
from pathlib import Path

from PIL import Image

from post.pipeline_layers import FIGURE_GENERATION_STAGE
from post.post_figures import run_figure_generation


def _write_subject_metrics(
    dataset_root: Path,
    subject: str,
    *,
    roi_name: str,
    status: str = "complete",
    qc_status: str = "complete",
    roi_peak: float = 0.31,
    roi_mean: float = 0.12,
    mni_baseline_roi_mean: float | None = None,
    overlap_fraction: float = 0.22,
    roi_threshold_voxels: int = 8,
    nonblocking_qc_checks: list[str] | None = None,
) -> None:
    post_root = dataset_root / subject / "anat" / "post"
    post_root.mkdir(parents=True)
    payload = {
        "subject": subject,
        "target_roi": roi_name,
        "top_percentile_voxels": 40,
        "subject_metrics_meta": {
            "status": status,
            "extended_metrics_status": "complete",
            "qc_status": qc_status,
            "nonblocking_qc_checks": nonblocking_qc_checks or [],
            "blocking_qc_checks": [],
        },
        "extended_metrics_meta": {"status": "complete"},
        "qc_meta": {"status": qc_status, "error_checks": []},
        "extended_metrics": {
            "roi_peak": roi_peak,
            "roi_mean": roi_mean,
            "mni_baseline_roi_mean": mni_baseline_roi_mean,
            "focality_voxels_gt_threshold": 120,
            "focality_volume_mm3_gt_threshold": 120.0,
            "electrode_distance_count": 4,
            "electrode_distance_mean_mm": 58.5,
            "electrode_distance_min_mm": 40.0,
            "electrode_distance_max_mm": 74.0,
        },
        "rois": {
            roi_name: {
                "roi_voxels": 500,
                "overlap_fraction": overlap_fraction,
                "focality_in_roi_voxels_gt_threshold": roi_threshold_voxels,
                "focality_in_roi_volume_mm3_gt_threshold": float(roi_threshold_voxels),
                "threshold_qc": {
                    "metric_threshold": {
                        "status": "ok",
                        "reason": "ok" if roi_threshold_voxels else "no_roi_voxels_above_threshold",
                    },
                    "overlay_threshold": {
                        "status": "ok",
                        "reason": "ok" if roi_threshold_voxels else "no_roi_voxels_above_threshold",
                    },
                },
            }
        },
    }
    (post_root / "subject_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_run_figure_generation_writes_generic_pngs_tables_and_summary(tmp_path):
    for repeat in ("01", "02"):
        dataset_root = tmp_path / f"Left_Hippocampus_Data_{repeat}"
        dataset_root.mkdir()
        _write_subject_metrics(
            dataset_root,
            "sub-01",
            roi_name="Left Hippocampus",
            roi_peak=0.31 if repeat == "01" else 0.34,
            roi_threshold_voxels=0,
            nonblocking_qc_checks=["overlays"] if repeat == "02" else None,
            qc_status="partial" if repeat == "02" else "complete",
        )
        _write_subject_metrics(
            dataset_root,
            "sub-02",
            roi_name="Left Hippocampus",
            roi_peak=0.28 if repeat == "01" else 0.30,
            roi_threshold_voxels=9,
        )

    result = run_figure_generation(
        batch_root=tmp_path,
        output_dir=tmp_path / "post_processing_figures",
        expected_repeats=["01", "02"],
    )

    assert result["stage"] == FIGURE_GENERATION_STAGE
    assert result["status"] == "ok"
    assert result["figure_count"] >= 7
    assert Path(result["figures"]["threshold_support"]).is_file()
    assert Path(result["tables"]["subject_metrics_long"]).is_file()
    assert Path(result["tables"]["roi_summary"]).is_file()
    assert Path(result["summary_path"]).is_file()

    with Image.open(result["figures"]["threshold_support"]) as image:
        assert image.size == (2400, 1350)

    with Path(result["tables"]["roi_summary"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["roi_label"] == "Left Hippocampus"
    assert int(float(rows[0]["complete_metrics"])) == 4
    assert int(float(rows[0]["zero_roi_threshold_rows"])) == 2
    assert int(float(rows[0]["nonblocking_overlay_events"])) == 1


def test_run_figure_generation_writes_aggregate_population_distribution_outputs(tmp_path):
    values = {
        ("01", "sub-01"): (0.10, 0.25),
        ("02", "sub-01"): (0.14, 0.35),
        ("01", "sub-02"): (0.20, 0.55),
        ("02", "sub-02"): (0.22, 0.65),
    }
    for repeat in ("01", "02"):
        dataset_root = tmp_path / f"Left_M1_Data_{repeat}"
        dataset_root.mkdir()
        for subject in ("sub-01", "sub-02"):
            roi_mean, overlap_fraction = values[(repeat, subject)]
            _write_subject_metrics(
                dataset_root,
                subject,
                roi_name="ctx_lh_G_precentral",
                roi_mean=roi_mean,
                mni_baseline_roi_mean=0.15,
                overlap_fraction=overlap_fraction,
            )

    result = run_figure_generation(
        batch_root=tmp_path,
        output_dir=tmp_path / "post_processing_figures",
        expected_repeats=["01", "02"],
    )

    assert Path(result["figures"]["aggregate_subject_variability"]).is_file()
    assert Path(result["figures"]["aggregate_mni_transfer_distribution"]).is_file()
    assert Path(result["tables"]["aggregate_subject_variability_summary"]).is_file()
    assert Path(result["tables"]["aggregate_mni_transfer_summary"]).is_file()

    with Path(result["tables"]["aggregate_subject_variability_summary"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        coverage_rows = list(csv.DictReader(handle))
    assert len(coverage_rows) == 1
    assert coverage_rows[0]["roi_label"] == "ctx_lh_G_precentral"
    assert float(coverage_rows[0]["population_median_overlap_percent"]) == 45.0
    assert int(float(coverage_rows[0]["n_subjects"])) == 2
    assert int(float(coverage_rows[0]["n_subject_repeat_rows"])) == 4

    with Path(result["tables"]["aggregate_mni_transfer_summary"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        mni_rows = list(csv.DictReader(handle))
    assert len(mni_rows) == 1
    assert float(mni_rows[0]["population_median_roi_mean_v_per_m"]) == 0.165
    assert float(mni_rows[0]["mni_baseline_roi_mean_v_per_m"]) == 0.15


def test_run_figure_generation_prefers_explicit_mni152_post_baseline(tmp_path):
    dataset_root = tmp_path / "Left_M1_Data_01"
    dataset_root.mkdir()
    _write_subject_metrics(
        dataset_root,
        "sub-01",
        roi_name="ctx_lh_G_precentral",
        roi_mean=0.10,
        mni_baseline_roi_mean=0.15,
        overlap_fraction=0.25,
    )
    mni_post_root = tmp_path / "MNI152-data" / "MNI152-left-m1" / "anat" / "post"
    mni_post_root.mkdir(parents=True)
    (mni_post_root / "subject_metrics.json").write_text(
        json.dumps(
            {
                "subject": "MNI152-left-m1",
                "target_roi": "ctx_lh_G_precentral",
                "extended_metrics": {"roi_mean": 0.18},
            }
        ),
        encoding="utf-8",
    )

    result = run_figure_generation(
        batch_root=tmp_path,
        output_dir=tmp_path / "post_processing_figures",
        expected_repeats=["01"],
        mni_baseline_post_root=tmp_path / "MNI152-data",
    )

    with Path(result["tables"]["aggregate_mni_transfer_summary"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["mni_baseline_roi_mean_v_per_m"]) == 0.18
    assert rows[0]["mni_baseline_source"].endswith(
        "MNI152-data/MNI152-left-m1/anat/post/subject_metrics.json"
    )
