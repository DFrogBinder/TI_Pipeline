import json

import numpy as np
import pandas as pd
import pytest

from post.post_population import run_population


def _write_subject(root, subject, index, *, status="complete"):
    post_dir = root / subject / "anat" / "post"
    post_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "label_id": 1,
                "label_name": "Target",
                "voxels": 10 + index,
                "volume_mm3": 100.0 + (10.0 * index),
                "mean": 0.10 + (0.02 * index),
                "max": 0.20 + (0.03 * index),
            },
            {
                "label_id": 2,
                "label_name": "Neighbor",
                "voxels": 20 + index,
                "volume_mm3": 50.0 + (5.0 * index),
                "mean": 0.05 + (0.01 * index),
                "max": 0.15 + (0.02 * index),
            },
        ]
    ).to_csv(post_dir / "region_stats_fastsurfer.csv", index=False)

    payload = {
        "schema_version": 3,
        "subject": subject,
        "target_roi": "Target",
        "percentile": 95.0,
        "percentile_value": 0.2 + (0.01 * index),
        "voxel_volume_mm3": 1.0,
        "top_percentile_voxels": 100 + index,
        "rois": {
            "Target": {
                "roi_voxels": 10 + index,
                "overlap_top_voxels": 3 + index,
                "roi_volume_mm3": 10.0 + index,
                "overlap_volume_mm3": 3.0 + index,
                "overlap_fraction": 0.20 + (0.05 * index),
                "roi_percentile_value": 0.30 + (0.01 * index),
            }
        },
        "extended_metrics": {
            "roi_peak": 0.40 + (0.10 * index),
            "roi_mean": 0.15 + (0.05 * index),
            "focality_voxels_gt_threshold": 20.0 + index,
            "focality_volume_mm3_gt_threshold": 20.0 + index,
            "csf_distance_mm": 4.0 + index,
            "skull_distance_mm": 8.0 + index,
            "electrode_distance_mean_mm": 60.0 + index,
            "electrode_distance_min_mm": 55.0 + index,
            "electrode_distance_max_mm": 65.0 + index,
            "neighbors": [
                {
                    "label_id": 2,
                    "label_name": "Neighbor",
                    "voxels": 20 + index,
                    "volume_mm3": 50.0 + (5.0 * index),
                    "mean": 0.20 + (0.20 * index),
                    "max": 0.50 + (0.20 * index),
                }
            ],
        },
        "extended_metrics_meta": {
            "schema_version": 3,
            "status": status,
        },
    }
    (post_dir / "subject_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_population_outputs_neighbor_stats_and_regional_correlations(tmp_path):
    root = tmp_path / "dataset"
    for index, subject in enumerate(["sub-01", "sub-02", "sub-03"], start=1):
        _write_subject(root, subject, index)

    out_dir = run_population(root=root, target_roi="Target")

    neighbor_summary = pd.read_csv(out_dir / "population_neighbor_summary.csv")
    expected_neighbor_columns = {
        "mean_of_mean",
        "median_of_mean",
        "iqr_mean",
        "std_mean",
        "cv_mean",
        "min_mean",
        "max_mean",
        "mean_peak",
        "median_peak",
        "iqr_peak",
        "std_peak",
        "cv_peak",
        "min_peak",
        "max_peak",
        "mean_volume_mm3",
    }
    assert expected_neighbor_columns.issubset(neighbor_summary.columns)

    row = neighbor_summary.loc[neighbor_summary["label_name"] == "Neighbor"].iloc[0]
    mean_values = np.array([0.40, 0.60, 0.80])
    peak_values = np.array([0.70, 0.90, 1.10])
    assert row["std_mean"] == pytest.approx(mean_values.std(ddof=1))
    assert row["cv_mean"] == pytest.approx(mean_values.std(ddof=0) / mean_values.mean())
    assert row["min_mean"] == pytest.approx(mean_values.min())
    assert row["max_mean"] == pytest.approx(mean_values.max())
    assert row["std_peak"] == pytest.approx(peak_values.std(ddof=1))
    assert row["cv_peak"] == pytest.approx(peak_values.std(ddof=0) / peak_values.mean())
    assert row["min_peak"] == pytest.approx(peak_values.min())
    assert row["max_peak"] == pytest.approx(peak_values.max())

    pooled = pd.read_csv(out_dir / "volume_intensity_correlation.csv")
    assert list(pooled.columns) == ["metric", "pearson_r"]
    assert set(pooled["metric"]) == {"mean", "max"}

    regional = pd.read_csv(out_dir / "regional_volume_intensity_correlation.csv")
    assert list(regional.columns) == [
        "label_id",
        "label_name",
        "subjects",
        "metric",
        "pearson_r",
    ]
    assert regional.shape[0] == 4
    assert set(zip(regional["label_name"], regional["metric"])) == {
        ("Target", "mean"),
        ("Target", "max"),
        ("Neighbor", "mean"),
        ("Neighbor", "max"),
    }
    assert set(regional["subjects"]) == {3}


def test_incomplete_subjects_are_excluded_from_manifest_and_summaries(tmp_path):
    root = tmp_path / "dataset"
    _write_subject(root, "sub-01", 1, status="complete")
    _write_subject(root, "sub-02", 2, status="partial")

    out_dir = run_population(
        root=root,
        subjects=["sub-01", "sub-02"],
        target_roi="Target",
    )

    manifest = pd.read_csv(out_dir / "population_cohort_manifest.csv")
    assert manifest["subject"].tolist() == ["sub-01"]
    assert manifest["has_region_table"].astype(bool).tolist() == [True]
    assert manifest["has_complete_subject_metrics"].astype(bool).tolist() == [True]
    assert manifest["included"].astype(bool).tolist() == [True]

    all_regions = pd.read_csv(out_dir / "all_region_values.csv")
    assert set(all_regions["subject"]) == {"sub-01"}

    region_summary = pd.read_csv(out_dir / "population_region_summary.csv")
    assert set(region_summary["subjects"]) == {1}

    subject_values = pd.read_csv(out_dir / "subject_metric_values.csv")
    assert subject_values["subject"].tolist() == ["sub-01"]

    subject_summary = pd.read_csv(out_dir / "population_subject_metric_summary.csv")
    roi_peak_summary = subject_summary.loc[subject_summary["metric"] == "roi_peak"].iloc[0]
    assert roi_peak_summary["subjects"] == 1
    assert roi_peak_summary["mean"] == pytest.approx(0.50)

    neighbor_metrics = pd.read_csv(out_dir / "subject_neighbor_metrics.csv")
    assert neighbor_metrics["subject"].tolist() == ["sub-01"]
