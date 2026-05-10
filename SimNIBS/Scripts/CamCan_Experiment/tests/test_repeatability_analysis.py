import importlib.util
import json
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _load_repeatability_module(monkeypatch):
    metric_extensions_stub = types.ModuleType("post.metric_extensions")

    def flatten_subject_metric_payload(payload, roi_key):
        roi_metrics = payload["rois"][roi_key]
        flattened = {
            "schema_version": payload.get("schema_version"),
            "target_roi": payload.get("target_roi"),
            "percentile": payload.get("percentile"),
            "percentile_value": payload.get("percentile_value"),
            "voxel_volume_mm3": payload.get("voxel_volume_mm3"),
            "whole_brain_voxels": payload.get("whole_brain_voxels"),
            "whole_brain_volume_mm3": payload.get("whole_brain_volume_mm3"),
            "top_percentile_voxels": payload.get("top_percentile_voxels"),
            "top_percentile_percent_of_whole_brain": payload.get("top_percentile_percent_of_whole_brain"),
            "roi_voxels": roi_metrics.get("roi_voxels"),
            "overlap_top_voxels": roi_metrics.get("overlap_top_voxels"),
            "roi_volume_mm3": roi_metrics.get("roi_volume_mm3"),
            "overlap_volume_mm3": roi_metrics.get("overlap_volume_mm3"),
            "overlap_fraction": roi_metrics.get("overlap_fraction"),
            "roi_percent_of_whole_brain": roi_metrics.get("roi_percent_of_whole_brain"),
            "overlap_top_percent_of_whole_brain": roi_metrics.get("overlap_top_percent_of_whole_brain"),
            "focality_in_roi_voxels_gt_threshold": roi_metrics.get("focality_in_roi_voxels_gt_threshold"),
            "focality_in_roi_volume_mm3_gt_threshold": roi_metrics.get("focality_in_roi_volume_mm3_gt_threshold"),
            "focality_in_roi_percent_of_whole_brain_gt_threshold": roi_metrics.get(
                "focality_in_roi_percent_of_whole_brain_gt_threshold"
            ),
        }
        flattened.update(payload.get("extended_metrics", {}))
        return flattened

    metric_extensions_stub.flatten_subject_metric_payload = flatten_subject_metric_payload
    monkeypatch.setitem(sys.modules, "post.metric_extensions", metric_extensions_stub)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "post"
        / "repeatability"
        / "analyze_subject_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("repeatability_analysis_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _save_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)


def _write_subject_run(
    post_dir: Path,
    *,
    subject: str,
    roi_name: str,
    field: np.ndarray,
    roi_mask: np.ndarray,
    top_mask: np.ndarray,
    percentile_value: float,
    percentile: float = 95.0,
) -> None:
    post_dir.mkdir(parents=True, exist_ok=True)
    overlap_mask = roi_mask & top_mask
    percentile_tag = f"top{int(percentile)}pct"
    finite_mask = np.isfinite(field)
    whole_brain_voxels = int(np.count_nonzero(finite_mask))
    top_percentile_voxels = int(np.count_nonzero(top_mask))
    roi_voxels = int(np.count_nonzero(roi_mask))
    overlap_top_voxels = int(np.count_nonzero(overlap_mask))
    threshold_mask = finite_mask & (field > 0.2)
    threshold_in_roi_mask = threshold_mask & roi_mask
    threshold_voxels = int(np.count_nonzero(threshold_mask))
    threshold_in_roi_voxels = int(np.count_nonzero(threshold_in_roi_mask))

    _save_nifti(post_dir / f"atlas_{roi_name}_mask.nii.gz", roi_mask.astype(np.float32))
    _save_nifti(post_dir / f"efield_{percentile_tag}_mask.nii.gz", top_mask.astype(np.float32))
    _save_nifti(
        post_dir / f"{roi_name}_overlap_{percentile_tag}_mask.nii.gz",
        overlap_mask.astype(np.float32),
    )
    _save_nifti(post_dir / f"TI_in_{roi_name}.nii.gz", field)

    roi_values = field[roi_mask]
    payload = {
        "schema_version": 4,
        "subject": subject,
        "target_roi": roi_name,
        "percentile": percentile,
        "percentile_value": percentile_value,
        "voxel_volume_mm3": 1.0,
        "whole_brain_voxels": whole_brain_voxels,
        "whole_brain_volume_mm3": float(whole_brain_voxels),
        "top_percentile_voxels": top_percentile_voxels,
        "top_percentile_percent_of_whole_brain": float(top_percentile_voxels / whole_brain_voxels * 100.0),
        "rois": {
            roi_name: {
                "roi_voxels": roi_voxels,
                "overlap_top_voxels": overlap_top_voxels,
                "roi_volume_mm3": float(roi_voxels),
                "overlap_volume_mm3": float(overlap_top_voxels),
                "overlap_fraction": float(overlap_top_voxels / roi_voxels),
                "roi_percent_of_whole_brain": float(roi_voxels / whole_brain_voxels * 100.0),
                "overlap_top_percent_of_whole_brain": float(overlap_top_voxels / whole_brain_voxels * 100.0),
                "focality_in_roi_voxels_gt_threshold": threshold_in_roi_voxels,
                "focality_in_roi_volume_mm3_gt_threshold": float(threshold_in_roi_voxels),
                "focality_in_roi_percent_of_whole_brain_gt_threshold": float(
                    threshold_in_roi_voxels / whole_brain_voxels * 100.0
                ),
            }
        },
        "extended_metrics": {
            "roi_peak": float(np.nanmax(roi_values)),
            "roi_mean": float(np.nanmean(roi_values)),
            "focality_percent_of_whole_brain_gt_threshold": float(
                threshold_voxels / whole_brain_voxels * 100.0
            ),
        },
        "extended_metric_status": {
            "roi_peak": "ok",
            "roi_mean": "ok",
        },
        "extended_metric_messages": {
            "roi_peak": None,
            "roi_mean": None,
        },
        "extended_metrics_meta": {
            "schema_version": 4,
            "status": "complete",
            "config_fingerprint": "test-fixture",
            "group_statuses": {
                "roi_intensity": "ok",
            },
            "group_messages": {
                "roi_intensity": None,
            },
        },
    }
    (post_dir / "subject_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_run_analysis_writes_image_repeatability_outputs(tmp_path, monkeypatch):
    module = _load_repeatability_module(monkeypatch)
    roi_name = "Left-Hippocampus"
    roi_mask = np.ones((2, 2, 2), dtype=bool)

    run_layout = {
        "Left_Hippocampus_Data_01": {
            "sub-01": {
                "field": np.array(
                    [[[5.0, 4.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    dtype=np.float32,
                ),
                "top_mask": np.array(
                    [[[1, 1], [0, 0]], [[0, 0], [0, 0]]],
                    dtype=bool,
                ),
                "percentile_value": 4.2,
            },
            "sub-02": {
                "field": np.array(
                    [[[1.0, 1.0], [1.0, 1.0]], [[5.2, 4.1], [1.0, 1.0]]],
                    dtype=np.float32,
                ),
                "top_mask": np.array(
                    [[[0, 0], [0, 0]], [[1, 1], [0, 0]]],
                    dtype=bool,
                ),
                "percentile_value": 4.0,
            },
        },
        "Left_Hippocampus_Data_02": {
            "sub-01": {
                "field": np.array(
                    [[[4.0, 5.0], [1.0, 4.5]], [[1.0, 1.0], [1.0, np.nan]]],
                    dtype=np.float32,
                ),
                "top_mask": np.array(
                    [[[0, 1], [0, 1]], [[0, 0], [0, 0]]],
                    dtype=bool,
                ),
                "percentile_value": 4.4,
            },
            "sub-02": {
                "field": np.array(
                    [[[1.0, 1.0], [1.0, 1.0]], [[4.2, 1.0], [5.4, np.nan]]],
                    dtype=np.float32,
                ),
                "top_mask": np.array(
                    [[[0, 0], [0, 0]], [[1, 0], [1, 0]]],
                    dtype=bool,
                ),
                "percentile_value": 4.1,
            },
        },
    }

    for run_name, subjects in run_layout.items():
        for subject, config in subjects.items():
            post_dir = tmp_path / run_name / subject / "anat" / "post"
            _write_subject_run(
                post_dir,
                subject=subject,
                roi_name=roi_name,
                field=config["field"],
                roi_mask=roi_mask,
                top_mask=config["top_mask"],
                percentile_value=config["percentile_value"],
            )

    result = module.run_analysis(dataset_root=tmp_path, roi=roi_name)
    output_dir = Path(result["output_dir"])

    for relative_path in [
        "image_repeatability_run_level.csv",
        "image_repeatability_pairwise_subject_run_pairs.csv",
        "image_repeatability_subject_level.csv",
        "image_repeatability_pairwise_run_summary.csv",
        "image_repeatability_cohort_summary.csv",
        "image_repeatability_issues.csv",
        "image_repeatability_report.md",
        "image_repeatability_methodology.md",
        "figures/08_image_repeatability_summary.png",
        "figures/09_whole_brain_occupancy_percentages.png",
    ]:
        assert (output_dir / relative_path).exists()

    pairwise = pd.read_csv(output_dir / "image_repeatability_pairwise_subject_run_pairs.csv")
    assert np.allclose(pairwise["roi_mask_dice"], 1.0)
    assert (pairwise["top_percentile_mask_dice"] < 1.0).all()
    assert (pairwise["overlap_mask_dice"] < 1.0).all()
    assert (pairwise["peak_displacement_mm"] > 0.0).all()

    subject_summary = pd.read_csv(output_dir / "image_repeatability_subject_level.csv")
    assert subject_summary["roi_mask_identical_all_runs"].all()
    assert subject_summary["grid_consistent_all_runs"].all()
    assert (subject_summary["reference_roi_voxels_excluded_nonfinite"] == 1).all()
    assert (subject_summary["within_roi_field_correlation_mean"] < 1.0).all()
    assert (subject_summary["peak_displacement_mm_mean"] > 0.0).all()

    issues = pd.read_csv(output_dir / "image_repeatability_issues.csv")
    assert issues.empty
