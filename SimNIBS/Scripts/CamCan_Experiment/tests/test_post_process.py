import gzip
import json
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

post_functions_stub = types.ModuleType("post.post_functions")
post_functions_stub._resolve_fastsurfer_atlas = lambda *args, **kwargs: None
post_functions_stub.build_context_scale_mask_from_fastsurfer = lambda *args, **kwargs: None
post_functions_stub.fastsurfer_dkt_labels = {}
post_functions_stub.make_outline = lambda *args, **kwargs: None
post_functions_stub.overlay_neighbor_union_on_t1_with_roi = lambda *args, **kwargs: "neighbor_overlay.png"
post_functions_stub.overlay_ti_full_field_true_vmax_reference_on_t1_with_roi = (
    lambda *args, **kwargs: None
)
post_functions_stub.overlay_ti_thresholds_on_t1_with_roi = lambda *args, **kwargs: (None, None, None)
post_functions_stub.overlay_ti_thresholds_on_t1_with_roi_individual_scale = (
    lambda *args, **kwargs: (None, None, None)
)
post_functions_stub.overlay_ti_thresholds_on_t1_with_roi_whole_brain_scale = (
    lambda *args, **kwargs: (None, None, None)
)
post_functions_stub.roi_masks_on_ti_grid = lambda *args, **kwargs: ({}, {})
post_functions_stub.write_csv = lambda *args, **kwargs: None
sys.modules.setdefault("post.post_functions", post_functions_stub)

ti_utils_stub = types.ModuleType("utils.ti_utils")
ti_utils_stub.ensure_dir = lambda *args, **kwargs: None
ti_utils_stub.extract_table = lambda *args, **kwargs: (None, None, None)
ti_utils_stub.load_ti_as_scalar = lambda *args, **kwargs: None
ti_utils_stub.normalize_roi_name = lambda value: "".join(
    c if c.isalnum() else "_" for c in value.strip().replace(" ", "_")
)
ti_utils_stub.resample_atlas_to_ti_grid = lambda *args, **kwargs: None
ti_utils_stub.save_masked_nii = lambda *args, **kwargs: None
ti_utils_stub.summarize_atlas_regions = lambda *args, **kwargs: None
ti_utils_stub.vol_mm3 = lambda *args, **kwargs: 1.0
sys.modules.setdefault("utils.ti_utils", ti_utils_stub)

import post.post_process as post_process_module
from post.post_process import (
    PostProcessConfig,
    _generate_selected_roi_overlays,
    _load_t1_image,
    _overlay_qc,
    _write_neighbor_visualization_outputs,
    run_post_process,
)


def test_load_t1_image_reads_standard_nifti(tmp_path):
    t1_path = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4)), t1_path)

    img = _load_t1_image(t1_path, PostProcessConfig(root_dir=str(tmp_path), verbose=False))

    assert img.shape == (4, 5, 6)


def test_load_t1_image_recovers_double_gzipped_nifti(tmp_path):
    source_path = tmp_path / "t1_source.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4)), source_path)

    double_gz_path = tmp_path / "t1_double.nii.gz"
    with gzip.open(double_gz_path, "wb") as f:
        f.write(source_path.read_bytes())

    img = _load_t1_image(double_gz_path, PostProcessConfig(root_dir=str(tmp_path), verbose=False))

    assert img.shape == (4, 5, 6)


def test_generate_selected_roi_overlays_retries_with_whole_brain_scale(monkeypatch, tmp_path):
    calls = []

    def fake_context_overlay(**kwargs):
        calls.append("context")
        return ("context_top95.png", "context_above200.png", None)

    def fake_roi_overlay(**kwargs):
        calls.append("roi_focus")
        raise ValueError("minvalue must be less than or equal to maxvalue")

    def fake_whole_brain_overlay(**kwargs):
        calls.append("whole_brain_fallback")
        return ("roi_top95.png", "roi_above200.png", None)

    def fake_reference_overlay(**kwargs):
        calls.append("reference")
        return "reference_full.png"

    monkeypatch.setattr(
        post_process_module,
        "overlay_ti_thresholds_on_t1_with_roi",
        fake_context_overlay,
    )
    monkeypatch.setattr(
        post_process_module,
        "overlay_ti_thresholds_on_t1_with_roi_individual_scale",
        fake_roi_overlay,
    )
    monkeypatch.setattr(
        post_process_module,
        "overlay_ti_thresholds_on_t1_with_roi_whole_brain_scale",
        fake_whole_brain_overlay,
    )
    monkeypatch.setattr(
        post_process_module,
        "overlay_ti_full_field_true_vmax_reference_on_t1_with_roi",
        fake_reference_overlay,
    )

    ti_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    t1_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    roi_mask = np.ones((2, 2, 2), dtype=bool)
    cfg = PostProcessConfig(root_dir=str(tmp_path), subject="sub-01", verbose=False)

    paths, strategy = _generate_selected_roi_overlays(
        cfg=cfg,
        ti_img=ti_img,
        ti_data=np.ones((2, 2, 2), dtype=np.float32),
        t1_img_full=t1_img,
        roi_mask=roi_mask,
        fs_atlas_img=None,
        out_dir=str(tmp_path),
        roi_name="Left-Hippocampus",
    )

    assert calls == ["context", "roi_focus", "whole_brain_fallback", "reference"]
    assert strategy == "whole_brain_fallback"
    assert paths == [
        "context_top95.png",
        "context_above200.png",
        "roi_top95.png",
        "roi_above200.png",
        "reference_full.png",
    ]


def test_overlay_qc_marks_partial_overlay_set_as_error(tmp_path):
    cfg = PostProcessConfig(root_dir=str(tmp_path), subject="sub-01", overlay_full_field=True)

    qc = _overlay_qc(
        cfg=cfg,
        overlay_paths=["ctx_TI_overlay_context_sub-01_top95.png"],
        attempted=True,
        error=None,
    )

    assert qc["status"] == "error"
    assert qc["expected_overlay_count"] == 7
    assert qc["written_overlay_count"] == 1
    assert "roi_focus_top95" in qc["missing_overlay_types"]


def test_write_neighbor_visualization_outputs_writes_union_and_categorical_masks(monkeypatch, tmp_path):
    neighbor_union = np.array(
        [
            [[False, True], [False, False]],
            [[True, False], [False, False]],
        ],
        dtype=bool,
    )
    neighbor_categorical = np.array(
        [
            [[0, 1002], [0, 0]],
            [[1003, 0], [0, 0]],
        ],
        dtype=np.int32,
    )

    def fake_build_fixed_neighbor_masks(**kwargs):
        return {
            "neighbor_template": [
                {"label_id": 1002, "label_name": "ctx-lh-caudalmiddlefrontal"},
                {"label_id": 1003, "label_name": "ctx-lh-cuneus"},
            ],
            "neighbor_label_ids": [1002, 1003],
            "neighbor_union_mask": neighbor_union,
            "neighbor_categorical_mask": neighbor_categorical,
        }

    monkeypatch.setattr(
        post_process_module,
        "build_fixed_neighbor_masks",
        fake_build_fixed_neighbor_masks,
    )

    cfg = PostProcessConfig(
        root_dir=str(tmp_path),
        subject="sub-01",
        mni_fixed_atlas_path=str(tmp_path / "mni_atlas.nii.gz"),
        verbose=False,
    )
    ti_img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    result = _write_neighbor_visualization_outputs(
        cfg=cfg,
        ti_img=ti_img,
        t1_img_full=None,
        roi_mask=np.zeros((2, 2, 2), dtype=bool),
        subject_atlas_data=np.zeros((2, 2, 2), dtype=np.int32),
        out_dir=str(tmp_path),
        roi_name="ctx-lh-precentral",
    )

    assert result["status"] == "mask_only"
    assert nib.load(result["union_mask_path"]).get_fdata().sum() == 2
    assert set(np.unique(nib.load(result["categorical_mask_path"]).get_fdata())) == {0.0, 1002.0, 1003.0}

    metadata = json.loads((tmp_path / "ctx_lh_precentral_fixed_neighbor_visualization.json").read_text())
    assert metadata["neighbor_union_voxels"] == 2
    assert metadata["neighbor_template"][0]["label_id"] == 1002


def test_run_post_process_writes_whole_brain_occupancy_metrics(monkeypatch, tmp_path):
    ti_data = np.array(
        [[[0.10, 0.30], [0.40, np.nan]], [[0.00, 0.25], [0.19, 0.50]]],
        dtype=np.float32,
    )
    roi_mask = np.array(
        [[[False, True], [True, False]], [[False, False], [False, True]]],
        dtype=bool,
    )
    ti_path = tmp_path / "ti.nii.gz"
    out_dir = tmp_path / "post"
    nib.save(nib.Nifti1Image(ti_data, np.eye(4)), ti_path)

    monkeypatch.setattr(
        post_process_module,
        "ensure_dir",
        lambda directory: Path(directory).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(post_process_module, "load_ti_as_scalar", lambda img: ti_data)
    monkeypatch.setattr(
        post_process_module,
        "roi_masks_on_ti_grid",
        lambda *args, **kwargs: ({"Target": roi_mask}, {}),
    )

    cfg = PostProcessConfig(
        root_dir=str(tmp_path),
        subject="sub-01",
        ti_path=str(ti_path),
        out_dir=str(out_dir),
        atlas_mode="mni",
        plot_roi="Target",
        percentile=95.0,
        offtarget_threshold=0.2,
        write_region_table=False,
        write_neighbor_visualization=False,
        overlay_full_field=False,
        verbose=False,
    )

    result = run_post_process(cfg)
    payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    roi_metrics = payload["rois"]["Target"]

    assert payload["whole_brain_voxels"] == 7
    assert payload["whole_brain_volume_mm3"] == 7.0
    assert payload["top_percentile_voxels"] == 1
    assert payload["top_percentile_percent_of_whole_brain"] == pytest.approx(1 / 7 * 100.0)
    assert roi_metrics["roi_percent_of_whole_brain"] == pytest.approx(3 / 7 * 100.0)
    assert roi_metrics["overlap_top_percent_of_whole_brain"] == pytest.approx(1 / 7 * 100.0)
    assert roi_metrics["focality_in_roi_voxels_gt_threshold"] == 3
    assert roi_metrics["focality_in_roi_percent_of_whole_brain_gt_threshold"] == pytest.approx(3 / 7 * 100.0)
    assert payload["extended_metrics"]["focality_percent_of_whole_brain_gt_threshold"] == pytest.approx(4 / 7 * 100.0)
