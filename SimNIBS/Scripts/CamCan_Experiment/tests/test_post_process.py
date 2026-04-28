import gzip
import sys
import types

import nibabel as nib
import numpy as np

post_functions_stub = types.ModuleType("post.post_functions")
post_functions_stub._resolve_fastsurfer_atlas = lambda *args, **kwargs: None
post_functions_stub.build_context_scale_mask_from_fastsurfer = lambda *args, **kwargs: None
post_functions_stub.fastsurfer_dkt_labels = {}
post_functions_stub.make_outline = lambda *args, **kwargs: None
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
ti_utils_stub.normalize_roi_name = lambda value: value
ti_utils_stub.resample_atlas_to_ti_grid = lambda *args, **kwargs: None
ti_utils_stub.save_masked_nii = lambda *args, **kwargs: None
ti_utils_stub.summarize_atlas_regions = lambda *args, **kwargs: None
ti_utils_stub.vol_mm3 = lambda *args, **kwargs: 1.0
sys.modules.setdefault("utils.ti_utils", ti_utils_stub)

import post.post_process as post_process_module
from post.post_process import PostProcessConfig, _generate_selected_roi_overlays, _load_t1_image


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
