import gzip
import sys
import types

import nibabel as nib
import numpy as np

post_functions_stub = types.ModuleType("post.post_functions")
post_functions_stub._resolve_fastsurfer_atlas = lambda *args, **kwargs: None
post_functions_stub.fastsurfer_dkt_labels = {}
post_functions_stub.make_outline = lambda *args, **kwargs: None
post_functions_stub.overlay_ti_thresholds_on_t1_with_roi = lambda *args, **kwargs: (None, None, None)
post_functions_stub.overlay_ti_thresholds_on_t1_with_roi_individual_scale = (
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

from post.post_process import PostProcessConfig, _load_t1_image


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
