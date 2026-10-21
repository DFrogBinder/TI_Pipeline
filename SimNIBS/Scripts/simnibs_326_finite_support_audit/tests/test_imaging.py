from __future__ import annotations

import numpy as np
import nibabel as nib

from simnibs_326_finite_support_audit.imaging import audit_nifti_stages


def _save(path, data, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine), path)


def test_nifti_audit_separates_interpolation_mask_and_unexplained_loss(tmp_path):
    shape = (9, 9, 9)
    atlas = np.zeros(shape, dtype=np.int16)
    atlas[2:7, 2:7, 2:7] = 17
    raw = np.ones(shape, dtype=np.float32)
    labels = np.ones(shape, dtype=np.int16)
    masked = raw.copy()
    raw[4, 4, 4] = np.nan
    masked[4, 4, 4] = np.nan
    labels[4, 4, 5] = 3
    masked[4, 4, 5] = np.nan
    masked[4, 5, 4] = np.nan

    paths = {name: tmp_path / f"{name}.nii.gz" for name in ("atlas", "raw", "labels", "masked")}
    _save(paths["atlas"], atlas)
    _save(paths["raw"], raw)
    _save(paths["labels"], labels)
    _save(paths["masked"], masked)

    result = audit_nifti_stages(
        raw_field_path=paths["raw"],
        masked_field_path=paths["masked"],
        tissue_labels_path=paths["labels"],
        atlas_path=paths["atlas"],
        roi="Left_Hippocampus",
        atlas_labels=(17,),
    )

    assert result["support"]["raw_nonfinite_voxels"] == 1
    assert result["support"]["mask_excluded_voxels"] == 1
    assert result["support"]["unexplained_nonfinite_voxels"] == 1
    assert result["grid"]["raw_masked_same_grid"] is True


def test_nifti_audit_fails_closed_when_raw_and_final_grids_differ(tmp_path):
    data = np.ones((5, 5, 5), dtype=np.float32)
    paths = {name: tmp_path / f"{name}.nii.gz" for name in ("atlas", "raw", "labels", "masked")}
    _save(paths["atlas"], np.full(data.shape, 17, dtype=np.int16))
    _save(paths["raw"], data)
    _save(paths["labels"], np.ones(data.shape, dtype=np.int16))
    _save(paths["masked"], data, affine=np.diag([2.0, 2.0, 2.0, 1.0]))

    import pytest

    with pytest.raises(ValueError, match="Raw and final fields differ"):
        audit_nifti_stages(
            raw_field_path=paths["raw"],
            masked_field_path=paths["masked"],
            tissue_labels_path=paths["labels"],
            atlas_path=paths["atlas"],
            roi="Left_Hippocampus",
            atlas_labels=(17,),
        )
