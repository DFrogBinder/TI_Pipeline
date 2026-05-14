import nibabel as nib
import numpy as np
import pytest

from post.metric_extensions import (
    build_fixed_neighbor_masks,
    compute_focality_metrics,
    flatten_subject_metric_payload,
)


def test_build_fixed_neighbor_masks_uses_fixed_template_labels_in_subject_space(tmp_path):
    mni_atlas = np.zeros((5, 5, 5), dtype=np.int32)
    mni_atlas[2, 2, 2] = 1022  # ctx-lh-precentral target
    mni_atlas[1, 2, 2] = 1002  # fixed-template neighbor
    mni_atlas[3, 2, 2] = 1003  # fixed-template neighbor
    mni_path = tmp_path / "mni_atlas.nii.gz"
    nib.save(nib.Nifti1Image(mni_atlas, np.eye(4)), mni_path)

    subject_atlas = np.zeros((5, 5, 5), dtype=np.int32)
    subject_atlas[2, 2, 2] = 1022
    subject_atlas[0, 0, 0] = 1002
    subject_atlas[4, 4, 4] = 1003

    masks = build_fixed_neighbor_masks(
        mni_fixed_atlas_path=str(mni_path),
        roi_name="ctx-lh-precentral",
        dilation_iter=1,
        subject_atlas_data=subject_atlas,
    )

    assert masks["neighbor_label_ids"] == [1002, 1003]
    assert masks["neighbor_union_mask"][0, 0, 0]
    assert masks["neighbor_union_mask"][4, 4, 4]
    assert not masks["neighbor_union_mask"][2, 2, 2]
    assert set(np.unique(masks["neighbor_categorical_mask"])) == {0, 1002, 1003}


def test_build_fixed_neighbor_masks_supports_direct_destrieux_roi_labels(tmp_path):
    mni_atlas = np.zeros((5, 5, 5), dtype=np.int32)
    mni_atlas[2, 2, 2] = 12115  # ctx_rh_G_front_middle target
    mni_atlas[1, 2, 2] = 12154  # neighboring Destrieux sulcus
    mni_atlas[3, 2, 2] = 12116  # neighboring Destrieux gyrus
    mni_path = tmp_path / "mni_destrieux_atlas.nii.gz"
    nib.save(nib.Nifti1Image(mni_atlas, np.eye(4)), mni_path)

    subject_atlas = np.zeros((5, 5, 5), dtype=np.int32)
    subject_atlas[2, 2, 2] = 12115
    subject_atlas[0, 0, 0] = 12154
    subject_atlas[4, 4, 4] = 12116

    masks = build_fixed_neighbor_masks(
        mni_fixed_atlas_path=str(mni_path),
        roi_name="right_dlpc",
        dilation_iter=1,
        subject_atlas_data=subject_atlas,
    )

    assert masks["neighbor_label_ids"] == [12116, 12154]
    assert masks["neighbor_union_mask"][0, 0, 0]
    assert masks["neighbor_union_mask"][4, 4, 4]
    assert not masks["neighbor_union_mask"][2, 2, 2]
    assert set(np.unique(masks["neighbor_categorical_mask"])) == {0, 12116, 12154}


def test_focality_percent_uses_finite_ti_voxels_as_whole_brain_denominator():
    ti_data = np.array(
        [[[0.10, 0.30], [0.40, np.nan]], [[0.00, 0.25], [0.19, 0.50]]],
        dtype=np.float32,
    )
    ti_img = nib.Nifti1Image(ti_data, np.eye(4))
    finite = np.isfinite(ti_data)

    metrics = compute_focality_metrics(
        ti_img=ti_img,
        ti_data=ti_data,
        finite_mask=finite,
        focality_threshold=0.2,
    )

    assert metrics["focality_voxels_gt_threshold"] == 4
    assert metrics["focality_percent_of_whole_brain_gt_threshold"] == pytest.approx(4 / 7 * 100.0)


def test_flatten_subject_metric_payload_includes_whole_brain_occupancy_fields():
    payload = {
        "subject": "sub-01",
        "target_roi": "Target",
        "percentile": 95.0,
        "percentile_value": 0.42,
        "voxel_volume_mm3": 1.0,
        "whole_brain_voxels": 100,
        "whole_brain_volume_mm3": 100.0,
        "top_percentile_voxels": 5,
        "top_percentile_percent_of_whole_brain": 5.0,
        "rois": {
            "Target": {
                "roi_voxels": 10,
                "overlap_top_voxels": 2,
                "roi_volume_mm3": 10.0,
                "overlap_volume_mm3": 2.0,
                "overlap_fraction": 0.2,
                "roi_percent_of_whole_brain": 10.0,
                "overlap_top_percent_of_whole_brain": 2.0,
                "focality_in_roi_voxels_gt_threshold": 3,
                "focality_in_roi_volume_mm3_gt_threshold": 3.0,
                "focality_in_roi_percent_of_whole_brain_gt_threshold": 3.0,
            }
        },
        "extended_metrics": {
            "focality_percent_of_whole_brain_gt_threshold": 12.0,
        },
    }

    flattened = flatten_subject_metric_payload(payload, "Target")

    assert flattened["whole_brain_voxels"] == 100
    assert flattened["top_percentile_percent_of_whole_brain"] == 5.0
    assert flattened["roi_percent_of_whole_brain"] == 10.0
    assert flattened["overlap_top_percent_of_whole_brain"] == 2.0
    assert flattened["focality_percent_of_whole_brain_gt_threshold"] == 12.0
    assert flattened["focality_in_roi_percent_of_whole_brain_gt_threshold"] == 3.0
