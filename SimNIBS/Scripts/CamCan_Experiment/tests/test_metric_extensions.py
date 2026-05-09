import nibabel as nib
import numpy as np

from post.metric_extensions import build_fixed_neighbor_masks


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
