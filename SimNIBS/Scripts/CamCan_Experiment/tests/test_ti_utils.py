import numpy as np
import nibabel as nib

from utils import ti_utils as t


def test_normalize_roi_name():
    assert t.normalize_roi_name("Left Hippocampus") == "Left_Hippocampus"
    assert t.normalize_roi_name("M1-L") == "M1_L"


def test_load_ti_as_scalar_handles_vector_and_scalar():
    scalar = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    vec = nib.Nifti1Image(np.ones((2, 2, 2, 3)), np.eye(4))
    np.testing.assert_allclose(t.load_ti_as_scalar(scalar), 1.0)
    np.testing.assert_allclose(t.load_ti_as_scalar(vec), np.sqrt(3))


def test_summarize_atlas_regions_simple_case():
    data = np.array([[[1, 2], [3, 4]], [[0, 1], [2, 3]]], dtype=np.float32)
    ti_img = nib.Nifti1Image(data, np.eye(4))

    atlas_data = np.array([[[10, 20], [10, 30]], [[40, 10], [20, 30]]], dtype=np.int16)
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    label_map = {10: "A", 20: "B", 30: "C", 40: "D"}

    df = t.summarize_atlas_regions(ti_img, atlas_img, label_map, percentile=50)
    names = set(df["label_name"])
    assert names == {"A", "B", "C", "D"}
    # Check one region's mean
    mean_a = df[df["label_name"] == "A"]["mean"].iloc[0]
    assert np.isclose(mean_a, (1 + 3 + 1) / 3)


def test_summarize_atlas_regions_includes_unmapped_destrieux_labels():
    ti_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    atlas_data = np.zeros((2, 2, 2), dtype=np.int32)
    atlas_data[0, 0, 0] = 12115
    atlas_data[1, 1, 1] = 12154
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))

    frame = t.summarize_atlas_regions(
        ti_img,
        atlas_img,
        {12115: "ctx_rh_G_front_middle"},
    )

    assert set(frame["label_id"]) == {12115, 12154}
    assert frame.set_index("label_id").loc[12154, "label_name"] == "Label-12154"
