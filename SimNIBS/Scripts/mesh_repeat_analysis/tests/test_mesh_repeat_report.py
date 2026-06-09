import numpy as np

from post import mesh_repeat_report


def test_roi_metric_data_falls_back_to_raw_ti_when_masked_roi_is_empty():
    roi_mask = np.array([True, True, False])
    masked_data = np.array([np.nan, np.nan, 3.0], dtype=np.float32)
    raw_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    roi_data, source = mesh_repeat_report._roi_metric_data(
        masked_data=masked_data,
        raw_data=raw_data,
        roi_mask=roi_mask,
    )

    assert source == "raw_ti"
    np.testing.assert_allclose(roi_data, raw_data)


def test_roi_metric_data_keeps_masked_ti_when_roi_has_finite_values():
    roi_mask = np.array([True, True, False])
    masked_data = np.array([1.0, 2.0, np.nan], dtype=np.float32)
    raw_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    roi_data, source = mesh_repeat_report._roi_metric_data(
        masked_data=masked_data,
        raw_data=raw_data,
        roi_mask=roi_mask,
    )

    assert source == "masked_ti"
    np.testing.assert_allclose(roi_data, masked_data)
