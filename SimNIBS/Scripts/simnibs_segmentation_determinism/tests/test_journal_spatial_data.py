from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from simnibs_segmentation_determinism.journal_spatial_data import (
    JournalSpatialDataError,
    prepare_journal_spatial_data,
    weighted_slice_indices,
)
from simnibs_segmentation_determinism.tests.test_visualizations import (
    _completed_source,
)


def _non_ras_spatial_fixture(config, *, t1_data=None):
    shape = (4, 5, 6)
    affine = np.asarray(
        [
            [-2.0, 0.0, 0.0, 6.0],
            [0.0, 2.0, 0.0, -4.0],
            [0.0, 0.0, 3.0, -9.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    arrays = []
    for repeat in range(10):
        labels = np.zeros(shape, dtype=np.uint8)
        labels[:, 1:4, 1:5] = 1
        if repeat < 4:
            labels[0, 2, 3] = 2
        arrays.append(labels)
    if t1_data is None:
        t1_data = np.linspace(1.0, 100.0, num=np.prod(shape), dtype=np.float32).reshape(
            shape
        )
    return _completed_source(
        config,
        label_arrays=arrays,
        label_affine=affine,
        t1_data=t1_data,
        t1_affine=affine,
    )


def test_weighted_slice_indices_are_deterministic_and_have_zero_fallback():
    assert weighted_slice_indices(np.ones(8), count=4) == (1, 3, 4, 6)
    assert weighted_slice_indices(np.zeros(7), count=4) == (0, 2, 4, 6)


def test_prepare_journal_spatial_data_reorients_ras_and_resamples_t1(
    configured_experiment,
):
    _non_ras_spatial_fixture(configured_experiment)

    data = prepare_journal_spatial_data(configured_experiment)

    assert nib.aff2axcodes(data.affine) == ("R", "A", "S")
    assert data.disagreement_counts.dtype == np.uint8
    assert data.disagreement_counts[3, 2, 3] == 4
    assert set(np.unique(data.disagreement_counts)) == {0, 4}
    assert data.t1_normalized.shape == data.modal_labels.shape == (4, 5, 6)
    assert np.min(data.t1_normalized) == pytest.approx(0.0)
    assert np.max(data.t1_normalized) == pytest.approx(1.0)
    assert data.repeat_count == 10
    assert data.t1_path.is_file()


def test_prepare_journal_spatial_data_clips_crop_and_selects_valid_slices(
    configured_experiment,
):
    _non_ras_spatial_fixture(configured_experiment)

    data = prepare_journal_spatial_data(configured_experiment)

    assert data.crop_bounds == ((0, 4), (0, 5), (0, 6))
    for axis, size in zip(("sagittal", "coronal", "axial"), data.modal_labels.shape):
        assert 0 <= data.main_slice_indices[axis] < size
        selected = data.atlas_slice_indices[axis]
        assert len(selected) == 4
        assert tuple(sorted(set(selected))) == selected
        assert all(0 <= index < size for index in selected)


def test_prepare_journal_spatial_data_fails_when_t1_has_no_positive_intensity(
    configured_experiment,
):
    _non_ras_spatial_fixture(
        configured_experiment,
        t1_data=np.zeros((4, 5, 6), dtype=np.float32),
    )

    with pytest.raises(JournalSpatialDataError, match="finite positive"):
        prepare_journal_spatial_data(configured_experiment)
