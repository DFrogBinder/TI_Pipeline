from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simnibs_segmentation_determinism.journal_spatial_data import (
    JournalSpatialDataError,
    prepare_journal_spatial_data,
    weighted_slice_indices,
)
from simnibs_segmentation_determinism.provenance import (
    sha256_file,
    write_json_atomic,
    write_sha256_sidecar,
)
from simnibs_segmentation_determinism.tests.test_visualizations import (
    _completed_source,
)


def _non_ras_spatial_fixture(config, *, t1_data=None, label_affine=None):
    shape = (4, 5, 6)
    affine = np.asarray(
        label_affine
        if label_affine is not None
        else [
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


def test_spatial_summary_coordinates_use_the_display_plane_centre(
    configured_experiment,
):
    affine = np.asarray(
        [
            [2.0, 0.25, 0.0, -8.0],
            [0.1, 2.0, 0.15, -6.0],
            [0.0, 0.2, 3.0, -10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _non_ras_spatial_fixture(configured_experiment, label_affine=affine)

    data = prepare_journal_spatial_data(configured_experiment)
    centre = np.asarray(
        [(start + stop - 1) / 2.0 for start, stop in data.crop_bounds]
    )
    for axis, orientation in enumerate(("sagittal", "coronal", "axial")):
        voxel = centre.copy()
        voxel[axis] = data.main_slice_indices[orientation]
        expected = nib.affines.apply_affine(data.affine, voxel)[axis]
        assert data.spatial_summary["main_slice_coordinates_mm"][orientation] == pytest.approx(
            expected
        )


def test_prepare_journal_spatial_data_fails_when_t1_has_no_positive_intensity(
    configured_experiment,
):
    _non_ras_spatial_fixture(
        configured_experiment,
        t1_data=np.zeros((4, 5, 6), dtype=np.float32),
    )

    with pytest.raises(JournalSpatialDataError, match="finite positive"):
        prepare_journal_spatial_data(configured_experiment)


def test_prepare_journal_spatial_data_reports_malformed_repeat_input_provenance(
    configured_experiment,
):
    root = _non_ras_spatial_fixture(configured_experiment)
    terminal = root / "_pipeline/complete.json"
    terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
    record = next(
        item
        for item in terminal_payload["repeat_receipts"]
        if item["arm"] == "simnibs401_charm"
    )
    receipt = root / Path(record["path"]).relative_to(root)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["inputs"] = []
    write_json_atomic(receipt, receipt_payload)
    write_sha256_sidecar(receipt)
    record["sha256"] = sha256_file(receipt)
    write_json_atomic(terminal, terminal_payload)
    write_sha256_sidecar(terminal)

    with pytest.raises(JournalSpatialDataError, match="lacks T1 provenance"):
        prepare_journal_spatial_data(configured_experiment)
