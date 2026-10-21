from __future__ import annotations

import importlib

import numpy as np
import pytest


def _audit_module():
    try:
        return importlib.import_module("simnibs_326_finite_support_audit.audit")
    except ModuleNotFoundError:
        pytest.fail("finite-support audit module has not been implemented")


def test_classify_voxel_support_separates_interpolation_mask_and_unexplained_gaps():
    """Catches collapsing every non-finite final voxel into one missing category."""

    audit = _audit_module()
    roi = np.ones((2, 2, 2), dtype=bool)
    raw = np.array(
        [
            [[0.30, np.nan], [0.20, 0.10]],
            [[0.15, 0.12], [0.11, 0.09]],
        ],
        dtype=float,
    )
    labels = np.array(
        [
            [[2, 2], [0, 1]],
            [[2, 5], [1, 2]],
        ],
        dtype=np.int16,
    )
    masked = np.array(
        [
            [[0.30, np.nan], [np.nan, 0.10]],
            [[np.nan, np.nan], [0.11, 0.09]],
        ],
        dtype=float,
    )

    result = audit.classify_voxel_support(
        roi_mask=roi,
        raw_field=raw,
        masked_field=masked,
        tissue_labels=labels,
    )

    assert result == {
        "roi_voxels": 8,
        "final_finite_voxels": 4,
        "raw_nonfinite_voxels": 1,
        "mask_excluded_voxels": 2,
        "unexplained_nonfinite_voxels": 1,
        "finite_outside_brain_mask_voxels": 0,
        "final_finite_raw_nonfinite_voxels": 0,
        "finite_value_mismatch_voxels": 0,
        "tissue_label_counts": {"0": 1, "1": 2, "2": 4, "5": 1},
    }


@pytest.mark.parametrize(
    ("mesh_nonfinite", "raw_nonfinite", "mask_excluded", "unexplained", "expected"),
    [
        (2, 0, 0, 0, "mesh_field_nonfinite"),
        (0, 3, 0, 0, "interpolation_nonfinite"),
        (0, 0, 4, 0, "brain_mask_exclusion"),
        (0, 2, 5, 0, "mixed_interpolation_and_mask_exclusion"),
        (0, 0, 0, 1, "masked_output_inconsistent"),
        (0, 0, 0, 0, "complete_support"),
    ],
)
def test_classify_run_cause_names_the_earliest_failed_boundary(
    mesh_nonfinite,
    raw_nonfinite,
    mask_excluded,
    unexplained,
    expected,
):
    """Catches reporting a downstream mask issue when the mesh/raw field failed first."""

    audit = _audit_module()
    support = {
        "raw_nonfinite_voxels": raw_nonfinite,
        "mask_excluded_voxels": mask_excluded,
        "unexplained_nonfinite_voxels": unexplained,
        "finite_outside_brain_mask_voxels": 0,
        "final_finite_raw_nonfinite_voxels": 0,
        "finite_value_mismatch_voxels": 0,
    }

    assert audit.classify_run_cause(
        mesh_nonfinite_voxels=mesh_nonfinite,
        support=support,
    ) == expected


def test_final_value_change_is_a_masked_output_inconsistency():
    audit = _audit_module()
    result = audit.classify_voxel_support(
        roi_mask=np.ones((1, 1, 1), dtype=bool),
        raw_field=np.array([[[0.2]]]),
        masked_field=np.array([[[0.3]]]),
        tissue_labels=np.array([[[2]]]),
    )
    assert result["finite_value_mismatch_voxels"] == 1
    assert (
        audit.classify_run_cause(mesh_nonfinite_voxels=0, support=result)
        == "masked_output_inconsistent"
    )
