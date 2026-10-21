from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from simnibs_326_finite_support_audit.mesh import audit_mesh_field, summarize_mesh_field


def test_mesh_summary_localizes_nonfinite_values_to_roi_tetrahedra():
    roi = np.zeros((5, 5, 5), dtype=bool)
    roi[1:4, 1:4, 1:4] = True
    summary = summarize_mesh_field(
        values=np.array([0.1, np.nan, np.nan, 0.4]),
        element_centres_world=np.array(
            [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [2.0, 2.0, 2.0]]
        ),
        element_types=np.array([4, 4, 4, 2]),
        roi_mask=roi,
        roi_affine=np.eye(4),
    )

    assert summary["mesh_field_nonfinite_elements"] == 2
    assert summary["mesh_roi_elements"] == 2
    assert summary["mesh_roi_nonfinite_elements"] == 1
    assert summary["mesh_roi_min_v_per_m"] == 0.1


def test_mesh_reader_fails_closed_outside_pinned_v3_version(tmp_path, monkeypatch):
    fake = types.ModuleType("simnibs")
    fake.__version__ = "4.0.1"
    monkeypatch.setitem(sys.modules, "simnibs", fake)
    mesh_path = tmp_path / "TI.msh"
    mesh_path.write_bytes(b"mesh")

    with pytest.raises(RuntimeError, match="requires SimNIBS 3.2.6"):
        audit_mesh_field(
            mesh_path,
            roi_mask=np.ones((2, 2, 2), dtype=bool),
            roi_affine=np.eye(4),
        )
