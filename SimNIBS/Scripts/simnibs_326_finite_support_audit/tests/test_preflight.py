from __future__ import annotations

import pytest

from simnibs_326_finite_support_audit.config import target_specs
from simnibs_326_finite_support_audit.preflight import assert_output_isolated, scope


def test_scope_is_full_and_explicitly_read_only():
    result = scope()
    assert result["v3_runs"] == 1600
    assert result["matched_v4_runs"] == 1600
    assert result["direct_v3_mesh_field_reads"] == 1600
    assert result["paired_diagnostic_panels"] == 40
    assert result["fem_tasks"] == 0
    assert result["source_outputs_modified"] is False


def test_output_cannot_overlap_a_simulation_source():
    v3_root = target_specs()["left-hippocampus"].v3_root
    with pytest.raises(RuntimeError, match="overlaps read-only sources"):
        assert_output_isolated(v3_root / "audit")
