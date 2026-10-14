from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_mni_field_representativeness.py"
SPEC = importlib.util.spec_from_file_location("representativeness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_empirical_percentile_uses_midrank() -> None:
    values = np.array([1.0, 2.0, 2.0, 4.0])
    assert MODULE.empirical_percentile(values, 2.0) == pytest.approx(50.0)
    assert MODULE.empirical_percentile(values, 0.0) == pytest.approx(0.0)
    assert MODULE.empirical_percentile(values, 5.0) == pytest.approx(100.0)


def test_mni_transform_directions() -> None:
    values = pd.Series([1.0, 1.2, 0.8])
    assert MODULE.transform_from_mni(values, 1.0, "percent").tolist() == pytest.approx(
        [0.0, 20.0, -20.0]
    )
    assert MODULE.transform_from_mni(values, 1.0, "percentage_points").tolist() == pytest.approx(
        [0.0, 0.2, -0.2]
    )


def test_adjusted_diagnostics_fit_reference_only() -> None:
    ages = np.arange(20.0, 80.0, 2.0)
    sex = np.where(np.arange(len(ages)) % 2 == 0, "FEMALE", "MALE")
    reference = pd.DataFrame(
        {
            "age_years_precise": ages,
            "recorded_sex": sex,
            "value": 10.0 + 0.2 * ages + (sex == "MALE") * 2.0 + np.sin(ages),
        }
    )
    selected = pd.DataFrame(
        {
            "age_years_precise": [25.0, 45.0, 65.0, 75.0],
            "recorded_sex": ["FEMALE", "MALE", "FEMALE", "MALE"],
            "value": [15.0, 21.0, 23.0, 27.0],
        }
    )
    ref_result, selected_result, diagnostics = MODULE.fit_adjusted_diagnostics(
        reference, selected, "value"
    )
    assert len(ref_result) == len(reference)
    assert len(selected_result) == len(selected)
    assert np.isfinite(ref_result["adjusted_residual_z"]).all()
    assert np.isfinite(selected_result["adjusted_residual_z"]).all()
    assert 0.0 <= diagnostics["selected_span_reference_percent"] <= 100.0
    assert 0.0 <= diagnostics["adjusted_ks_p_value"] <= 1.0
