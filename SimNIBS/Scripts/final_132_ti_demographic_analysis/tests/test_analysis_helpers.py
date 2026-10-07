from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_final132_demographics.py"
SPEC = importlib.util.spec_from_file_location("final132_demographic_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_safe_ratio_handles_valid_and_invalid_denominators() -> None:
    assert MODULE.safe_ratio(1.0, 4.0) == 0.25
    assert np.isnan(MODULE.safe_ratio(1.0, 0.0))


def test_mean_ci_uses_participant_count() -> None:
    n, mean_value, sd_value, low, high = MODULE.mean_ci([1.0, 2.0, 3.0, 4.0])
    assert n == 4
    assert mean_value == 2.5
    assert sd_value > 0
    assert low < mean_value < high


def test_expected_full_scope_is_5280() -> None:
    assert MODULE.EXPECTED_RUNS == 132 * 4 * 10 == 5280


def test_primary_endpoint_is_target_roi_mean() -> None:
    assert MODULE.PRIMARY_METRIC == "roi_mean_v_per_m"
    assert MODULE.METRIC_INFO[MODULE.PRIMARY_METRIC]["primary"] is True
