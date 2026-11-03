from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from repeatability_combined_analysis.settings import METRICS, SUBJECTS
from repeatability_combined_analysis.statistics import (
    build_contrasts,
    build_support_audit,
    exact_sign_flip,
    holm_adjust,
    summarize_cells,
)


def complete_run_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_index, subject in enumerate(SUBJECTS):
        for version_index, version in enumerate(("simnibs326", "simnibs4")):
            for target_index, target in enumerate(
                ("left_hippocampus", "right_m1")
            ):
                for condition_index, condition in enumerate(
                    ("remesh", "fixed_mesh")
                ):
                    for repeat in range(1, 41):
                        central = (
                            0.10
                            + 0.002 * subject_index
                            + 0.020 * version_index
                            + 0.030 * target_index
                            + 0.010 * condition_index
                            + 0.001 * version_index * condition_index
                        )
                        # Different run slopes make technical SD contrasts
                        # non-zero while preserving simple central contrasts.
                        slope = (
                            0.0001
                            + 0.00002 * version_index
                            + 0.00001 * condition_index
                        )
                        value = central + slope * (repeat - 20.5)
                        roi_voxels = 201 if target_index == 0 else 102
                        nonfinite = int(
                            subject_index == 0
                            and version_index == 0
                            and target_index == 0
                            and condition_index == 0
                            and repeat in (1, 2)
                        )
                        finite = roi_voxels - nonfinite
                        rows.append(
                            {
                                "version": version,
                                "target": target,
                                "subject": subject,
                                "condition": condition,
                                "repeat_tag": f"repeat_{repeat:03d}",
                                "roi_voxels": roi_voxels,
                                "achieved_roi_volume_mm3": float(roi_voxels),
                                "finite_roi_voxels": finite,
                                "nonfinite_roi_voxels": nonfinite,
                                "finite_roi_fraction": finite / roi_voxels,
                                "roi_median_v_per_m": value,
                                "roi_mean_v_per_m": value + 0.01,
                                "roi_p95_v_per_m": value + 0.03,
                            }
                        )
    return pd.DataFrame(rows)


def test_cell_summary_has_80_cells_per_metric_and_exact_statistics() -> None:
    frame = complete_run_frame()
    summary = summarize_cells(frame)

    assert len(summary) == 80 * len(METRICS)
    assert set(summary.groupby("metric").size()) == {80}
    row = summary.loc[
        (summary["version"] == "simnibs326")
        & (summary["target"] == "left_hippocampus")
        & (summary["condition"] == "remesh")
        & (summary["subject"] == SUBJECTS[0])
        & (summary["metric"] == "roi_median_v_per_m")
    ].iloc[0]
    values = frame.loc[
        (frame["version"] == "simnibs326")
        & (frame["target"] == "left_hippocampus")
        & (frame["condition"] == "remesh")
        & (frame["subject"] == SUBJECTS[0]),
        "roi_median_v_per_m",
    ].to_numpy()
    assert row["run_count"] == 40
    assert row["cell_mean"] == pytest.approx(np.mean(values))
    assert row["cell_median"] == pytest.approx(np.median(values))
    assert row["cell_sd"] == pytest.approx(np.std(values, ddof=1))
    assert row["cell_cv"] == pytest.approx(np.std(values, ddof=1) / np.mean(values))
    assert row["cell_iqr"] == pytest.approx(
        np.percentile(values, 75) - np.percentile(values, 25)
    )
    assert row["cell_range"] == pytest.approx(np.ptp(values))
    assert row["minimum_finite_roi_voxels"] == 200
    assert row["minimum_finite_fraction"] == pytest.approx(200 / 201)
    assert row["affected_run_count"] == 2


def test_cell_summary_uses_null_cv_when_cell_mean_is_zero() -> None:
    frame = complete_run_frame()
    mask = (
        (frame["version"] == "simnibs326")
        & (frame["target"] == "left_hippocampus")
        & (frame["condition"] == "remesh")
        & (frame["subject"] == SUBJECTS[0])
    )
    frame.loc[mask, "roi_median_v_per_m"] = 0.0

    row = summarize_cells(frame).loc[
        lambda current: (current["version"] == "simnibs326")
        & (current["target"] == "left_hippocampus")
        & (current["condition"] == "remesh")
        & (current["subject"] == SUBJECTS[0])
        & (current["metric"] == "roi_median_v_per_m")
    ].iloc[0]
    assert row["cell_mean"] == 0.0
    assert math.isnan(row["cell_cv"])


@pytest.mark.parametrize(
    ("outcome", "expected_prefix"),
    (("cell_median", "central"), ("cell_sd", "variability")),
)
def test_contrast_shapes_signs_and_raw_participant_values(
    outcome: str, expected_prefix: str
) -> None:
    summary = summarize_cells(complete_run_frame())
    contrasts = build_contrasts(summary, outcome)

    assert set(contrasts) == {"version", "condition", "interaction"}
    assert len(contrasts["version"]) == 12
    assert len(contrasts["condition"]) == 12
    assert len(contrasts["interaction"]) == 6
    assert set(contrasts["version"]["family"]) == {
        f"{expected_prefix}_version"
    }
    assert set(contrasts["condition"]["family"]) == {
        f"{expected_prefix}_condition"
    }
    assert set(contrasts["interaction"]["family"]) == {
        f"{expected_prefix}_interaction"
    }

    version_row = contrasts["version"].loc[
        lambda current: (current["target"] == "left_hippocampus")
        & (current["condition"] == "remesh")
        & (current["metric"] == "roi_median_v_per_m")
    ].iloc[0]
    condition_row = contrasts["condition"].loc[
        lambda current: (current["target"] == "left_hippocampus")
        & (current["version"] == "simnibs326")
        & (current["metric"] == "roi_median_v_per_m")
    ].iloc[0]
    interaction_row = contrasts["interaction"].loc[
        lambda current: (current["target"] == "left_hippocampus")
        & (current["metric"] == "roi_median_v_per_m")
    ].iloc[0]

    version_values = json.loads(version_row["participant_contrasts_json"])
    condition_values = json.loads(condition_row["participant_contrasts_json"])
    interaction_values = json.loads(
        interaction_row["participant_contrasts_json"]
    )
    assert list(version_values) == list(SUBJECTS)
    assert len(version_values) == len(condition_values) == len(interaction_values) == 10
    if outcome == "cell_median":
        assert np.allclose(list(version_values.values()), -0.02)
        assert np.allclose(list(condition_values.values()), 0.01)
        assert np.allclose(list(interaction_values.values()), -0.001)
    else:
        assert version_row["mean_difference"] < 0
        assert condition_row["mean_difference"] > 0
        assert interaction_row["mean_difference"] < 0


def test_exact_sign_flip_enumerates_all_1024_assignments() -> None:
    result = exact_sign_flip(np.ones(10))
    assert result["assignments"] == 1024
    assert result["n"] == 10
    assert result["exact_p_value"] == pytest.approx(2 / 1024)
    assert result["mean_difference"] == 1.0
    assert result["median_difference"] == 1.0
    assert result["sd_difference"] == 0.0
    assert math.isnan(result["cohen_dz"])


def test_exact_sign_flip_all_zero_and_nonfinite_rejection() -> None:
    result = exact_sign_flip(np.zeros(10))
    assert result["exact_p_value"] == 1.0
    assert math.isnan(result["cohen_dz"])
    with pytest.raises(ValueError, match="finite"):
        exact_sign_flip(np.array([0.0, np.nan]))


def test_holm_adjust_is_stable_and_monotonic_in_rank_order() -> None:
    raw = np.array([0.04, 0.01, 0.03, 0.01])
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.06, 0.04, 0.06, 0.04])
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1.0)


def test_zero_sd_cells_still_produce_raw_sd_contrasts() -> None:
    frame = complete_run_frame()
    frame.loc[:, list(METRICS)] = frame.groupby(
        ["version", "target", "condition", "subject"]
    )[list(METRICS)].transform("mean")
    contrast_tables = build_contrasts(summarize_cells(frame), "cell_sd")
    for table in contrast_tables.values():
        assert np.allclose(table["mean_difference"], 0.0)
        assert np.allclose(table["sd_difference"], 0.0)
        assert table["cohen_dz"].isna().all()
        assert np.allclose(table["exact_p_value"], 1.0)


def test_support_audit_counts_incomplete_support_without_causal_claims() -> None:
    audit, summary = build_support_audit(complete_run_frame())
    assert len(audit) == 80
    affected = audit.loc[
        (audit["version"] == "simnibs326")
        & (audit["target"] == "left_hippocampus")
        & (audit["condition"] == "remesh")
        & (audit["subject"] == SUBJECTS[0])
    ].iloc[0]
    assert affected["runs"] == 40
    assert affected["runs_with_incomplete_support"] == 2
    assert affected["runs_with_complete_support"] == 38
    assert affected["maximum_nonfinite_roi_voxels"] == 1
    assert affected["minimum_finite_fraction"] == pytest.approx(200 / 201)
    assert summary["groups"] == 80
    assert summary["groups_with_incomplete_support"] == 1
    wording = json.dumps(summary).lower()
    assert "does not establish causation" in wording
    assert "causes field" not in wording

