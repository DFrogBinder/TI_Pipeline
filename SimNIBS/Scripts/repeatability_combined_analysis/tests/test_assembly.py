from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from repeatability_combined_analysis.assembly import (
    AssemblyError,
    build_assembly_plan,
    materialize_inputs,
)
from repeatability_combined_analysis.settings import (
    EXCLUDED_SUBJECT,
    REPLACEMENT_SUBJECT,
    SUBJECTS,
)


def test_build_and_materialize_exact_four_table_9_plus_1_scope(
    analysis_config,
    tmp_path: Path,
) -> None:
    plan = build_assembly_plan(analysis_config)

    assert plan.total_rows == 3200
    assert len(plan.selections) == 10
    assert {table.row_count for table in plan.tables} == {800}
    assert not analysis_config.output_root.exists()

    paths = materialize_inputs(plan, tmp_path / "canonical")
    assert set(paths) == {
        "simnibs326_left_hippocampus",
        "simnibs326_right_m1",
        "simnibs4_left_hippocampus",
        "simnibs4_right_m1",
    }
    for path in paths.values():
        frame = pd.read_csv(path)
        assert len(frame) == 800
        assert tuple(frame["subject"].drop_duplicates()) == SUBJECTS
        assert set(frame["condition"]) == {"remesh", "fixed_mesh"}
        assert set(frame.groupby(["subject", "condition"]).size()) == {40}
        assert EXCLUDED_SUBJECT not in set(frame["subject"])
        assert REPLACEMENT_SUBJECT in set(frame["subject"])

    manifest = json.loads(
        (tmp_path / "canonical" / "assembly_manifest.json").read_text()
    )
    assert manifest["status"] == "complete"
    assert manifest["total_rows"] == 3200
    assert len(manifest["sources"]) == 10


def test_v4_rows_are_selected_from_the_approved_source_roles(analysis_config) -> None:
    plan = build_assembly_plan(analysis_config)
    table = next(
        item for item in plan.tables if item.key == "simnibs4_left_hippocampus"
    )
    frame = table.to_frame()

    retained = frame[frame["subject"] != REPLACEMENT_SUBJECT]
    assert set(retained.loc[retained["condition"] == "remesh", "source_origin"]) == {
        "v4_historical_left"
    }
    assert set(
        retained.loc[retained["condition"] == "fixed_mesh", "source_origin"]
    ) == {"v4_corrected_fixed_left"}
    assert set(
        frame.loc[frame["subject"] == REPLACEMENT_SUBJECT, "source_origin"]
    ) == {"v4_replacement_left"}

    source = next(
        item for item in plan.selections if item.name == "v4_historical_left"
    )
    assert source.path == analysis_config.sources.v4_historical_left_csv
    assert source.size_bytes > 0
    assert len(source.sha256) == 64
    assert source.observed_rows == 800
    assert source.selected_rows == 360
    assert source.rejected_rows == 440
    assert source.conditions == ("remesh",)
    assert len(source.subjects) == 9
    assert len(source.selected_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("wrong_subject", "subject"),
        ("duplicate", "duplicate"),
        ("missing_repeat", "repeat"),
        ("wrong_roi", "ROI"),
        ("wrong_volume", "volume"),
        ("missing_mean", "missing columns"),
        ("negative", "negative"),
        ("nonfinite", "non-finite"),
        ("bad_support", "support"),
        ("changing_geometry", "geometry"),
        ("old_schema", "schema"),
    ],
)
def test_invalid_metric_sources_fail_closed(
    analysis_config,
    mutation: str,
    match: str,
) -> None:
    path = analysis_config.sources.v3_base9_left_csv
    frame = pd.read_csv(path)
    if mutation == "wrong_subject":
        frame.loc[frame["subject"] == frame["subject"].iloc[0], "subject"] = "sub-CC999999"
    elif mutation == "duplicate":
        frame.iloc[1] = frame.iloc[0]
    elif mutation == "missing_repeat":
        frame = frame.drop(frame.index[39])
    elif mutation == "wrong_roi":
        frame["roi"] = "Wrong_ROI"
    elif mutation == "wrong_volume":
        frame["requested_roi_volume_mm3"] = 201.0
    elif mutation == "missing_mean":
        frame = frame.drop(columns=["roi_mean_v_per_m"])
    elif mutation == "negative":
        frame.loc[0, "roi_median_v_per_m"] = -0.1
    elif mutation == "nonfinite":
        frame.loc[0, "roi_p95_v_per_m"] = float("nan")
    elif mutation == "bad_support":
        frame.loc[0, "finite_roi_voxels"] = 1
    elif mutation == "changing_geometry":
        frame.loc[0, "roi_radius_mm"] = 9.0
    elif mutation == "old_schema":
        frame["schema_version"] = 1
    frame.to_csv(path, index=False)

    with pytest.raises(AssemblyError, match=match):
        build_assembly_plan(analysis_config)
    assert not analysis_config.output_root.exists()


def test_missing_replacement_schema_table_reports_exact_refresh_scope(
    analysis_config,
) -> None:
    path = analysis_config.sources.v4_replacement_left_csv
    pd.DataFrame(
        {"repeat_tag": ["repeat_001"], "median_roi": [0.2]}
    ).to_csv(path, index=False)

    with pytest.raises(AssemblyError, match="160 existing NIfTI metrics"):
        build_assembly_plan(analysis_config)
    assert not analysis_config.output_root.exists()


def test_materialization_rejects_a_source_changed_after_planning(
    analysis_config,
    tmp_path: Path,
) -> None:
    plan = build_assembly_plan(analysis_config)
    with analysis_config.sources.v3_replacement_left_csv.open("a") as handle:
        handle.write("\n")

    with pytest.raises(AssemblyError, match="changed after planning"):
        materialize_inputs(plan, tmp_path / "canonical")
    assert not (tmp_path / "canonical" / "assembly_manifest.json").exists()

