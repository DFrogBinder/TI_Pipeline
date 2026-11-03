from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from repeatability_combined_analysis import render
from repeatability_combined_analysis.tests.test_statistics import (
    complete_run_frame,
)


def test_render_writes_four_figure_families_per_target_and_portable_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    saved: list[Path] = []

    def fake_save(_figure: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG placeholder")
        saved.append(path)

    monkeypatch.setattr(render, "_save_figure", fake_save)
    result = render.render_comprehensive_outputs(
        complete_run_frame(), tmp_path / "comprehensive"
    )

    expected_names = {
        f"figures/{target}/{family}.png"
        for target in ("left_hippocampus", "right_m1")
        for family in (
            "central_values",
            "technical_sd",
            "paired_version_contrasts",
            "version_condition_interaction",
        )
    }
    assert {path.relative_to(tmp_path / "comprehensive").as_posix() for path in saved} == expected_names
    assert len(result["figures"]) == 8
    assert set(result["figures"]) == expected_names
    for artifact in result["artifacts"]:
        assert not Path(artifact).is_absolute()
        assert (tmp_path / "comprehensive" / artifact).is_file()

    summary = pd.read_csv(
        tmp_path / "comprehensive" / "participant_cell_summary.csv"
    )
    central = pd.read_csv(
        tmp_path / "comprehensive" / "central_value_contrasts.csv"
    )
    variability = pd.read_csv(
        tmp_path / "comprehensive" / "technical_variability_contrasts.csv"
    )
    assert len(summary) == 240
    assert len(central) == 30
    assert len(variability) == 30

    report = (tmp_path / "comprehensive" / "analysis_report.md").read_text()
    captions = (tmp_path / "comprehensive" / "figure_captions.md").read_text()
    combined = f"{report}\n{captions}".lower()
    assert "participant is the inferential unit" in combined
    assert "40 technical repeats" in combined
    assert "primary" in combined and "sensitivity" in combined
    assert "fixed-mesh" in combined and "mechanistic control" in combined
    assert "aligned" in combined and "not identical" in combined
    assert "therapeutic" in combined and "continuous" in combined

    support = json.loads(
        (tmp_path / "comprehensive" / "finite_support_summary.json").read_text()
    )
    assert support["groups"] == 80


def test_render_manifest_records_interpretation_and_relative_paths(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_save(_figure: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG placeholder")

    monkeypatch.setattr(render, "_save_figure", fake_save)
    root = tmp_path / "comprehensive"
    result = render.render_comprehensive_outputs(complete_run_frame(), root)
    manifest = json.loads((root / "comprehensive_manifest.json").read_text())

    assert manifest["status"] == "complete"
    assert manifest["run_level_rows"] == 3200
    assert manifest["participant_cells"] == 80
    assert manifest["participant_cell_metric_rows"] == 240
    assert manifest["participant_is_inferential_unit"] is True
    assert manifest["technical_repeats_per_cell"] == 40
    assert manifest["primary_metric"] == "roi_median_v_per_m"
    assert manifest["sensitivity_metrics"] == [
        "roi_mean_v_per_m",
        "roi_p95_v_per_m",
    ]
    assert result["manifest"] == "comprehensive_manifest.json"
    assert all(not Path(path).is_absolute() for path in manifest["artifacts"])

