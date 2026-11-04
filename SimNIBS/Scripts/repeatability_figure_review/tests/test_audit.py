from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from repeatability_figure_review.audit import (
    EXPECTED_ANALYSIS_FIGURES,
    SourceAuditError,
    audit_sources,
)


PAPER_FIGURES = (
    "figure_target_field_repeatability.png",
    "figure_single_run_ranking_uncertainty.png",
    "figure_bootstrap_precision.png",
    "figure_mesh_element_count_variability.png",
    "figure_tissue_volume_deviations.png",
)


def _write_png(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _make_source_tree(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "analysis"
    manuscript = tmp_path / "manuscript"
    for relative in EXPECTED_ANALYSIS_FIGURES:
        _write_png(source / relative)
    (source / "inputs").mkdir(parents=True)
    for name in (
        "simnibs326_left_hippocampus.csv",
        "simnibs326_right_m1.csv",
        "simnibs4_left_hippocampus.csv",
        "simnibs4_right_m1.csv",
    ):
        (source / "inputs" / name).write_text("header\n", encoding="utf-8")
    (source / "analysis_manifest.json").write_text("{}\n", encoding="utf-8")
    for name in PAPER_FIGURES:
        _write_png(manuscript / "figures" / "combined" / name, (40, 30))
    (manuscript / "main.tex").write_text("paper\n", encoding="utf-8")
    return source, manuscript


def test_audit_inventories_all_expected_analysis_pngs(tmp_path: Path) -> None:
    source, manuscript = _make_source_tree(tmp_path)

    bundle = audit_sources(source, manuscript)

    assert len(bundle.figures) == 20
    assert {record.target for record in bundle.figures} == {
        "left_hippocampus",
        "right_m1",
    }
    assert all(record.width > 0 and record.height > 0 for record in bundle.figures)
    assert all(len(record.sha256) == 64 for record in bundle.figures)
    assert {record.family for record in bundle.figures} == {
        "comprehensive",
        "cross_version_remesh",
        "within_simnibs326",
        "within_simnibs4",
    }


def test_audit_maps_all_five_paper_figures_with_honest_statuses(tmp_path: Path) -> None:
    source, manuscript = _make_source_tree(tmp_path)

    bundle = audit_sources(source, manuscript)
    by_name = {item.paper_figure: item for item in bundle.paper_equivalences}

    assert set(by_name) == set(PAPER_FIGURES)
    assert by_name["figure_target_field_repeatability.png"].status == "updated_current"
    assert by_name["figure_single_run_ranking_uncertainty.png"].status == "updated_current"
    assert by_name["figure_bootstrap_precision.png"].status == "regenerate_current"
    assert by_name["figure_mesh_element_count_variability.png"].status == "legacy_data_gap"
    assert by_name["figure_tissue_volume_deviations.png"].status == "legacy_retained_subjects"
    assert "not an updated 9+1" in by_name[
        "figure_mesh_element_count_variability.png"
    ].rationale


def test_audit_fails_closed_when_expected_figure_is_missing(tmp_path: Path) -> None:
    source, manuscript = _make_source_tree(tmp_path)
    missing = source / EXPECTED_ANALYSIS_FIGURES[0]
    missing.unlink()

    with pytest.raises(SourceAuditError, match="Required analysis figure is missing"):
        audit_sources(source, manuscript)


def test_audit_fails_closed_when_required_input_table_is_empty(tmp_path: Path) -> None:
    source, manuscript = _make_source_tree(tmp_path)
    table = source / "inputs" / "simnibs4_left_hippocampus.csv"
    table.write_text("", encoding="utf-8")

    with pytest.raises(SourceAuditError, match="Required source is missing or empty"):
        audit_sources(source, manuscript)


def test_dispositions_flag_known_presentation_failures(tmp_path: Path) -> None:
    source, manuscript = _make_source_tree(tmp_path)

    bundle = audit_sources(source, manuscript)
    by_type = {(item.family, item.figure_type): item for item in bundle.figures}

    assert by_type[("cross_version_remesh", "v3_percentile_location")].disposition == "repair_boundary_clipping"
    assert by_type[("within_simnibs326", "single_repeat_subject_ranking_uncertainty")].disposition == "replace_deterministic_blank_panel"
    assert by_type[("comprehensive", "technical_sd")].disposition == "repair_dynamic_range"
