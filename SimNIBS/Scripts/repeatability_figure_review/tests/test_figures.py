from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from repeatability_figure_review.audit import (
    EXPECTED_ANALYSIS_FIGURES,
    audit_sources,
    sha256_file,
)
from repeatability_figure_review.figures import (
    bootstrap_subject_curve,
    load_v4_ranking_tables,
    percentile_plot_bounds,
    prepare_sd_for_log_plot,
    render_revised_figures,
)


PAPER_FIGURES = (
    "figure_target_field_repeatability.png",
    "figure_single_run_ranking_uncertainty.png",
    "figure_bootstrap_precision.png",
    "figure_mesh_element_count_variability.png",
    "figure_tissue_volume_deviations.png",
)

BANNED_READER_TERMS = (
    "optimizer-ROI",
    "Optimizer-ROI",
    "|E|",
    "within-cell",
    "Within-cell",
    "difference-in-differences",
    "Subject IDs",
    "Participant 40-run median",
    "v3 − v4 participant contrast",
)


def _write_png(path: Path, size: tuple[int, int] = (180, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def _metric_table(version: str, target: str) -> pd.DataFrame:
    rows = []
    roi = "Left_Hippocampus" if target == "left_hippocampus" else "Right_M1"
    volume = 200.0 if target == "left_hippocampus" else 100.0
    version_shift = 0.045 if version == "simnibs326" else 0.0
    target_shift = 0.02 if target == "left_hippocampus" else 0.0
    for subject_index in range(10):
        subject = f"sub-CC{100000 + subject_index}"
        base = 0.16 + 0.006 * subject_index + version_shift + target_shift
        for condition in ("remesh", "fixed_mesh"):
            for repeat in range(1, 41):
                if version == "simnibs4" and condition == "remesh":
                    delta = 0.006 * np.sin(repeat + subject_index)
                elif version == "simnibs326" and condition == "remesh":
                    delta = 1e-6 * np.sin(repeat)
                else:
                    delta = 0.0
                value = base + delta
                rows.append(
                    {
                        "schema_version": 2,
                        "subject": subject,
                        "condition": condition,
                        "repeat_tag": f"repeat_{repeat:03d}",
                        "roi": roi,
                        "requested_roi_volume_mm3": volume,
                        "roi_mean_v_per_m": value + 0.002,
                        "roi_median_v_per_m": value,
                        "roi_p95_v_per_m": value + 0.01,
                    }
                )
    return pd.DataFrame(rows)


def _make_complete_sources(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "analysis"
    manuscript = tmp_path / "manuscript"
    for relative in EXPECTED_ANALYSIS_FIGURES:
        _write_png(source / relative)
    (source / "analysis_manifest.json").write_text("{}\n", encoding="utf-8")
    for version in ("simnibs326", "simnibs4"):
        for target in ("left_hippocampus", "right_m1"):
            path = source / "inputs" / f"{version}_{target}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            _metric_table(version, target).to_csv(path, index=False)
    ranking_root = source / "established" / "within_simnibs4"
    pair_rows = []
    draw_rows = []
    subjects = [f"sub-CC{100000 + index}" for index in range(10)]
    for target_index, target in enumerate(("left_hippocampus", "right_m1")):
        for higher_index, higher in enumerate(subjects):
            for lower_index, lower in enumerate(subjects[higher_index + 1 :], start=higher_index + 1):
                pair_rows.append(
                    {
                        "target": target,
                        "higher_mean_subject": higher,
                        "lower_mean_subject": lower,
                        "higher_mean_v_per_m": 0.30 - higher_index * 0.01,
                        "lower_mean_v_per_m": 0.30 - lower_index * 0.01,
                        "mean_difference_v_per_m": (lower_index - higher_index) * 0.01,
                        "single_run_reversal_probability": min(
                            0.55, 0.02 * (lower_index - higher_index + target_index)
                        ),
                    }
                )
        draw_rows.extend(
            {
                "target": target,
                "kendall_rank_agreement": 1.0 - 2.0 * (draw % 8) / 45.0,
                "pairwise_reversals": draw % 8,
            }
            for draw in range(20_000)
        )
    pd.DataFrame(pair_rows).to_csv(
        ranking_root / "pairwise_rank_reversal_probabilities.csv", index=False
    )
    pd.DataFrame(draw_rows).to_csv(
        ranking_root / "rank_selection_draws.csv", index=False
    )
    for name in PAPER_FIGURES:
        _write_png(manuscript / "figures" / "combined" / name, (200, 120))
    (manuscript / "main.tex").write_text("paper\n", encoding="utf-8")
    return source, manuscript


def test_percentile_bounds_keep_100th_percentile_inside_axis() -> None:
    lower, upper = percentile_plot_bounds(np.asarray([100.0, 100.0]))

    assert lower <= 50.0
    assert upper > 100.0


def test_log_sd_preparation_marks_zero_values_at_explicit_floor() -> None:
    plotted, at_floor, floor = prepare_sd_for_log_plot(
        np.asarray([0.0, 1e-8, 2e-4, 7e-3])
    )

    assert floor == 1e-8
    assert plotted.tolist() == [1e-8, 1e-8, 2e-4, 7e-3]
    assert at_floor.tolist() == [True, True, False, False]


def test_bootstrap_curve_is_reproducible_and_covers_requested_counts() -> None:
    values = np.linspace(0.15, 0.25, 40)

    first = bootstrap_subject_curve(
        values, iterations=100, seed=7, min_repeats=2, max_repeats=5
    )
    second = bootstrap_subject_curve(
        values, iterations=100, seed=7, min_repeats=2, max_repeats=5
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["n_repeats"].tolist() == [2, 3, 4, 5]
    assert (first[["ci_half_width_rel", "abs_error_quantile_rel"]].to_numpy() > 0).all()


def test_v4_ranking_tables_are_loaded_with_complete_scope(tmp_path: Path) -> None:
    source, manuscript = _make_complete_sources(tmp_path)
    bundle = audit_sources(source, manuscript)

    pairs, draws = load_v4_ranking_tables(bundle)

    assert len(pairs) == 90
    assert len(draws) == 40_000
    assert set(pairs["target"]) == {"left_hippocampus", "right_m1"}
    assert set(draws["target"]) == {"left_hippocampus", "right_m1"}


def test_render_revised_figures_builds_current_and_legacy_outputs(tmp_path: Path) -> None:
    source, manuscript = _make_complete_sources(tmp_path)
    bundle = audit_sources(source, manuscript)
    output = tmp_path / "review"

    artifacts = render_revised_figures(
        bundle,
        output,
        bootstrap_iterations=100,
        bootstrap_max_repeats=5,
    )
    by_path = {item.relative_path: item for item in artifacts}

    expected = {
        "revised_figures/comprehensive/central_values.png",
        "revised_figures/comprehensive/technical_sd_log_scale.png",
        "revised_figures/comprehensive/paired_version_contrasts.png",
        "revised_figures/comprehensive/version_condition_interaction.png",
        "revised_figures/cross_version/remesh_distribution_overlay.png",
        "revised_figures/cross_version/v3_percentile_location.png",
        "revised_figures/within_simnibs326/deterministic_repeatability.png",
        "revised_figures/within_simnibs326/deterministic_ranking_summary.png",
        "paper_equivalents/figure_target_field_repeatability.png",
        "paper_equivalents/figure_single_run_ranking_uncertainty.png",
        "paper_equivalents/figure_bootstrap_precision.png",
        "legacy_paper_figures/figure_mesh_element_count_variability.png",
        "legacy_paper_figures/figure_tissue_volume_deviations.png",
        "figure_build_metadata.json",
        "bootstrap_precision/per_subject_curve.csv",
        "bootstrap_precision/precision_curve_summary.csv",
    }
    assert expected.issubset(by_path)
    assert all((output / relative).is_file() for relative in expected)
    assert all(
        (output / relative).stat().st_size > 0
        for relative in expected
    )

    for relative in expected:
        if relative.endswith(".png"):
            with Image.open(output / relative) as image:
                assert image.width >= 180
                assert image.height >= 100

    for legacy in (
        "figure_mesh_element_count_variability.png",
        "figure_tissue_volume_deviations.png",
    ):
        original = manuscript / "figures" / "combined" / legacy
        copied = output / "legacy_paper_figures" / legacy
        assert sha256_file(original) == sha256_file(copied)


def test_regenerated_figures_use_reader_facing_language_and_are_direct_renders(
    tmp_path: Path,
) -> None:
    source, manuscript = _make_complete_sources(tmp_path)
    bundle = audit_sources(source, manuscript)
    output = tmp_path / "review"

    artifacts = render_revised_figures(
        bundle,
        output,
        bootstrap_iterations=100,
        bootstrap_max_repeats=5,
    )
    regenerated = [
        item for item in artifacts if item.relative_path.endswith(".png")
        and item.category != "legacy_paper_figure"
    ]
    descriptions = {}
    for artifact in regenerated:
        with Image.open(output / artifact.relative_path) as image:
            description = image.info.get("Description", "")
        assert description, artifact.relative_path
        descriptions[artifact.relative_path] = description

    primary_text = descriptions[
        "paper_equivalents/figure_target_field_repeatability.png"
    ]
    bootstrap_text = descriptions[
        "paper_equivalents/figure_bootstrap_precision.png"
    ]
    all_text = "\n".join(descriptions.values())
    assert "Median temporal-interference field in target region (V/m)" in primary_text
    assert "Mean temporal-interference field in target region" in bootstrap_text
    assert "New mesh each run" in primary_text
    assert "Same mesh reused" in primary_text
    assert not any(term in all_text for term in BANNED_READER_TERMS)

    placeholders = [
        source
        / "established"
        / "within_simnibs4"
        / "figures"
        / target
        / source_name
        for target in ("left_hippocampus", "right_m1")
        for source_name in (
            "01_primary_median_roi_repeat_distributions.png",
            "02_single_repeat_subject_ranking_uncertainty.png",
        )
    ]
    direct_outputs = [
        output / "paper_equivalents" / "figure_target_field_repeatability.png",
        output / "paper_equivalents" / "figure_single_run_ranking_uncertainty.png",
    ]
    assert all(
        sha256_file(rendered) not in {sha256_file(source_path) for source_path in placeholders}
        for rendered in direct_outputs
    )
