"""Assemble the complete repeatability figure-review companion package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import AuditBundle, FigureRecord, audit_sources, sha256_file
from .figures import (
    Artifact,
    TARGET_LABELS,
    _cell_summary,
    _cross_version_summary,
    _load_frame,
    render_revised_figures,
)
from .report import render_pdf


def tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    """Return a deterministic file manifest used to prove source immutability."""

    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_originals(bundle: AuditBundle, output_root: Path) -> list[Artifact]:
    artifacts = []
    for record in bundle.figures:
        relative = f"original_figures/{record.relative_path}"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle.source_root / record.relative_path, destination)
        if sha256_file(destination) != record.sha256:
            raise RuntimeError(f"Original figure checksum changed during copy: {record.relative_path}")
        artifacts.append(
            Artifact(
                relative_path=relative,
                category="original_source_figure",
                size_bytes=destination.stat().st_size,
                sha256=record.sha256,
                width=record.width,
                height=record.height,
            )
        )
    return artifacts


def _rank_context(bundle: AuditBundle, version: str, target: str) -> str:
    root = bundle.source_root / "established" / version
    pair_path = root / "pairwise_rank_reversal_probabilities.csv"
    draw_path = root / "rank_selection_draws.csv"
    if not pair_path.is_file() or not draw_path.is_file():
        return "The ranking panels summarize pairwise reversal probabilities and whole-cohort Kendall rank agreement."
    pair = pd.read_csv(pair_path)
    draws = pd.read_csv(draw_path)
    pair_values = pair.loc[pair["target"] == target, "single_run_reversal_probability"]
    draw_values = draws.loc[draws["target"] == target, "kendall_rank_agreement"]
    return (
        f"The largest pairwise reversal probability is {pair_values.max() * 100:.1f}%; "
        f"the median Kendall rank agreement is {draw_values.median():.3f} "
        f"(range {draw_values.min():.3f}–{draw_values.max():.3f})."
    )


def _figure_entries(bundle: AuditBundle, output_root: Path) -> list[dict[str, str]]:
    frame = _load_frame(bundle)
    summary = _cell_summary(frame)
    cross = _cross_version_summary(frame)
    entries = []
    for record in bundle.figures:
        target_label = TARGET_LABELS[record.target]
        target_summary = summary.loc[summary["target"] == record.target]
        key = (record.family, record.figure_type)
        purpose = encoding = finding = audit = replacement = ""
        if key == ("comprehensive", "central_values"):
            cell_medians = {
                f"{version}:{condition}": float(group["cell_median"].median())
                for (version, condition), group in target_summary.groupby(["version", "condition"])
            }
            purpose = "Compare participant-level central optimizer-ROI fields across both SimNIBS versions and both meshing conditions."
            encoding = "Each point is one participant after taking the median across 40 technical repeats; a black horizontal bar is the across-participant median."
            finding = (
                f"The participant median is {cell_medians['simnibs326:remesh']:.4f} V/m for v3 remesh and "
                f"{cell_medians['simnibs4:remesh']:.4f} V/m for v4 remesh. Remesh and fixed-mesh centres are close within each version."
            )
            audit = "Scientifically valid, but separate target files and unexplained black bars weaken manuscript readability."
            replacement = "Use revised_figures/comprehensive/central_values.png."
        elif key == ("comprehensive", "technical_sd"):
            v4 = target_summary.loc[(target_summary.version == "simnibs4") & (target_summary.condition == "remesh"), "cell_sd"]
            controls = target_summary.loc[~((target_summary.version == "simnibs4") & (target_summary.condition == "remesh")), "cell_sd"]
            purpose = "Compare technical run-to-run standard deviation across the four version-condition cells."
            encoding = "Each point is one participant's SD across 40 runs; black bars are participant medians."
            finding = f"v4 remesh SD spans {v4.min():.4f}–{v4.max():.4f} V/m, whereas all other cells are at or below {controls.max():.2e} V/m."
            audit = "The linear axis collapses three near-zero groups onto the baseline, making valid observations look absent."
            replacement = "Use revised_figures/comprehensive/technical_sd_log_scale.png, which has a log axis and explicit floor symbols."
        elif key == ("comprehensive", "paired_version_contrasts"):
            means = []
            for condition in ("remesh", "fixed_mesh"):
                pivot = target_summary.loc[target_summary.condition == condition].pivot(index="subject", columns="version", values="cell_median")
                means.append(float((pivot.simnibs326 - pivot.simnibs4).mean()))
            purpose = "Show the participant-level v3-minus-v4 contrast separately for remesh and fixed mesh."
            encoding = "Two points joined by one line belong to the same participant."
            finding = f"Mean paired differences are {means[0]:.4f} V/m for remesh and {means[1]:.4f} V/m for fixed mesh."
            audit = "The original assigns arbitrary colors to participants without a legend, so color cannot be interpreted."
            replacement = "Use revised_figures/comprehensive/paired_version_contrasts.png with neutral paired lines and condition-coded endpoints."
        elif key == ("comprehensive", "version_condition_interaction"):
            contrasts = {}
            for condition in ("remesh", "fixed_mesh"):
                pivot = target_summary.loc[target_summary.condition == condition].pivot(index="subject", columns="version", values="cell_median")
                contrasts[condition] = pivot.simnibs326 - pivot.simnibs4
            interaction = contrasts["fixed_mesh"] - contrasts["remesh"]
            purpose = "Display each participant's version-by-condition difference-in-differences."
            encoding = "Each bar is (v3−v4)fixed minus (v3−v4)remesh; zero means the version contrast is unchanged by condition."
            finding = f"Effects range from {interaction.min():.2e} to {interaction.max():.2e} V/m; values are small relative to the central version contrast."
            audit = "Rotated identifiers and one relatively large effect compress the remaining bars."
            replacement = "Use revised_figures/comprehensive/version_condition_interaction.png, a sorted horizontal dot plot."
        elif key == ("cross_version_remesh", "v4_distribution_with_v3_overlay"):
            selected = cross.loc[cross.target == record.target]
            purpose = "Locate the 40 v3 remesh results relative to the matched 40-run v4 distribution for every participant."
            encoding = "Blue box plots represent v4; red v3 points and diamonds represent individual v3 runs and their median."
            finding = f"v3 values lie above the matched v4 distribution for all {len(selected)} participants."
            audit = "Valid but visually dense because near-identical v3 points overlap."
            replacement = "Use revised_figures/cross_version/remesh_distribution_overlay.png, which summarizes v3 by range, IQR and median."
        elif key == ("cross_version_remesh", "v3_percentile_location"):
            selected = cross.loc[cross.target == record.target]
            purpose = "Express each v3 run as an empirical percentile of the matched v4 remesh distribution."
            encoding = "Dots are participant medians; thick and thin lines are IQR and range; the dashed line marks the v4 median."
            finding = f"All participant median percentiles are {selected.percentile_median.min():.0f}–{selected.percentile_median.max():.0f}."
            audit = "The x-axis ends at 100, so markers and intervals at the 100th percentile are clipped against the frame."
            replacement = "Use revised_figures/cross_version/v3_percentile_location.png, which extends beyond 100 and annotates the boundary result."
        elif key == ("within_simnibs326", "primary_median_roi_repeat_distributions"):
            selected = frame.loc[(frame.version == "simnibs326") & (frame.target == record.target) & (frame.condition == "remesh")]
            max_range = selected.groupby("subject").roi_median_v_per_m.agg(lambda values: values.max() - values.min()).max()
            purpose = "Show v3 remesh and fixed-mesh run-level optimizer-ROI medians for each participant."
            encoding = "Pale points are runs; open markers and bars summarize mean and SD for each condition."
            finding = f"The largest v3 remesh 40-run range is {max_range:.2e} V/m, so points overlap almost completely."
            audit = "The overlap can look like missing data unless effective invariance is stated explicitly."
            replacement = "Use revised_figures/within_simnibs326/deterministic_repeatability.png for the deterministic result; retain the original as detailed provenance."
        elif key == ("within_simnibs326", "single_repeat_subject_ranking_uncertainty"):
            purpose = "Assess whether selecting one v3 remesh run changes participant ordering."
            encoding = "The heatmap is pairwise reversal probability; the histogram is Kendall agreement across 20,000 one-run selections."
            finding = _rank_context(bundle, "within_simnibs326", record.target)
            audit = "A white heatmap and single bar look like a rendering failure even though they encode a deterministic result."
            replacement = "Use revised_figures/within_simnibs326/deterministic_ranking_summary.png."
        elif key == ("within_simnibs4", "primary_median_roi_repeat_distributions"):
            selected = target_summary.loc[(target_summary.version == "simnibs4")]
            purpose = "Show updated 9+1 v4 run-level remesh and fixed-mesh target-field distributions."
            encoding = "Pale points are individual runs; open markers show condition means and error bars show ±1 SD."
            finding = "Remesh runs show visible dispersion while fixed-mesh runs collapse to near-identical values."
            audit = "Clear and scientifically useful, but supplied as separate target files rather than the paper's combined layout."
            replacement = "Use paper_equivalents/figure_target_field_repeatability.png for the manuscript-equivalent combined panel."
        elif key == ("within_simnibs4", "single_repeat_subject_ranking_uncertainty"):
            purpose = "Quantify participant-ordering uncertainty when one v4 remesh realization is selected."
            encoding = "The heatmap gives exact pairwise reversal risks and the histogram gives whole-cohort Kendall agreement across 20,000 selections."
            finding = _rank_context(bundle, "within_simnibs4", record.target)
            audit = "Clear, but supplied as separate target files rather than the paper's combined layout."
            replacement = "Use paper_equivalents/figure_single_run_ranking_uncertainty.png."
        else:
            raise RuntimeError(f"No guide description for {key}")

        entries.append(
            {
                "title": f"{target_label}: {record.figure_type.replace('_', ' ')}",
                "image_path": str(output_root / "original_figures" / record.relative_path),
                "source_path": record.relative_path,
                "purpose": purpose,
                "encoding": encoding,
                "finding": finding,
                "audit": audit,
                "replacement": replacement,
            }
        )
    return entries


REVISED_DESCRIPTIONS = {
    "revised_figures/comprehensive/central_values.png": "Combined two-target view of participant 40-run cell medians. Black bars are across-participant medians; repeats are not treated as independent participants.",
    "revised_figures/comprehensive/technical_sd_log_scale.png": "Combined two-target technical-SD view on a logarithmic scale. Open downward triangles explicitly mark observations at or below the stated 1e-8 V/m display floor.",
    "revised_figures/comprehensive/paired_version_contrasts.png": "Neutral lines pair the remesh and fixed-mesh v3-minus-v4 contrasts within each participant; endpoint color identifies condition rather than participant.",
    "revised_figures/comprehensive/version_condition_interaction.png": "Sorted horizontal participant effects for the version-by-condition difference-in-differences, with a zero reference line.",
    "revised_figures/cross_version/remesh_distribution_overlay.png": "Matched v4 box plots with v3 range, interquartile range and median; this removes redundant overplotting of effectively coincident v3 runs.",
    "revised_figures/cross_version/v3_percentile_location.png": "Empirical v3 percentiles with the 100th percentile inside the plotting frame and an explicit boundary annotation.",
    "revised_figures/within_simnibs326/deterministic_repeatability.png": "Participant v3 40-run ranges on a log scale, making effective invariance visible without pretending observations are absent.",
    "revised_figures/within_simnibs326/deterministic_ranking_summary.png": "Direct statement of the deterministic v3 ranking outcome: zero pairwise reversals and Kendall tau of 1.00 in every single-run selection.",
    "paper_equivalents/figure_target_field_repeatability.png": "Updated combined 9+1 SimNIBS 4 equivalent of the paper's target-field repeatability figure.",
    "paper_equivalents/figure_single_run_ranking_uncertainty.png": "Updated combined 9+1 SimNIBS 4 equivalent of the paper's ranking-uncertainty figure.",
    "paper_equivalents/figure_bootstrap_precision.png": "Updated 9+1 SimNIBS 4 bootstrap precision figure using 20,000 resamples per participant and run count in the production build.",
    "legacy_paper_figures/figure_mesh_element_count_variability.png": "Pixel-identical legacy paper figure. It is not an updated 9+1 equivalent because current mesh-element metrics are absent from the downloaded package.",
    "legacy_paper_figures/figure_tissue_volume_deviations.png": "Pixel-identical legacy v4 illustration. Its selected participants remain in the 9+1 cohort, but it is not a new cross-version structural analysis.",
}


def _markdown_guide(
    output_root: Path,
    bundle: AuditBundle,
    original_entries: list[dict[str, str]],
    revised_entries: list[dict[str, str]],
) -> str:
    lines = [
        "# Final-132 combined repeatability figure guide",
        "",
        "This companion reviews the 20 PNG figures in the downloaded combined analysis, provides repaired versions where presentation was misleading, and maps all five current manuscript figures to current or legacy evidence. The downloaded analysis and manuscript were read-only.",
        "",
        "The participant is the inferential unit. Forty repeats estimate within-cell technical variability. SimNIBS-version ROI constructions are aligned to the same anatomy and requested physical volume, but their voxel masks are not assumed identical. Fixed mesh is a mechanistic control, not a general modelling recommendation.",
        "",
        "## Current-paper figure equivalence",
        "",
        "| Paper figure | Status | Companion output | Interpretation |",
        "| --- | --- | --- | --- |",
    ]
    for item in bundle.paper_equivalences:
        lines.append(
            f"| `{item.paper_figure}` | `{item.status}` | `{item.companion_path}` | {item.rationale} |"
        )
    lines.extend(["", "## Original downloaded analysis figures", ""])
    for index, entry in enumerate(original_entries, start=1):
        relative_image = Path(entry["image_path"]).relative_to(output_root).as_posix()
        lines.extend(
            [
                f"### {index}. {entry['title']}",
                "",
                f"Source: `{entry['source_path']}`",
                "",
                f"![{entry['title']}]({relative_image})",
                "",
                f"- **Purpose:** {entry['purpose']}",
                f"- **How to read it:** {entry['encoding']}",
                f"- **Result:** {entry['finding']}",
                f"- **Audit finding:** {entry['audit']}",
                f"- **Recommended use:** {entry['replacement']}",
                "",
            ]
        )
    lines.extend(["## Revised and paper-equivalent figures", ""])
    for index, entry in enumerate(revised_entries, start=1):
        lines.extend(
            [
                f"### R{index}. {entry['title']}",
                "",
                f"Path: `{entry['relative_path']}`",
                "",
                f"![{entry['title']}]({entry['relative_path']})",
                "",
                entry["description"],
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- These simulations measure repeatability, not accuracy or biological efficacy.",
            "- The technical repeats are not additional participants.",
            "- A fixed mesh isolates one computational mechanism but is not a universal recommendation.",
            "- The cross-version comparison uses aligned optimizer-matched ROI construction, not numerically identical masks.",
            "- Element-count and tissue-volume metrics are unavailable in the downloaded combined-analysis package and were not inferred.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(output_root: Path) -> None:
    files = [path for path in sorted(output_root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    lines = [f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}" for path in files]
    (output_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(
    source_root: Path,
    manuscript_root: Path,
    output_root: Path,
    *,
    bootstrap_iterations: int = 20_000,
    bootstrap_max_repeats: int = 40,
) -> dict[str, Any]:
    """Build the full companion package without modifying either source tree."""

    source_root = Path(source_root).resolve()
    manuscript_root = Path(manuscript_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    source_before = tree_manifest(source_root)
    manuscript_before = tree_manifest(manuscript_root)
    bundle = audit_sources(source_root, manuscript_root)
    output_root.mkdir(parents=True)

    original_artifacts = _copy_originals(bundle, output_root)
    revised_artifacts = list(
        render_revised_figures(
            bundle,
            output_root,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_max_repeats=bootstrap_max_repeats,
        )
    )
    _write_csv(
        output_root / "figure_inventory.csv",
        [asdict(item) for item in bundle.figures],
        list(asdict(bundle.figures[0]).keys()),
    )
    _write_csv(
        output_root / "paper_equivalence.csv",
        [asdict(item) for item in bundle.paper_equivalences],
        list(asdict(bundle.paper_equivalences[0]).keys()),
    )

    original_entries = _figure_entries(bundle, output_root)
    revised_entries = [
        {
            "title": Path(item.relative_path).stem.replace("_", " ").title(),
            "relative_path": item.relative_path,
            "image_path": str(output_root / item.relative_path),
            "description": REVISED_DESCRIPTIONS[item.relative_path],
        }
        for item in revised_artifacts
        if item.relative_path in REVISED_DESCRIPTIONS
    ]
    guide_markdown = _markdown_guide(output_root, bundle, original_entries, revised_entries)
    (output_root / "FIGURE_GUIDE.md").write_text(guide_markdown, encoding="utf-8")
    guide_spec = {
        "title": "Final-132 combined repeatability figure guide",
        "subtitle": "Audit, repaired figures, and current-paper equivalence",
        "overview": [
            "This source-bound companion reviews all 20 figures in the downloaded 9+1 combined analysis. It preserves the original figures, repairs identified presentation failures, and provides updated paper equivalents where the downloaded data support them.",
            "The participant is the inferential unit; the 40 repeats are technical replicates. Fixed mesh is a mechanistic control. Version-specific optimizer ROIs use aligned construction but are not identical masks.",
        ],
        "paper_equivalence": [asdict(item) for item in bundle.paper_equivalences],
        "original_figures": original_entries,
        "revised_figures": revised_entries,
    }
    provenance = output_root / "provenance"
    provenance.mkdir(exist_ok=True)
    spec_path = provenance / "guide_spec.json"
    spec_path.write_text(json.dumps(guide_spec, indent=2) + "\n", encoding="utf-8")
    render_pdf(spec_path, output_root / "FIGURE_GUIDE.pdf")

    readme = """# Final-132 combined repeatability figure review v1

This is a companion to `final_132_repeatability_combined_analysis_v1`.
It does not replace or modify the downloaded analysis. Start with
`FIGURE_GUIDE.pdf` for the illustrated audit or `FIGURE_GUIDE.md` for the
searchable source. `paper_equivalence.csv` records which current manuscript
figures have updated 9+1 equivalents and which remain legacy evidence.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    if tree_manifest(source_root) != source_before:
        raise RuntimeError("Downloaded analysis changed during companion build")
    if tree_manifest(manuscript_root) != manuscript_before:
        raise RuntimeError("Manuscript changed during companion build")

    all_artifacts = original_artifacts + revised_artifacts
    manifest = {
        "status": "complete",
        "workflow": "repeatability_figure_review",
        "schema_version": 1,
        "source_root": str(source_root),
        "source_tree_file_count": len(source_before),
        "source_tree_manifest_sha256": hashlib.sha256(
            json.dumps(source_before, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "manuscript_root": str(manuscript_root),
        "manuscript_tree_file_count": len(manuscript_before),
        "original_figure_count": len(bundle.figures),
        "paper_figure_count": len(bundle.paper_equivalences),
        "revised_png_count": sum(item.relative_path.endswith(".png") for item in revised_artifacts),
        "source_outputs_modified": False,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_max_repeats": bootstrap_max_repeats,
        "artifacts": [asdict(item) for item in all_artifacts],
        "interpretation_boundaries": [
            "participant is the inferential unit",
            "technical repeats are not independent participants",
            "fixed mesh is a mechanistic control",
            "version-specific ROI masks are aligned but not identical",
            "no therapeutic or biological threshold is inferred",
            "mesh and tissue metrics were not inferred from unavailable data",
        ],
    }
    (output_root / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_checksums(output_root)
    return {
        "status": "complete",
        "output_root": str(output_root),
        "original_figure_count": len(bundle.figures),
        "revised_png_count": manifest["revised_png_count"],
        "source_outputs_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manuscript-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--bootstrap-max-repeats", type=int, default=40)
    args = parser.parse_args()
    result = build_package(
        args.source_root,
        args.manuscript_root,
        args.output_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_max_repeats=args.bootstrap_max_repeats,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
