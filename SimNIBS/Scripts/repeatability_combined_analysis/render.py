"""Render the condition-inclusive participant-level analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .settings import METRICS, PRIMARY_METRIC, SUBJECTS, TARGETS
from .statistics import build_contrasts, build_support_audit, summarize_cells


TARGET_LABELS = {
    "left_hippocampus": "Left hippocampus",
    "right_m1": "Right M1",
}
CELL_LABELS = (
    ("simnibs326", "remesh", "v3 remesh"),
    ("simnibs326", "fixed_mesh", "v3 fixed"),
    ("simnibs4", "remesh", "v4 remesh"),
    ("simnibs4", "fixed_mesh", "v4 fixed"),
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _cell_figure(
    summary: pd.DataFrame, target: str, outcome: str, title: str, ylabel: str
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    primary = summary.loc[
        (summary["target"] == target) & (summary["metric"] == PRIMARY_METRIC)
    ]
    for position, (version, condition, label) in enumerate(CELL_LABELS, start=1):
        values = primary.loc[
            (primary["version"] == version)
            & (primary["condition"] == condition),
            outcome,
        ].to_numpy(dtype=float)
        axis.scatter(
            np.full(len(values), position),
            values,
            s=28,
            alpha=0.8,
            label=label,
        )
        axis.plot(
            [position - 0.20, position + 0.20],
            [np.median(values), np.median(values)],
            color="black",
            linewidth=2,
        )
    axis.set_xticks(range(1, 5), [item[2] for item in CELL_LABELS])
    axis.set_ylabel(ylabel)
    axis.set_title(f"{TARGET_LABELS[target]}: {title}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _participant_contrast_matrix(table: pd.DataFrame) -> np.ndarray:
    rows = []
    for payload in table["participant_contrasts_json"]:
        values = json.loads(payload)
        rows.append([float(values[subject]) for subject in SUBJECTS])
    return np.asarray(rows, dtype=float)


def _version_contrast_figure(
    table: pd.DataFrame, target: str
) -> plt.Figure:
    subset = table.loc[
        (table["target"] == target) & (table["metric"] == PRIMARY_METRIC)
    ]
    matrix = _participant_contrast_matrix(subset)
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for participant in range(matrix.shape[1]):
        axis.plot([1, 2], matrix[:, participant], marker="o", alpha=0.55)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks([1, 2], subset["condition"].str.replace("_", " "))
    axis.set_ylabel("Participant v3 - v4 contrast (V/m)")
    axis.set_title(
        f"{TARGET_LABELS[target]}: paired version contrasts by condition"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _interaction_figure(table: pd.DataFrame, target: str) -> plt.Figure:
    row = table.loc[
        (table["target"] == target) & (table["metric"] == PRIMARY_METRIC)
    ].iloc[0]
    values = json.loads(row["participant_contrasts_json"])
    ordered = np.asarray([values[subject] for subject in SUBJECTS], dtype=float)
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.bar(np.arange(len(SUBJECTS)), ordered, color="#4472C4")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(
        np.arange(len(SUBJECTS)),
        [subject.removeprefix("sub-") for subject in SUBJECTS],
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Difference-in-differences (V/m)")
    axis.set_title(
        f"{TARGET_LABELS[target]}: participant version-condition interaction"
    )
    figure.tight_layout()
    return figure


def _write_report(path: Path) -> None:
    path.write_text(
        """# Comprehensive 9+1 repeatability analysis

The participant is the inferential unit (n = 10). The 40 technical repeats in
each participant-version-target-condition cell estimate that participant's
cell centre and technical variability; they are not treated as independent
participants.

The primary field outcome is the median optimizer-ROI field
(`roi_median_v_per_m`). ROI mean and p95 are prespecified sensitivity
outcomes. Run-level standard deviation is the primary technical-variability
outcome, while coefficient of variation is descriptive.

Version contrasts are SimNIBS 3.2.6 minus SimNIBS 4. Condition contrasts are
fixed mesh minus remesh. The interaction is `(v3 - v4)_fixed -
(v3 - v4)_remesh`. Paired exact two-sided sign-flip tests use all 1,024 sign
assignments, with Holm adjustment within each named family.

The fixed-mesh arm is a mechanistic control for identifying computational
variation; it is not a general recommendation for applied modelling. Target
ROIs use aligned optimizer-matched, parcel-clipped spherical construction, but
the version-specific voxel masks are not identical. Simulated fields are
continuous quantities, and no therapeutic or biological threshold is inferred.

Finite-support results are descriptive provenance and sensitivity checks. An
association between support loss and field values does not establish causation.
""",
        encoding="utf-8",
    )


def _write_captions(path: Path) -> None:
    path.write_text(
        """# Figure captions

## Central values

Participant cell centres for the primary optimizer-ROI median field across the
four version-condition cells. Each point is one participant after combining 40
technical repeats; black bars show the participant median.

## Technical SD

Within-participant standard deviation across 40 technical repeats. This is the
primary technical-variability summary and does not represent between-person
uncertainty.

## Paired version contrasts

Participant-level SimNIBS 3.2.6 minus SimNIBS 4 contrasts for remesh and
fixed-mesh conditions. The fixed-mesh comparison is a mechanistic control.

## Version-condition interaction

Participant difference-in-differences, `(v3 - v4)_fixed - (v3 - v4)_remesh`.
Targets are aligned optimizer-matched spherical ROIs, not identical voxel
masks. Fields are continuous simulations and are not labelled therapeutic.
Mean and p95 ROI metrics are available as sensitivity outcomes in the tables.
""",
        encoding="utf-8",
    )


def _combine_contrasts(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [tables[name] for name in ("version", "condition", "interaction")],
        ignore_index=True,
    )


def render_comprehensive_outputs(
    frame: pd.DataFrame, output_dir: Path
) -> dict[str, Any]:
    """Write all comprehensive tables, figures, captions, and manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_cells(frame)
    central_tables = build_contrasts(summary, "cell_median")
    variability_tables = build_contrasts(summary, "cell_sd")
    central = _combine_contrasts(central_tables)
    variability = _combine_contrasts(variability_tables)
    support, support_summary = build_support_audit(frame)

    artifacts = [
        "participant_cell_summary.csv",
        "central_value_contrasts.csv",
        "technical_variability_contrasts.csv",
        "finite_support_audit.csv",
        "finite_support_summary.json",
        "analysis_report.md",
        "figure_captions.md",
    ]
    summary.to_csv(output_dir / artifacts[0], index=False)
    central.to_csv(output_dir / artifacts[1], index=False)
    variability.to_csv(output_dir / artifacts[2], index=False)
    support.to_csv(output_dir / artifacts[3], index=False)
    (output_dir / artifacts[4]).write_text(
        json.dumps(support_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / artifacts[5])
    _write_captions(output_dir / artifacts[6])

    figure_paths: list[str] = []
    for target in TARGETS:
        figures = {
            "central_values": _cell_figure(
                summary,
                target,
                "cell_median",
                "participant central field values",
                "Participant cell median (V/m)",
            ),
            "technical_sd": _cell_figure(
                summary,
                target,
                "cell_sd",
                "technical repeat variability",
                "Within-cell run SD (V/m)",
            ),
            "paired_version_contrasts": _version_contrast_figure(
                central_tables["version"], target
            ),
            "version_condition_interaction": _interaction_figure(
                central_tables["interaction"], target
            ),
        }
        for name, figure in figures.items():
            relative = f"figures/{target}/{name}.png"
            _save_figure(figure, output_dir / relative)
            figure_paths.append(relative)
    artifacts.extend(figure_paths)

    manifest = {
        "status": "complete",
        "schema_version": 1,
        "run_level_rows": int(len(frame)),
        "participant_cells": 80,
        "participant_cell_metric_rows": int(len(summary)),
        "participant_is_inferential_unit": True,
        "technical_repeats_per_cell": 40,
        "primary_metric": PRIMARY_METRIC,
        "sensitivity_metrics": [metric for metric in METRICS if metric != PRIMARY_METRIC],
        "central_contrasts": int(len(central)),
        "technical_variability_contrasts": int(len(variability)),
        "artifacts": artifacts,
        "interpretation": {
            "fixed_mesh": "mechanistic control",
            "roi_masks": "aligned but not identical version-specific voxel masks",
            "field_values": "continuous simulated fields without a therapeutic threshold",
        },
    }
    manifest_path = output_dir / "comprehensive_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts.append("comprehensive_manifest.json")
    return {
        "status": "complete",
        "manifest": "comprehensive_manifest.json",
        "figures": figure_paths,
        "artifacts": artifacts,
    }

