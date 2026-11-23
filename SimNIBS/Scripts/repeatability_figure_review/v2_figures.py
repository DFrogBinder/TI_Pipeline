"""Versioned figures for native v3.2.6-versus-v4.6 repeatability analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from repeatability_combined_analysis_v2.settings import SUBJECTS, TARGETS
from ti_current_repair.repeatability_v2.precision import write_metric_table


VERSION_COLORS = {"simnibs326": "#0072B2", "simnibs46": "#D55E00"}
CONDITION_MARKERS = {"remesh": "o", "fixed_mesh": "D"}
TARGET_LABELS = {
    "left_hippocampus": "Left hippocampus",
    "right_m1": "Right M1",
}


def _style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)


def _participant_axis(axis: plt.Axes) -> None:
    axis.set_xticks(np.arange(len(SUBJECTS)))
    axis.set_xticklabels(
        [subject.removeprefix("sub-") for subject in SUBJECTS],
        rotation=45,
        ha="right",
        fontsize=7,
    )


def _save_grayscale(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        grayscale = image.convert("L")
        grayscale.save(destination)


def _matched_figure(frame: pd.DataFrame, output: Path) -> None:
    subset = frame.loc[
        (frame["metric"] == "roi_median_v_per_m")
        & frame["outcome_mode"].isin(
            ("ti_target", "tdcs_pair1_matched_target")
        )
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=False)
    for axis, target in zip(axes, TARGETS):
        target_frame = subset.loc[subset["target"] == target]
        for version, color in VERSION_COLORS.items():
            for condition, marker in CONDITION_MARKERS.items():
                for outcome, linestyle, label_suffix in (
                    ("ti_target", "-", "TI target"),
                    ("tdcs_pair1_matched_target", "--", "Pair 1 matched target"),
                ):
                    values = (
                        target_frame.loc[
                            (target_frame["version"] == version)
                            & (target_frame["condition"] == condition)
                            & (target_frame["outcome_mode"] == outcome)
                        ]
                        .set_index("subject")["cell_sd"]
                        .reindex(SUBJECTS)
                    )
                    axis.plot(
                        np.arange(len(SUBJECTS)),
                        values,
                        color=color,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.2,
                        markersize=4.5,
                        label=f"{version} | {condition} | {label_suffix}",
                    )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("Participant (one point per 40-repeat cell summary)")
        axis.set_ylabel("Technical-repeat SD of ROI median (V/m)")
        _participant_axis(axis)
        _style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    figure.suptitle("Matched-location TI and pair-1 repeatability decomposition")
    figure.tight_layout(rect=(0, 0.18, 1, 0.94))
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _under_figure(frame: pd.DataFrame, output: Path) -> None:
    subset = frame.loc[
        (frame["metric"] == "roi_median_v_per_m")
        & (frame["outcome_mode"] == "tdcs_pair1_under_anode")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=False)
    for axis, target in zip(axes, TARGETS):
        target_frame = subset.loc[subset["target"] == target]
        for version, color in VERSION_COLORS.items():
            for condition, marker in CONDITION_MARKERS.items():
                values = (
                    target_frame.loc[
                        (target_frame["version"] == version)
                        & (target_frame["condition"] == condition)
                    ]
                    .set_index("subject")["cell_sd"]
                    .reindex(SUBJECTS)
                )
                axis.plot(
                    np.arange(len(SUBJECTS)),
                    values,
                    color=color,
                    marker=marker,
                    linewidth=1.2,
                    markersize=5,
                    label=f"{version} | {condition}",
                )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("Participant (one point per 40-repeat cell summary)")
        axis.set_ylabel("Under-anode pair-1 technical SD (V/m)")
        _participant_axis(axis)
        _style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle("Separate application-facing tDCS under-anode analysis")
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _pipeline_diagram(output: Path) -> None:
    stages = (
        "Corrected\nsegmentation",
        "Head\nmeshing",
        "Electrode\nconstruction",
        "Component\nsolving",
        "TI\ncombination",
        "Native float64\nextraction",
        "ROI + participant\naggregation",
    )
    figure, axis = plt.subplots(figsize=(13.0, 4.2))
    axis.set_xlim(-0.5, len(stages) - 0.5)
    axis.set_ylim(-0.8, 1.0)
    axis.axis("off")
    for index, label in enumerate(stages):
        axis.text(
            index,
            0.35,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#E8F1F8" if index < 5 else "#FDEBD0",
                "edgecolor": "#333333",
            },
        )
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(index + 0.72, 0.35),
                xytext=(index + 0.28, 0.35),
                arrowprops={"arrowstyle": "->", "color": "#333333"},
            )
    axis.text(
        5,
        -0.35,
        "float32 NIfTI branches here for qualitative QC only",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    axis.annotate(
        "",
        xy=(5, -0.16),
        xytext=(5, 0.12),
        arrowprops={"arrowstyle": "->", "color": "#777777"},
    )
    figure.suptitle("Repeatability v2 provenance and precision path", fontsize=14)
    figure.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def render_v2_figure_package(
    *, cells: pd.DataFrame, output_root: Path
) -> dict[str, object]:
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"Figure output root must be new: {output_root}")
    required = {
        "subject",
        "version",
        "target",
        "condition",
        "outcome_mode",
        "metric",
        "run_count",
        "cell_sd",
    }
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"Participant-cell table is missing columns: {missing}")
    if set(cells["run_count"]) != {40} or set(cells["subject"]) != set(SUBJECTS):
        raise ValueError("Figure input is not the complete 40-repeat canonical cohort")
    output_root.mkdir(parents=True)
    matched = output_root / "matched_target_decomposition.png"
    under = output_root / "under_anode_application.png"
    pipeline = output_root / "pipeline_v2.png"
    _matched_figure(cells, matched)
    _under_figure(cells, under)
    _pipeline_diagram(pipeline)
    matched_gray = output_root / "matched_target_decomposition_grayscale.png"
    under_gray = output_root / "under_anode_application_grayscale.png"
    _save_grayscale(matched, matched_gray)
    _save_grayscale(under, under_gray)
    write_metric_table(
        output_root / "figure_source_participant_cells.csv",
        cells.to_dict(orient="records"),
    )
    notes = output_root / "manuscript_notes.md"
    notes.write_text(
        """# Repeatability v2 manuscript boundaries

- Native processing is described as lowering the analysis precision floor, not removing all variability.
- Exact electrode-augmented-mesh reuse is the solver-only control.
- Matched-target component results support TI decomposition claims without assuming that TI doubles component variability.
- Under-anode results support a separate application-facing tDCS use case and are not a location-matched TI comparison.
- Preliminary SimNIBS 4.0.1 results must be replaced before final submission.
- Points in the figures are participant-level 40-repeat cell summaries; participants are the inferential units.
""",
        encoding="utf-8",
    )
    receipt: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "point_unit": "participant_cell_summary",
        "participant_order": list(SUBJECTS),
        "version_colors": VERSION_COLORS,
        "condition_markers": CONDITION_MARKERS,
        "matched_target_figure": str(matched.resolve()),
        "under_anode_figure": str(under.resolve()),
        "pipeline_diagram": str(pipeline.resolve()),
        "matched_target_grayscale": str(matched_gray.resolve()),
        "under_anode_grayscale": str(under_gray.resolve()),
        "figure_source_table": str(
            (output_root / "figure_source_participant_cells.csv").resolve()
        ),
        "manuscript_notes": str(notes.resolve()),
    }
    (output_root / "figure_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
