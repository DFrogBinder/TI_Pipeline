#!/usr/bin/env python3
"""Build the final MNI-threshold CamCan manuscript figures.

This renderer implements the figure decisions from the supervisor meeting:

* superficial ROIs precede deep ROIs;
* each ROI is evaluated at its SimNIBS 4.0.1 MNI152 mean target field;
* mean target fields, coverage percentages, and coverage ratios are shown as
  absolute values so that their physical and percentage context is retained;
* MNI152 is plotted at its absolute ROI-specific value as the population
  reference;
* one combined personalized figure contains target coverage, off-target
  coverage, and target/off-target coverage ratio for all four ROIs;
* repeat error bars, repeat-distribution plots, standalone MNI field plots,
  percentile-context plots, effectiveness/spread scatter plots, and separate
  minimum/maximum relationships are not generated;
* all axes are linear;
* target/off-target ratios with a zero off-target denominator are drawn as
  censored infinity observations at the finite plotting boundary.

Coverage at an ROI-specific threshold cannot be reconstructed from the older
0.20/0.18/0.15 V/m aggregate tables. The loader therefore requires exact
columns produced by a fresh image-level metric extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


FIGURE_SCHEMA_VERSION = 9
ROI_ORDER = [
    "Left_M1",
    "Right_DLPC",
    "Left_Hippocampus",
    "Right_Thalamus",
]
ROI_LABELS = {
    "Left_M1": "Left M1",
    "Right_DLPC": "Right DLPFC",
    "Left_Hippocampus": "Left hippocampus",
    "Right_Thalamus": "Right thalamus",
}
ROI_SLUGS = {
    "Left_M1": "left_m1",
    "Right_DLPC": "right_dlpc",
    "Left_Hippocampus": "left_hippocampus",
    "Right_Thalamus": "right_thalamus",
}
CONDITIONS = ("generic", "personalized")

BLUE = "#2F6B9A"
ORANGE = "#D97706"
GREEN = "#00876C"
PURPLE = "#8E5EA2"
GRAY = "#5F6873"
LIGHT_GRAY = "#CBD1D8"
GRID = "#E4E8ED"
BLACK = "#222222"
EXPECTED_FIGURE_STEMS = [
    "figure_personalization_subject_changes_all_rois_at_mni_roi_threshold",
    "figure_population_mean_field_and_target_offtarget_ratio_absolute",
    "figure_population_mean_field_offtarget_relationship_at_mni_roi_threshold",
    "figure_population_target_offtarget_relationship_at_mni_roi_threshold",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_slug(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def short_subject(subject: str) -> str:
    return str(subject).replace("sub-", "")


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.8,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figures_dir / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(figure)


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
    *,
    finite: bool = True,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{label} is missing exact required columns: {missing}. "
            "Run the MNI-threshold image-level metric extraction; do not "
            "interpolate or relabel the older 0.20/0.18/0.15 V/m metrics."
        )
    if finite:
        numeric_columns = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        values = frame[numeric_columns].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            bad = [
                column
                for column in numeric_columns
                if not np.isfinite(frame[column].to_numpy(dtype=float)).all()
            ]
            raise RuntimeError(f"{label} contains non-finite data in {bad}")


def load_thresholds(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "roi",
        "roi_group",
        "roi_order",
        "threshold_v_per_m",
        "mni_min_v_per_m",
        "mni_mean_v_per_m",
        "mni_max_p99_9_v_per_m",
        "simnibs_version",
        "source_table_sha256",
    ]
    _require_columns(frame, required, "MNI threshold table", finite=False)
    if (
        len(frame) != 4
        or frame["roi"].tolist() != ROI_ORDER
        or frame["roi_order"].tolist() != [1, 2, 3, 4]
        or frame["roi_group"].tolist()
        != ["superficial", "superficial", "deep", "deep"]
        or set(frame["simnibs_version"].astype(str)) != {"4.0.1"}
    ):
        raise RuntimeError(
            "MNI threshold table must contain the four ROIs in the standard "
            "superficial-then-deep order and identify SimNIBS 4.0.1."
        )
    numeric = [
        "threshold_v_per_m",
        "mni_min_v_per_m",
        "mni_mean_v_per_m",
        "mni_max_p99_9_v_per_m",
    ]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("MNI threshold table contains non-finite values")
    if not np.allclose(
        frame["threshold_v_per_m"],
        frame["mni_mean_v_per_m"],
        rtol=0.0,
        atol=1e-14,
    ):
        raise RuntimeError("Every evaluation threshold must equal the MNI mean")
    return frame.set_index("roi", drop=False)


def _validate_manifest(
    path: Path,
    expected: dict[str, object],
    thresholds: pd.DataFrame,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"{path} has {key}={payload.get(key)!r}; expected {value!r}"
            )
    available = np.asarray(payload.get("thresholds_v_per_m", []), dtype=float)
    if available.size == 0:
        raise RuntimeError(f"{path} does not declare analysis thresholds")
    for threshold in thresholds["threshold_v_per_m"].to_numpy(dtype=float):
        if not np.isclose(available, threshold, rtol=0.0, atol=1e-12).any():
            raise RuntimeError(
                f"{path} does not contain required threshold "
                f"{threshold:.17g} V/m"
            )
    return payload


def coverage_columns(roi: str, thresholds: pd.DataFrame) -> tuple[str, str]:
    slug = threshold_slug(float(thresholds.loc[roi, "threshold_v_per_m"]))
    return (
        f"target_coverage_percent_ge_{slug}",
        f"off_target_coverage_percent_ge_{slug}",
    )


def load_inputs(
    cohort_dir: Path,
    personalized_dir: Path,
    thresholds: pd.DataFrame,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_manifest = _validate_manifest(
        cohort_dir / "analysis_manifest.json",
        {
            "analysis_schema_version": 4,
            "status": "complete",
            "subjects": 132,
            "repeat_level_records": 5280,
            "subject_level_records": 528,
            "mni_baselines": 4,
        },
        thresholds,
    )
    personalized_manifest = _validate_manifest(
        personalized_dir / "analysis_manifest.json",
        {
            "comparison_schema_version": 3,
            "manuscript_analysis_schema_version": 4,
            "status": "complete",
            "subject_roi_configurations": 28,
            "repeat_level_records": 560,
            "condition_repeat_mean_records": 56,
        },
        thresholds,
    )
    subjects = pd.read_csv(
        cohort_dir / "subject_level_repeat_mean_metrics.csv"
    )
    mni = pd.read_csv(cohort_dir / "mni152_baseline_metrics.csv")
    paired = pd.read_csv(
        personalized_dir / "paired_personalized_vs_generic.csv"
    )
    field_columns = [
        "roi_min_v_per_m",
        "roi_mean_v_per_m",
        "roi_robust_max_p99_9_v_per_m",
    ]
    _require_columns(subjects, ["subject", "roi", *field_columns], "cohort")
    _require_columns(mni, ["subject", "roi", *field_columns], "MNI152")
    paired_required = ["subject", "roi"]
    for roi in ROI_ORDER:
        target_column, off_target_column = coverage_columns(roi, thresholds)
        _require_columns(
            subjects.loc[subjects["roi"] == roi],
            [target_column, off_target_column],
            f"cohort {roi}",
        )
        _require_columns(
            mni.loc[mni["roi"] == roi],
            [target_column, off_target_column],
            f"MNI152 {roi}",
        )
        for metric in (target_column, off_target_column):
            for condition in CONDITIONS:
                paired_required.append(
                    f"{metric}__{condition}_repeat_mean"
                )
    _require_columns(paired, paired_required, "personalized paired metrics")
    if (
        len(subjects) != 528
        or subjects.groupby("roi").size().to_dict()
        != {roi: 132 for roi in ROI_ORDER}
        or len(mni) != 4
        or set(mni["roi"]) != set(ROI_ORDER)
        or len(paired) != 28
        or paired["subject"].nunique() != 7
        or paired.groupby("roi").size().to_dict()
        != {roi: 7 for roi in ROI_ORDER}
    ):
        raise RuntimeError("Input tables do not match the required study scope")
    mni = mni.set_index("roi", drop=False).loc[ROI_ORDER]
    for roi in ROI_ORDER:
        source = thresholds.loc[roi]
        observed = mni.loc[roi]
        checks = (
            ("roi_min_v_per_m", "mni_min_v_per_m"),
            ("roi_mean_v_per_m", "mni_mean_v_per_m"),
            (
                "roi_robust_max_p99_9_v_per_m",
                "mni_max_p99_9_v_per_m",
            ),
        )
        for metric, reference in checks:
            if not math.isclose(
                float(observed[metric]),
                float(source[reference]),
                rel_tol=0.0,
                abs_tol=2e-12,
            ):
                raise RuntimeError(
                    f"MNI152 {roi} {metric} does not match the validated "
                    "SimNIBS 4.0.1 threshold source table"
                )
    roi_rank = {roi: index for index, roi in enumerate(ROI_ORDER)}
    subjects = subjects.assign(_roi_rank=subjects["roi"].map(roi_rank))
    subjects = subjects.sort_values(["_roi_rank", "subject"])
    paired = paired.assign(_roi_rank=paired["roi"].map(roi_rank))
    paired = paired.sort_values(["_roi_rank", "subject"])
    return cohort_manifest, personalized_manifest, subjects, mni, paired


def ratio_values(
    target: np.ndarray,
    off_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target/off-target ratio and status labels.

    `infinite` means target > 0 and off-target == 0. `undefined` means 0/0.
    These cases must not be silently replaced with an arbitrary finite value.
    """

    target = np.asarray(target, dtype=float)
    off_target = np.asarray(off_target, dtype=float)
    if np.any(target < 0) or np.any(off_target < 0):
        raise RuntimeError("Coverage percentages cannot be negative")
    ratio = np.full(target.shape, np.nan, dtype=float)
    status = np.full(target.shape, "undefined", dtype=object)
    finite = off_target > 0
    ratio[finite] = target[finite] / off_target[finite]
    status[finite] = "finite"
    infinite = (off_target == 0) & (target > 0)
    ratio[infinite] = np.inf
    status[infinite] = "infinite"
    return ratio, status


def centered_ratio(
    target: np.ndarray,
    off_target: np.ndarray,
    *,
    mni_target: float,
    mni_off_target: float,
    roi: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    mni_ratio, mni_status = ratio_values(
        np.asarray([mni_target]), np.asarray([mni_off_target])
    )
    if mni_status[0] != "finite":
        raise RuntimeError(
            f"Cannot MNI-centre the target/off-target ratio for {roi}: "
            f"the MNI denominator is zero ({mni_status[0]} ratio). This "
            "requires an explicit scientific decision, not an epsilon."
        )
    values, status = ratio_values(target, off_target)
    finite = status == "finite"
    values[finite] -= float(mni_ratio[0])
    return values, status, float(mni_ratio[0])


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.ptp(x) == 0:
        raise RuntimeError("Insufficient variation for a descriptive fit")
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(float(x.min()), float(x.max()), 240)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum((y - fitted) ** 2)) / denominator
        if denominator > 0
        else math.nan
    )
    rho = float(spearmanr(x, y).statistic)
    return grid, slope * grid + intercept, {
        "model": "linear",
        "n": int(len(x)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "spearman_rho": rho,
    }


def plot_mni_absolute(
    mni: pd.DataFrame,
    thresholds: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    figure, axis = plt.subplots(figsize=(7.3, 3.9))
    x = np.arange(len(ROI_ORDER), dtype=float)
    for index, roi in enumerate(ROI_ORDER):
        value = float(mni.loc[roi, "roi_mean_v_per_m"])
        axis.scatter(
            index,
            value,
            marker="D",
            s=52,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        rows.append(
            {
                "roi": roi,
                "summary": "Mean",
                "value_v_per_m": value,
                "evaluation_threshold": True,
                "simnibs_version": thresholds.loc[roi, "simnibs_version"],
            }
        )
    axis.axvline(1.5, color=LIGHT_GRAY, lw=0.8)
    axis.text(
        0.5,
        1.04,
        "Superficial targets",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=GRAY,
        fontsize=8,
    )
    axis.text(
        2.5,
        1.04,
        "Deep targets",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=GRAY,
        fontsize=8,
    )
    axis.set_xticks(x, [ROI_LABELS[roi] for roi in ROI_ORDER])
    axis.set_ylabel("E-field inside MNI152 target ROI (V/m)")
    axis.set_title(
        "MNI152 mean target field used for ROI-specific evaluation",
        weight="bold",
        pad=24,
    )
    axis.grid(axis="y", color=GRID, lw=0.55)
    axis.set_axisbelow(True)
    axis.axhline(
        0.2,
        color=GRAY,
        lw=0.9,
        ls=(0, (4, 3)),
        zorder=1,
        label="0.2 V/m optimization target",
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor=ORANGE,
                markeredgecolor="white",
                label="MNI152 mean",
                markersize=6.5,
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                lw=0.9,
                ls=(0, (4, 3)),
                label="0.2 V/m optimization target",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.25)
    save_figure(
        figure,
        figures_dir,
        "figure_mni152_absolute_mean_target_field",
    )
    return pd.DataFrame(rows)


def mni_absolute_mean_table(
    mni: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "roi": roi,
                "summary": "Mean",
                "value_v_per_m": float(mni.loc[roi, "roi_mean_v_per_m"]),
                "evaluation_threshold": True,
                "simnibs_version": thresholds.loc[roi, "simnibs_version"],
            }
            for roi in ROI_ORDER
        ]
    )


def _arrow(
    axis: plt.Axes,
    start: float,
    end: float,
    y: float,
    *,
    color: str = GRAY,
) -> None:
    axis.annotate(
        "",
        xy=(end, y),
        xytext=(start, y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 0.9,
            "shrinkA": 4.5,
            "shrinkB": 4.5,
            "mutation_scale": 8,
            "alpha": 0.72,
        },
        zorder=1,
    )


def plot_personalized_roi(
    paired: pd.DataFrame,
    mni: pd.DataFrame,
    thresholds: pd.DataFrame,
    roi: str,
    figures_dir: Path,
) -> pd.DataFrame:
    rows = paired.loc[paired["roi"] == roi].sort_values("subject").copy()
    subjects = rows["subject"].tolist()
    y = np.arange(len(subjects), dtype=float)
    target_metric, off_metric = coverage_columns(roi, thresholds)
    mni_target = float(mni.loc[roi, target_metric])
    mni_off = float(mni.loc[roi, off_metric])
    source: list[dict] = []
    figure, axes = plt.subplots(1, 3, figsize=(10.9, 4.15), sharey=True)
    figure.subplots_adjust(
        left=0.115,
        right=0.985,
        top=0.87,
        bottom=0.24,
        wspace=0.28,
    )

    coverage_specs = [
        (target_metric, mni_target, "Target coverage (%)"),
        (off_metric, mni_off, "Off-target coverage (%)"),
    ]
    for panel_index, (metric, baseline, label) in enumerate(coverage_specs):
        axis = axes[panel_index]
        generic = rows[
            f"{metric}__generic_repeat_mean"
        ].to_numpy(dtype=float)
        personalized = rows[
            f"{metric}__personalized_repeat_mean"
        ].to_numpy(dtype=float)
        for row_index, subject in enumerate(subjects):
            _arrow(axis, generic[row_index], personalized[row_index], y[row_index])
            for condition, value in (
                ("generic", generic[row_index]),
                ("personalized", personalized[row_index]),
            ):
                source.append(
                    {
                        "subject": subject,
                        "roi": roi,
                        "panel": "target_coverage"
                        if panel_index == 0
                        else "off_target_coverage",
                        "condition": condition,
                        "absolute_value": value,
                        "ratio_status": "",
                        "mni_reference": baseline,
                        "threshold_v_per_m": thresholds.loc[
                            roi, "threshold_v_per_m"
                        ],
                    }
                )
        axis.scatter(
            generic,
            y,
            s=35,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=1.2,
            zorder=3,
        )
        axis.scatter(
            personalized,
            y,
            s=38,
            facecolor=ORANGE,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        if panel_index == 0:
            axis.set_xlim(-4.0, 104.0)
        else:
            axis.set_xlim(
                min(-0.05, float(min(generic.min(), personalized.min()))),
                max(
                    float(max(generic.max(), personalized.max(), baseline))
                    * 1.08,
                    0.5,
                ),
            )
        axis.set_xlabel(label)
        axis.grid(axis="x", color=GRID, lw=0.55)
        axis.set_axisbelow(True)

    ratio_axis = axes[2]
    ratio_by_condition: dict[str, np.ndarray] = {}
    status_by_condition: dict[str, np.ndarray] = {}
    mni_ratio_values, mni_ratio_status = ratio_values(
        np.asarray([mni_target]), np.asarray([mni_off])
    )
    if mni_ratio_status[0] != "finite":
        raise RuntimeError(
            f"Cannot plot the absolute MNI152 target/off-target ratio for "
            f"{roi}: {mni_ratio_status[0]}"
        )
    mni_ratio = float(mni_ratio_values[0])
    for condition in CONDITIONS:
        target = rows[
            f"{target_metric}__{condition}_repeat_mean"
        ].to_numpy(dtype=float)
        off_target = rows[
            f"{off_metric}__{condition}_repeat_mean"
        ].to_numpy(dtype=float)
        values, statuses = ratio_values(target, off_target)
        ratio_by_condition[condition] = values
        status_by_condition[condition] = statuses
    finite_values = np.concatenate(
        [
            values[status_by_condition[condition] == "finite"]
            for condition, values in ratio_by_condition.items()
        ]
    )
    if finite_values.size == 0:
        raise RuntimeError(f"No finite personalized ratios for {roi}")
    span = max(float(np.ptp(finite_values)), 1.0)
    right_edge = float(np.max(finite_values)) + 0.12 * span
    left_edge = float(np.min(finite_values)) - 0.08 * span
    for row_index, subject in enumerate(subjects):
        plotted: dict[str, float | None] = {}
        for condition in CONDITIONS:
            status = str(status_by_condition[condition][row_index])
            value = float(ratio_by_condition[condition][row_index])
            if status == "finite":
                plotted[condition] = value
            elif status == "infinite":
                plotted[condition] = right_edge
            else:
                plotted[condition] = None
            source.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "panel": "target_to_off_target_ratio",
                    "condition": condition,
                    "absolute_value": value,
                    "ratio_status": status,
                    "mni_reference": mni_ratio,
                    "threshold_v_per_m": thresholds.loc[
                        roi, "threshold_v_per_m"
                    ],
                }
            )
        if plotted["generic"] is not None and plotted["personalized"] is not None:
            _arrow(
                ratio_axis,
                float(plotted["generic"]),
                float(plotted["personalized"]),
                y[row_index],
            )
    for condition, color, filled in (
        ("generic", BLUE, False),
        ("personalized", ORANGE, True),
    ):
        values = ratio_by_condition[condition]
        statuses = status_by_condition[condition]
        finite = statuses == "finite"
        ratio_axis.scatter(
            values[finite],
            y[finite],
            s=35 if condition == "generic" else 38,
            facecolor=color if filled else "white",
            edgecolor="white" if filled else color,
            linewidth=0.7 if filled else 1.2,
            zorder=4,
        )
        infinite = statuses == "infinite"
        if infinite.any():
            ratio_axis.scatter(
                np.full(int(infinite.sum()), right_edge),
                y[infinite],
                marker=">",
                s=48,
                facecolor=color if filled else "white",
                edgecolor=color,
                linewidth=1.1,
                zorder=5,
            )
        undefined = statuses == "undefined"
        for y_value in y[undefined]:
            ratio_axis.text(
                left_edge,
                y_value,
                "0/0",
                color=color,
                fontsize=6.5,
                ha="left",
                va="center",
            )
    ratio_axis.set_xlim(left_edge, right_edge + 0.02 * span)
    ratio_axis.set_xlabel("Target/off-target coverage ratio")
    ratio_axis.grid(axis="x", color=GRID, lw=0.55)
    ratio_axis.set_axisbelow(True)
    if any(
        (status_by_condition[condition] == "infinite").any()
        for condition in CONDITIONS
    ):
        ratio_axis.text(
            right_edge,
            1.025,
            "∞ (censored)",
            transform=ratio_axis.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color=GRAY,
        )

    axes[0].set_yticks(y, [short_subject(subject) for subject in subjects])
    axes[0].set_ylabel("Subject")
    axes[0].invert_yaxis()
    for index, axis in enumerate(axes):
        axis.text(
            0.0,
            1.055,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    figure.suptitle(
        f"{ROI_LABELS[roi]}: generic-to-personalized change",
        weight="bold",
        y=0.965,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=BLUE,
                markeredgewidth=1.2,
                label="Generic",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=ORANGE,
                markeredgecolor="white",
                label="Personalized",
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                marker=">",
                label="Generic → personalized",
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                label=(
                    "Coverage threshold: MNI152 mean = "
                    f"{float(thresholds.loc[roi, 'threshold_v_per_m']):.3f} V/m"
                ),
            ),
        ],
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        handlelength=2.0,
        columnspacing=1.25,
    )
    stem = (
        "figure_personalization_subject_changes_"
        f"{ROI_SLUGS[roi]}_at_mni_roi_threshold"
    )
    save_figure(figure, figures_dir, stem)
    return pd.DataFrame(source)


def plot_personalized_combined(
    paired: pd.DataFrame,
    mni: pd.DataFrame,
    thresholds: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    """Combine the four three-panel personalization plots in one figure."""
    figure, axes = plt.subplots(
        len(ROI_ORDER),
        3,
        figsize=(10.9, 12.3),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.895,
        bottom=0.075,
        hspace=0.54,
        wspace=0.28,
    )
    source: list[dict] = []
    panel_index = 0

    for roi_index, roi in enumerate(ROI_ORDER):
        rows = paired.loc[paired["roi"] == roi].sort_values("subject").copy()
        subjects = rows["subject"].tolist()
        y = np.arange(len(subjects), dtype=float)
        target_metric, off_metric = coverage_columns(roi, thresholds)
        mni_target = float(mni.loc[roi, target_metric])
        mni_off = float(mni.loc[roi, off_metric])
        row_axes = axes[roi_index]

        coverage_specs = [
            (target_metric, mni_target, "target_coverage"),
            (off_metric, mni_off, "off_target_coverage"),
        ]
        for column_index, (metric, baseline, panel_name) in enumerate(
            coverage_specs
        ):
            axis = row_axes[column_index]
            generic = rows[
                f"{metric}__generic_repeat_mean"
            ].to_numpy(dtype=float)
            personalized = rows[
                f"{metric}__personalized_repeat_mean"
            ].to_numpy(dtype=float)
            for subject_index, subject in enumerate(subjects):
                _arrow(
                    axis,
                    generic[subject_index],
                    personalized[subject_index],
                    y[subject_index],
                )
                for condition, value in (
                    ("generic", generic[subject_index]),
                    ("personalized", personalized[subject_index]),
                ):
                    source.append(
                        {
                            "subject": subject,
                            "roi": roi,
                            "panel": panel_name,
                            "condition": condition,
                            "absolute_value": value,
                            "ratio_status": "",
                            "mni_reference": baseline,
                            "threshold_v_per_m": thresholds.loc[
                                roi, "threshold_v_per_m"
                            ],
                        }
                    )
            axis.scatter(
                generic,
                y,
                s=35,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=1.2,
                zorder=3,
            )
            axis.scatter(
                personalized,
                y,
                s=38,
                facecolor=ORANGE,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            if column_index == 0:
                axis.set_xlim(-4.0, 104.0)
            else:
                axis.set_xlim(
                    min(
                        -0.05,
                        float(min(generic.min(), personalized.min())),
                    ),
                    max(
                        float(
                            max(
                                generic.max(),
                                personalized.max(),
                                baseline,
                            )
                        )
                        * 1.08,
                        0.5,
                    ),
                )
            axis.grid(axis="x", color=GRID, lw=0.55)
            axis.set_axisbelow(True)

        ratio_axis = row_axes[2]
        ratio_by_condition: dict[str, np.ndarray] = {}
        status_by_condition: dict[str, np.ndarray] = {}
        mni_ratio_values, mni_ratio_status = ratio_values(
            np.asarray([mni_target]),
            np.asarray([mni_off]),
        )
        if mni_ratio_status[0] != "finite":
            raise RuntimeError(
                "Cannot plot the absolute MNI152 target/off-target ratio "
                f"for {roi}: {mni_ratio_status[0]}"
            )
        mni_ratio = float(mni_ratio_values[0])
        for condition in CONDITIONS:
            target = rows[
                f"{target_metric}__{condition}_repeat_mean"
            ].to_numpy(dtype=float)
            off_target = rows[
                f"{off_metric}__{condition}_repeat_mean"
            ].to_numpy(dtype=float)
            values, statuses = ratio_values(target, off_target)
            ratio_by_condition[condition] = values
            status_by_condition[condition] = statuses
        finite_values = np.concatenate(
            [
                values[status_by_condition[condition] == "finite"]
                for condition, values in ratio_by_condition.items()
            ]
        )
        if finite_values.size == 0:
            raise RuntimeError(f"No finite personalized ratios for {roi}")
        span = max(float(np.ptp(finite_values)), 1.0)
        right_edge = float(np.max(finite_values)) + 0.12 * span
        left_edge = float(np.min(finite_values)) - 0.08 * span
        for subject_index, subject in enumerate(subjects):
            plotted: dict[str, float | None] = {}
            for condition in CONDITIONS:
                status = str(status_by_condition[condition][subject_index])
                value = float(ratio_by_condition[condition][subject_index])
                if status == "finite":
                    plotted[condition] = value
                elif status == "infinite":
                    plotted[condition] = right_edge
                else:
                    plotted[condition] = None
                source.append(
                    {
                        "subject": subject,
                        "roi": roi,
                        "panel": "target_to_off_target_ratio",
                        "condition": condition,
                        "absolute_value": value,
                        "ratio_status": status,
                        "mni_reference": mni_ratio,
                        "threshold_v_per_m": thresholds.loc[
                            roi, "threshold_v_per_m"
                        ],
                    }
                )
            if (
                plotted["generic"] is not None
                and plotted["personalized"] is not None
            ):
                _arrow(
                    ratio_axis,
                    float(plotted["generic"]),
                    float(plotted["personalized"]),
                    y[subject_index],
                )
        for condition, color, filled in (
            ("generic", BLUE, False),
            ("personalized", ORANGE, True),
        ):
            values = ratio_by_condition[condition]
            statuses = status_by_condition[condition]
            finite = statuses == "finite"
            ratio_axis.scatter(
                values[finite],
                y[finite],
                s=35 if condition == "generic" else 38,
                facecolor=color if filled else "white",
                edgecolor="white" if filled else color,
                linewidth=0.7 if filled else 1.2,
                zorder=4,
            )
            infinite = statuses == "infinite"
            if infinite.any():
                ratio_axis.scatter(
                    np.full(int(infinite.sum()), right_edge),
                    y[infinite],
                    marker=">",
                    s=48,
                    facecolor=color if filled else "white",
                    edgecolor=color,
                    linewidth=1.1,
                    zorder=5,
                )
            undefined = statuses == "undefined"
            for y_value in y[undefined]:
                ratio_axis.text(
                    left_edge,
                    y_value,
                    "0/0",
                    color=color,
                    fontsize=6.5,
                    ha="left",
                    va="center",
                )
        ratio_axis.set_xlim(left_edge, right_edge + 0.02 * span)
        ratio_axis.grid(axis="x", color=GRID, lw=0.55)
        ratio_axis.set_axisbelow(True)
        if any(
            (status_by_condition[condition] == "infinite").any()
            for condition in CONDITIONS
        ):
            ratio_axis.text(
                right_edge,
                1.025,
                "∞ (censored)",
                transform=ratio_axis.get_xaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7,
                color=GRAY,
            )

        row_axes[0].set_yticks(
            y,
            [short_subject(subject) for subject in subjects],
        )
        row_axes[0].set_ylabel(f"{ROI_LABELS[roi]}\nSubject")
        row_axes[0].invert_yaxis()
        for axis in row_axes[1:]:
            axis.set_yticks(y)
            axis.tick_params(axis="y", labelleft=False)
            axis.set_ylim(row_axes[0].get_ylim())
        for column_index, axis in enumerate(row_axes):
            axis.text(
                0.0,
                1.035,
                chr(ord("A") + panel_index),
                transform=axis.transAxes,
                weight="bold",
                fontsize=10,
            )
            panel_index += 1
        if roi_index == 0:
            for axis, title in zip(
                row_axes,
                (
                    "Target coverage (%)",
                    "Off-target coverage (%)",
                    "Target/off-target coverage ratio",
                ),
            ):
                axis.set_title(title, weight="bold", pad=14)
        else:
            row_axes[0].set_xlabel("Target coverage (%)")
            row_axes[1].set_xlabel("Off-target coverage (%)")
            row_axes[2].set_xlabel("Target/off-target coverage ratio")

    figure.suptitle(
        "Generic-to-personalized changes across target regions",
        weight="bold",
        y=0.985,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=BLUE,
                markeredgewidth=1.2,
                label="Generic",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=ORANGE,
                markeredgecolor="white",
                label="Personalized",
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                marker=">",
                label="Generic → personalized",
            ),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        handlelength=2.0,
        columnspacing=1.4,
    )
    threshold_text = "; ".join(
        f"{ROI_LABELS[roi]} {float(thresholds.loc[roi, 'threshold_v_per_m']):.3f}"
        for roi in ROI_ORDER
    )
    figure.text(
        0.5,
        0.928,
        f"Coverage thresholds (V/m): {threshold_text}",
        ha="center",
        va="center",
        fontsize=7.5,
        color=GRAY,
    )
    stem = (
        "figure_personalization_subject_changes_all_rois_"
        "at_mni_roi_threshold"
    )
    save_figure(figure, figures_dir, stem)
    return pd.DataFrame(source)


def _violin(
    axis: plt.Axes,
    values_by_roi: list[np.ndarray],
    *,
    color: str,
    rng: np.random.Generator,
    mni_values: np.ndarray,
) -> dict[str, object]:
    """Draw the established CamCAN violin style and absolute MNI references."""
    positions = np.arange(1, len(ROI_ORDER) + 1)
    parts = axis.violinplot(
        values_by_roi,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(0.8)
    iqr_artists = []
    median_artists = []
    sample_artists = []
    for position, values in zip(positions, values_by_roi):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr_artists.append(
            axis.vlines(position, q1, q3, color=color, lw=5.0, zorder=3)
        )
        median_artists.append(axis.scatter(
            position,
            median,
            s=18,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=4,
        ))
        sample = rng.choice(
            values, size=min(44, len(values)), replace=False
        )
        sample_artists.append(axis.scatter(
            position + rng.uniform(-0.16, 0.16, len(sample)),
            sample,
            s=5,
            color=color,
            alpha=0.22,
            edgecolors="none",
            zorder=2,
        ))
    mni_artist = axis.scatter(
        positions,
        np.asarray(mni_values, dtype=float),
        marker="D",
        s=39,
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.65,
        zorder=5,
    )
    return {
        "bodies": parts["bodies"],
        "iqr": iqr_artists,
        "medians": median_artists,
        "samples": sample_artists,
        "mni": mni_artist,
    }


def plot_population_summary(
    subjects: pd.DataFrame,
    mni: pd.DataFrame,
    thresholds: pd.DataFrame,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    figure, axes = plt.subplots(1, 2, figsize=(8.45, 4.35))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.84,
        bottom=0.25,
        wspace=0.24,
    )
    field_values: list[np.ndarray] = []
    ratio_values_by_roi: list[np.ndarray] = []
    population_rows: list[dict] = []
    ratio_audit: list[dict] = []
    ratio_statuses: dict[str, np.ndarray] = {}
    mni_ratios: dict[str, float] = {}
    mni_fields: list[float] = []
    for roi in ROI_ORDER:
        roi_rows = subjects.loc[subjects["roi"] == roi]
        field = roi_rows["roi_mean_v_per_m"].to_numpy(dtype=float)
        field_values.append(field)
        mni_fields.append(float(mni.loc[roi, "roi_mean_v_per_m"]))
        target_metric, off_metric = coverage_columns(roi, thresholds)
        target = roi_rows[target_metric].to_numpy(dtype=float)
        off_target = roi_rows[off_metric].to_numpy(dtype=float)
        ratios, statuses = ratio_values(target, off_target)
        mni_ratio_values, mni_ratio_status = ratio_values(
            np.asarray([float(mni.loc[roi, target_metric])]),
            np.asarray([float(mni.loc[roi, off_metric])]),
        )
        if mni_ratio_status[0] != "finite":
            raise RuntimeError(
                f"Cannot plot the absolute MNI152 target/off-target ratio "
                f"for {roi}: {mni_ratio_status[0]}"
            )
        mni_ratio = float(mni_ratio_values[0])
        finite = statuses == "finite"
        if not finite.any():
            raise RuntimeError(f"No finite cohort ratios for {roi}")
        ratio_values_by_roi.append(ratios[finite])
        ratio_statuses[roi] = statuses
        mni_ratios[roi] = mni_ratio
        for index, subject in enumerate(roi_rows["subject"]):
            population_rows.extend(
                [
                    {
                        "subject": subject,
                        "roi": roi,
                        "outcome": "mean_target_field",
                        "absolute_value": field[index],
                        "status": "finite",
                        "mni_reference": mni_fields[-1],
                    },
                    {
                        "subject": subject,
                        "roi": roi,
                        "outcome": "target_to_off_target_coverage_ratio",
                        "absolute_value": ratios[index],
                        "status": statuses[index],
                        "mni_reference": mni_ratio,
                    },
                ]
            )
        ratio_audit.append(
            {
                "roi": roi,
                "threshold_v_per_m": thresholds.loc[
                    roi, "threshold_v_per_m"
                ],
                "mni_target_coverage_percent": mni.loc[roi, target_metric],
                "mni_off_target_coverage_percent": mni.loc[roi, off_metric],
                "mni_target_to_off_target_ratio": mni_ratio,
                "finite_subjects": int(finite.sum()),
                "infinite_censored_subjects": int(
                    (statuses == "infinite").sum()
                ),
                "undefined_zero_over_zero_subjects": int(
                    (statuses == "undefined").sum()
                ),
            }
        )

    rng = np.random.default_rng(20260729)
    _violin(
        axes[0],
        field_values,
        color=BLUE,
        rng=rng,
        mni_values=np.asarray(mni_fields),
    )
    axes[0].axhline(
        0.2,
        color=GRAY,
        lw=0.9,
        ls=(0, (4, 3)),
        zorder=1,
    )
    axes[0].set_ylabel("Mean target E-field (V/m)")
    axes[0].set_title("Mean target field", weight="bold")

    _violin(
        axes[1],
        ratio_values_by_roi,
        color=GREEN,
        rng=rng,
        mni_values=np.asarray([mni_ratios[roi] for roi in ROI_ORDER]),
    )
    all_finite_ratios = np.concatenate(ratio_values_by_roi)
    ratio_span = max(float(np.ptp(all_finite_ratios)), 1.0)
    ratio_edge = float(np.max(all_finite_ratios)) + 0.12 * ratio_span
    axes[1].set_ylim(
        float(np.min(all_finite_ratios)) - 0.08 * ratio_span,
        ratio_edge + 0.05 * ratio_span,
    )
    for position, roi in enumerate(ROI_ORDER, start=1):
        count = int((ratio_statuses[roi] == "infinite").sum())
        if count:
            axes[1].scatter(
                np.full(count, position),
                np.full(count, ratio_edge),
                marker="^",
                s=32,
                color=PURPLE,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            axes[1].text(
                position,
                ratio_edge + 0.018 * ratio_span,
                f"∞ × {count}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=PURPLE,
            )
        undefined_count = int(
            (ratio_statuses[roi] == "undefined").sum()
        )
        if undefined_count:
            axes[1].text(
                position,
                0.015,
                f"0/0 excluded: {undefined_count}",
                transform=axes[1].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=GRAY,
                rotation=90,
            )
    axes[1].set_ylabel("Target/off-target coverage ratio")
    axes[1].set_title("Thresholded target selectivity", weight="bold")

    for index, axis in enumerate(axes):
        axis.set_xticks(
            np.arange(1, 5),
            [ROI_LABELS[roi] for roi in ROI_ORDER],
            rotation=20,
            ha="right",
        )
        axis.axvline(2.5, color=LIGHT_GRAY, lw=0.8)
        axis.grid(axis="y", color=GRID, lw=0.55)
        axis.set_axisbelow(True)
        axis.text(
            0.0,
            1.055,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    figure.suptitle(
        "CamCan population outcomes with MNI152 reference",
        weight="bold",
        y=0.965,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=GRAY,
                markerfacecolor="white",
                lw=4,
                label="CamCAN median and interquartile range",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=ORANGE,
                markeredgecolor="white",
                label="MNI152 reference",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
    )
    save_figure(
        figure,
        figures_dir,
        "figure_population_mean_field_and_target_offtarget_ratio_absolute",
    )
    return pd.DataFrame(population_rows), pd.DataFrame(ratio_audit)


def plot_relationship(
    subjects: pd.DataFrame,
    mni: pd.DataFrame,
    thresholds: pd.DataFrame,
    figures_dir: Path,
    *,
    x_kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if x_kind not in {"mean_field", "target_coverage"}:
        raise ValueError(x_kind)
    stem = (
        "figure_population_mean_field_offtarget_relationship_"
        "at_mni_roi_threshold"
        if x_kind == "mean_field"
        else "figure_population_target_offtarget_relationship_"
        "at_mni_roi_threshold"
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.45, 6.35))
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.86,
        bottom=0.12,
        hspace=0.38,
        wspace=0.30,
    )
    fit_rows: list[dict] = []
    source_rows: list[dict] = []
    for index, roi in enumerate(ROI_ORDER):
        axis = axes.flat[index]
        roi_rows = subjects.loc[subjects["roi"] == roi]
        target_metric, off_metric = coverage_columns(roi, thresholds)
        y = roi_rows[off_metric].to_numpy(dtype=float)
        mni_y = float(mni.loc[roi, off_metric])
        if x_kind == "mean_field":
            x = roi_rows["roi_mean_v_per_m"].to_numpy(dtype=float)
            mni_x = float(mni.loc[roi, "roi_mean_v_per_m"])
            x_label = "Mean target E-field (V/m)"
        else:
            x = roi_rows[target_metric].to_numpy(dtype=float)
            mni_x = float(mni.loc[roi, target_metric])
            x_label = "Target coverage (%)"
        grid, fitted, statistics = _linear_fit(x, y)
        fit_rows.append(
            {
                "figure": stem,
                "roi": roi,
                "x_outcome": x_kind,
                "y_outcome": "off_target_coverage",
                "threshold_v_per_m": thresholds.loc[
                    roi, "threshold_v_per_m"
                ],
                **statistics,
            }
        )
        for subject, x_value, y_value in zip(
            roi_rows["subject"], x, y
        ):
            source_rows.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "x_outcome": x_kind,
                    "x_absolute_value": x_value,
                    "off_target_coverage_percent": y_value,
                    "mni_x_reference": mni_x,
                    "mni_off_target_coverage_percent": mni_y,
                    "threshold_v_per_m": thresholds.loc[
                        roi, "threshold_v_per_m"
                    ],
                }
            )
        axis.scatter(
            x,
            y,
            s=18,
            color=BLUE,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        axis.plot(grid, fitted, color=GRAY, lw=1.55, zorder=3)
        axis.scatter(
            mni_x,
            mni_y,
            marker="D",
            s=48,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        axis.set_title(ROI_LABELS[roi], weight="bold")
        axis.grid(color=GRID, lw=0.55)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0.0)
        if x_kind == "target_coverage":
            axis.set_xlim(0.0, 100.0)
        axis.text(
            0.04,
            0.96,
            (
                rf"Linear fit  $R^2$={statistics['r_squared']:.2f}"
                "\n"
                rf"Spearman $\rho$={statistics['spearman_rho']:.2f}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=7.1,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        axis.text(
            0.97,
            0.04,
            f"threshold {float(thresholds.loc[roi, 'threshold_v_per_m']):.3f} V/m",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color=GRAY,
        )
        if index % 2 == 0:
            axis.set_ylabel("Off-target coverage (%)")
        axis.text(
            0.0,
            1.055,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    figure.supxlabel(x_label, y=0.025)
    figure.suptitle(
        (
            "Mean target field and off-target exposure"
            if x_kind == "mean_field"
            else "Target and off-target coverage"
        ),
        weight="bold",
        y=0.965,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=BLUE,
                markeredgecolor="none",
                label="CamCan subject",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor=ORANGE,
                markeredgecolor="white",
                label="MNI152",
            ),
            Line2D([0], [0], color=GRAY, lw=1.55, label="Linear fit"),
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
    )
    save_figure(figure, figures_dir, stem)
    return pd.DataFrame(source_rows), pd.DataFrame(fit_rows)


def captions(thresholds: pd.DataFrame) -> dict[str, str]:
    threshold_text = "; ".join(
        f"{ROI_LABELS[roi]} {float(thresholds.loc[roi, 'threshold_v_per_m']):.3f} V/m"
        for roi in ROI_ORDER
    )
    result = {
        "figure_personalization_subject_changes_all_rois_at_mni_roi_threshold": (
            "Generic-to-personalized changes for the seven personalized "
            "subjects, with one row per target region. Columns show target "
            "coverage, off-target coverage, and their target/off-target ratio. "
            "Open blue circles are generic-montage values, filled orange "
            "circles are personalized-montage values, and arrows connect the "
            "two values for each subject. Coverage is evaluated at the "
            f"ROI-specific MNI152 mean ({threshold_text}). A boundary triangle "
            "labelled ∞ denotes positive target coverage with zero off-target "
            "coverage; 0/0 denotes an undefined ratio."
        ),
        "figure_population_mean_field_and_target_offtarget_ratio_absolute": (
            "CamCan distributions of mean target-ROI TI E-field (A) and "
            "target/off-target coverage ratio (B). Pale violins show the "
            "population density, thick bars the interquartile range, white "
            "circles the median, and translucent points a representative "
            "subject subset. Orange diamonds mark MNI152. The dashed line in "
            "A marks 0.2 V/m. Ratios in B use coverage evaluated at each ROI's "
            "MNI152 mean; boundary triangles labelled ∞ denote positive target "
            "coverage with zero off-target coverage."
        ),
        "figure_population_mean_field_offtarget_relationship_at_mni_roi_threshold": (
            "Mean target-ROI TI E-field versus off-target coverage for Left "
            "M1 (A), Right DLPFC (B), Left hippocampus (C), and Right thalamus "
            "(D). Blue points represent CamCan subjects, orange diamonds mark "
            "MNI152, and grey lines are ordinary least-squares fits. Insets "
            "report R² and Spearman's ρ. Off-target coverage is evaluated at "
            f"the ROI-specific MNI152 mean ({threshold_text})."
        ),
        "figure_population_target_offtarget_relationship_at_mni_roi_threshold": (
            "Target coverage versus off-target coverage for Left M1 (A), "
            "Right DLPFC (B), Left hippocampus (C), and Right thalamus (D). "
            "Blue points represent CamCan subjects, orange diamonds mark "
            "MNI152, and grey lines are ordinary least-squares fits. Insets "
            "report R² and Spearman's ρ. Both coverage percentages are "
            f"evaluated at the ROI-specific MNI152 mean ({threshold_text})."
        ),
    }
    return result


def write_text_outputs(
    output_dir: Path,
    caption_values: dict[str, str],
    thresholds: pd.DataFrame,
) -> None:
    rows = [
        {"figure": stem, "caption": caption}
        for stem, caption in caption_values.items()
    ]
    pd.DataFrame(rows).to_csv(
        output_dir / "figure_captions.csv", index=False
    )
    lines = ["# Self-contained figure captions", ""]
    for row in rows:
        lines.extend([f"## {row['figure']}", "", row["caption"], ""])
    (output_dir / "figure_captions.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (output_dir / "SUPERVISOR_NOTE_RATIO_ZERO_DENOMINATORS.md").write_text(
        """# Decision requested: zero off-target coverage in selectivity ratios

The revised figures define thresholded selectivity as:

**target coverage / off-target coverage**

At the ROI-specific MNI152 threshold, some observations can have positive
target coverage but exactly zero off-target coverage. Their mathematical ratio
is positive infinity. Replacing zero with an arbitrary epsilon would create a
large but method-dependent finite value, while dropping the observations would
hide the strongest possible selectivity result.

The current figures therefore use a linear axis and mark these observations
with a triangle at the upper/right plotting boundary labelled **∞ (censored)**.
They remain in the source tables with `ratio_status=infinite`. If both target
and off-target coverage are zero, the ratio is `0/0` and is genuinely
undefined; those observations are counted separately and are not included in
the finite violin density.

The absolute MNI152 ratio is also displayed when its denominator is non-zero.
If the MNI152 off-target denominator is zero, the MNI152 reference itself is
infinite and the renderer deliberately stops rather than applying an epsilon
or inventing a finite value.

Please confirm whether this censored-infinity convention should be retained in
the manuscript. Alternatives would require an explicit methodological choice,
such as reporting the two coverage components without a ratio or defining a
pre-specified denominator regularization constant.
""",
        encoding="utf-8",
    )
    (output_dir / "SIMNIBS_CURRENT_CALIBRATION_ERROR_NOTE.md").write_text(
        """# Where SimNIBS reports the current calibration error

For each simulated tDCS electrode pair, SimNIBS integrates current flux at
the active and reference electrodes. It estimates the calibration error as
the absolute difference between the magnitudes of those two fluxes divided by
their mean magnitude. SimNIBS then rescales the potential solution by the
requested current divided by the estimated mean current.

In SimNIBS 4.5 this is implemented in
`simnibs/simulation/fem.py`, function `_sim_tdcs_pair`. The normal log message
is:

`Estimated current calibration error: ...`

Errors above 10% are emitted as warnings. For a standard `SESSION` run, the
message is written to:

`<SESSION.pathfem>/simnibs_simulation_<timestamp>.log`

This quantity describes FEM current calibration for each electrode-pair
solve. It is not an uncertainty estimate for the later TI-max combination of
the two electric fields.
""",
        encoding="utf-8",
    )
    (output_dir / "SUPERVISOR_REQUEST_AUDIT.md").write_text(
        """# Final supervisor-request figure audit

- Existing v3 and v4 figure directories are not modified.
- ROI order is Left M1, Right DLPFC, Left hippocampus, Right thalamus:
  superficial targets first, then deep targets.
- All axes are linear; no log or symmetric-log scales are used.
- Coverage is calculated from the source images at each ROI's SimNIBS 4.0.1
  MNI152 mean target field. No interpolation from 0.20/0.18/0.15 V/m tables is
  permitted.
- The standalone MNI target-field figure is absent; its values are retained in
  the source and manuscript tables.
- One combined personalization figure contains target coverage, off-target
  coverage, and target/off-target coverage ratio for all four ROIs. It uses
  condition means and generic-to-personalized arrows, with no repeat error
  bars.
- The effectiveness/spread scatter and all repeat-distribution figures are
  absent.
- The population field distribution reports mean only and is combined with a
  violin plot of the target/off-target coverage ratio.
- Population violins retain the established presentation: pale coloured
  density, translucent subject points, coloured interquartile-range bars,
  white CamCAN median circles, and an orange MNI152 diamond at its absolute
  value for every ROI.
- Target E-field, target coverage, off-target coverage, and target/off-target
  ratios are plotted as absolute values. The mean-field and target-coverage
  relationships are retained; separate minimum and maximum relationships are
  absent.
- Target/off-target ratios with a zero off-target denominator are represented
  as censored positive infinity at the linear plotting boundary. Undefined
  0/0 cases are reported separately.
- Figures are written as 400-dpi PNG only, with exact source tables and concise
  captions that describe the displayed plot without repeating the study design.
""",
        encoding="utf-8",
    )
    thresholds.reset_index(drop=True).to_csv(
        output_dir / "tables" / "table_mni152_roi_thresholds.csv",
        index=False,
    )


def build(
    cohort_dir: Path,
    personalized_dir: Path,
    threshold_table: Path,
    output_dir: Path,
    *,
    force: bool,
) -> dict:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory exists: {output_dir}; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True)
    tables_dir.mkdir()
    thresholds = load_thresholds(threshold_table)
    (
        cohort_manifest,
        personalized_manifest,
        subjects,
        mni,
        paired,
    ) = load_inputs(cohort_dir, personalized_dir, thresholds)
    set_style()

    mni_absolute_mean_table(mni, thresholds).to_csv(
        tables_dir / "table_mni152_absolute_mean_target_field.csv",
        index=False,
    )
    plot_personalized_combined(
        paired,
        mni,
        thresholds,
        figures_dir,
    ).to_csv(
        tables_dir / "table_personalization_subject_changes.csv",
        index=False,
    )
    population, ratio_audit = plot_population_summary(
        subjects, mni, thresholds, figures_dir
    )
    population.to_csv(
        tables_dir / "table_population_mean_field_and_ratio.csv",
        index=False,
    )
    ratio_audit.to_csv(
        tables_dir / "table_ratio_zero_denominator_audit.csv",
        index=False,
    )
    relationship_tables = []
    fit_tables = []
    for x_kind in ("mean_field", "target_coverage"):
        relationship, fits = plot_relationship(
            subjects,
            mni,
            thresholds,
            figures_dir,
            x_kind=x_kind,
        )
        relationship_tables.append(relationship)
        fit_tables.append(fits)
    pd.concat(relationship_tables, ignore_index=True).to_csv(
        tables_dir / "table_population_relationship_source_data.csv",
        index=False,
    )
    pd.concat(fit_tables, ignore_index=True).to_csv(
        tables_dir / "table_population_linear_fit_statistics.csv",
        index=False,
    )

    caption_values = captions(thresholds)
    write_text_outputs(output_dir, caption_values, thresholds)
    actual_stems = sorted(path.stem for path in figures_dir.glob("*.png"))
    if actual_stems != sorted(EXPECTED_FIGURE_STEMS):
        raise RuntimeError(
            f"Figure set mismatch: got {actual_stems}, expected "
            f"{sorted(EXPECTED_FIGURE_STEMS)}"
        )
    result = {
        "status": "complete",
        "figure_revision_schema_version": FIGURE_SCHEMA_VERSION,
        "figure_revision_variant": "mni401_roi_threshold_absolute_v7",
        "cohort_analysis_schema_version": cohort_manifest[
            "analysis_schema_version"
        ],
        "personalized_comparison_schema_version": personalized_manifest[
            "comparison_schema_version"
        ],
        "roi_order": ROI_ORDER,
        "coverage_threshold_policy": (
            "each ROI evaluated at its SimNIBS 4.0.1 MNI152 mean target field"
        ),
        "thresholds_v_per_m": {
            roi: float(thresholds.loc[roi, "threshold_v_per_m"])
            for roi in ROI_ORDER
        },
        "population_centering": "none; all outcomes are absolute values",
        "population_mni_markers": (
            "orange diamond at the absolute MNI152 value for every ROI"
        ),
        "population_violin_style": (
            "pale density, deterministic subject subsample, coloured IQR, "
            "white CamCAN median"
        ),
        "ratio_definition": "target coverage / off-target coverage",
        "ratio_infinite_policy": (
            "positive target and zero off-target is positive infinity, "
            "censored at the finite linear-axis boundary"
        ),
        "ratio_undefined_policy": (
            "zero target and zero off-target is undefined 0/0, excluded from "
            "finite violin density and counted explicitly"
        ),
        "axis_scale_policy": "linear only",
        "personalized_repeat_error_bars": False,
        "personalized_trajectory_arrows": True,
        "field_relationships": ["mean only"],
        "figures_removed": [
            "figure_mni152_percentile_context_ge_0p20",
            "figure_mni152_absolute_mean_target_field",
            "figure_personalization_effectiveness_spread_ge_0p20",
            "separate per-ROI personalization figures",
            "all figure_personalization_repeat_distributions_*",
            "separate minimum and maximum population relationships",
        ],
        "threshold_table": {
            "path": str(threshold_table),
            "sha256": sha256_file(threshold_table),
        },
        "figures": sorted(
            str(path.relative_to(output_dir))
            for path in figures_dir.iterdir()
            if path.is_file()
        ),
        "tables": sorted(
            str(path.relative_to(output_dir))
            for path in tables_dir.iterdir()
            if path.is_file()
        ),
        "captions": len(caption_values),
    }
    (output_dir / "figure_revision_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--personalized-dir", required=True, type=Path)
    parser.add_argument(
        "--mni-threshold-table",
        type=Path,
        default=Path(__file__).with_name(
            "mni152_simnibs401_roi_thresholds.csv"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(
        args.cohort_dir.resolve(),
        args.personalized_dir.resolve(),
        args.mni_threshold_table.resolve(),
        args.out_dir.resolve(),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
