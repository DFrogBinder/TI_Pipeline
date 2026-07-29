#!/usr/bin/env python3
"""Build the final MNI-threshold CamCan manuscript figures.

This renderer implements the figure decisions from the supervisor meeting:

* superficial ROIs precede deep ROIs;
* each ROI is evaluated at its SimNIBS 4.0.1 MNI152 mean target field;
* population outcomes are expressed relative to the corresponding MNI152
  result;
* the personalized figure contains target coverage, off-target coverage, and
  target/off-target coverage ratio only;
* repeat error bars, repeat-distribution plots, percentile-context plots,
  effectiveness/spread scatter plots, and separate minimum/maximum
  relationships are not generated;
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


FIGURE_SCHEMA_VERSION = 6
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
FIELD_COLORS = {
    "Minimum": BLUE,
    "Mean": GREEN,
    "Maximum (P99.9)": PURPLE,
}
EXPECTED_FIGURE_STEMS = [
    "figure_mni152_absolute_field_summaries",
    *[
        "figure_personalization_subject_changes_"
        f"{ROI_SLUGS[roi]}_at_mni_roi_threshold"
        for roi in ROI_ORDER
    ],
    "figure_population_mean_field_and_target_offtarget_ratio_mni_relative",
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
    for suffix, kwargs in (
        (".png", {"dpi": 400}),
        (".pdf", {}),
    ):
        figure.savefig(
            figures_dir / f"{stem}{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            **kwargs,
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
        values = {
            "Minimum": float(mni.loc[roi, "roi_min_v_per_m"]),
            "Mean": float(mni.loc[roi, "roi_mean_v_per_m"]),
            "Maximum (P99.9)": float(
                mni.loc[roi, "roi_robust_max_p99_9_v_per_m"]
            ),
        }
        axis.vlines(
            index,
            values["Minimum"],
            values["Maximum (P99.9)"],
            color=LIGHT_GRAY,
            lw=5.0,
            zorder=1,
        )
        for offset, (label, value) in zip(
            (-0.11, 0.0, 0.11), values.items()
        ):
            axis.scatter(
                index + offset,
                value,
                s=50 if label == "Mean" else 37,
                color=FIELD_COLORS[label],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            rows.append(
                {
                    "roi": roi,
                    "summary": label,
                    "value_v_per_m": value,
                    "evaluation_threshold": (
                        label == "Mean"
                    ),
                    "simnibs_version": thresholds.loc[
                        roi, "simnibs_version"
                    ],
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
        "MNI152 target-field summaries used for ROI-specific evaluation",
        weight="bold",
        pad=24,
    )
    axis.grid(axis="y", color=GRID, lw=0.55)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="white",
                label=label,
                markersize=6.5,
            )
            for label, color in FIELD_COLORS.items()
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.25)
    save_figure(figure, figures_dir, EXPECTED_FIGURE_STEMS[0])
    return pd.DataFrame(rows)


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
        top=0.82,
        bottom=0.24,
        wspace=0.28,
    )

    coverage_specs = [
        (
            target_metric,
            mni_target,
            "Target coverage difference\nfrom MNI152 (percentage points)",
        ),
        (
            off_metric,
            mni_off,
            "Off-target coverage difference\nfrom MNI152 (percentage points)",
        ),
    ]
    for panel_index, (metric, baseline, label) in enumerate(coverage_specs):
        axis = axes[panel_index]
        generic = (
            rows[f"{metric}__generic_repeat_mean"].to_numpy(dtype=float)
            - baseline
        )
        personalized = (
            rows[f"{metric}__personalized_repeat_mean"].to_numpy(dtype=float)
            - baseline
        )
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
                        "mni_relative_value": value,
                        "ratio_status": "",
                        "mni_absolute_reference": baseline,
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
        axis.axvline(0.0, color=LIGHT_GRAY, lw=0.9, zorder=0)
        axis.set_xlabel(label)
        axis.grid(axis="x", color=GRID, lw=0.55)
        axis.set_axisbelow(True)

    ratio_axis = axes[2]
    ratio_by_condition: dict[str, np.ndarray] = {}
    status_by_condition: dict[str, np.ndarray] = {}
    mni_ratio = math.nan
    for condition in CONDITIONS:
        target = rows[
            f"{target_metric}__{condition}_repeat_mean"
        ].to_numpy(dtype=float)
        off_target = rows[
            f"{off_metric}__{condition}_repeat_mean"
        ].to_numpy(dtype=float)
        values, statuses, reference = centered_ratio(
            target,
            off_target,
            mni_target=mni_target,
            mni_off_target=mni_off,
            roi=roi,
        )
        ratio_by_condition[condition] = values
        status_by_condition[condition] = statuses
        mni_ratio = reference
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
                    "mni_relative_value": value,
                    "ratio_status": status,
                    "mni_absolute_reference": mni_ratio,
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
    ratio_axis.axvline(0.0, color=LIGHT_GRAY, lw=0.9, zorder=0)
    ratio_axis.set_xlim(left_edge, right_edge + 0.02 * span)
    ratio_axis.set_xlabel(
        "Target/off-target coverage ratio\ndifference from MNI152"
    )
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
    figure.text(
        0.5,
        0.875,
        (
            "Coverage evaluated at the MNI152 mean field "
            f"({float(thresholds.loc[roi, 'threshold_v_per_m']):.3f} V/m)"
        ),
        ha="center",
        color=GRAY,
        fontsize=8,
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
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
    )
    stem = (
        "figure_personalization_subject_changes_"
        f"{ROI_SLUGS[roi]}_at_mni_roi_threshold"
    )
    save_figure(figure, figures_dir, stem)
    return pd.DataFrame(source)


def _violin(
    axis: plt.Axes,
    values_by_roi: list[np.ndarray],
    *,
    color: str,
) -> None:
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
        body.set_edgecolor("white")
        body.set_alpha(0.60)
        body.set_linewidth(0.7)
    for position, values in zip(positions, values_by_roi):
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        axis.vlines(position, q1, q3, color=BLACK, lw=4.2, zorder=3)
        axis.scatter(
            position,
            median,
            s=22,
            facecolor="white",
            edgecolor=BLACK,
            linewidth=0.7,
            zorder=4,
        )


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
    for roi in ROI_ORDER:
        roi_rows = subjects.loc[subjects["roi"] == roi]
        field = (
            roi_rows["roi_mean_v_per_m"].to_numpy(dtype=float)
            - float(mni.loc[roi, "roi_mean_v_per_m"])
        )
        field_values.append(field)
        target_metric, off_metric = coverage_columns(roi, thresholds)
        target = roi_rows[target_metric].to_numpy(dtype=float)
        off_target = roi_rows[off_metric].to_numpy(dtype=float)
        ratios, statuses, mni_ratio = centered_ratio(
            target,
            off_target,
            mni_target=float(mni.loc[roi, target_metric]),
            mni_off_target=float(mni.loc[roi, off_metric]),
            roi=roi,
        )
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
                        "mni_relative_value": field[index],
                        "status": "finite",
                    },
                    {
                        "subject": subject,
                        "roi": roi,
                        "outcome": "target_to_off_target_coverage_ratio",
                        "mni_relative_value": ratios[index],
                        "status": statuses[index],
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

    _violin(axes[0], field_values, color=BLUE)
    axes[0].axhline(0.0, color=ORANGE, lw=1.1, zorder=1)
    axes[0].set_ylabel("Mean target E-field difference from MNI152 (V/m)")
    axes[0].set_title("Mean target field", weight="bold")

    _violin(axes[1], ratio_values_by_roi, color=GREEN)
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
    axes[1].axhline(0.0, color=ORANGE, lw=1.1, zorder=1)
    axes[1].set_ylabel(
        "Target/off-target coverage ratio\ndifference from MNI152"
    )
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
        "CamCan population outcomes relative to MNI152",
        weight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.06,
        (
            "Violin width denotes density; black bars show the interquartile "
            "range; white circles show the median. Triangles mark censored "
            "infinite ratios."
        ),
        ha="center",
        fontsize=7.2,
        color=GRAY,
    )
    save_figure(
        figure,
        figures_dir,
        "figure_population_mean_field_and_target_offtarget_ratio_mni_relative",
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
        y = (
            roi_rows[off_metric].to_numpy(dtype=float)
            - float(mni.loc[roi, off_metric])
        )
        if x_kind == "mean_field":
            x = (
                roi_rows["roi_mean_v_per_m"].to_numpy(dtype=float)
                - float(mni.loc[roi, "roi_mean_v_per_m"])
            )
            x_label = "Mean target E-field difference from MNI152 (V/m)"
        else:
            x = (
                roi_rows[target_metric].to_numpy(dtype=float)
                - float(mni.loc[roi, target_metric])
            )
            x_label = (
                "Target coverage difference from MNI152 (percentage points)"
            )
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
                    "x_mni_relative_value": x_value,
                    "off_target_coverage_mni_relative_pp": y_value,
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
            0.0,
            0.0,
            marker="D",
            s=48,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        axis.axvline(0.0, color=LIGHT_GRAY, lw=0.8, zorder=0)
        axis.axhline(0.0, color=LIGHT_GRAY, lw=0.8, zorder=0)
        axis.set_title(ROI_LABELS[roi], weight="bold")
        axis.grid(color=GRID, lw=0.55)
        axis.set_axisbelow(True)
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
            axis.set_ylabel(
                "Off-target coverage difference from MNI152\n"
                "(percentage points)"
            )
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
                label="MNI152 (zero reference)",
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
        "figure_mni152_absolute_field_summaries": (
            "Absolute MNI152 E-field summaries inside each parcel-clipped "
            "target ROI using SimNIBS 4.0.1. Points show the minimum, "
            "arithmetic mean, and maximum (P99.9). The MNI152 mean for each "
            "ROI defines that ROI's subsequent coverage threshold."
        ),
        "figure_population_mean_field_and_target_offtarget_ratio_mni_relative": (
            "CamCan population distributions relative to MNI152. The left "
            "panel shows each subject's mean target E-field minus the "
            "ROI-specific MNI152 mean. The right panel shows target coverage "
            "divided by off-target coverage, followed by subtraction of the "
            "corresponding finite MNI152 ratio. Ratios with positive target "
            "coverage and zero off-target coverage are positive infinity and "
            "are censored at the upper linear-axis boundary; 0/0 observations "
            "are undefined and counted separately. "
            f"ROI-specific thresholds were {threshold_text}."
        ),
        "figure_population_mean_field_offtarget_relationship_at_mni_roi_threshold": (
            "Relationship between mean target E-field and off-target coverage "
            "across 132 CamCan subjects. Both axes show differences from the "
            "corresponding MNI152 result. Coverage is evaluated separately at "
            f"each ROI's MNI152 mean field ({threshold_text}). Solid lines "
            "are descriptive linear fits; diamonds mark the MNI152 zero "
            "reference."
        ),
        "figure_population_target_offtarget_relationship_at_mni_roi_threshold": (
            "Relationship between target and off-target coverage across 132 "
            "CamCan subjects. Both axes show percentage-point differences "
            "from MNI152, with coverage evaluated at the ROI-specific MNI152 "
            f"mean field ({threshold_text}). Solid lines are descriptive "
            "linear fits; diamonds mark the MNI152 zero reference."
        ),
    }
    for roi in ROI_ORDER:
        threshold = float(thresholds.loc[roi, "threshold_v_per_m"])
        result[
            "figure_personalization_subject_changes_"
            f"{ROI_SLUGS[roi]}_at_mni_roi_threshold"
        ] = (
            f"Generic-to-personalized changes for {ROI_LABELS[roi]}. "
            f"Coverage is evaluated at the MNI152 mean field ({threshold:.3f} "
            "V/m). Target and off-target coverage are shown as "
            "percentage-point differences from MNI152. Selectivity is target "
            "coverage divided by off-target coverage and is shown relative "
            "to the finite MNI152 ratio. Each marker is the arithmetic mean "
            "of ten independently remeshed repeat-level metrics; repeat error "
            "bars are omitted. Arrows connect the generic and personalized "
            "condition means for the same subject. Right-edge triangles mark "
            "ratios censored at positive infinity when off-target coverage is "
            "zero."
        )
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

MNI-relative ratios additionally require the MNI152 off-target coverage to be
non-zero. If the MNI152 denominator itself is zero, subtracting an infinite MNI
ratio is undefined; the renderer deliberately stops instead of applying an
epsilon or inventing a finite baseline. That case would require a separate
supervisor decision.

Please confirm whether this censored-infinity convention should be retained in
the manuscript. Alternatives would require an explicit methodological choice,
such as reporting the two coverage components without a ratio or defining a
pre-specified denominator regularization constant.
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
- The MNI percentile-context figure is replaced by an absolute MNI152
  minimum/mean/maximum (P99.9) target-field figure.
- Each personalized ROI figure contains target coverage, off-target coverage,
  and target/off-target coverage ratio only. It uses condition means and
  generic-to-personalized arrows, with no repeat error bars.
- The effectiveness/spread scatter and all repeat-distribution figures are
  absent.
- The population field distribution reports mean only and is combined with a
  violin plot of the target/off-target coverage ratio.
- Population relationship figures are MNI-centred. The mean-field and
  target-coverage relationships are retained; separate minimum and maximum
  relationships are absent.
- Target/off-target ratios with a zero off-target denominator are represented
  as censored positive infinity at the linear plotting boundary. Undefined
  0/0 cases are reported separately.
- Figures are written as 400-dpi PNG and vector PDF with exact source tables
  and self-contained captions.
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

    plot_mni_absolute(mni, thresholds, figures_dir).to_csv(
        tables_dir / "table_mni152_absolute_field_summaries.csv",
        index=False,
    )
    personalized_tables = [
        plot_personalized_roi(
            paired, mni, thresholds, roi, figures_dir
        )
        for roi in ROI_ORDER
    ]
    pd.concat(personalized_tables, ignore_index=True).to_csv(
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
        "figure_revision_variant": "mni401_roi_threshold_mni_relative_v5",
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
        "population_centering": (
            "observed outcome minus ROI-specific MNI152 outcome"
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
            "figure_personalization_effectiveness_spread_ge_0p20",
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
