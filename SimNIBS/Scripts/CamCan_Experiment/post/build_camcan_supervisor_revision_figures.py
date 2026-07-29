#!/usr/bin/env python3
"""Build the supervisor-requested optimizer-matched CamCan figure revision.

Inputs are the refreshed schema-4 final-132 cohort analysis and schema-3
personalized comparison.  The figure layer is descriptive and uses the
parcel-clipped target ROI and a single evaluation threshold of 0.20 V/m.
"""

from __future__ import annotations

import argparse
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
from scipy.optimize import curve_fit
from scipy.stats import percentileofscore, spearmanr


ROI_ORDER = [
    "Left_Hippocampus",
    "Left_M1",
    "Right_DLPC",
    "Right_Thalamus",
]
ROI_LABELS = {
    "Left_Hippocampus": "Left hippocampus",
    "Left_M1": "Left M1",
    "Right_DLPC": "Right DLPFC",
    "Right_Thalamus": "Right thalamus",
}
DEEP_ROIS = {"Left_Hippocampus", "Right_Thalamus"}
CONDITIONS = ("generic", "personalized")
THRESHOLD = 0.20
THRESHOLD_TOKEN = "0p2"

BLUE = "#2F6B9A"
ORANGE = "#D97706"
GRAY = "#5F6873"
LIGHT_GRAY = "#CBD1D8"
GRID = "#E4E8ED"
RED = "#B64A4A"
GREEN = "#00876C"
SUBJECT_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#6F6259",
]


def short_subject(subject: str) -> str:
    return str(subject).replace("sub-", "")


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(figure)


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        hint = (
            " Re-run image-level extraction with schema 4/3; target means "
            "cannot be reconstructed from the older CSV files."
            if any("roi_mean_v_per_m" in item for item in missing)
            else ""
        )
        raise RuntimeError(f"{label} missing columns: {missing}.{hint}")
    values = frame[columns].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        bad = frame[columns].columns[~np.isfinite(values).all(axis=0)].tolist()
        raise RuntimeError(f"{label} contains non-finite values in {bad}")


def load_cohort(
    input_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((input_dir / "analysis_manifest.json").read_text())
    expected = {
        "analysis_schema_version": 4,
        "status": "complete",
        "subjects": 132,
        "repeat_level_records": 5280,
        "subject_level_records": 528,
        "mni_baselines": 4,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Unexpected cohort manifest {key}={manifest.get(key)!r}; "
                f"expected {value!r}"
            )
    if manifest.get("thresholds_v_per_m") != [0.2, 0.18, 0.15]:
        raise RuntimeError("Cohort thresholds must be [0.2, 0.18, 0.15]")
    subjects = pd.read_csv(input_dir / "subject_level_repeat_mean_metrics.csv")
    mni = pd.read_csv(input_dir / "mni152_baseline_metrics.csv")
    columns = [
        "roi_mean_v_per_m",
        "roi_median_v_per_m",
        "target_coverage_percent_ge_0p2",
        "off_target_coverage_percent_ge_0p2",
    ]
    _require_columns(subjects, columns, "cohort subject metrics")
    _require_columns(mni, columns, "MNI152 metrics")
    if (
        len(subjects) != 528
        or subjects.groupby("roi").size().to_dict()
        != {roi: 132 for roi in ROI_ORDER}
        or len(mni) != 4
        or set(mni["roi"]) != set(ROI_ORDER)
    ):
        raise RuntimeError("Cohort data do not contain the exact 132 × 4 scope")
    return manifest, subjects, mni.set_index("roi")


def load_personalized(
    input_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((input_dir / "analysis_manifest.json").read_text())
    expected = {
        "comparison_schema_version": 3,
        "manuscript_analysis_schema_version": 4,
        "status": "complete",
        "subject_roi_configurations": 28,
        "repeat_level_records": 560,
        "condition_repeat_mean_records": 56,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Unexpected personalized manifest {key}={manifest.get(key)!r}; "
                f"expected {value!r}"
            )
    paired = pd.read_csv(input_dir / "paired_personalized_vs_generic.csv")
    repeats = pd.read_csv(input_dir / "repeat_level_metrics.csv")
    base_metrics = [
        "roi_mean_v_per_m",
        "roi_median_v_per_m",
        "target_coverage_percent_ge_0p2",
        "off_target_coverage_percent_ge_0p2",
    ]
    paired_columns: list[str] = []
    for metric in base_metrics:
        paired_columns.extend(
            [
                f"{metric}__generic_repeat_mean",
                f"{metric}__generic_repeat_sd",
                f"{metric}__personalized_repeat_mean",
                f"{metric}__personalized_repeat_sd",
            ]
        )
    _require_columns(paired, paired_columns, "paired personalized metrics")
    _require_columns(
        repeats,
        ["roi_mean_v_per_m", "roi_median_v_per_m"],
        "personalized repeat metrics",
    )
    if (
        len(paired) != 28
        or paired["subject"].nunique() != 7
        or not paired.groupby("roi").size().eq(7).all()
        or len(repeats) != 560
        or not repeats.groupby(["subject", "roi", "condition"]).size().eq(10).all()
    ):
        raise RuntimeError("Personalized data do not contain the exact 7 × 4 × 2 × 10 scope")
    roi_rank = {roi: index for index, roi in enumerate(ROI_ORDER)}
    subjects = sorted(paired["subject"].unique())
    subject_rank = {subject: index for index, subject in enumerate(subjects)}
    paired = paired.assign(
        _roi_order=paired["roi"].map(roi_rank),
        _subject_order=paired["subject"].map(subject_rank),
    ).sort_values(["_roi_order", "_subject_order"])
    repeats = repeats.assign(
        _roi_order=repeats["roi"].map(roi_rank),
        _subject_order=repeats["subject"].map(subject_rank),
    ).sort_values(["_roi_order", "_subject_order", "condition", "repeat"])
    return manifest, paired, repeats


def _exp_model(x: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    return c + a * np.expm1(b * x)


def descriptive_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    exponential: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str | int]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.linspace(float(x.min()), float(x.max()), 240)
    if exponential:
        scale = max(float(np.ptp(x)), 1e-9)
        x0 = float(x.min())
        normalized = (x - x0) / scale
        initial = [
            max(0.0, float(np.percentile(y, 5))),
            max(0.01, float(np.ptp(y))),
            1.0,
        ]
        parameters, _ = curve_fit(
            _exp_model,
            normalized,
            y,
            p0=initial,
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, 20.0]),
            maxfev=100_000,
        )
        fitted = _exp_model(normalized, *parameters)
        grid_y = _exp_model((grid - x0) / scale, *parameters)
        parameter_text = {
            "intercept_c": float(parameters[0]),
            "amplitude_a": float(parameters[1]),
            "rate_b_normalized": float(parameters[2]),
        }
        model = "exponential"
    else:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        grid_y = slope * grid + intercept
        parameter_text = {"slope": float(slope), "intercept": float(intercept)}
        model = "linear"
    denominator = float(np.sum((y - y.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum((y - fitted) ** 2)) / denominator
        if denominator > 0
        else math.nan
    )
    rho = float(spearmanr(x, y).statistic)
    return (
        grid,
        grid_y,
        {
            "model": model,
            "n": int(len(x)),
            "r_squared": r_squared,
            "spearman_rho": rho,
            **parameter_text,
        },
    )


def _cohort_relationship_figure(
    subjects: pd.DataFrame,
    mni: pd.DataFrame,
    figures_dir: Path,
    *,
    x_metric: str,
    x_label: str,
    stem: str,
) -> pd.DataFrame:
    y_metric = "off_target_coverage_percent_ge_0p2"
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.25))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.87,
        bottom=0.09,
        hspace=0.37,
        wspace=0.30,
    )
    fit_rows: list[dict[str, float | str | int]] = []
    for index, roi in enumerate(ROI_ORDER):
        axis = axes.flat[index]
        rows = subjects.loc[subjects["roi"] == roi]
        x = rows[x_metric].to_numpy(dtype=float)
        y = rows[y_metric].to_numpy(dtype=float)
        exponential = roi in DEEP_ROIS
        grid, fitted, statistics = descriptive_fit(
            x,
            y,
            exponential=exponential,
        )
        fit_rows.append(
            {
                "figure": stem,
                "roi": roi,
                "x_metric": x_metric,
                "y_metric": y_metric,
                **statistics,
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
        axis.plot(
            grid,
            fitted,
            color=GRAY,
            lw=1.55,
            linestyle="--" if not exponential else "-",
            zorder=3,
        )
        axis.scatter(
            float(mni.loc[roi, x_metric]),
            float(mni.loc[roi, y_metric]),
            marker="D",
            s=51,
            facecolor=ORANGE,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        axis.text(
            0.04,
            0.96,
            (
                f"{statistics['model'].capitalize()} fit\n"
                rf"$R^2$={statistics['r_squared']:.2f}; "
                rf"$\rho$={statistics['spearman_rho']:.2f}"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        axis.set_title(ROI_LABELS[roi], weight="bold")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Off-target coverage ≥ 0.20 V/m (%)")
        if "coverage" in x_metric:
            axis.set_xlim(-3, 103)
        else:
            padding = max(0.005, float(np.ptp(x)) * 0.05)
            axis.set_xlim(max(0.0, float(x.min()) - padding), float(x.max()) + padding)
        y_padding = max(0.2, float(np.ptp(y)) * 0.08)
        axis.set_ylim(max(0.0, float(y.min()) - y_padding), float(y.max()) + y_padding)
        axis.grid(color=GRID, lw=0.55)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            1.03,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                alpha=0.65,
                label="CamCan subject (mean of 10 repeat-level metrics)",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=ORANGE,
                label="MNI152 reference",
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                linestyle="--",
                label="Linear fit (superficial targets)",
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                linestyle="-",
                label="Exponential fit (deep targets)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    save_figure(figure, figures_dir, stem)
    return pd.DataFrame(fit_rows)


def plot_population_ratio(
    subjects: pd.DataFrame,
    mni: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    target = "target_coverage_percent_ge_0p2"
    off_target = "off_target_coverage_percent_ge_0p2"
    figure, axis = plt.subplots(figsize=(7.35, 4.5))
    rng = np.random.default_rng(20260729)
    rows: list[dict[str, float | str | int]] = []
    finite_groups: list[np.ndarray] = []
    zero_counts: list[int] = []
    for roi in ROI_ORDER:
        selected = subjects.loc[subjects["roi"] == roi]
        target_values = selected[target].to_numpy(dtype=float)
        off_values = selected[off_target].to_numpy(dtype=float)
        valid = target_values > 0
        ratio = off_values[valid] / target_values[valid]
        finite_groups.append(ratio)
        zero_count = int((~valid).sum())
        zero_counts.append(zero_count)
        rows.append(
            {
                "roi": roi,
                "subjects": int(len(selected)),
                "finite_ratio_subjects": int(valid.sum()),
                "undefined_target_zero_subjects": zero_count,
                "median_off_target_to_target_ratio": float(np.median(ratio)),
                "q1": float(np.quantile(ratio, 0.25)),
                "q3": float(np.quantile(ratio, 0.75)),
            }
        )
    box = axis.boxplot(
        finite_groups,
        positions=np.arange(1, 5),
        widths=0.52,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "lw": 1.2},
        boxprops={"edgecolor": GRAY, "lw": 0.8},
        whiskerprops={"color": GRAY, "lw": 0.8},
        capprops={"color": GRAY, "lw": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.82)
    tick_labels: list[str] = []
    for position, roi, values, zero_count in zip(
        np.arange(1, 5),
        ROI_ORDER,
        finite_groups,
        zero_counts,
    ):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        axis.scatter(
            position + jitter,
            values,
            s=13,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.55,
            alpha=0.75,
            zorder=3,
        )
        mni_target = float(mni.loc[roi, target])
        mni_ratio = (
            float(mni.loc[roi, off_target]) / mni_target
            if mni_target > 0
            else math.nan
        )
        if math.isfinite(mni_ratio):
            axis.scatter(
                position,
                mni_ratio,
                marker="D",
                s=45,
                facecolor=ORANGE,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        tick_labels.append(
            f"{ROI_LABELS[roi]}\n$n_0$={zero_count}/132"
        )
    axis.set_yscale("symlog", linthresh=1e-4, linscale=0.6, base=10)
    axis.set_xticks(np.arange(1, 5), tick_labels, fontsize=7.1)
    axis.set_ylabel("Off-target coverage / target coverage at 0.20 V/m")
    axis.grid(axis="y", color=GRID, lw=0.55)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=ORANGE,
                label="MNI152 reference",
            ),
            Line2D(
                [0],
                [0],
                color="none",
                lw=0,
                label="$n_0$: target coverage=0; ratio undefined",
            ),
        ],
        loc="upper left",
        frameon=False,
    )
    save_figure(
        figure,
        figures_dir,
        "figure_population_offtarget_target_ratio_ge_0p20",
    )
    return pd.DataFrame(rows)


def plot_mni_percentiles(
    subjects: pd.DataFrame,
    mni: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    metrics = [
        ("Mean target field", "roi_mean_v_per_m", BLUE),
        ("Median target field", "roi_median_v_per_m", GREEN),
        ("Target coverage ≥0.20", "target_coverage_percent_ge_0p2", ORANGE),
        ("Off-target coverage ≥0.20", "off_target_coverage_percent_ge_0p2", RED),
    ]
    records: list[dict[str, float | str]] = []
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 5.7))
    figure.subplots_adjust(
        left=0.15,
        right=0.985,
        top=0.90,
        bottom=0.10,
        hspace=0.38,
        wspace=0.31,
    )
    for index, roi in enumerate(ROI_ORDER):
        axis = axes.flat[index]
        rows = subjects.loc[subjects["roi"] == roi]
        percentiles = []
        for label, metric, color in metrics:
            value = float(mni.loc[roi, metric])
            percentile = float(
                percentileofscore(
                    rows[metric].to_numpy(dtype=float),
                    value,
                    kind="mean",
                )
            )
            percentiles.append(percentile)
            records.append(
                {
                    "roi": roi,
                    "metric": metric,
                    "metric_label": label,
                    "mni152_value": value,
                    "mni152_percentile_within_camcan": percentile,
                }
            )
        y = np.arange(len(metrics))[::-1]
        axis.axvspan(25, 75, color="#F3F4F6", zorder=0)
        axis.axvline(50, color=GRAY, lw=1, linestyle="--")
        axis.scatter(
            percentiles,
            y,
            c=[item[2] for item in metrics],
            s=39,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.set_yticks(y, [item[0] for item in metrics])
        axis.set_xlim(0, 100)
        axis.set_xlabel("MNI152 percentile within CamCan (%)")
        axis.set_title(ROI_LABELS[roi], weight="bold")
        axis.grid(axis="x", color=GRID, lw=0.55)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            1.03,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    save_figure(
        figure,
        figures_dir,
        "figure_mni152_percentile_context_ge_0p20",
    )
    return pd.DataFrame(records)


def _subject_colors(paired: pd.DataFrame) -> dict[str, str]:
    subjects = sorted(paired["subject"].unique())
    if len(subjects) != 7:
        raise RuntimeError("Expected seven personalized subjects")
    return dict(zip(subjects, SUBJECT_PALETTE))


def plot_personalized_summary(paired: pd.DataFrame, figures_dir: Path) -> None:
    panels = [
        ("roi_mean_v_per_m", "Mean target field\n(V/m)"),
        ("roi_median_v_per_m", "Median target field\n(V/m)"),
        ("target_coverage_percent_ge_0p2", "Target coverage\n≥0.20 V/m (%)"),
        (
            "off_target_coverage_percent_ge_0p2",
            "Off-target coverage\n≥0.20 V/m (%)",
        ),
    ]
    figure, axes = plt.subplots(4, 4, figsize=(7.35, 8.8), sharey="row")
    figure.subplots_adjust(
        left=0.17,
        right=0.99,
        top=0.92,
        bottom=0.075,
        hspace=0.34,
        wspace=0.28,
    )
    for roi_index, roi in enumerate(ROI_ORDER):
        subset = paired.loc[paired["roi"] == roi].sort_values("_subject_order")
        y = np.arange(len(subset))[::-1]
        labels = [short_subject(subject) for subject in subset["subject"]]
        for metric_index, (metric, title) in enumerate(panels):
            axis = axes[roi_index, metric_index]
            generic = subset[f"{metric}__generic_repeat_mean"].to_numpy(dtype=float)
            personalized = subset[
                f"{metric}__personalized_repeat_mean"
            ].to_numpy(dtype=float)
            generic_sd = subset[f"{metric}__generic_repeat_sd"].to_numpy(dtype=float)
            personalized_sd = subset[
                f"{metric}__personalized_repeat_sd"
            ].to_numpy(dtype=float)
            for yi, start, end in zip(y, generic, personalized):
                axis.plot([start, end], [yi, yi], color=LIGHT_GRAY, lw=1.25)
            axis.errorbar(
                generic,
                y,
                xerr=generic_sd,
                fmt="o",
                ms=4,
                mfc="white",
                mec=BLUE,
                mew=1.05,
                ecolor=BLUE,
                elinewidth=0.7,
                capsize=1.5,
                zorder=3,
            )
            axis.errorbar(
                personalized,
                y,
                xerr=personalized_sd,
                fmt="o",
                ms=4,
                mfc=ORANGE,
                mec=ORANGE,
                mew=0.9,
                ecolor=ORANGE,
                elinewidth=0.7,
                capsize=1.5,
                zorder=4,
            )
            if roi_index == 0:
                axis.set_title(title, weight="bold", pad=6)
            axis.set_yticks(y)
            axis.set_yticklabels(labels if metric_index == 0 else [])
            axis.grid(axis="x", color=GRID, lw=0.55)
            if "coverage_percent" in metric:
                axis.set_xlim(-3, 103)
            else:
                values = np.concatenate(
                    [
                        generic - generic_sd,
                        generic + generic_sd,
                        personalized - personalized_sd,
                        personalized + personalized_sd,
                    ]
                )
                padding = max(0.005, float(np.ptp(values)) * 0.07)
                axis.set_xlim(
                    max(0.0, float(values.min()) - padding),
                    float(values.max()) + padding,
                )
            if metric_index == 0:
                axis.set_ylabel(ROI_LABELS[roi], weight="bold", labelpad=7)
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=BLUE,
                markerfacecolor="white",
                label="Generic montage",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=ORANGE,
                markerfacecolor=ORANGE,
                label="Personalized montage",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.985),
        ncol=2,
        frameon=False,
    )
    save_figure(
        figure,
        figures_dir,
        "figure_personalization_all_subject_changes",
    )


def plot_personalized_trajectories(
    paired: pd.DataFrame,
    figures_dir: Path,
) -> None:
    x_metric = "target_coverage_percent_ge_0p2"
    y_metric = "off_target_coverage_percent_ge_0p2"
    colors = _subject_colors(paired)
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.2))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.80,
        bottom=0.10,
        hspace=0.37,
        wspace=0.30,
    )
    for index, roi in enumerate(ROI_ORDER):
        axis = axes.flat[index]
        subset = paired.loc[paired["roi"] == roi].sort_values("_subject_order")
        for _, row in subset.iterrows():
            subject = str(row["subject"])
            color = colors[subject]
            x0 = float(row[f"{x_metric}__generic_repeat_mean"])
            y0 = float(row[f"{y_metric}__generic_repeat_mean"])
            x1 = float(row[f"{x_metric}__personalized_repeat_mean"])
            y1 = float(row[f"{y_metric}__personalized_repeat_mean"])
            axis.plot([x0, x1], [y0, y1], color=color, lw=1.3, alpha=0.78)
            axis.scatter(
                x0,
                y0,
                s=31,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=3,
            )
            axis.scatter(
                x1,
                y1,
                s=33,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        axis.set_title(ROI_LABELS[roi], weight="bold")
        axis.set_xlim(-3, 103)
        all_y = np.concatenate(
            [
                subset[f"{y_metric}__generic_repeat_mean"].to_numpy(dtype=float),
                subset[
                    f"{y_metric}__personalized_repeat_mean"
                ].to_numpy(dtype=float),
            ]
        )
        y_padding = max(0.1, float(np.ptp(all_y)) * 0.08)
        axis.set_ylim(max(0.0, float(all_y.min()) - y_padding), float(all_y.max()) + y_padding)
        axis.set_xlabel("Target coverage ≥ 0.20 V/m (%)")
        axis.set_ylabel("Off-target coverage ≥ 0.20 V/m (%)")
        axis.grid(color=GRID, lw=0.55)
        axis.text(
            0.01,
            1.03,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    figure.legend(
        handles=[
            *[
                Line2D([0], [0], color=colors[s], lw=2, label=short_subject(s))
                for s in sorted(colors)
            ],
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=GRAY,
                label="Generic",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=GRAY,
                label="Personalized",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
    )
    save_figure(
        figure,
        figures_dir,
        "figure_personalization_effectiveness_spread_ge_0p20",
    )


def plot_personalized_ratio(
    paired: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    target = "target_coverage_percent_ge_0p2"
    off_target = "off_target_coverage_percent_ge_0p2"
    colors = _subject_colors(paired)
    figure, axes = plt.subplots(2, 2, figsize=(7.35, 6.0), sharey=True)
    figure.subplots_adjust(
        left=0.15,
        right=0.985,
        top=0.84,
        bottom=0.09,
        hspace=0.35,
        wspace=0.22,
    )
    records: list[dict[str, float | str]] = []
    for index, roi in enumerate(ROI_ORDER):
        axis = axes.flat[index]
        subset = paired.loc[paired["roi"] == roi].sort_values("_subject_order")
        y = np.arange(len(subset))[::-1]
        for yi, (_, row) in zip(y, subset.iterrows()):
            subject = str(row["subject"])
            color = colors[subject]
            generic = float(
                row[f"{target}__generic_repeat_mean"]
                / row[f"{off_target}__generic_repeat_mean"]
            )
            personalized = float(
                row[f"{target}__personalized_repeat_mean"]
                / row[f"{off_target}__personalized_repeat_mean"]
            )
            records.append(
                {
                    "subject": subject,
                    "roi": roi,
                    "generic_target_to_off_target_ratio": generic,
                    "personalized_target_to_off_target_ratio": personalized,
                    "change": personalized - generic,
                    "fold_change": personalized / generic if generic > 0 else math.nan,
                }
            )
            axis.plot(
                [generic, personalized],
                [yi, yi],
                color=LIGHT_GRAY,
                lw=1.25,
            )
            axis.scatter(
                generic,
                yi,
                s=31,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=3,
            )
            axis.scatter(
                personalized,
                yi,
                s=33,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        axis.set_xscale("symlog", linthresh=0.5, linscale=0.6, base=10)
        axis.set_yticks(y, [short_subject(item) for item in subset["subject"]])
        if index % 2:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_title(ROI_LABELS[roi], weight="bold")
        axis.set_xlabel("Target coverage / off-target coverage at 0.20 V/m")
        axis.grid(axis="x", color=GRID, lw=0.55)
        axis.text(
            0.01,
            1.03,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=GRAY,
                label="Generic",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=GRAY,
                label="Personalized",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    save_figure(
        figure,
        figures_dir,
        "figure_personalization_target_offtarget_ratio_ge_0p20",
    )
    return pd.DataFrame(records)


def plot_repeat_mean_median(
    repeats: pd.DataFrame,
    figures_dir: Path,
    roi: str,
) -> None:
    subset = repeats.loc[repeats["roi"] == roi]
    subjects = sorted(subset["subject"].unique())
    figure, axes = plt.subplots(2, 4, figsize=(7.35, 4.8), sharey=True)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.88,
        bottom=0.14,
        hspace=0.45,
        wspace=0.18,
    )
    rng = np.random.default_rng(20260729)
    all_values = subset[
        ["roi_mean_v_per_m", "roi_median_v_per_m"]
    ].to_numpy(dtype=float)
    padding = max(0.006, float(np.ptp(all_values)) * 0.08)
    limits = (
        max(0.0, float(all_values.min()) - padding),
        float(all_values.max()) + padding,
    )
    for index, subject in enumerate(subjects):
        axis = axes.flat[index]
        rows = subset.loc[subset["subject"] == subject]
        groups = [
            rows.loc[rows["condition"] == "generic", "roi_mean_v_per_m"].to_numpy(),
            rows.loc[
                rows["condition"] == "personalized", "roi_mean_v_per_m"
            ].to_numpy(),
            rows.loc[
                rows["condition"] == "generic", "roi_median_v_per_m"
            ].to_numpy(),
            rows.loc[
                rows["condition"] == "personalized", "roi_median_v_per_m"
            ].to_numpy(),
        ]
        positions = [1, 2.2, 4.4, 5.8]
        colors = [BLUE, ORANGE, BLUE, ORANGE]
        box = axis.boxplot(
            groups,
            positions=positions,
            widths=0.52,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "white", "lw": 1.1},
            boxprops={"edgecolor": GRAY, "lw": 0.7},
            whiskerprops={"color": GRAY, "lw": 0.7},
            capprops={"color": GRAY, "lw": 0.7},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
        for position, group, color in zip(positions, groups, colors):
            jitter = rng.uniform(-0.08, 0.08, len(group))
            axis.scatter(
                position + jitter,
                group,
                s=11,
                facecolor="white",
                edgecolor=color,
                linewidth=0.55,
                zorder=3,
            )
            axis.scatter(
                position,
                float(np.mean(group)),
                marker="D",
                s=20,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        axis.set_title(short_subject(subject))
        axis.set_xticks(
            positions,
            ["G\nmean", "P\nmean", "G\nmedian", "P\nmedian"],
            fontsize=5.6,
        )
        axis.set_xlim(0.45, 6.35)
        axis.set_ylim(*limits)
        axis.grid(axis="y", color=GRID, lw=0.55)
        if index % 4:
            axis.tick_params(axis="y", labelleft=False)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color=BLUE,
                markerfacecolor=BLUE,
                lw=0,
                label="Generic (G)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=ORANGE,
                markerfacecolor=ORANGE,
                lw=0,
                label="Personalized (P)",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color=GRAY,
                markerfacecolor=GRAY,
                lw=0,
                label="Mean across repeats",
            ),
        ],
        loc="center",
        frameon=False,
    )
    figure.suptitle(
        f"{ROI_LABELS[roi]}: repeat distributions of target mean and median",
        weight="bold",
        y=0.97,
    )
    figure.supylabel("Target-ROI field (V/m)", x=0.02)
    save_figure(
        figure,
        figures_dir,
        f"figure_personalization_technical_repeats_{roi.lower()}",
    )


def captions() -> dict[str, str]:
    common_cohort = (
        "The fixed MNI152-derived temporal-interference montage was simulated "
        "on 132 corrected CamCan heads. Each subject value is the arithmetic "
        "mean of a metric calculated independently in ten remeshing repeats. "
        "The target is a parcel-clipped sphere centred on the volume centroid "
        "(100 mm³ for cortical targets; 200 mm³ for subcortical targets)."
    )
    common_personal = (
        "Seven CamCan subjects were simulated with the fixed MNI152-derived "
        "generic montage and an ROI-specific personalized montage on the same "
        "corrected head. All seven subjects are shown for every target. Metrics "
        "were calculated independently in ten remeshing repeats and then "
        "arithmetic-mean aggregated within condition. The sample was selected "
        "for outcome extremes in an earlier analysis and supports descriptive, "
        "not population-level, inference."
    )
    result = {
        "figure_population_target_offtarget_relationship_ge_0p20": (
            "Target coverage and off-target exposure at 0.20 V/m. "
            f"{common_cohort} Blue points represent subjects and the orange "
            "diamond is the MNI152 reference. Target coverage is the percentage "
            "of target voxels reaching 0.20 V/m; off-target coverage is the "
            "percentage of finite brain voxels outside the target reaching the "
            "same threshold. Dashed lines are descriptive linear fits for "
            "superficial cortical targets; solid curves are descriptive "
            "exponential fits for deep targets. R² and Spearman ρ are shown "
            "within each panel."
        ),
        "figure_population_mean_field_offtarget_relationship_ge_0p20": (
            "Mean target field and off-target exposure at 0.20 V/m. "
            f"{common_cohort} Horizontal position is the arithmetic mean field "
            "over finite target voxels; vertical position is off-target "
            "coverage. Blue points represent subjects and the orange diamond "
            "is MNI152. Fits use the same linear-superficial and "
            "exponential-deep descriptive models as the coverage analysis."
        ),
        "figure_population_offtarget_target_ratio_ge_0p20": (
            "Population distribution of the off-target-to-target coverage "
            f"ratio at 0.20 V/m. {common_cohort} Lower finite ratios indicate "
            "less off-target exposure per percentage point of target coverage. "
            "Boxes summarize subjects with non-zero target coverage; individual "
            "subjects are overlaid and orange diamonds show MNI152. A ratio is "
            "undefined when target coverage is zero, so these complete target "
            "failures are not silently omitted: n₀ beneath each target gives "
            "their count out of 132 subjects."
        ),
        "figure_mni152_percentile_context_ge_0p20": (
            "Position of the MNI152 reference within the CamCan population. "
            f"{common_cohort} Each point is the percentile rank of the single "
            "MNI152 value within the 132 subject values for the indicated "
            "target and outcome. The shaded band denotes the interquartile "
            "range and the dashed line marks the population median."
        ),
        "figure_personalization_all_subject_changes": (
            "Generic-versus-personalized changes across all 28 subject-target "
            f"configurations. {common_personal} Rows are targets and columns "
            "show mean target field, median target field, target coverage and "
            "off-target coverage at 0.20 V/m. Open blue circles are generic "
            "condition means, filled orange circles are personalized condition "
            "means, gray segments connect conditions for the same subject, and "
            "error bars show sample SD across ten repeats. Subjects are not "
            "labelled best or worst because the historical ranking used a "
            "different ROI definition and target-only criterion."
        ),
        "figure_personalization_effectiveness_spread_ge_0p20": (
            "Subject-level changes in target coverage and off-target exposure "
            f"at 0.20 V/m. {common_personal} An open point and filled point show "
            "the generic and personalized condition means for one subject; a "
            "thin line links the two conditions without implying a ranked "
            "direction of improvement. Rightward movement is greater target "
            "coverage and downward movement is less off-target coverage. "
            "Subject identity is encoded consistently by colour across panels."
        ),
        "figure_personalization_target_offtarget_ratio_ge_0p20": (
            "Subject-level change in target-to-off-target coverage ratio at "
            f"0.20 V/m. {common_personal} The ratio divides percentage target "
            "coverage by percentage off-target coverage, so higher values "
            "indicate greater thresholded target selectivity. Open and filled "
            "points show generic and personalized condition means; gray "
            "segments link the same subject. A symmetric-logarithmic x-axis is "
            "used because ratios span several orders of magnitude and may be "
            "zero when target coverage is zero."
        ),
    }
    for roi in ROI_ORDER:
        result[f"figure_personalization_technical_repeats_{roi.lower()}"] = (
            f"Technical-repeat distributions of mean and median target field "
            f"for {ROI_LABELS[roi]}. {common_personal} Each subject panel shows "
            "ten repeat-level values for the generic (G) and personalized (P) "
            "conditions. Boxes show interquartile ranges, internal white lines "
            "show repeat medians, whiskers extend to 1.5 interquartile ranges, "
            "open points are individual repeats, and diamonds are the "
            "arithmetic means used in condition-level comparisons."
        )
    return result


def write_captions(output_dir: Path, values: dict[str, str]) -> None:
    rows = []
    lines = ["# Self-contained figure captions", ""]
    for stem, caption in values.items():
        rows.append({"figure": stem, "caption": caption})
        lines.extend([f"## {stem}", "", caption, ""])
    pd.DataFrame(rows).to_csv(output_dir / "figure_captions.csv", index=False)
    (output_dir / "figure_captions.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build(
    cohort_dir: Path,
    personalized_dir: Path,
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
    cohort_manifest, subjects, mni = load_cohort(cohort_dir)
    personalized_manifest, paired, repeats = load_personalized(personalized_dir)
    set_style()

    fit_coverage = _cohort_relationship_figure(
        subjects,
        mni,
        figures_dir,
        x_metric="target_coverage_percent_ge_0p2",
        x_label="Target coverage ≥ 0.20 V/m (%)",
        stem="figure_population_target_offtarget_relationship_ge_0p20",
    )
    fit_mean = _cohort_relationship_figure(
        subjects,
        mni,
        figures_dir,
        x_metric="roi_mean_v_per_m",
        x_label="Mean target field (V/m)",
        stem="figure_population_mean_field_offtarget_relationship_ge_0p20",
    )
    pd.concat([fit_coverage, fit_mean], ignore_index=True).to_csv(
        tables_dir / "table_descriptive_fit_statistics.csv",
        index=False,
    )
    plot_population_ratio(subjects, mni, figures_dir).to_csv(
        tables_dir / "table_population_offtarget_target_ratio.csv",
        index=False,
    )
    plot_mni_percentiles(subjects, mni, figures_dir).to_csv(
        tables_dir / "table_mni152_percentile_context.csv",
        index=False,
    )
    plot_personalized_summary(paired, figures_dir)
    plot_personalized_trajectories(paired, figures_dir)
    plot_personalized_ratio(paired, figures_dir).to_csv(
        tables_dir / "table_personalized_target_offtarget_ratio.csv",
        index=False,
    )
    for roi in ROI_ORDER:
        plot_repeat_mean_median(repeats, figures_dir, roi)
    caption_values = captions()
    write_captions(output_dir, caption_values)
    result = {
        "status": "complete",
        "figure_revision_schema_version": 1,
        "threshold_v_per_m": THRESHOLD,
        "cohort_analysis_schema_version": cohort_manifest[
            "analysis_schema_version"
        ],
        "personalized_comparison_schema_version": personalized_manifest[
            "comparison_schema_version"
        ],
        "subjects": 132,
        "personalized_subjects": 7,
        "figures": sorted(
            str(path.relative_to(output_dir))
            for path in figures_dir.glob("*")
            if path.is_file()
        ),
        "tables": sorted(
            str(path.relative_to(output_dir))
            for path in tables_dir.glob("*")
            if path.is_file()
        ),
        "captions": len(caption_values),
        "best_worst_visual_encoding": False,
        "trajectory_arrows": False,
        "deep_target_fit": "exponential",
        "superficial_target_fit": "linear",
        "ratio_zero_target_policy": (
            "off-target/target population ratio undefined when target coverage "
            "is zero; excluded from finite box and explicitly counted"
        ),
    }
    (output_dir / "figure_revision_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--personalized-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(
        args.cohort_dir.resolve(),
        args.personalized_dir.resolve(),
        args.out_dir.resolve(),
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
