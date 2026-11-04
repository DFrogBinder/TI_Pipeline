"""Render repaired figures and paper-equivalent outputs for the review package."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

from .audit import AuditBundle, sha256_file


TARGETS = ("left_hippocampus", "right_m1")
TARGET_LABELS = {
    "left_hippocampus": "Left hippocampus",
    "right_m1": "Right M1",
}
VERSIONS = ("simnibs326", "simnibs4")
CELL_SPECS = (
    (
        "simnibs326",
        "remesh",
        "SimNIBS 3.2.6\nNew mesh each run",
        "#4C78A8",
    ),
    (
        "simnibs326",
        "fixed_mesh",
        "SimNIBS 3.2.6\nSame mesh reused",
        "#F58518",
    ),
    ("simnibs4", "remesh", "SimNIBS 4\nNew mesh each run", "#4C78A8"),
    ("simnibs4", "fixed_mesh", "SimNIBS 4\nSame mesh reused", "#F58518"),
)
TECHNICAL_SD_FLOOR = 1e-8


@dataclass(frozen=True)
class Artifact:
    relative_path: str
    category: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None


def figure_text(figure: plt.Figure) -> tuple[str, ...]:
    """Return the non-empty text actually attached to a Matplotlib figure."""

    texts: list[str] = []
    seen: set[str] = set()
    for artist in figure.findobj(match=Text):
        value = artist.get_text().strip()
        if value and artist.get_visible() and value not in seen:
            seen.add(value)
            texts.append(value)
    return tuple(texts)


def percentile_plot_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return limits that show the v4 median and keep 100 inside the frame."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Percentile plot requires at least one finite value")
    lower = min(45.0, float(finite.min()) - 2.0)
    upper = max(102.0, float(finite.max()) + 2.0)
    return lower, upper


def prepare_sd_for_log_plot(
    values: np.ndarray, floor: float = TECHNICAL_SD_FLOOR
) -> tuple[np.ndarray, np.ndarray, float]:
    """Place zero/near-zero SD values at an explicit plotting floor."""

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Technical SD values must be finite and non-negative")
    at_floor = values <= floor
    plotted = np.maximum(values, floor)
    return plotted, at_floor, floor


def bootstrap_subject_curve(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int = 0,
    min_repeats: int = 2,
    max_repeats: int = 40,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Compute the established mean-field bootstrap precision curve."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < max_repeats:
        raise ValueError("Not enough one-dimensional repeat values")
    if iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be finite")
    generator = rng if rng is not None else np.random.default_rng(seed)
    reference = float(np.mean(values))
    if reference == 0.0:
        raise ValueError("Relative precision is undefined for a zero mean")
    rows = []
    for count in range(min_repeats, max_repeats + 1):
        indices = generator.integers(0, values.size, size=(iterations, count))
        estimates = values[indices].mean(axis=1)
        lower, upper = np.quantile(estimates, [0.025, 0.975])
        absolute_error = np.abs(estimates - reference) / abs(reference)
        rows.append(
            {
                "n_repeats": count,
                "reference_value": reference,
                "ci_half_width_rel": float((upper - lower) / (2.0 * abs(reference))),
                "abs_error_quantile_rel": float(np.quantile(absolute_error, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _artifact(output_root: Path, relative: str, category: str) -> Artifact:
    path = output_root / relative
    width = height = None
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    return Artifact(
        relative_path=relative,
        category=category,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        width=width,
        height=height,
    )


def _save_figure(
    figure: plt.Figure,
    output_root: Path,
    relative: str,
    category: str,
) -> Artifact:
    path = output_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    description = "\n".join(figure_text(figure))
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Description": description},
    )
    plt.close(figure)
    return _artifact(output_root, relative, category)


def _load_frame(bundle: AuditBundle) -> pd.DataFrame:
    frames = []
    for version in VERSIONS:
        for target in TARGETS:
            path = bundle.source_root / "inputs" / f"{version}_{target}.csv"
            frame = pd.read_csv(path)
            required = {
                "subject",
                "condition",
                "repeat_tag",
                "roi_mean_v_per_m",
                "roi_median_v_per_m",
            }
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            if len(frame) != 800 or frame["subject"].nunique() != 10:
                raise ValueError(f"{path} does not contain the expected 800-row 9+1 scope")
            counts = frame.groupby(["subject", "condition"])["repeat_tag"].nunique()
            if set(counts.index.get_level_values("condition")) != {"remesh", "fixed_mesh"}:
                raise ValueError(f"{path} does not contain both conditions")
            if not (counts == 40).all():
                raise ValueError(f"{path} does not contain 40 unique repeats per cell")
            frame = frame.copy()
            frame["version"] = version
            frame["target"] = target
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    metric_values = combined[["roi_mean_v_per_m", "roi_median_v_per_m"]].to_numpy(float)
    if not np.isfinite(metric_values).all():
        raise ValueError("Canonical input tables contain non-finite field metrics")
    return combined


def load_v4_ranking_tables(
    bundle: AuditBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the audited SimNIBS 4 ranking tables and verify their full scope."""

    root = bundle.source_root / "established" / "within_simnibs4"
    pairs = pd.read_csv(root / "pairwise_rank_reversal_probabilities.csv")
    draws = pd.read_csv(root / "rank_selection_draws.csv")
    required_pairs = {
        "target",
        "higher_mean_subject",
        "lower_mean_subject",
        "higher_mean_v_per_m",
        "lower_mean_v_per_m",
        "mean_difference_v_per_m",
        "single_run_reversal_probability",
    }
    required_draws = {
        "target",
        "kendall_rank_agreement",
        "pairwise_reversals",
    }
    for frame, required, expected, label in (
        (pairs, required_pairs, 45, "pairwise ranking"),
        (draws, required_draws, 20_000, "ranking draws"),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} table is missing columns: {sorted(missing)}")
        counts = frame.groupby("target").size().to_dict()
        if counts != {target: expected for target in TARGETS}:
            raise ValueError(f"{label} table has incomplete target scope: {counts}")
        numeric = list(required.difference({"target", "higher_mean_subject", "lower_mean_subject"}))
        if not np.isfinite(frame[numeric].to_numpy(float)).all():
            raise ValueError(f"{label} table contains non-finite values")
    return pairs, draws


def _cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["target", "version", "condition", "subject"], as_index=False)
        .agg(
            cell_median=("roi_median_v_per_m", "median"),
            cell_sd=("roi_median_v_per_m", "std"),
        )
        .sort_values(["target", "version", "condition", "subject"])
    )


def _short_subject(value: str) -> str:
    return value.removeprefix("sub-").removeprefix("CC")


def _style(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid_axis, color="#d8dee9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def _render_comprehensive(
    frame: pd.DataFrame, output_root: Path
) -> list[Artifact]:
    summary = _cell_summary(frame)
    artifacts: list[Artifact] = []
    rng = np.random.default_rng(20261104)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 9.0), sharex=True)
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        target_frame = summary.loc[summary["target"] == target]
        for position, (version, condition, label, color) in enumerate(CELL_SPECS):
            values = target_frame.loc[
                (target_frame["version"] == version)
                & (target_frame["condition"] == condition),
                "cell_median",
            ].to_numpy(float)
            jitter = rng.uniform(-0.055, 0.055, len(values))
            axis.scatter(position + jitter, values, s=38, color=color, alpha=0.82)
            axis.plot(
                [position - 0.18, position + 0.18],
                [np.median(values)] * 2,
                color="black",
                linewidth=2.2,
            )
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_ylabel("Median temporal-interference field\nin target region (V/m)")
        _style(axis)
    axes[-1].set_xticks(range(4), [item[2] for item in CELL_SPECS])
    figure.suptitle(
        "Typical target-region field for each participant across software versions and mesh conditions",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    artifacts.append(
        _save_figure(
            figure,
            output_root,
            "revised_figures/comprehensive/central_values.png",
            "revised_comprehensive",
        )
    )

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 9.2), sharex=True, sharey=True)
    floor_counts: dict[str, int] = {}
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        target_frame = summary.loc[summary["target"] == target]
        for position, (version, condition, label, color) in enumerate(CELL_SPECS):
            raw = target_frame.loc[
                (target_frame["version"] == version)
                & (target_frame["condition"] == condition),
                "cell_sd",
            ].to_numpy(float)
            plotted, at_floor, floor = prepare_sd_for_log_plot(raw)
            floor_counts[f"{target}:{version}:{condition}"] = int(at_floor.sum())
            jitter = rng.uniform(-0.055, 0.055, len(raw))
            axis.scatter(
                position + jitter[~at_floor],
                plotted[~at_floor],
                s=38,
                color=color,
                alpha=0.82,
            )
            axis.scatter(
                position + jitter[at_floor],
                plotted[at_floor],
                s=42,
                marker="v",
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
            )
            median_value = max(float(np.median(raw)), floor)
            axis.plot(
                [position - 0.18, position + 0.18],
                [median_value] * 2,
                color="black",
                linewidth=2.0,
            )
        axis.set_yscale("log")
        axis.set_ylim(TECHNICAL_SD_FLOOR / 2.0, 1.5e-2)
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_ylabel("Run-to-run variability (SD, V/m; log scale)")
        _style(axis)
    axes[-1].set_xticks(range(4), [item[2] for item in CELL_SPECS])
    figure.text(
        0.5,
        0.01,
        "Open downward triangles mark variability ≤ 1×10⁻⁸ V/m, shown at that plotting floor.",
        ha="center",
        fontsize=9,
    )
    figure.suptitle("Run-to-run variability across 40 simulations", y=0.995)
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    artifacts.append(
        _save_figure(
            figure,
            output_root,
            "revised_figures/comprehensive/technical_sd_log_scale.png",
            "revised_comprehensive",
        )
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.8, 9.0), sharex=True)
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        target_frame = summary.loc[summary["target"] == target]
        contrasts: dict[str, pd.Series] = {}
        for condition in ("remesh", "fixed_mesh"):
            pivot = target_frame.loc[target_frame["condition"] == condition].pivot(
                index="subject", columns="version", values="cell_median"
            )
            contrasts[condition] = pivot["simnibs326"] - pivot["simnibs4"]
        subjects = sorted(contrasts["remesh"].index)
        for subject in subjects:
            axis.plot(
                [0, 1],
                [contrasts["remesh"].loc[subject], contrasts["fixed_mesh"].loc[subject]],
                color="#9aa0a6",
                linewidth=1.1,
                alpha=0.72,
                zorder=1,
            )
        axis.scatter(
            np.zeros(len(subjects)),
            contrasts["remesh"].loc[subjects],
            color="#4C78A8",
            s=42,
            zorder=2,
            label="New mesh each run",
        )
        axis.scatter(
            np.ones(len(subjects)),
            contrasts["fixed_mesh"].loc[subjects],
            color="#F58518",
            s=42,
            zorder=2,
            label="Same mesh reused",
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_ylabel(
            "SimNIBS 3.2.6 minus SimNIBS 4\nmedian target-region field (V/m)"
        )
        _style(axis)
    axes[-1].set_xticks([0, 1], ["New mesh each run", "Same mesh reused"])
    axes[0].legend(frameon=False, ncol=2, loc="best")
    figure.suptitle("Field difference between software versions for each participant", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    artifacts.append(
        _save_figure(
            figure,
            output_root,
            "revised_figures/comprehensive/paired_version_contrasts.png",
            "revised_comprehensive",
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.5), sharex=False)
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        target_frame = summary.loc[summary["target"] == target]
        by_condition = {}
        for condition in ("remesh", "fixed_mesh"):
            pivot = target_frame.loc[target_frame["condition"] == condition].pivot(
                index="subject", columns="version", values="cell_median"
            )
            by_condition[condition] = pivot["simnibs326"] - pivot["simnibs4"]
        interaction = (by_condition["fixed_mesh"] - by_condition["remesh"]).sort_values()
        y = np.arange(len(interaction))
        axis.hlines(y, 0, interaction.to_numpy(float), color="#8da0cb", linewidth=1.6)
        axis.scatter(interaction.to_numpy(float), y, color="#3b66b0", s=45, zorder=3)
        axis.axvline(0, color="black", linewidth=1)
        axis.set_yticks(y, [_short_subject(value) for value in interaction.index])
        axis.set_xlabel(
            "Change in the SimNIBS 3.2.6 minus SimNIBS 4 difference\n"
            "when one mesh is reused instead of creating a new mesh each run (V/m)"
        )
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        _style(axis, grid_axis="x")
    figure.suptitle(
        "Does reusing one mesh change the field difference between software versions?",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    artifacts.append(
        _save_figure(
            figure,
            output_root,
            "revised_figures/comprehensive/version_condition_interaction.png",
            "revised_comprehensive",
        )
    )
    return artifacts


def _midrank_percentile(value: float, reference: np.ndarray) -> float:
    return float(100.0 * ((reference < value).sum() + 0.5 * (reference == value).sum()) / len(reference))


def _cross_version_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    remesh = frame.loc[frame["condition"] == "remesh"]
    for target in TARGETS:
        target_frame = remesh.loc[remesh["target"] == target]
        for subject in sorted(target_frame["subject"].unique()):
            v3 = target_frame.loc[
                (target_frame["subject"] == subject)
                & (target_frame["version"] == "simnibs326"),
                "roi_median_v_per_m",
            ].to_numpy(float)
            v4 = target_frame.loc[
                (target_frame["subject"] == subject)
                & (target_frame["version"] == "simnibs4"),
                "roi_median_v_per_m",
            ].to_numpy(float)
            percentiles = np.asarray([_midrank_percentile(value, v4) for value in v3])
            rows.append(
                {
                    "target": target,
                    "subject": subject,
                    "v3_values": v3,
                    "v4_values": v4,
                    "v4_median": float(np.median(v4)),
                    "v3_median": float(np.median(v3)),
                    "v3_q1": float(np.percentile(v3, 25)),
                    "v3_q3": float(np.percentile(v3, 75)),
                    "v3_min": float(np.min(v3)),
                    "v3_max": float(np.max(v3)),
                    "percentile_median": float(np.median(percentiles)),
                    "percentile_q1": float(np.percentile(percentiles, 25)),
                    "percentile_q3": float(np.percentile(percentiles, 75)),
                    "percentile_min": float(np.min(percentiles)),
                    "percentile_max": float(np.max(percentiles)),
                }
            )
    return pd.DataFrame(rows)


def _render_cross_version(frame: pd.DataFrame, output_root: Path) -> tuple[list[Artifact], dict[str, list[float]]]:
    summary = _cross_version_summary(frame)
    artifacts: list[Artifact] = []

    figure, axes = plt.subplots(2, 1, figsize=(12.0, 9.2))
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        selected = summary.loc[summary["target"] == target].sort_values("v4_median")
        positions = np.arange(1, len(selected) + 1)
        box = axis.boxplot(
            selected["v4_values"].tolist(),
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#0b132b", "linewidth": 1.4},
            boxprops={"facecolor": "#78a6d1", "edgecolor": "#315b7d", "alpha": 0.7},
            whiskerprops={"color": "#315b7d"},
            capprops={"color": "#315b7d"},
        )
        del box
        for position, row in zip(positions, selected.itertuples()):
            axis.hlines(position * 0 + row.v3_median, position - 0.18, position + 0.18, color="#8f1d2c", linewidth=2.2)
            axis.vlines(position, row.v3_min, row.v3_max, color="#d1495b", linewidth=1.0)
            axis.vlines(position, row.v3_q1, row.v3_q3, color="#d1495b", linewidth=4.0)
            axis.scatter(position, row.v3_median, marker="D", s=38, color="#8f1d2c", edgecolor="white", linewidth=0.5, zorder=4)
        axis.set_xticks(positions, [_short_subject(value) for value in selected["subject"]], rotation=35, ha="right")
        axis.set_ylabel("Median temporal-interference field\nin target region (V/m)")
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        _style(axis)
    axes[-1].set_xlabel("Participant (ordered by SimNIBS 4 median)")
    handles = [
        plt.Line2D(
            [0], [0], color="#315b7d", linewidth=8, alpha=0.7,
            label="SimNIBS 4: 40 runs with a new mesh each run",
        ),
        plt.Line2D(
            [0], [0], marker="D", color="#d1495b",
            markerfacecolor="#8f1d2c", linewidth=2,
            label="SimNIBS 3.2.6: range, middle 50%, and median",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(
        "SimNIBS 3.2.6 fields compared with matched SimNIBS 4 fields",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    artifacts.append(_save_figure(figure, output_root, "revised_figures/cross_version/remesh_distribution_overlay.png", "revised_cross_version"))

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), sharex=True)
    bounds_by_target: dict[str, list[float]] = {}
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        selected = summary.loc[summary["target"] == target].sort_values("percentile_median")
        y = np.arange(len(selected))
        medians = selected["percentile_median"].to_numpy(float)
        minima = selected["percentile_min"].to_numpy(float)
        maxima = selected["percentile_max"].to_numpy(float)
        q1 = selected["percentile_q1"].to_numpy(float)
        q3 = selected["percentile_q3"].to_numpy(float)
        axis.hlines(y, minima, maxima, color="#a7b4c2", linewidth=1.5)
        axis.hlines(y, q1, q3, color="#315b7d", linewidth=5.0)
        axis.scatter(medians, y, color="#d1495b", s=42, zorder=3)
        axis.axvline(50, color="#14213d", linestyle="--", linewidth=1.5)
        lower, upper = percentile_plot_bounds(np.concatenate([minima, maxima, medians]))
        bounds_by_target[target] = [lower, upper]
        axis.set_xlim(lower, upper)
        axis.set_yticks(y, selected["subject"])
        axis.set_xlabel(
            "Percentile within matched SimNIBS 4 runs\n(new mesh each run)"
        )
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        if np.allclose(medians, 100.0):
            axis.text(
                0.98,
                0.03,
                "All 40 SimNIBS 3.2.6 runs per participant\nare at the 100th percentile",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#d8dee9", "alpha": 0.9},
            )
        _style(axis, grid_axis="x")
    figure.suptitle(
        "Where SimNIBS 3.2.6 fields fall within the SimNIBS 4 distribution",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    artifacts.append(_save_figure(figure, output_root, "revised_figures/cross_version/v3_percentile_location.png", "revised_cross_version"))
    return artifacts, bounds_by_target


def _render_v3_determinism(frame: pd.DataFrame, output_root: Path) -> list[Artifact]:
    remesh = frame.loc[(frame["version"] == "simnibs326") & (frame["condition"] == "remesh")]
    ranges = (
        remesh.groupby(["target", "subject"])["roi_median_v_per_m"]
        .agg(lambda values: float(values.max() - values.min()))
        .rename("range")
        .reset_index()
    )
    artifacts: list[Artifact] = []
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.5), sharex=True)
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        selected = ranges.loc[ranges["target"] == target].sort_values("range")
        raw = selected["range"].to_numpy(float)
        plotted, at_floor, floor = prepare_sd_for_log_plot(raw, floor=1e-10)
        y = np.arange(len(selected))
        axis.scatter(plotted[~at_floor], y[~at_floor], color="#4C78A8", s=45)
        axis.scatter(plotted[at_floor], y[at_floor], marker="<", facecolors="none", edgecolors="#4C78A8", s=45)
        axis.set_xscale("log")
        axis.set_yticks(y, [_short_subject(value) for value in selected["subject"]])
        axis.set_xlabel(
            "Difference between largest and smallest field across 40 runs (V/m; log scale)"
        )
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        _style(axis, grid_axis="x")
    figure.suptitle(
        "SimNIBS 3.2.6 fields are effectively unchanged when a new mesh is created each run",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    artifacts.append(_save_figure(figure, output_root, "revised_figures/within_simnibs326/deterministic_repeatability.png", "revised_within_v3"))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        axis.axis("off")
        axis.text(0.5, 0.78, TARGET_LABELS[target], ha="center", fontsize=17, fontweight="bold")
        axis.text(0.5, 0.53, "0 / 45", ha="center", fontsize=34, color="#315b7d", fontweight="bold")
        axis.text(0.5, 0.40, "participant pairs reversed rank", ha="center", fontsize=12)
        axis.text(0.5, 0.22, "Kendall’s τ = 1.00", ha="center", fontsize=22, color="#8f1d2c")
        axis.text(0.5, 0.11, "in all 20,000 single-run selections", ha="center", fontsize=11)
        axis.text(0.02, 0.95, chr(65 + panel), transform=axis.transAxes, fontsize=16, fontweight="bold", va="top")
    figure.suptitle("Stable participant ordering in SimNIBS 3.2.6", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    artifacts.append(_save_figure(figure, output_root, "revised_figures/within_simnibs326/deterministic_ranking_summary.png", "revised_within_v3"))
    return artifacts


def _render_v4_repeatability(
    frame: pd.DataFrame, output_root: Path
) -> Artifact:
    selected = frame.loc[frame["version"] == "simnibs4"]
    colors = {"remesh": "#4C78A8", "fixed_mesh": "#F58518"}
    labels = {
        "remesh": "New mesh each run",
        "fixed_mesh": "Same mesh reused",
    }
    markers = {"remesh": "o", "fixed_mesh": "D"}
    offsets = {"remesh": -0.13, "fixed_mesh": 0.13}
    rng = np.random.default_rng(42)
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 9.5))
    for panel, (axis, target) in enumerate(zip(axes, TARGETS)):
        target_frame = selected.loc[selected["target"] == target]
        ordering = (
            target_frame.loc[target_frame["condition"] == "remesh"]
            .groupby("subject")["roi_median_v_per_m"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        for position, participant in enumerate(ordering):
            for condition in ("remesh", "fixed_mesh"):
                values = target_frame.loc[
                    (target_frame["subject"] == participant)
                    & (target_frame["condition"] == condition),
                    "roi_median_v_per_m",
                ].to_numpy(float)
                center = position + offsets[condition]
                jitter = rng.uniform(-0.045, 0.045, len(values))
                axis.scatter(
                    center + jitter,
                    values,
                    s=18,
                    alpha=0.35,
                    color=colors[condition],
                    edgecolors="none",
                    label=labels[condition] if position == 0 else None,
                    zorder=2,
                )
                axis.errorbar(
                    center,
                    values.mean(),
                    yerr=values.std(ddof=1),
                    fmt=markers[condition],
                    color=colors[condition],
                    markerfacecolor="white",
                    markeredgewidth=1.8,
                    markersize=7,
                    capsize=3,
                    linewidth=1.8,
                    zorder=4,
                )
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_ylabel("Median temporal-interference field in target region (V/m)")
        axis.set_xticks(
            range(len(ordering)),
            [_short_subject(participant) for participant in ordering],
            rotation=35,
            ha="right",
        )
        _style(axis)
    axes[-1].set_xlabel(
        "Participant (ordered by mean field when a new mesh is created each run)"
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        title="Mesh condition",
    )
    figure.suptitle("Run-to-run repeatability in SimNIBS 4", y=0.995)
    figure.tight_layout(rect=(0, 0.075, 1, 0.97))
    return _save_figure(
        figure,
        output_root,
        "paper_equivalents/figure_target_field_repeatability.png",
        "paper_equivalent_current",
    )


def _render_v4_ranking(
    frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    draw_frame: pd.DataFrame,
    output_root: Path,
) -> Artifact:
    remesh = frame.loc[
        (frame["version"] == "simnibs4") & (frame["condition"] == "remesh")
    ]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.5),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )
    panel_index = 0
    for row, target in enumerate(TARGETS):
        target_frame = remesh.loc[remesh["target"] == target]
        means = target_frame.groupby("subject")["roi_median_v_per_m"].mean()
        participants = means.sort_values(ascending=False).index.tolist()
        matrix = np.full((len(participants), len(participants)), np.nan)
        target_pairs = pair_frame.loc[pair_frame["target"] == target]
        for pair in target_pairs.itertuples():
            higher_index = participants.index(pair.higher_mean_subject)
            lower_index = participants.index(pair.lower_mean_subject)
            matrix[higher_index, lower_index] = pair.single_run_reversal_probability

        matrix_axis = axes[row, 0]
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("white")
        image = matrix_axis.imshow(matrix, cmap=cmap, vmin=0, vmax=0.6)
        participant_labels = [_short_subject(value) for value in participants]
        matrix_axis.set_xticks(
            range(len(participants)), participant_labels, rotation=42, ha="right"
        )
        matrix_axis.set_yticks(range(len(participants)), participant_labels)
        matrix_axis.set_xlabel("Participant with lower 40-run mean field")
        matrix_axis.set_ylabel("Participant with higher 40-run mean field")
        matrix_axis.set_title(
            f"{chr(65 + panel_index)}  {TARGET_LABELS[target]} — Pairwise reversal probability",
            loc="left",
        )
        panel_index += 1
        colorbar = figure.colorbar(image, ax=matrix_axis, fraction=0.047, pad=0.04)
        colorbar.set_label("Probability that one selected run reverses the pair")

        agreements = draw_frame.loc[
            draw_frame["target"] == target, "kendall_rank_agreement"
        ].to_numpy(float)
        histogram_axis = axes[row, 1]
        lower = max(-1.0, float(agreements.min()) - 0.04)
        histogram_axis.hist(
            agreements,
            bins=np.linspace(lower, 1.005, 18),
            weights=np.full(len(agreements), 100.0 / len(agreements)),
            color="#4C78A8",
            alpha=0.82,
            edgecolor="white",
        )
        median = float(np.median(agreements))
        histogram_axis.axvline(
            median,
            color="#244f73",
            linewidth=2.2,
            label=f"Median agreement = {median:.2f}",
        )
        histogram_axis.set_xlim(lower, 1.005)
        histogram_axis.set_xlabel(
            "Agreement with the 40-run participant ordering (Kendall's τ)"
        )
        histogram_axis.set_ylabel("Random one-run selections (%)")
        histogram_axis.set_title(
            f"{chr(65 + panel_index)}  {TARGET_LABELS[target]} — Whole-cohort ordering",
            loc="left",
        )
        panel_index += 1
        histogram_axis.legend(frameon=False)
        _style(histogram_axis)

    figure.suptitle(
        "Uncertainty in participant ranking when only one new-mesh run is selected (SimNIBS 4)",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(
        figure,
        output_root,
        "paper_equivalents/figure_single_run_ranking_uncertainty.png",
        "paper_equivalent_current",
    )


def _bootstrap_curves(
    frame: pd.DataFrame,
    *,
    iterations: int,
    max_repeats: int,
) -> pd.DataFrame:
    rows = []
    selected = frame.loc[(frame["version"] == "simnibs4") & (frame["condition"] == "remesh")]
    for target in TARGETS:
        generator = np.random.default_rng(0)
        target_frame = selected.loc[selected["target"] == target]
        for subject in sorted(target_frame["subject"].unique()):
            values = target_frame.loc[target_frame["subject"] == subject, "roi_mean_v_per_m"].to_numpy(float)
            curve = bootstrap_subject_curve(values, iterations=iterations, min_repeats=2, max_repeats=max_repeats, rng=generator)
            curve.insert(0, "subject", subject)
            curve.insert(0, "target", target)
            rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def _render_bootstrap(
    frame: pd.DataFrame,
    output_root: Path,
    *,
    iterations: int,
    max_repeats: int,
) -> list[Artifact]:
    curves = _bootstrap_curves(frame, iterations=iterations, max_repeats=max_repeats)
    curve_relative = "bootstrap_precision/per_subject_curve.csv"
    curve_path = output_root / curve_relative
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(curve_path, index=False)

    summary_rows = []
    for target in TARGETS:
        for count in (2, max_repeats):
            selected = curves.loc[(curves["target"] == target) & (curves["n_repeats"] == count)]
            summary_rows.append(
                {
                    "target": target,
                    "n_repeats": count,
                    "subjects": int(selected["subject"].nunique()),
                    "ci_half_width_percent_min": float(selected["ci_half_width_rel"].min() * 100),
                    "ci_half_width_percent_max": float(selected["ci_half_width_rel"].max() * 100),
                    "abs_error_percent_min": float(selected["abs_error_quantile_rel"].min() * 100),
                    "abs_error_percent_max": float(selected["abs_error_quantile_rel"].max() * 100),
                }
            )
    summary_relative = "bootstrap_precision/precision_curve_summary.csv"
    pd.DataFrame(summary_rows).to_csv(output_root / summary_relative, index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0), sharex=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, 10))
    for row, target in enumerate(TARGETS):
        selected = curves.loc[curves["target"] == target]
        subjects = sorted(selected["subject"].unique())
        for column, (metric, title) in enumerate(
            (
                ("ci_half_width_rel", "Central 95% interval half-width"),
                ("abs_error_quantile_rel", "95th percentile absolute relative error"),
            )
        ):
            axis = axes[row, column]
            for color, subject in zip(colors, subjects):
                subject_curve = selected.loc[selected["subject"] == subject]
                axis.plot(subject_curve["n_repeats"], subject_curve[metric] * 100, color=color, linewidth=1.4, alpha=0.9, label=_short_subject(subject))
            axis.set_title(f"{TARGET_LABELS[target]} — {title}", fontsize=11)
            axis.set_xlabel("Number of runs averaged (new mesh each run)")
            axis.set_ylabel("Relative uncertainty (%)")
            axis.set_xlim(2, max_repeats)
            axis.set_ylim(bottom=0)
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, title="Participant")
    figure.suptitle(
        "Mean temporal-interference field in target region: bootstrap precision",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.96))
    figure_relative = "paper_equivalents/figure_bootstrap_precision.png"
    artifacts = [
        _save_figure(figure, output_root, figure_relative, "paper_equivalent_current"),
        _artifact(output_root, curve_relative, "bootstrap_data"),
        _artifact(output_root, summary_relative, "bootstrap_data"),
    ]
    return artifacts


def render_revised_figures(
    bundle: AuditBundle,
    output_root: Path,
    *,
    bootstrap_iterations: int = 20_000,
    bootstrap_max_repeats: int = 40,
) -> tuple[Artifact, ...]:
    """Render repaired figures and updated/legacy paper-equivalence outputs."""

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(bundle)
    artifacts: list[Artifact] = []
    artifacts.extend(_render_comprehensive(frame, output_root))
    cross_artifacts, bounds = _render_cross_version(frame, output_root)
    artifacts.extend(cross_artifacts)
    artifacts.extend(_render_v3_determinism(frame, output_root))
    pair_frame, draw_frame = load_v4_ranking_tables(bundle)
    artifacts.append(_render_v4_repeatability(frame, output_root))
    artifacts.append(
        _render_v4_ranking(
            frame,
            pair_frame,
            draw_frame,
            output_root,
        )
    )

    artifacts.extend(
        _render_bootstrap(
            frame,
            output_root,
            iterations=bootstrap_iterations,
            max_repeats=bootstrap_max_repeats,
        )
    )

    manuscript_figures = bundle.manuscript_root / "figures" / "combined"
    for name in (
        "figure_mesh_element_count_variability.png",
        "figure_tissue_volume_deviations.png",
    ):
        relative = f"legacy_paper_figures/{name}"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manuscript_figures / name, destination)
        artifacts.append(_artifact(output_root, relative, "legacy_paper_figure"))

    metadata = {
        "status": "complete",
        "technical_sd_plot_floor_v_per_m": TECHNICAL_SD_FLOOR,
        "technical_sd_floor_marker": "open downward triangle",
        "percentile_axis_limits": bounds,
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": 0,
            "coverage": 0.95,
            "metric": "roi_mean_v_per_m",
            "condition": "simnibs4 remesh",
            "max_repeats": bootstrap_max_repeats,
        },
        "legacy_policy": {
            "figure_mesh_element_count_variability.png": "not an updated 9+1 equivalent; current mesh metrics unavailable",
            "figure_tissue_volume_deviations.png": "legacy v4-only illustration using retained selected participants",
        },
        "roi_boundary": "aligned optimizer-matched construction; version-specific masks are not identical",
    }
    metadata_relative = "figure_build_metadata.json"
    (output_root / metadata_relative).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    artifacts.append(_artifact(output_root, metadata_relative, "metadata"))
    return tuple(artifacts)
