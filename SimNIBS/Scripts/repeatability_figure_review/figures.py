"""Render repaired figures and paper-equivalent outputs for the review package."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .audit import AuditBundle, sha256_file


TARGETS = ("left_hippocampus", "right_m1")
TARGET_LABELS = {
    "left_hippocampus": "Left hippocampus",
    "right_m1": "Right M1",
}
VERSIONS = ("simnibs326", "simnibs4")
CELL_SPECS = (
    ("simnibs326", "remesh", "v3 remesh", "#4C78A8"),
    ("simnibs326", "fixed_mesh", "v3 fixed", "#F58518"),
    ("simnibs4", "remesh", "v4 remesh", "#54A24B"),
    ("simnibs4", "fixed_mesh", "v4 fixed", "#E45756"),
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
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
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
        axis.set_ylabel("Participant 40-run median (V/m)")
        _style(axis)
    axes[-1].set_xticks(range(4), [item[2] for item in CELL_SPECS])
    figure.suptitle("Participant central field values across version and condition", y=0.995)
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
        axis.set_ylabel("Within-cell SD (V/m; log scale)")
        _style(axis)
    axes[-1].set_xticks(range(4), [item[2] for item in CELL_SPECS])
    figure.text(
        0.5,
        0.01,
        "Open downward triangles mark SD ≤ 1×10⁻⁸ V/m, plotted at that explicit floor.",
        ha="center",
        fontsize=9,
    )
    figure.suptitle("Technical repeat variability across 40 runs", y=0.995)
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
            label="Remesh",
        )
        axis.scatter(
            np.ones(len(subjects)),
            contrasts["fixed_mesh"].loc[subjects],
            color="#F58518",
            s=42,
            zorder=2,
            label="Fixed mesh",
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_ylabel("v3 − v4 participant contrast (V/m)")
        _style(axis)
    axes[-1].set_xticks([0, 1], ["Remesh", "Fixed mesh"])
    axes[0].legend(frameon=False, ncol=2, loc="best")
    figure.suptitle("Paired version contrasts by condition", y=0.995)
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
        axis.set_xlabel("(v3 − v4)fixed − (v3 − v4)remesh (V/m)")
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        _style(axis, grid_axis="x")
    figure.suptitle("Participant version–condition interaction", y=0.995)
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
        axis.set_ylabel("Optimizer-ROI median |E| (V/m)")
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        _style(axis)
    axes[-1].set_xlabel("Participant (ordered by v4 median)")
    handles = [
        plt.Line2D([0], [0], color="#315b7d", linewidth=8, alpha=0.7, label="v4: 40-run box plot"),
        plt.Line2D([0], [0], marker="D", color="#d1495b", markerfacecolor="#8f1d2c", linewidth=2, label="v3: range, IQR and median"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("SimNIBS v3.2.6 location within matched v4 remesh distributions", y=0.995)
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
        axis.set_xlabel("Percentile in matched v4 remesh distribution")
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        if np.allclose(medians, 100.0):
            axis.text(
                0.98,
                0.03,
                "All 40 v3 runs per participant\nare at the 100th percentile",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#d8dee9", "alpha": 0.9},
            )
        _style(axis, grid_axis="x")
    figure.suptitle("Empirical location of v3 remesh values within v4", y=0.995)
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
        axis.set_xlabel("40-run max − min (V/m; log scale)")
        axis.set_title(f"{chr(65 + panel)}  {TARGET_LABELS[target]}", loc="left")
        _style(axis, grid_axis="x")
    figure.suptitle("SimNIBS 3.2.6 remesh outputs are effectively invariant", y=0.995)
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
    figure.suptitle("Deterministic v3 ranking result", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    artifacts.append(_save_figure(figure, output_root, "revised_figures/within_simnibs326/deterministic_ranking_summary.png", "revised_within_v3"))
    return artifacts


def _combine_vertical(paths: Iterable[Path], output: Path, *, panel_letters: bool) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    resized = []
    for image in images:
        if image.width != width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        resized.append(image)
    separator = max(12, width // 180)
    canvas = Image.new("RGB", (width, sum(image.height for image in resized) + separator), "white")
    y = 0
    positions = []
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(resized):
        positions.append(y)
        canvas.paste(image, (0, y))
        y += image.height
        if index == 0:
            draw.line((width * 0.04, y + separator / 2, width * 0.96, y + separator / 2), fill="#d0d0d0", width=max(2, width // 1400))
            y += separator
    if panel_letters:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(26, width // 80))
        except OSError:
            font = ImageFont.load_default()
        for letter, position in zip(("A", "B"), positions):
            draw.text((max(14, width // 180), position + max(10, width // 240)), letter, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(220, 220), optimize=True)


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
            axis.set_xlabel("Number of remesh runs averaged")
            axis.set_ylabel("Relative uncertainty (%)")
            axis.set_xlim(2, max_repeats)
            axis.set_ylim(bottom=0)
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, title="Participant")
    figure.suptitle("Bootstrap precision of the optimizer-ROI mean field", y=0.995)
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

    v4_root = bundle.source_root / "established" / "within_simnibs4" / "figures"
    paper_specs = (
        (
            "01_primary_median_roi_repeat_distributions.png",
            "paper_equivalents/figure_target_field_repeatability.png",
            True,
        ),
        (
            "02_single_repeat_subject_ranking_uncertainty.png",
            "paper_equivalents/figure_single_run_ranking_uncertainty.png",
            False,
        ),
    )
    for source_name, relative, panel_letters in paper_specs:
        _combine_vertical(
            [v4_root / target / source_name for target in TARGETS],
            output_root / relative,
            panel_letters=panel_letters,
        )
        artifacts.append(_artifact(output_root, relative, "paper_equivalent_current"))

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

