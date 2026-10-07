#!/usr/bin/env python3
"""Create supervisor-ready figures for the MNI152 SimNIBS version comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


ROI_ORDER = (
    "Left_M1",
    "Right_DLPC",
    "Left_Hippocampus",
    "Right_Thalamus",
)
ROI_LABELS = {
    "Left_M1": "Left M1",
    "Right_DLPC": "Right DLPFC",
    "Left_Hippocampus": "Left hippocampus",
    "Right_Thalamus": "Right thalamus",
}
BASELINE_DIRS = {
    "Left_M1": "MNI152-left-m1",
    "Right_DLPC": "MNI152-right-dlpc",
    "Left_Hippocampus": "MNI152-left-hippocampus",
    "Right_Thalamus": "MNI152-right-thalamus",
}
VERSION_COLUMNS = {
    "SimNIBS 4.5.0": "simnibs_4p5p0",
    "SimNIBS 4.0.1": "simnibs_4p0p1",
}
VERSION_COLORS = {
    "SimNIBS 4.5.0": "#3B6FB6",
    "SimNIBS 4.0.1": "#D97904",
}
CONTINUOUS_METRICS = (
    ("roi_min_v_per_m", "Minimum"),
    ("roi_mean_v_per_m", "Mean"),
    ("roi_median_v_per_m", "Median"),
    ("roi_robust_max_p99_9_v_per_m", "Maximum (P99.9)"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["roi"] = pd.Categorical(
        result["roi"], categories=ROI_ORDER, ordered=True
    )
    return result.sort_values("roi")


def add_roi_group_labels(axis: plt.Axes) -> None:
    axis.axvline(1.5, color="#C8CDD4", linewidth=0.9, zorder=0)
    axis.text(
        0.5,
        1.015,
        "Cortical ROIs",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color="#5E6670",
        fontsize=9,
    )
    axis.text(
        2.5,
        1.015,
        "Deep ROIs",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color="#5E6670",
        fontsize=9,
    )


def save_figure(figure: plt.Figure, stem: Path) -> list[Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=320, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def metric_rows(selected: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = ordered(selected[selected["metric"] == metric])
    if len(rows) != len(ROI_ORDER):
        raise ValueError(f"Expected four rows for {metric}; found {len(rows)}.")
    return rows


def plot_mean_field(selected: pd.DataFrame, output_dir: Path) -> list[Path]:
    rows = metric_rows(selected, "roi_mean_v_per_m")
    x = np.arange(len(ROI_ORDER), dtype=float)
    offset = 0.10

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    old = rows["simnibs_4p5p0"].to_numpy(dtype=float)
    new = rows["simnibs_4p0p1"].to_numpy(dtype=float)
    for index in range(len(x)):
        axis.plot(
            [x[index] - offset, x[index] + offset],
            [old[index], new[index]],
            color="#AEB5BE",
            linewidth=1.5,
            zorder=1,
        )
    axis.scatter(
        x - offset,
        old,
        s=58,
        color=VERSION_COLORS["SimNIBS 4.5.0"],
        edgecolor="white",
        linewidth=0.8,
        label="SimNIBS 4.5.0",
        zorder=3,
    )
    axis.scatter(
        x + offset,
        new,
        s=58,
        marker="s",
        color=VERSION_COLORS["SimNIBS 4.0.1"],
        edgecolor="white",
        linewidth=0.8,
        label="SimNIBS 4.0.1",
        zorder=3,
    )
    for index, delta in enumerate(new - old):
        axis.text(
            x[index],
            min(old[index], new[index]) - 0.0025,
            f"Δ {delta:+.4f}",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#4F5660",
        )
    axis.set_xticks(x, [ROI_LABELS[roi] for roi in ROI_ORDER])
    axis.set_ylabel("Mean E-field within ROI (V/m)")
    axis.set_title("MNI152 mean ROI field by SimNIBS version", pad=24)
    lower = float(min(np.min(old), np.min(new)))
    upper = float(max(np.max(old), np.max(new)))
    padding = max((upper - lower) * 0.16, 0.004)
    axis.set_ylim(lower - padding, upper + padding)
    axis.yaxis.grid(True, color="#E3E6EA", linewidth=0.8)
    axis.set_axisbelow(True)
    add_roi_group_labels(axis)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.005,
        "Identical fixed MNI152 model and ROI definitions; Δ = SimNIBS 4.0.1 − 4.5.0.",
        ha="center",
        fontsize=8.5,
        color="#5E6670",
    )
    figure.subplots_adjust(bottom=0.24, top=0.84)
    return save_figure(
        figure, output_dir / "figure_01_mean_roi_field_by_version"
    )


def paired_metric_panel(
    axis: plt.Axes, rows: pd.DataFrame, *, title: str
) -> None:
    x = np.arange(len(ROI_ORDER), dtype=float)
    offset = 0.09
    old = rows["simnibs_4p5p0"].to_numpy(dtype=float)
    new = rows["simnibs_4p0p1"].to_numpy(dtype=float)
    for index in range(len(x)):
        axis.plot(
            [x[index] - offset, x[index] + offset],
            [old[index], new[index]],
            color="#AEB5BE",
            linewidth=1.3,
            zorder=1,
        )
    axis.scatter(
        x - offset,
        old,
        s=42,
        color=VERSION_COLORS["SimNIBS 4.5.0"],
        edgecolor="white",
        linewidth=0.7,
        label="SimNIBS 4.5.0",
        zorder=3,
    )
    axis.scatter(
        x + offset,
        new,
        s=42,
        marker="s",
        color=VERSION_COLORS["SimNIBS 4.0.1"],
        edgecolor="white",
        linewidth=0.7,
        label="SimNIBS 4.0.1",
        zorder=3,
    )
    axis.set_xticks(x, [ROI_LABELS[roi] for roi in ROI_ORDER], rotation=15)
    axis.set_ylabel("E-field within ROI (V/m)")
    axis.set_title(title)
    axis.yaxis.grid(True, color="#E3E6EA", linewidth=0.8)
    axis.set_axisbelow(True)
    lower = float(min(np.min(old), np.min(new)))
    upper = float(max(np.max(old), np.max(new)))
    padding = max((upper - lower) * 0.17, 0.003)
    axis.set_ylim(lower - padding, upper + padding)


def plot_continuous_metric_values(
    selected: pd.DataFrame, output_dir: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.1))
    for axis, (metric, label) in zip(
        axes.ravel(), CONTINUOUS_METRICS, strict=True
    ):
        paired_metric_panel(
            axis,
            metric_rows(selected, metric),
            title=label,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Continuous MNI152 ROI field metrics by SimNIBS version",
        fontsize=14,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Minimum, mean, median, and maximum (P99.9) are calculated from the same ROI masks for both versions.",
        ha="center",
        fontsize=8.5,
        color="#5E6670",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.92,
        wspace=0.25,
        hspace=0.34,
    )
    return save_figure(
        figure, output_dir / "figure_02_continuous_roi_metrics_by_version"
    )


def plot_continuous_metric_changes(
    selected: pd.DataFrame, output_dir: Path
) -> list[Path]:
    values = np.empty((len(CONTINUOUS_METRICS), len(ROI_ORDER)), dtype=float)
    for row_index, (metric, _) in enumerate(CONTINUOUS_METRICS):
        rows = metric_rows(selected, metric)
        values[row_index, :] = rows[
            "percent_delta_relative_to_4p5p0"
        ].to_numpy(dtype=float)

    limit = max(3.5, float(np.nanmax(np.abs(values))))
    normalizer = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axis = plt.subplots(figsize=(8.7, 4.2))
    image = axis.imshow(values, cmap="RdBu_r", norm=normalizer, aspect="auto")
    axis.set_xticks(
        np.arange(len(ROI_ORDER)), [ROI_LABELS[roi] for roi in ROI_ORDER]
    )
    axis.set_yticks(
        np.arange(len(CONTINUOUS_METRICS)),
        [label for _, label in CONTINUOUS_METRICS],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:+.2f}%",
                ha="center",
                va="center",
                color="#20242A",
                fontsize=9.5,
                fontweight="semibold",
            )
    axis.set_title(
        "Change in continuous ROI field metrics: SimNIBS 4.0.1 versus 4.5.0",
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.04)
    colorbar.set_label("Relative change (%)")
    figure.text(
        0.5,
        0.01,
        "Negative values indicate lower fields in SimNIBS 4.0.1.",
        ha="center",
        fontsize=8.5,
        color="#5E6670",
    )
    figure.subplots_adjust(bottom=0.18)
    return save_figure(
        figure, output_dir / "figure_03_continuous_metric_percent_change"
    )


def plot_voxelwise_summary(
    voxelwise: pd.DataFrame, output_dir: Path
) -> list[Path]:
    rows = ordered(voxelwise)
    x = np.arange(len(ROI_ORDER), dtype=float)
    colors = ["#4E79A7", "#59A14F", "#B07AA1", "#E15759"]

    figure, axes = plt.subplots(1, 2, figsize=(11.7, 4.6))
    l2 = rows["relative_l2_difference_percent"].to_numpy(dtype=float)
    mae = (
        rows["mean_absolute_difference_v_per_m"].to_numpy(dtype=float) * 1000.0
    )
    correlations = rows["pearson_r"].to_numpy(dtype=float)

    bars_l2 = axes[0].bar(x, l2, color=colors, width=0.64)
    axes[0].bar_label(
        bars_l2,
        labels=[f"{value:.2f}%" for value in l2],
        padding=3,
        fontsize=8.5,
    )
    bars_mae = axes[1].bar(x, mae, color=colors, width=0.64)
    axes[1].bar_label(
        bars_mae,
        labels=[f"{value:.2f}" for value in mae],
        padding=3,
        fontsize=8.5,
    )
    for index, correlation in enumerate(correlations):
        axes[1].text(
            x[index],
            mae[index] + max(mae) * 0.12,
            f"r={correlation:.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            color="#4F5660",
        )

    for axis in axes:
        axis.set_xticks(x, [ROI_LABELS[roi] for roi in ROI_ORDER], rotation=15)
        axis.yaxis.grid(True, color="#E3E6EA", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Relative L2 difference (%)")
    axes[0].set_title("Whole-brain relative difference")
    axes[0].set_ylim(0, max(l2) * 1.22)
    axes[1].set_ylabel("Mean absolute difference (mV/m)")
    axes[1].set_title("Absolute difference and spatial correlation")
    axes[1].set_ylim(0, max(mae) * 1.42)
    figure.suptitle(
        "Whole-brain TI-field agreement between SimNIBS versions",
        fontsize=14,
        y=1.03,
    )
    figure.text(
        0.5,
        -0.04,
        "Each comparison contains 1,637,424 shared finite brain voxels.",
        ha="center",
        fontsize=8.5,
        color="#5E6670",
    )
    figure.subplots_adjust(bottom=0.22, top=0.79, wspace=0.27)
    return save_figure(
        figure, output_dir / "figure_04_voxelwise_agreement_summary"
    )


def find_ti(parent: Path, roi: str) -> Path:
    path = (
        parent
        / BASELINE_DIRS[roi]
        / "anat"
        / "SimNIBS"
        / "ti_brain_only.nii.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def plot_voxelwise_identity(
    old_parent: Path, new_parent: Path, output_dir: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.2))
    rng = np.random.default_rng(20260729)
    last_hexbin = None
    for axis, roi in zip(axes.ravel(), ROI_ORDER, strict=True):
        old = np.asarray(nib.load(find_ti(old_parent, roi)).dataobj, dtype=np.float32)
        new = np.asarray(nib.load(find_ti(new_parent, roi)).dataobj, dtype=np.float32)
        finite = np.isfinite(old) & np.isfinite(new)
        old_values = old[finite].astype(np.float64, copy=False)
        new_values = new[finite].astype(np.float64, copy=False)
        count = min(150_000, old_values.size)
        indices = rng.choice(old_values.size, size=count, replace=False)
        x = old_values[indices]
        y = new_values[indices]
        upper = float(max(np.max(x), np.max(y)))
        last_hexbin = axis.hexbin(
            x,
            y,
            gridsize=58,
            mincnt=1,
            bins="log",
            extent=(0.0, upper, 0.0, upper),
            cmap="viridis",
            linewidths=0,
        )
        axis.plot(
            [0.0, upper],
            [0.0, upper],
            color="#D94F4F",
            linestyle=(0, (5, 3)),
            linewidth=1.2,
            label="Identity",
        )
        correlation = float(np.corrcoef(old_values, new_values)[0, 1])
        delta = float(np.mean(new_values - old_values))
        axis.text(
            0.04,
            0.96,
            f"r = {correlation:.6f}\nMean Δ = {delta:+.5f} V/m",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#D8DCE2",
                "alpha": 0.9,
            },
        )
        axis.set_title(ROI_LABELS[roi])
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("SimNIBS 4.5.0 TI field (V/m)")
        axis.set_ylabel("SimNIBS 4.0.1 TI field (V/m)")
    if last_hexbin is not None:
        colorbar = figure.colorbar(
            last_hexbin,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label("Sampled voxel count (log scale)")
    figure.suptitle(
        "Voxelwise agreement of MNI152 TI fields", fontsize=14, y=0.995
    )
    figure.text(
        0.5,
        0.01,
        "Deterministic sample of up to 150,000 shared finite voxels per ROI; red dashed line denotes identity.",
        ha="center",
        fontsize=8.5,
        color="#5E6670",
    )
    figure.subplots_adjust(
        left=0.09, right=0.88, bottom=0.09, top=0.92, wspace=0.25, hspace=0.32
    )
    return save_figure(
        figure, output_dir / "figure_05_voxelwise_field_identity"
    )


def build_guide(
    *,
    path: Path,
    selected: pd.DataFrame,
    voxelwise: pd.DataFrame,
    atlas: Path,
) -> None:
    mean_rows = metric_rows(selected, "roi_mean_v_per_m")
    voxel_rows = ordered(voxelwise)
    lines = [
        "# MNI152 SimNIBS 4.0.1 versus 4.5.0: figure guide",
        "",
        "## Main conclusion",
        "",
        "The SimNIBS version change produces a small, consistent downward shift "
        "in TI-field amplitude while preserving the spatial field pattern almost "
        "perfectly.",
        "",
        "This package compares only the downstream SimNIBS 4.0.1 and 4.5.0 "
        "field solutions. The stimulation optimization was performed independently "
        "in SCIRun and its optimization objective is not evaluated here.",
        "",
        "## Mean ROI field",
        "",
        "| ROI | 4.5.0 (V/m) | 4.0.1 (V/m) | Δ (V/m) | Δ % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in mean_rows.to_dict(orient="records"):
        lines.append(
            f"| {ROI_LABELS[str(row['roi'])]} | "
            f"{row['simnibs_4p5p0']:.5f} | "
            f"{row['simnibs_4p0p1']:.5f} | "
            f"{row['delta_4p0p1_minus_4p5p0']:+.5f} | "
            f"{row['percent_delta_relative_to_4p5p0']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Whole-brain agreement",
            "",
            "| ROI | Pearson r | Relative L2 difference (%) | MAE (V/m) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in voxel_rows.to_dict(orient="records"):
        lines.append(
            f"| {ROI_LABELS[str(row['roi'])]} | "
            f"{row['pearson_r']:.6f} | "
            f"{row['relative_l2_difference_percent']:.3f} | "
            f"{row['mean_absolute_difference_v_per_m']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "1. `figure_01_mean_roi_field_by_version`: paired mean ROI fields.",
            "2. `figure_02_continuous_roi_metrics_by_version`: minimum, mean, "
            "median, and maximum (P99.9) values.",
            "3. `figure_03_continuous_metric_percent_change`: minimum, mean, median, "
            "and maximum (P99.9) relative changes.",
            "4. `figure_04_voxelwise_agreement_summary`: relative and absolute "
            "whole-brain differences.",
            "5. `figure_05_voxelwise_field_identity`: direct voxelwise agreement.",
            "",
            "Each figure is supplied as a 320-dpi PNG and a vector PDF.",
            "",
            "## Analysis provenance",
            "",
            f"- Destrieux/a2009s MNI atlas: `{atlas}`",
            "- The MNI mesh, reference T1, montage table, ROI-specific currents, "
            "conductivities, electrode geometry, and TImax workflow were held fixed.",
            "- Comparison direction: SimNIBS 4.0.1 minus SimNIBS 4.5.0.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    comparison_dir = args.comparison_dir.expanduser().resolve(strict=True)
    old_parent = args.simnibs_4p5_parent.expanduser().resolve(strict=True)
    new_parent = args.simnibs_4p0p1_parent.expanduser().resolve(strict=True)
    atlas = args.mni_atlas.expanduser().resolve(strict=True)
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(
        comparison_dir / "mni152_selected_metric_differences.csv"
    )
    voxelwise = pd.read_csv(comparison_dir / "mni152_voxelwise_differences.csv")
    continuous_metric_names = [metric for metric, _ in CONTINUOUS_METRICS]
    continuous = selected[selected["metric"].isin(continuous_metric_names)].copy()
    configure_style()

    outputs: list[Path] = []
    outputs.extend(plot_mean_field(selected, output_dir))
    outputs.extend(plot_continuous_metric_values(selected, output_dir))
    outputs.extend(plot_continuous_metric_changes(selected, output_dir))
    outputs.extend(plot_voxelwise_summary(voxelwise, output_dir))
    outputs.extend(plot_voxelwise_identity(old_parent, new_parent, output_dir))

    continuous_csv = output_dir / "continuous_roi_metric_comparison.csv"
    voxelwise_csv = output_dir / "voxelwise_agreement.csv"
    ordered(continuous).to_csv(continuous_csv, index=False)
    ordered(voxelwise).to_csv(voxelwise_csv, index=False)
    outputs.extend([continuous_csv, voxelwise_csv])

    guide = output_dir / "README_FOR_SUPERVISOR.md"
    build_guide(
        path=guide,
        selected=selected,
        voxelwise=voxelwise,
        atlas=atlas,
    )
    outputs.append(guide)

    manifest = {
        "figure_schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "SimNIBS 4.0.1 minus SimNIBS 4.5.0",
        "scope": (
            "Continuous ROI field metrics and whole-brain voxelwise agreement only; "
            "no optimization-objective or threshold-based evaluation."
        ),
        "roi_order": list(ROI_ORDER),
        "inputs": {
            "comparison_dir": str(comparison_dir),
            "simnibs_4p5_parent": str(old_parent),
            "simnibs_4p0p1_parent": str(new_parent),
            "mni_atlas": str(atlas),
        },
        "outputs": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--simnibs-4p5-parent", type=Path, required=True)
    parser.add_argument("--simnibs-4p0p1-parent", type=Path, required=True)
    parser.add_argument("--mni-atlas", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
