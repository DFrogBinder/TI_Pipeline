#!/usr/bin/env python3
"""Create lightweight presentation figures from completed repeatability analysis."""

from __future__ import annotations

import argparse
import binascii
import csv
import json
import math
import random
import shutil
import statistics
import struct
import sys
import zlib
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from post import aggregate_paired_analysis


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_condition_rows(experiment_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    analysis_root = experiment_root / "_analysis"
    for summary_csv in sorted(analysis_root.glob("sub-*/**/summary.csv")):
        try:
            subject = summary_csv.parents[1].name
            condition = summary_csv.parent.name
        except IndexError:
            continue
        for row in _read_csv(summary_csv):
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "repeat_tag": row.get("repeat_tag", ""),
                    "median_roi": row.get("median_roi", ""),
                    "mean_roi": row.get("mean_roi", ""),
                    "p95_roi": row.get("p95_roi", ""),
                    "peak_roi": row.get("peak_roi", ""),
                    "p95_head": row.get("p95_head", ""),
                    "mesh_nodes": row.get("mesh_nodes", ""),
                }
            )
    return rows


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _repeat_number(value: object) -> int:
    text = str(value)
    try:
        return int(text.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _draw_line(canvas: bytearray, width: int, height: int, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int]) -> None:
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            offset = (y0 * width + x0) * 3
            canvas[offset : offset + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _write_png(path: Path, width: int, height: int, canvas: bytearray) -> None:
    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + bytes(canvas[y * stride : (y + 1) * stride]))
    raw = b"".join(rows)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_fallback_line_plot(path: Path, rows: list[dict[str, object]], metric: str) -> bool:
    if not rows:
        return False
    width, height = 900, 480
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 30, 70
    canvas = bytearray([255] * width * height * 3)
    axis_color = (55, 65, 81)
    _draw_line(canvas, width, height, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), axis_color)
    _draw_line(canvas, width, height, (margin_left, margin_top), (margin_left, height - margin_bottom), axis_color)

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    values: list[float] = []
    for row in rows:
        value = _safe_float(row.get(metric))
        if math.isfinite(value):
            groups.setdefault((str(row["subject"]), str(row["condition"])), []).append(row)
            values.append(value)
    if not values:
        _write_png(path, width, height, canvas)
        return True

    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    colors = [(37, 99, 235), (220, 38, 38), (22, 163, 74), (147, 51, 234), (234, 88, 12)]
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    for group_index, (_key, group_rows) in enumerate(sorted(groups.items())):
        group_rows = sorted(group_rows, key=lambda row: str(row["repeat_tag"]))
        points: list[tuple[int, int]] = []
        denominator = max(1, len(group_rows) - 1)
        for index, row in enumerate(group_rows):
            value = _safe_float(row.get(metric))
            if not math.isfinite(value):
                continue
            x = margin_left + int(index * plot_width / denominator)
            y = margin_top + int((vmax - value) * plot_height / (vmax - vmin))
            points.append((x, y))
        color = colors[group_index % len(colors)]
        for a, b in zip(points, points[1:]):
            _draw_line(canvas, width, height, a, b, color)
        for x, y in points:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        offset = ((y + dy) * width + (x + dx)) * 3
                        canvas[offset : offset + 3] = bytes(color)
    _write_png(path, width, height, canvas)
    return True


def _write_line_plot(path: Path, rows: list[dict[str, object]], metric: str, ylabel: str) -> bool:
    if not rows:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _write_fallback_line_plot(path, rows, metric)

    axis_label = ylabel.removesuffix(" by Repeat")
    groups: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        subject = str(row["subject"])
        condition = str(row["condition"])
        groups.setdefault(subject, {}).setdefault(condition, []).append(row)

    subjects = sorted(groups)
    columns = 2 if len(subjects) > 1 else 1
    rows_count = math.ceil(len(subjects) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(12, max(4.0, rows_count * 3.0)),
        squeeze=False,
    )
    colors = {"remesh": "#2563eb", "fixed_mesh": "#dc2626"}
    labels_seen: set[str] = set()
    legend_handles = []
    legend_labels = []
    condition_order = ("remesh", "fixed_mesh")

    for subject_index, subject in enumerate(subjects):
        axis = axes.flat[subject_index]
        subject_groups = groups[subject]
        ordered_conditions = [name for name in condition_order if name in subject_groups]
        ordered_conditions.extend(sorted(set(subject_groups) - set(ordered_conditions)))
        max_repeat = 1
        for condition in ordered_conditions:
            group_rows = sorted(subject_groups[condition], key=lambda row: _repeat_number(row["repeat_tag"]))
            repeat_numbers = [_repeat_number(row["repeat_tag"]) for row in group_rows]
            max_repeat = max(max_repeat, *(repeat_numbers or [1]))
            line = axis.plot(
                repeat_numbers,
                [_safe_float(row[metric]) for row in group_rows],
                color=colors.get(condition),
                marker="o",
                markersize=2.5,
                linewidth=1.25,
                label=condition,
            )[0]
            if condition not in labels_seen:
                labels_seen.add(condition)
                legend_handles.append(line)
                legend_labels.append(condition)
        tick_step = 10 if max_repeat >= 20 else max(1, max_repeat // 4)
        ticks = list(range(1, max_repeat + 1, tick_step))
        if max_repeat not in ticks:
            ticks.append(max_repeat)
        axis.set_xticks(ticks)
        axis.set_title(subject, fontsize=10)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
        axis.tick_params(labelsize=8)
        if subject_index % columns == 0:
            axis.set_ylabel(axis_label, fontsize=9)
        if subject_index // columns == rows_count - 1:
            axis.set_xlabel("Repeat", fontsize=9)

    for unused_index in range(len(subjects), rows_count * columns):
        axes.flat[unused_index].set_visible(False)

    figure.suptitle(ylabel, fontsize=14)
    if legend_handles:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.975),
        )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _write_primary_repeat_distribution(path: Path, rows: list[dict[str, object]]) -> bool:
    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = _safe_float(row.get("median_roi"))
        if not math.isfinite(value):
            continue
        subject = str(row["subject"])
        condition = str(row["condition"])
        groups.setdefault(subject, {}).setdefault(condition, []).append(value)
    subjects = [
        subject
        for subject in sorted(groups)
        if groups[subject].get("remesh") and groups[subject].get("fixed_mesh")
    ]
    if not subjects:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _write_fallback_line_plot(path, rows, "median_roi")

    blue = "#1f77b4"
    orange = "#ff7f0e"
    light_gray = "#d9d9d9"
    rng = random.Random(42)
    x_positions = list(range(len(subjects)))
    with plt.rc_context(
        {
            "font.size": 15,
            "axes.titlesize": 22,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(16, 9))
        for index, subject in enumerate(subjects):
            remesh = groups[subject]["remesh"]
            fixed = groups[subject]["fixed_mesh"]
            remesh_x = [x_positions[index] - 0.08 + rng.gauss(0.0, 0.045) for _ in remesh]
            fixed_x = [x_positions[index] + 0.14 + rng.gauss(0.0, 0.045) for _ in fixed]
            axis.scatter(
                remesh_x,
                remesh,
                color=blue,
                alpha=0.45,
                s=34,
                linewidths=0,
                label="Remesh repeats" if index == 0 else None,
            )
            axis.scatter(
                fixed_x,
                fixed,
                color=orange,
                alpha=0.52,
                s=28,
                linewidths=0,
                label="Fixed-mesh repeats" if index == 0 else None,
            )
            axis.errorbar(
                x_positions[index] - 0.08,
                statistics.fmean(remesh),
                yerr=statistics.stdev(remesh) if len(remesh) > 1 else 0.0,
                color=blue,
                marker="o",
                markersize=7,
                capsize=4,
                linewidth=2,
                label="Remesh mean +/- SD" if index == 0 else None,
            )
            axis.errorbar(
                x_positions[index] + 0.14,
                statistics.fmean(fixed),
                yerr=statistics.stdev(fixed) if len(fixed) > 1 else 0.0,
                color=orange,
                marker="D",
                markersize=7,
                capsize=4,
                linewidth=2,
                label="Fixed-mesh mean +/- SD" if index == 0 else None,
            )

        axis.set_title("Hippocampal TI varies across remeshed runs but not fixed-mesh runs", pad=18)
        axis.set_ylabel("Median left-hippocampus TI (V/m)")
        axis.set_xlabel("Subject")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [subject.removeprefix("sub-CC") for subject in subjects],
            rotation=35,
            ha="right",
        )
        axis.grid(axis="y", color=light_gray, linewidth=0.8)
        axis.legend(frameon=False, loc="upper right")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return True


def make_figures(*, experiment_root: Path, output_dir: Path | None = None) -> dict[str, object]:
    output_dir = output_dir or experiment_root / "_figures" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((experiment_root / "_analysis").glob("sub-*/condition_comparison.json")):
        aggregate_paired_analysis.aggregate_paired_summary(experiment_root=experiment_root)
    condition_rows = _collect_condition_rows(experiment_root)
    condition_summary = output_dir / "presentation_condition_summary.csv"
    _write_csv(
        condition_summary,
        condition_rows,
        ["subject", "condition", "repeat_tag", "median_roi", "mean_roi", "p95_roi", "peak_roi", "p95_head", "mesh_nodes"],
    )

    figures = []
    primary_figure = output_dir / "01_primary_median_roi_repeat_distributions.png"
    if _write_primary_repeat_distribution(primary_figure, condition_rows):
        figures.append(str(primary_figure))
    if _write_line_plot(
        output_dir / "condition_median_roi_by_repeat.png",
        condition_rows,
        "median_roi",
        "Median ROI TI by Repeat",
    ):
        figures.append(str(output_dir / "condition_median_roi_by_repeat.png"))
    if _write_line_plot(
        output_dir / "condition_mesh_nodes_by_repeat.png",
        condition_rows,
        "mesh_nodes",
        "Mesh Nodes by Repeat",
    ):
        figures.append(str(output_dir / "condition_mesh_nodes_by_repeat.png"))

    paired_source = experiment_root / "_analysis" / "paired_condition_summary.csv"
    paired_dest = output_dir / "presentation_paired_condition_summary.csv"
    if paired_source.is_file():
        shutil.copyfile(paired_source, paired_dest)

    manifest = {
        "experiment_root": str(experiment_root),
        "output_dir": str(output_dir),
        "condition_rows": len(condition_rows),
        "figures": figures,
        "figures_written": len(figures),
        "summary_csv": str(condition_summary),
        "paired_summary_csv": str(paired_dest) if paired_dest.is_file() else None,
    }
    (output_dir / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = make_figures(
        experiment_root=args.experiment_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
