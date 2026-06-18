#!/usr/bin/env python3
"""Create lightweight presentation figures from completed repeatability analysis."""

from __future__ import annotations

import argparse
import binascii
import csv
import json
import math
import shutil
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
                    "peak_roi": row.get("peak_roi", ""),
                    "mesh_nodes": row.get("mesh_nodes", ""),
                }
            )
    return rows


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


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

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["subject"]), str(row["condition"])), []).append(row)

    plt.figure(figsize=(max(8, len(rows) * 0.18), 4.8))
    for (subject, condition), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda row: str(row["repeat_tag"]))
        plt.plot(
            [str(row["repeat_tag"]) for row in group_rows],
            [_safe_float(row[metric]) for row in group_rows],
            marker="o",
            linewidth=1.5,
            label=f"{subject} {condition}",
        )
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
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
        ["subject", "condition", "repeat_tag", "median_roi", "mean_roi", "peak_roi", "mesh_nodes"],
    )

    figures = []
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
