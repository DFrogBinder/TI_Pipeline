#!/usr/bin/env python3
"""Create lightweight presentation figures from completed repeatability analysis."""

from __future__ import annotations

import argparse
import ast
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


ROI_DISPLAY_NAMES = {
    "left-hippocampus": "left hippocampus",
    "right-hippocampus": "right hippocampus",
    "left-m1": "left M1",
    "right-m1": "right M1",
}

TISSUE_NAMES = {
    1: "White matter",
    2: "Grey matter",
    3: "CSF",
    4: "Bone",
    5: "Scalp",
    6: "Eyes",
    7: "Compact bone",
    8: "Spongy bone",
    9: "Blood",
    10: "Muscle",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_condition_rows(
    experiment_root: Path,
    *,
    mesh_metrics_csv: Path | None = None,
) -> list[dict[str, object]]:
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
                    "mesh_elements": row.get("mesh_elements", ""),
                    "mesh_elements_by_tissue": row.get(
                        "mesh_elements_by_tissue", ""
                    ),
                    "mesh_volume_mm3_by_tissue": row.get(
                        "mesh_volume_mm3_by_tissue", ""
                    ),
                }
            )
    if mesh_metrics_csv is not None:
        mesh_rows = _read_csv(mesh_metrics_csv)
        mesh_fields = [
            "mesh_nodes",
            "mesh_elements",
            "mesh_elements_by_tissue",
            "mesh_volume_mm3_by_tissue",
        ]
        mesh_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
        for mesh_row in mesh_rows:
            key = (
                mesh_row.get("subject", ""),
                mesh_row.get("condition", ""),
                mesh_row.get("repeat_tag", ""),
            )
            if not all(key):
                raise RuntimeError(
                    f"Incomplete mesh-metric key in {mesh_metrics_csv}: {key}"
                )
            if key in mesh_by_key:
                raise RuntimeError(
                    f"Duplicate mesh-metric key in {mesh_metrics_csv}: {key}"
                )
            mesh_by_key[key] = mesh_row
        condition_keys = {
            (
                str(row["subject"]),
                str(row["condition"]),
                str(row["repeat_tag"]),
            )
            for row in rows
        }
        if set(mesh_by_key) != condition_keys:
            missing = condition_keys - set(mesh_by_key)
            unexpected = set(mesh_by_key) - condition_keys
            raise RuntimeError(
                "Mesh-metric overlay does not exactly match the repeatability "
                f"analysis: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        for row in rows:
            key = (
                str(row["subject"]),
                str(row["condition"]),
                str(row["repeat_tag"]),
            )
            overlay = mesh_by_key[key]
            for field in mesh_fields:
                value = overlay.get(field, "")
                if value == "":
                    raise RuntimeError(
                        f"Missing {field} for mesh-metric key {key}"
                    )
                row[field] = value
    return rows


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_mapping(value: object) -> dict[int, float]:
    if isinstance(value, dict):
        source = value
    else:
        text = str(value).strip()
        if not text:
            return {}
        try:
            source = json.loads(text)
        except json.JSONDecodeError:
            try:
                source = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return {}
    result: dict[int, float] = {}
    for key, item in source.items():
        try:
            numeric = float(item)
            integer_key = int(key)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result[integer_key] = numeric
    return result


def _roi_context(experiment_root: Path) -> tuple[str | None, str]:
    manifest_path = experiment_root / "_pipeline" / "experiment_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None, "target ROI"

    roi_preset = manifest.get("roi_preset")
    if not isinstance(roi_preset, str) or not roi_preset.strip():
        stimulation = manifest.get("stimulation")
        if isinstance(stimulation, dict):
            roi_preset = stimulation.get("montage_preset")
    if not isinstance(roi_preset, str) or not roi_preset.strip():
        return None, "target ROI"

    normalized = roi_preset.strip().lower().replace("_", "-").replace(" ", "-")
    return normalized, ROI_DISPLAY_NAMES.get(
        normalized,
        normalized.replace("-", " "),
    )


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
            points = axis.scatter(
                repeat_numbers,
                [_safe_float(row[metric]) for row in group_rows],
                color=colors.get(condition),
                marker="o",
                s=10,
                linewidths=0,
                alpha=0.75,
                label=condition,
            )
            if condition not in labels_seen:
                labels_seen.add(condition)
                legend_handles.append(points)
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


def _write_primary_repeat_distribution(
    path: Path,
    rows: list[dict[str, object]],
    *,
    roi_display_name: str,
    metric: str = "median_roi",
    ylabel: str | None = None,
    title: str | None = None,
) -> bool:
    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = _safe_float(row.get(metric))
        if not math.isfinite(value):
            continue
        subject = str(row["subject"])
        condition = str(row["condition"])
        groups.setdefault(subject, {}).setdefault(condition, []).append(value)
    subjects = [
        subject for subject in groups
        if groups[subject].get("remesh") and groups[subject].get("fixed_mesh")
    ]
    subjects.sort(
        key=lambda subject: statistics.fmean(groups[subject]["remesh"]),
        reverse=True,
    )
    if not subjects:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _write_fallback_line_plot(path, rows, metric)

    blue = "#1f77b4"
    orange = "#ff7f0e"
    dark_blue = "#14527a"
    dark_orange = "#a84f00"
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
        figure, axis = plt.subplots(figsize=(12.6, 8.2))
        for index, subject in enumerate(subjects):
            remesh = groups[subject]["remesh"]
            fixed = groups[subject]["fixed_mesh"]
            remesh_x = [
                x_positions[index] - 0.055 + rng.gauss(0.0, 0.028)
                for _ in remesh
            ]
            fixed_x = [
                x_positions[index] + 0.085 + rng.gauss(0.0, 0.028)
                for _ in fixed
            ]
            axis.scatter(
                remesh_x,
                remesh,
                color=blue,
                alpha=0.28,
                s=31,
                linewidths=0,
                label="Remesh repeats" if index == 0 else None,
            )
            axis.scatter(
                fixed_x,
                fixed,
                color=orange,
                alpha=0.32,
                s=27,
                linewidths=0,
                label="Fixed-mesh repeats" if index == 0 else None,
            )
            axis.errorbar(
                x_positions[index] - 0.055,
                statistics.fmean(remesh),
                yerr=statistics.stdev(remesh) if len(remesh) > 1 else 0.0,
                color=dark_blue,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=dark_blue,
                markeredgewidth=2.0,
                markersize=8.5,
                capsize=4.5,
                capthick=2.4,
                linewidth=2.4,
                label="Remesh mean +/- SD" if index == 0 else None,
                zorder=5,
            )
            axis.errorbar(
                x_positions[index] + 0.085,
                statistics.fmean(fixed),
                yerr=statistics.stdev(fixed) if len(fixed) > 1 else 0.0,
                color=dark_orange,
                marker="D",
                markerfacecolor="white",
                markeredgecolor=dark_orange,
                markeredgewidth=2.0,
                markersize=8.0,
                capsize=4.5,
                capthick=2.4,
                linewidth=2.4,
                label="Fixed-mesh mean +/- SD" if index == 0 else None,
                zorder=5,
            )

        axis.set_title(
            (
                title.format(roi=roi_display_name)
                if title
                else (
                    f"{roi_display_name.capitalize()} TI across remeshed "
                    "and fixed-mesh runs"
                )
            ),
            pad=18,
        )
        axis.set_ylabel(
            ylabel.format(roi=roi_display_name)
            if ylabel
            else f"Median {roi_display_name} TI (V/m)"
        )
        axis.set_xlabel("Subject (ordered by decreasing remesh mean)")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [subject.removeprefix("sub-CC") for subject in subjects],
            rotation=35,
            ha="right",
        )
        axis.grid(axis="y", color=light_gray, linewidth=0.8)
        axis.legend(frameon=False, loc="upper right")
        axis.margins(x=0.025)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return True


def _write_single_repeat_rank_uncertainty(
    figure_path: Path,
    rows: list[dict[str, object]],
    *,
    roi_display_name: str,
    metric: str = "median_roi",
) -> dict[str, object] | None:
    """Quantify subject-order reversals caused by choosing one random repeat."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("condition")) != "remesh":
            continue
        value = _safe_float(row.get(metric))
        if math.isfinite(value):
            groups.setdefault(str(row["subject"]), []).append(value)
    subjects = [subject for subject, values in groups.items() if values]
    subjects.sort(key=lambda subject: statistics.fmean(groups[subject]), reverse=True)
    if len(subjects) < 2:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    count = len(subjects)
    reversal = np.full((count, count), np.nan, dtype=float)
    pair_rows: list[dict[str, object]] = []
    for i in range(count):
        for j in range(i + 1, count):
            left = np.asarray(groups[subjects[i]], dtype=float)
            right = np.asarray(groups[subjects[j]], dtype=float)
            probability = float(np.mean(left[:, None] <= right[None, :]))
            reversal[i, j] = probability
            pair_rows.append(
                {
                    "higher_mean_subject": subjects[i],
                    "lower_mean_subject": subjects[j],
                    "higher_mean_v_per_m": statistics.fmean(groups[subjects[i]]),
                    "lower_mean_v_per_m": statistics.fmean(groups[subjects[j]]),
                    "single_repeat_order_reversal_probability": probability,
                }
            )

    rng = np.random.default_rng(20260730)
    bootstrap_count = 20_000
    tau_values = np.empty(bootstrap_count, dtype=float)
    inversion_counts = np.empty(bootstrap_count, dtype=int)
    denominator = count * (count - 1) / 2
    for draw_index in range(bootstrap_count):
        draw = np.asarray(
            [rng.choice(groups[subject]) for subject in subjects],
            dtype=float,
        )
        inversions = int(
            sum(
                draw[i] <= draw[j]
                for i in range(count)
                for j in range(i + 1, count)
            )
        )
        inversion_counts[draw_index] = inversions
        tau_values[draw_index] = 1.0 - 2.0 * inversions / denominator

    with plt.rc_context(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12.6, 5.3),
            gridspec_kw={"width_ratios": [1.12, 1.0]},
        )
        heatmap = plt.get_cmap("Blues").copy()
        heatmap.set_bad("white")
        image = axes[0].imshow(
            reversal,
            cmap=heatmap,
            vmin=0.0,
            vmax=0.6,
            interpolation="nearest",
        )
        labels = [subject.removeprefix("sub-CC") for subject in subjects]
        axes[0].set_xticks(range(count), labels, rotation=42, ha="right")
        axes[0].set_yticks(range(count), labels)
        axes[0].set_xlabel("Lower mean-ranked subject")
        axes[0].set_ylabel("Higher mean-ranked subject")
        axes[0].set_title("A  Pairwise rank-reversal probability", loc="left")
        colorbar = figure.colorbar(image, ax=axes[0], fraction=0.047, pad=0.04)
        colorbar.set_label("Probability")

        lower_tau = max(-1.0, float(np.min(tau_values)) - 0.04)
        axes[1].hist(
            tau_values,
            bins=np.linspace(lower_tau, 1.005, 18),
            weights=np.full(len(tau_values), 100.0 / len(tau_values)),
            color="#1f77b4",
            alpha=0.78,
            edgecolor="white",
        )
        axes[1].axvline(
            float(np.median(tau_values)),
            color="#14527a",
            lw=2.2,
            label=f"Median = {np.median(tau_values):.2f}",
        )
        axes[1].set_xlim(lower_tau, 1.005)
        axes[1].set_xlabel("Kendall rank agreement with 40-repeat means")
        axes[1].set_ylabel("Random single-repeat selections (%)")
        axes[1].set_title("B  Whole-cohort ordering uncertainty", loc="left")
        axes[1].legend(frameon=False)
        figure.suptitle(
            f"{roi_display_name.capitalize()}: uncertainty from selecting "
            "one remesh repeat",
            fontsize=17,
            y=1.01,
        )
        figure.tight_layout()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(figure)

    pair_csv = figure_path.with_name(
        "single_repeat_pairwise_rank_reversal_probabilities.csv"
    )
    _write_csv(
        pair_csv,
        pair_rows,
        [
            "higher_mean_subject",
            "lower_mean_subject",
            "higher_mean_v_per_m",
            "lower_mean_v_per_m",
            "single_repeat_order_reversal_probability",
        ],
    )
    summary = {
        "metric": metric,
        "condition": "remesh",
        "subjects": subjects,
        "repeats_per_subject": {
            subject: len(groups[subject]) for subject in subjects
        },
        "random_single_repeat_selections": bootstrap_count,
        "median_kendall_rank_agreement": float(np.median(tau_values)),
        "kendall_rank_agreement_iqr": [
            float(np.percentile(tau_values, 25)),
            float(np.percentile(tau_values, 75)),
        ],
        "probability_of_any_subject_order_reversal": float(
            np.mean(inversion_counts > 0)
        ),
        "mean_number_of_pairwise_reversals": float(
            np.mean(inversion_counts)
        ),
        "maximum_pairwise_reversal_probability": float(
            max(
                row["single_repeat_order_reversal_probability"]
                for row in pair_rows
            )
        ),
        "pairwise_csv": str(pair_csv),
        "figure": str(figure_path),
    }
    summary_path = figure_path.with_name(
        "single_repeat_ranking_uncertainty.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_tissue_repeat_distributions(
    path: Path,
    rows: list[dict[str, object]],
    *,
    mapping_field: str,
    value_label: str,
    title: str,
) -> bool:
    """Draw repeat distributions by tissue when element metadata are present."""
    parsed = [
        (
            str(row.get("subject")),
            str(row.get("condition")),
            _safe_mapping(row.get(mapping_field)),
        )
        for row in rows
    ]
    tissue_ids = sorted(
        {
            tissue
            for _subject, _condition, mapping in parsed
            for tissue, value in mapping.items()
            if value > 0
        }
    )
    if not tissue_ids:
        return False
    subjects = sorted(
        {subject for subject, _condition, mapping in parsed if mapping}
    )
    if not subjects:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    columns = 3
    row_count = math.ceil(len(tissue_ids) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(13.2, 3.65 * row_count),
        squeeze=False,
    )
    rng = np.random.default_rng(42)
    colors = {"remesh": "#1f77b4", "fixed_mesh": "#ff7f0e"}
    offsets = {"remesh": -0.08, "fixed_mesh": 0.10}
    for panel_index, tissue_id in enumerate(tissue_ids):
        axis = axes.flat[panel_index]
        for subject_index, subject in enumerate(subjects):
            for condition in ("remesh", "fixed_mesh"):
                values = np.asarray(
                    [
                        mapping[tissue_id]
                        for row_subject, row_condition, mapping in parsed
                        if row_subject == subject
                        and row_condition == condition
                        and tissue_id in mapping
                    ],
                    dtype=float,
                )
                if not values.size:
                    continue
                center = subject_index + offsets[condition]
                axis.scatter(
                    center + rng.uniform(-0.028, 0.028, len(values)),
                    values,
                    s=9,
                    alpha=0.23,
                    color=colors[condition],
                    edgecolors="none",
                )
                axis.scatter(
                    center,
                    values.mean(),
                    s=25,
                    facecolor="white",
                    edgecolor=colors[condition],
                    linewidth=1.3,
                    zorder=4,
                )
        axis.set_title(
            TISSUE_NAMES.get(tissue_id, f"Tissue {tissue_id}"),
            loc="left",
            fontsize=12,
        )
        axis.set_xticks(
            range(len(subjects)),
            [subject.removeprefix("sub-CC") for subject in subjects],
            rotation=42,
            ha="right",
            fontsize=7,
        )
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        if panel_index % columns == 0:
            axis.set_ylabel(value_label)
    for unused in range(len(tissue_ids), row_count * columns):
        axes.flat[unused].set_visible(False)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[condition],
            markeredgecolor="none",
            label=condition.replace("_", " ").title(),
        )
        for condition in ("remesh", "fixed_mesh")
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    figure.suptitle(title, fontsize=17, y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return True


def _write_example_tissue_composition(
    path: Path,
    rows: list[dict[str, object]],
) -> bool:
    """Show 100% tissue composition over repeats for one variable subject."""
    records = []
    for row in rows:
        if str(row.get("condition")) != "remesh":
            continue
        mapping = _safe_mapping(row.get("mesh_volume_mm3_by_tissue"))
        total = sum(mapping.values())
        if mapping and total > 0:
            records.append((str(row["subject"]), str(row["repeat_tag"]), mapping))
    if not records:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    by_subject: dict[str, list[tuple[str, dict[int, float]]]] = {}
    for subject, repeat_tag, mapping in records:
        by_subject.setdefault(subject, []).append((repeat_tag, mapping))
    subject = max(
        by_subject,
        key=lambda item: sum(
            np.std(
                [
                    mapping.get(tissue, 0.0) / sum(mapping.values())
                    for _repeat, mapping in by_subject[item]
                ]
            )
            for tissue in TISSUE_NAMES
        ),
    )
    subject_rows = sorted(
        by_subject[subject],
        key=lambda item: _repeat_number(item[0]),
    )
    tissue_ids = [
        tissue
        for tissue in TISSUE_NAMES
        if any(mapping.get(tissue, 0.0) > 0 for _repeat, mapping in subject_rows)
    ]
    x = np.arange(len(subject_rows))
    bottom = np.zeros(len(subject_rows), dtype=float)
    colors = plt.get_cmap("tab10").colors
    figure, axis = plt.subplots(figsize=(12.6, 5.2))
    for index, tissue in enumerate(tissue_ids):
        fractions = np.asarray(
            [
                100.0 * mapping.get(tissue, 0.0) / sum(mapping.values())
                for _repeat, mapping in subject_rows
            ]
        )
        axis.bar(
            x,
            fractions,
            bottom=bottom,
            width=0.84,
            color=colors[index % len(colors)],
            label=TISSUE_NAMES.get(tissue, f"Tissue {tissue}"),
        )
        bottom += fractions
    axis.set_ylim(0, 100)
    axis.set_ylabel("Mesh volume composition (%)")
    axis.set_xlabel("Remesh repeat")
    tick_step = max(1, len(x) // 10)
    ticks = x[::tick_step]
    axis.set_xticks(
        ticks,
        [str(_repeat_number(subject_rows[index][0])) for index in ticks],
    )
    axis.set_title(
        f"Example subject {subject.removeprefix('sub-CC')}: "
        "tissue composition across remesh repeats"
    )
    axis.legend(
        frameon=False,
        ncol=min(5, len(tissue_ids)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return True


def _condition_repeat_description(
    rows: list[dict[str, object]],
    condition: str,
) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        if str(row.get("condition")) == condition:
            subject = str(row.get("subject"))
            counts[subject] = counts.get(subject, 0) + 1
    unique = sorted(set(counts.values()))
    if not unique:
        return "no available repeats"
    if len(unique) == 1:
        return f"{unique[0]} repeats per subject"
    return f"{min(unique)}–{max(unique)} repeats per subject"


def _repeatability_captions(
    *,
    rows: list[dict[str, object]],
    figures: list[str],
    roi_display_name: str,
    rank_summary: dict[str, object] | None,
) -> dict[str, str]:
    generated = {Path(path).stem for path in figures}
    captions: dict[str, str] = {}
    primary_stem = "01_primary_median_roi_repeat_distributions"
    if primary_stem in generated:
        captions[primary_stem] = (
            f"Repeat-level median TI E-field in the {roi_display_name} target "
            "ROI. Pale blue and orange points show remesh and fixed-mesh "
            "repeats, respectively. Dark open markers show each condition "
            "mean and error bars show ±1 SD. Subjects are ordered by "
            "decreasing remesh mean."
        )
    rank_stem = "02_single_repeat_subject_ranking_uncertainty"
    if rank_stem in generated and rank_summary is not None:
        captions[rank_stem] = (
            f"Uncertainty in the between-subject ordering of median "
            f"{roi_display_name} TI E-field when one remesh repeat is "
            "selected. Panel A shows the probability that each subject pair "
            "reverses its ordering relative to the 40-repeat means; darker "
            "cells indicate less stable ordering. Panel B shows Kendall rank "
            "agreement for 20,000 random selections of one repeat per "
            "subject; +1 denotes identical ordering and the vertical line "
            "marks the median."
        )
    element_stem = "03_primary_mesh_element_repeat_distributions"
    if element_stem in generated:
        captions[element_stem] = (
            "Repeat-level total tetrahedral element count. Pale blue and "
            "orange points show remesh and fixed-mesh repeats, respectively. "
            "Dark open markers show each condition mean and error bars show "
            "±1 SD. Subjects are ordered by decreasing remesh mean."
        )
    tissue_element_stem = "04_tissue_element_repeat_distributions"
    if tissue_element_stem in generated:
        captions[tissue_element_stem] = (
            "Tissue-specific tetrahedral element counts. Each panel represents "
            "one SimNIBS tissue tag; pale points show repeat values and open "
            "markers show condition means."
        )
    tissue_volume_stem = "05_tissue_volume_repeat_distributions"
    if tissue_volume_stem in generated:
        captions[tissue_volume_stem] = (
            "Tissue-specific tetrahedral mesh volume. Each panel represents "
            "one SimNIBS tissue tag; pale points show repeat values in mm³ and "
            "open markers show condition means."
        )
    composition_stem = "06_example_subject_tissue_composition"
    if composition_stem in generated:
        captions[composition_stem] = (
            "Tissue-volume composition across remesh repeats for the subject "
            "with the largest within-subject variation. Each bar is one "
            "repeat and sums to 100%; colours identify SimNIBS tissue tags."
        )
    return captions


def _write_caption_outputs(
    output_dir: Path,
    caption_values: dict[str, str],
) -> tuple[Path, Path]:
    caption_csv = output_dir / "figure_captions.csv"
    caption_md = output_dir / "figure_captions.md"
    rows = [
        {"figure": stem, "caption": caption}
        for stem, caption in caption_values.items()
    ]
    _write_csv(caption_csv, rows, ["figure", "caption"])
    lines = ["# Self-contained repeatability figure captions", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['figure']}",
                "",
                str(row["caption"]),
                "",
            ]
        )
    caption_md.write_text("\n".join(lines), encoding="utf-8")
    return caption_md, caption_csv


def make_figures(
    *,
    experiment_root: Path,
    output_dir: Path | None = None,
    mesh_metrics_csv: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir or experiment_root / "_figures" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_preset, roi_display_name = _roi_context(experiment_root)
    paired_source = experiment_root / "_analysis" / "paired_condition_summary.csv"
    if (
        not paired_source.is_file()
        and any(
            (experiment_root / "_analysis").glob(
                "sub-*/condition_comparison.json"
            )
        )
    ):
        aggregate_paired_analysis.aggregate_paired_summary(experiment_root=experiment_root)
    condition_rows = _collect_condition_rows(
        experiment_root,
        mesh_metrics_csv=mesh_metrics_csv,
    )
    condition_summary = output_dir / "presentation_condition_summary.csv"
    _write_csv(
        condition_summary,
        condition_rows,
        [
            "subject",
            "condition",
            "repeat_tag",
            "median_roi",
            "mean_roi",
            "p95_roi",
            "peak_roi",
            "p95_head",
            "mesh_nodes",
            "mesh_elements",
            "mesh_elements_by_tissue",
            "mesh_volume_mm3_by_tissue",
        ],
    )

    figures = []
    data_availability = {
        "median_roi": any(
            math.isfinite(_safe_float(row.get("median_roi")))
            for row in condition_rows
        ),
        "mesh_elements": any(
            math.isfinite(_safe_float(row.get("mesh_elements")))
            for row in condition_rows
        ),
        "mesh_elements_by_tissue": any(
            bool(_safe_mapping(row.get("mesh_elements_by_tissue")))
            for row in condition_rows
        ),
        "mesh_volume_mm3_by_tissue": any(
            bool(_safe_mapping(row.get("mesh_volume_mm3_by_tissue")))
            for row in condition_rows
        ),
    }
    primary_figure = output_dir / "01_primary_median_roi_repeat_distributions.png"
    if _write_primary_repeat_distribution(
        primary_figure,
        condition_rows,
        roi_display_name=roi_display_name,
    ):
        figures.append(str(primary_figure))
    rank_figure = output_dir / "02_single_repeat_subject_ranking_uncertainty.png"
    rank_summary = _write_single_repeat_rank_uncertainty(
        rank_figure,
        condition_rows,
        roi_display_name=roi_display_name,
    )
    if rank_summary is not None:
        figures.append(str(rank_figure))

    element_figure = (
        output_dir / "03_primary_mesh_element_repeat_distributions.png"
    )
    if _write_primary_repeat_distribution(
        element_figure,
        condition_rows,
        roi_display_name=roi_display_name,
        metric="mesh_elements",
        ylabel="Tetrahedral elements",
        title="Tetrahedral mesh elements across remeshed and fixed-mesh runs",
    ):
        figures.append(str(element_figure))

    tissue_element_figure = (
        output_dir / "04_tissue_element_repeat_distributions.png"
    )
    if _write_tissue_repeat_distributions(
        tissue_element_figure,
        condition_rows,
        mapping_field="mesh_elements_by_tissue",
        value_label="Tetrahedral elements",
        title="Tissue-specific tetrahedral elements across repeats",
    ):
        figures.append(str(tissue_element_figure))

    tissue_volume_figure = (
        output_dir / "05_tissue_volume_repeat_distributions.png"
    )
    if _write_tissue_repeat_distributions(
        tissue_volume_figure,
        condition_rows,
        mapping_field="mesh_volume_mm3_by_tissue",
        value_label=r"Tetrahedral volume (mm$^3$)",
        title="Tissue-specific mesh volume across repeats",
    ):
        figures.append(str(tissue_volume_figure))

    composition_figure = (
        output_dir / "06_example_subject_tissue_composition.png"
    )
    if _write_example_tissue_composition(composition_figure, condition_rows):
        figures.append(str(composition_figure))

    status_note = output_dir / "ELEMENT_TISSUE_DATA_STATUS.md"
    missing = [
        key for key, available in data_availability.items()
        if key != "median_roi" and not available
    ]
    status_lines = [
        "# Element and tissue repeatability data status",
        "",
        (
            "The revised repeatability analysis now extracts tetrahedral "
            "element counts and tetrahedral volumes by SimNIBS tissue tag "
            "directly from each `TI.msh` file."
        ),
        "",
    ]
    if missing:
        status_lines.extend(
            [
                (
                    "The currently downloaded summary archives predate that "
                    "extraction and do not contain: "
                    + ", ".join(f"`{field}`" for field in missing)
                    + "."
                ),
                "",
                (
                    "Those figures are therefore intentionally omitted from "
                    "this render. They require rerunning the analysis stage "
                    "against the retained HPC meshes; no FEM simulation or "
                    "remeshing is required."
                ),
            ]
        )
    else:
        status_lines.append(
            "All requested element-count and tissue-volume inputs were "
            "available and their figures were rendered."
        )
    status_note.write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    supervisor_note = output_dir / "SUPERVISOR_REPEATABILITY_UPDATE.md"
    supervisor_lines = [
        "# Repeatability figure update",
        "",
        (
            "The primary field plot retains the agreed remesh/fixed-mesh "
            "styling. Individual repeat points are lighter, while the "
            "mean ± SD markers are darker and outlined so the mean is "
            "visually distinct. Subjects are ordered by decreasing "
            "40-repeat remesh mean and placed more closely together."
        ),
        "",
    ]
    if rank_summary is not None:
        supervisor_lines.extend(
            [
                "## Single-repeat ranking uncertainty",
                "",
                (
                    "For each pair of subjects, the rank-reversal probability "
                    "is calculated exactly over all 40 × 40 possible pairs "
                    "of remesh repeats. The whole-cohort panel additionally "
                    "draws one repeat independently for every subject 20,000 "
                    "times and compares each resulting subject ordering with "
                    "the ordering based on the 40-repeat means."
                ),
                "",
                (
                    "- Probability that at least one subject pair reverses "
                    "order: "
                    f"{100.0 * float(rank_summary['probability_of_any_subject_order_reversal']):.1f}%"
                ),
                (
                    "- Mean number of pairwise reversals per random "
                    "single-repeat selection: "
                    f"{float(rank_summary['mean_number_of_pairwise_reversals']):.2f}"
                ),
                (
                    "- Median Kendall rank agreement with the 40-repeat "
                    "means: "
                    f"{float(rank_summary['median_kendall_rank_agreement']):.2f}"
                ),
                (
                    "- Largest pairwise rank-reversal probability: "
                    f"{100.0 * float(rank_summary['maximum_pairwise_reversal_probability']):.1f}%"
                ),
                "",
            ]
        )
    supervisor_lines.extend(
        [
            "## Mesh element and tissue analyses",
            "",
            (
                "The analysis now extracts tetrahedral element counts and "
                "tetrahedral volume by tissue directly from each `TI.msh`. "
                "See `ELEMENT_TISSUE_DATA_STATUS.md` for whether those fields "
                "were present in this downloaded analysis bundle."
            ),
        ]
    )
    supervisor_note.write_text(
        "\n".join(supervisor_lines) + "\n",
        encoding="utf-8",
    )

    paired_dest = output_dir / "presentation_paired_condition_summary.csv"
    if paired_source.is_file():
        shutil.copyfile(paired_source, paired_dest)

    caption_values = _repeatability_captions(
        rows=condition_rows,
        figures=figures,
        roi_display_name=roi_display_name,
        rank_summary=rank_summary,
    )
    caption_md, caption_csv = _write_caption_outputs(
        output_dir,
        caption_values,
    )
    manifest = {
        "experiment_root": str(experiment_root),
        "output_dir": str(output_dir),
        "mesh_metrics_csv": (
            str(mesh_metrics_csv) if mesh_metrics_csv is not None else None
        ),
        "roi_preset": roi_preset,
        "roi_display_name": roi_display_name,
        "condition_rows": len(condition_rows),
        "figures": figures,
        "figures_written": len(figures),
        "data_availability": data_availability,
        "rank_uncertainty_summary": rank_summary,
        "element_tissue_status_note": str(status_note),
        "supervisor_update_note": str(supervisor_note),
        "captions": len(caption_values),
        "caption_markdown": str(caption_md),
        "caption_csv": str(caption_csv),
        "summary_csv": str(condition_summary),
        "paired_summary_csv": str(paired_dest) if paired_dest.is_file() else None,
    }
    (output_dir / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mesh-metrics-csv", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = make_figures(
        experiment_root=args.experiment_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        mesh_metrics_csv=(
            args.mesh_metrics_csv.expanduser().resolve()
            if args.mesh_metrics_csv
            else None
        ),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
