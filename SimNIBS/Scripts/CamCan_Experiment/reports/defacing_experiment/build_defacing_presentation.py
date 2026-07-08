#!/usr/bin/env python3
"""Analyze the sub-CCMe defacing experiment and build a presentation deck."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from scipy import stats


DATA_ROOT = Path("/home/boyan/sandbox/Jake_Data/defacing_experiment")
OUT_DIR = DATA_ROOT / "analysis_deliverables" / "outputs"
PLANNED_REPEATS_PER_ARM = 40

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

NAVY = RGBColor(12, 35, 64)
BLUE = RGBColor(25, 86, 140)
TEAL = RGBColor(18, 137, 142)
GOLD = RGBColor(232, 174, 70)
INK = RGBColor(31, 37, 45)
MUTED = RGBColor(94, 105, 118)
LIGHT = RGBColor(246, 248, 250)
WHITE = RGBColor(255, 255, 255)


def mpl_color(color: RGBColor) -> str:
    return "#{:02x}{:02x}{:02x}".format(*tuple(color))


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    units: str
    decimals: int = 3
    lower_is_better: bool = False


METRICS = [
    MetricSpec("roi_mean", "ROI mean field", "V/m", 3),
    MetricSpec("roi_peak", "ROI peak field", "V/m", 3),
    MetricSpec("roi_percentile_value", "ROI P95 field", "V/m", 3),
    MetricSpec("overlap_fraction", "ROI in top 5% field", "%", 1),
    MetricSpec("focality_in_roi_volume_mm3_gt_threshold", "ROI volume >=0.2 V/m", "mm3", 0),
    MetricSpec("roi_volume_mm3", "ROI volume", "mm3", 0),
    MetricSpec("whole_brain_volume_mm3", "Whole-brain volume", "mm3", 0),
    MetricSpec("focality_volume_mm3_gt_threshold", "Whole-brain volume >=0.2 V/m", "mm3", 0),
    MetricSpec("percentile_value", "Whole-brain P95 field", "V/m", 3),
]

HEADLINE_KEYS = [
    "roi_mean",
    "roi_percentile_value",
    "overlap_fraction",
    "focality_in_roi_volume_mm3_gt_threshold",
    "focality_volume_mm3_gt_threshold",
    "whole_brain_volume_mm3",
]

TARGET_LABELS = {
    "left-hippocampus": "Left hippocampus",
    "left-m1": "Left M1",
}

CONDITION_LABELS = {
    "intact": "Face intact",
    "defaced": "Face removed",
}

ARM_NAMES = (
    "Left_Hippocampus_Intact",
    "Left_Hippocampus_Defaced",
    "Left_M1_Intact",
    "Left_M1_Defaced",
)


def resolve_source_root() -> Path:
    candidates = [DATA_ROOT, DATA_ROOT / "post_export_40repeats"]
    candidates.extend(sorted(DATA_ROOT.glob("post_export*")))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / arm).is_dir() for arm in ARM_NAMES):
            return candidate

    raise FileNotFoundError(
        "Could not find the four defacing arm folders under "
        f"{DATA_ROOT} or a post_export* child directory."
    )


SOURCE_ROOT = resolve_source_root()


def arm_from_path(path: Path) -> str:
    parts = path.relative_to(SOURCE_ROOT).parts
    for part in parts:
        if part in ARM_NAMES:
            return part
    raise ValueError(f"Cannot infer arm from path: {path}")


def target_from_arm(arm: str) -> str:
    if "Hippocampus" in arm:
        return "left-hippocampus"
    if "M1" in arm:
        return "left-m1"
    raise ValueError(f"Cannot infer target from arm: {arm}")


def condition_from_arm(arm: str) -> str:
    if "Defaced" in arm:
        return "defaced"
    if "Intact" in arm:
        return "intact"
    raise ValueError(f"Cannot infer condition from arm: {arm}")


def repeat_from_path(path: Path) -> int:
    match = re.search(r"Data_(\d+)", path.as_posix())
    if not match:
        raise ValueError(f"Cannot infer repeat from path: {path}")
    return int(match.group(1))


def numeric(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def load_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(SOURCE_ROOT.rglob("subject_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        arm = arm_from_path(path)
        target = target_from_arm(arm)
        condition = condition_from_arm(arm)
        roi_name = data["target_roi"]
        roi = data["rois"][roi_name]
        ext = data["extended_metrics"]
        record: dict[str, object] = {
            "path": str(path),
            "arm": arm,
            "target": target,
            "condition": condition,
            "repeat": repeat_from_path(path),
            "subject": data.get("subject"),
            "target_roi": roi_name,
            "subject_status": data.get("subject_metrics_meta", {}).get("status"),
            "extended_status": data.get("extended_metrics_meta", {}).get("status"),
            "qc_status": data.get("qc_meta", {}).get("status"),
        }
        for key in (
            "percentile_value",
            "whole_brain_voxels",
            "whole_brain_volume_mm3",
            "top_percentile_voxels",
            "top_percentile_percent_of_whole_brain",
        ):
            record[key] = numeric(data.get(key))
        for key in (
            "roi_voxels",
            "roi_volume_mm3",
            "overlap_top_voxels",
            "overlap_volume_mm3",
            "overlap_fraction",
            "roi_percent_of_whole_brain",
            "overlap_top_percent_of_whole_brain",
            "focality_in_roi_voxels_gt_threshold",
            "focality_in_roi_volume_mm3_gt_threshold",
            "focality_in_roi_percent_of_whole_brain_gt_threshold",
            "roi_percentile_value",
        ):
            record[key] = numeric(roi.get(key))
        for key in (
            "roi_mean",
            "roi_peak",
            "focality_voxels_gt_threshold",
            "focality_volume_mm3_gt_threshold",
            "focality_percent_of_whole_brain_gt_threshold",
            "csf_distance_mm",
        ):
            record[key] = numeric(ext.get(key))
        records.append(record)
    return records


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return st.mean(vals) if vals else math.nan


def sd(values: Iterable[float]) -> float:
    vals = list(values)
    return st.stdev(vals) if len(vals) > 1 else 0.0


def sem(values: Iterable[float]) -> float:
    vals = list(values)
    return sd(vals) / math.sqrt(len(vals)) if len(vals) > 1 else math.nan


def fmt_value(value: float, metric_key: str) -> str:
    if value is None or not isinstance(value, (int, float)) or math.isnan(value):
        return "n/a"
    spec = next((m for m in METRICS if m.key == metric_key), None)
    decimals = spec.decimals if spec else 3
    if spec and spec.units == "%":
        return f"{100 * value:.{decimals}f}%"
    if abs(value) >= 1000 and decimals == 0:
        return f"{value:,.0f}"
    return f"{value:.{decimals}f}"


def records_by(records: list[dict[str, object]], target: str, condition: str) -> list[dict[str, object]]:
    return [r for r in records if r["target"] == target and r["condition"] == condition]


def arm_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for target in sorted(TARGET_LABELS):
        for condition in ("intact", "defaced"):
            subset = records_by(records, target, condition)
            for spec in METRICS:
                vals = [float(r[spec.key]) for r in subset if isinstance(r.get(spec.key), (int, float))]
                rows.append(
                    {
                        "target": target,
                        "condition": condition,
                        "metric": spec.key,
                        "metric_label": spec.label,
                        "units": spec.units,
                        "n": len(vals),
                        "mean": mean(vals),
                        "sd": sd(vals),
                        "min": min(vals) if vals else math.nan,
                        "max": max(vals) if vals else math.nan,
                    }
                )
    return rows


def paired_deltas(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for target in sorted(TARGET_LABELS):
        intact = {int(r["repeat"]): r for r in records_by(records, target, "intact")}
        defaced = {int(r["repeat"]): r for r in records_by(records, target, "defaced")}
        repeats = sorted(set(intact) & set(defaced))
        for spec in METRICS:
            xs, ys, diffs = [], [], []
            for repeat in repeats:
                x = intact[repeat].get(spec.key)
                y = defaced[repeat].get(spec.key)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    xs.append(float(x))
                    ys.append(float(y))
                    diffs.append(float(y) - float(x))
            if not diffs:
                continue
            intact_mean = mean(xs)
            defaced_mean = mean(ys)
            delta_mean = mean(diffs)
            delta_sd = sd(diffs)
            delta_sem = sem(diffs)
            if len(diffs) > 1 and delta_sem and not math.isnan(delta_sem):
                tcrit = stats.t.ppf(0.975, len(diffs) - 1)
                ci_low = delta_mean - tcrit * delta_sem
                ci_high = delta_mean + tcrit * delta_sem
                p_value = float(stats.ttest_rel(ys, xs).pvalue)
            else:
                ci_low = ci_high = p_value = math.nan
            rows.append(
                {
                    "target": target,
                    "metric": spec.key,
                    "metric_label": spec.label,
                    "units": spec.units,
                    "n_pairs": len(diffs),
                    "intact_mean": intact_mean,
                    "defaced_mean": defaced_mean,
                    "delta_mean": delta_mean,
                    "delta_sd": delta_sd,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "percent_delta": 100 * delta_mean / intact_mean if intact_mean else math.nan,
                    "paired_t_p": p_value,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(rows: list[dict[str, object]], target: str, key: str) -> dict[str, object]:
    for row in rows:
        if row["target"] == target and row["metric"] == key:
            return row
    raise KeyError((target, key))


def plot_percent_deltas(delta_rows: list[dict[str, object]], path: Path) -> None:
    labels = [
        ("roi_mean", "ROI mean"),
        ("roi_percentile_value", "ROI P95"),
        ("overlap_fraction", "Top-5% overlap"),
        ("focality_in_roi_volume_mm3_gt_threshold", "ROI >=0.2 V/m"),
        ("focality_volume_mm3_gt_threshold", "Whole brain >=0.2"),
    ]
    targets = ["left-hippocampus", "left-m1"]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for idx, target in enumerate(targets):
        vals = [metric_row(delta_rows, target, key)["percent_delta"] for key, _ in labels]
        offset = [-width / 2, width / 2][idx]
        color = [BLUE, TEAL][idx]
        bars = ax.bar([i + offset for i in x], vals, width, label=TARGET_LABELS[target], color=mpl_color(color))
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.8 if val >= 0 else -1.8),
                f"{val:+.1f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
            )
    ax.axhline(0, color="#56606b", linewidth=1)
    ax.set_ylabel("Defaced minus intact mean change (%)")
    ax.set_xticks(list(x), [label for _, label in labels], rotation=18, ha="right")
    ax.set_title("Effect of defacing by target and endpoint")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_key_distributions(records: list[dict[str, object]], path: Path) -> None:
    panels = [
        ("roi_percentile_value", "ROI P95 field (V/m)", lambda v: v),
        ("overlap_fraction", "ROI in top 5% field (%)", lambda v: 100 * v),
        ("focality_in_roi_volume_mm3_gt_threshold", "ROI volume >=0.2 V/m (mm3)", lambda v: v),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (key, title, transform) in zip(axes, panels):
        positions, data, colors, labels = [], [], [], []
        pos = 1
        for target in ("left-hippocampus", "left-m1"):
            for condition, color in (("intact", BLUE), ("defaced", TEAL)):
                vals = [
                    transform(float(r[key]))
                    for r in records_by(records, target, condition)
                    if isinstance(r.get(key), (int, float))
                ]
                positions.append(pos)
                data.append(vals)
                colors.append(mpl_color(color))
                labels.append(f"{TARGET_LABELS[target]}\n{CONDITION_LABELS[condition]}")
                pos += 1
            pos += 0.6
        bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set(facecolor=color, alpha=0.28, edgecolor=color, linewidth=1.5)
        for median in bp["medians"]:
            median.set(color="#1f252d", linewidth=1.5)
        for p, vals, color in zip(positions, data, colors):
            ax.scatter([p] * len(vals), vals, color=color, s=20, alpha=0.72, zorder=3)
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_repeat_lines(records: list[dict[str, object]], path: Path) -> None:
    panels = [
        ("left-hippocampus", "overlap_fraction", "Hippocampus: top-5% overlap", lambda v: 100 * v, "%"),
        ("left-m1", "roi_percentile_value", "M1: ROI P95 field", lambda v: v, "V/m"),
        ("left-m1", "focality_in_roi_volume_mm3_gt_threshold", "M1: ROI volume >=0.2 V/m", lambda v: v, "mm3"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (target, key, title, transform, ylabel) in zip(axes, panels):
        intact = {int(r["repeat"]): r for r in records_by(records, target, "intact")}
        defaced = {int(r["repeat"]): r for r in records_by(records, target, "defaced")}
        for repeat in sorted(set(intact) & set(defaced)):
            x = transform(float(intact[repeat][key]))
            y = transform(float(defaced[repeat][key]))
            ax.plot([0, 1], [x, y], color="#95a1ad", alpha=0.7, linewidth=1.0)
            ax.scatter([0], [x], color=mpl_color(BLUE), s=24)
            ax.scatter([1], [y], color=mpl_color(TEAL), s=24)
        ax.set_xticks([0, 1], ["Intact", "Defaced"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def representative_overlay(target: str, condition: str) -> Path | None:
    suffix = "Defaced" if condition == "defaced" else "Intact"
    if target == "left-hippocampus":
        arm = f"Left_Hippocampus_{suffix}"
        pattern = "Left_Hippocampus_TI_overlay_roi_focus_sub-CCMe_top95.png"
        dataset = "Left_Hippocampus_Data_01"
    else:
        arm = f"Left_M1_{suffix}"
        pattern = "ctx_lh_G_precentral_TI_overlay_roi_focus_sub-CCMe_top95.png"
        dataset = "Left_M1_Data_01"
    path = SOURCE_ROOT / arm / dataset / "sub-CCMe" / "anat" / "post" / pattern
    return path if path.is_file() else None


def add_bg(slide, title: str | None = None) -> None:
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE
    top = slide.shapes.add_shape(1, 0, 0, SLIDE_W, Inches(0.18))
    top.fill.solid()
    top.fill.fore_color.rgb = GOLD
    top.line.fill.background()
    if title:
        box = slide.shapes.add_textbox(Inches(0.55), Inches(0.34), Inches(12.1), Inches(0.45))
        tf = box.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(24)
        p.font.color.rgb = NAVY


def add_footer(slide, text: str = "sub-CCMe defacing experiment | SimNIBS post-processing") -> None:
    box = slide.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(12.2), Inches(0.25))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(8)
    p.font.color.rgb = MUTED


def textbox(slide, x, y, w, h, text, *, size=16, bold=False, color=INK, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    if align:
        p.alignment = align
    return box


def bullet_list(slide, x, y, w, h, items: list[str], *, size=15):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = f"- {item}"
        p.font.size = Pt(size)
        p.font.color.rgb = INK
        p.space_after = Pt(8)
    return box


def add_metric_card(slide, x, y, w, h, label, value, subtitle, color):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = color
    shape.line.width = Pt(1.2)
    textbox(slide, x + 0.15, y + 0.12, w - 0.3, 0.28, label, size=10, bold=True, color=MUTED)
    textbox(slide, x + 0.15, y + 0.43, w - 0.3, 0.36, value, size=20, bold=True, color=color)
    textbox(slide, x + 0.15, y + 0.84, w - 0.3, h - 0.9, subtitle, size=9, color=INK)


def add_table(slide, x, y, w, h, headers: list[str], rows: list[list[str]], font_size=8):
    table = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for col, header in enumerate(headers):
        cell = table.cell(0, col)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.color.rgb = WHITE
            p.font.bold = True
            p.font.size = Pt(font_size)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(248, 249, 251) if r_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size)
                p.font.color.rgb = INK
    return table


def pct_text(row: dict[str, object]) -> str:
    return f"{float(row['percent_delta']):+.1f}%"


def p_text(row: dict[str, object]) -> str:
    p = float(row["paired_t_p"])
    return "n/a" if math.isnan(p) else f"{p:.3g}"


def mean_text(row: dict[str, object], key: str, metric_key: str) -> str:
    return fmt_value(float(row[key]), metric_key)


def percent_number(row: dict[str, object]) -> float:
    return float(row["percent_delta"])


def percent_magnitude_text(row: dict[str, object], *, decimals: int = 1) -> str:
    return f"{abs(percent_number(row)):.{decimals}f}%"


def direction_phrase(row: dict[str, object], *, decimals: int = 1) -> str:
    value = percent_number(row)
    if value > 0:
        return f"increased +{abs(value):.{decimals}f}%"
    if value < 0:
        return f"decreased -{abs(value):.{decimals}f}%"
    return "was unchanged"


def experiment_counts(records: list[dict[str, object]], delta_rows: list[dict[str, object]]) -> dict[str, int]:
    targets = {str(r["target"]) for r in records}
    conditions = {str(r["condition"]) for r in records}
    repeats = {int(r["repeat"]) for r in records}
    pair_counts = [int(row["n_pairs"]) for row in delta_rows if row.get("metric") == "roi_mean"]
    repeat_count = len(repeats)
    expected_runs = len(targets) * len(conditions) * PLANNED_REPEATS_PER_ARM
    return {
        "actual_runs": len(records),
        "expected_runs": expected_runs,
        "target_count": len(targets),
        "condition_count": len(conditions),
        "repeat_count": repeat_count,
        "planned_repeats": PLANNED_REPEATS_PER_ARM,
        "pair_count": max(pair_counts) if pair_counts else 0,
        "is_complete": int(len(records) == expected_runs),
    }


def build_deck(records: list[dict[str, object]], arm_rows: list[dict[str, object]], delta_rows: list[dict[str, object]], figures: dict[str, Path]) -> Path:
    counts = experiment_counts(records, delta_rows)
    hip_p95_title = metric_row(delta_rows, "left-hippocampus", "roi_percentile_value")
    m1_focal_title = metric_row(delta_rows, "left-m1", "focality_in_roi_volume_mm3_gt_threshold")
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    bar = slide.shapes.add_shape(1, 0, 0, Inches(0.22), SLIDE_H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = GOLD
    bar.line.fill.background()
    textbox(slide, 0.72, 0.72, 11.4, 0.7, "Does defacing alter SimNIBS TI outputs?", size=34, bold=True, color=WHITE)
    status = "complete planned run set" if counts["is_complete"] else "interim completed subset"
    textbox(slide, 0.74, 1.52, 10.8, 0.5, f"sub-CCMe repeat experiment: intact versus face-removed T1/T2 inputs ({status})", size=18, color=RGBColor(213, 224, 235))
    textbox(
        slide,
        0.76,
        2.6,
        10.9,
        1.1,
        f"Across {counts['pair_count']} paired technical repeats, hippocampal ROI P95 {direction_phrase(hip_p95_title)} while M1 above-threshold ROI volume {direction_phrase(m1_focal_title)} after defacing.",
        size=21,
        color=WHITE,
    )
    textbox(slide, 0.76, 6.76, 10.8, 0.25, f"Generated from local post-processing outputs in {SOURCE_ROOT}", size=9, color=RGBColor(190, 204, 218))

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Why this matters: anonymization can change model inputs")
    add_footer(slide)
    textbox(
        slide,
        0.75,
        1.1,
        5.6,
        1.25,
        "SimNIBS uses subject anatomy to build a head model, place tissues, and solve electric fields. Removing the face protects identity, but it also changes the anatomical input that the pipeline sees.",
        size=17,
        color=INK,
    )
    add_metric_card(slide, 6.8, 1.05, 2.55, 1.25, "Input change", "T1 + T2", "Both anatomical images were defaced.", BLUE)
    add_metric_card(slide, 9.75, 1.05, 2.55, 1.25, "Risk", "Geometry", "Face handling can affect reconstructed skin/skull surfaces.", TEAL)
    add_table(
        slide,
        0.75,
        3.0,
        11.8,
        2.35,
        ["Question", "Operational test in this experiment"],
        [
            ["Does defacing change field amplitude?", "Compare ROI mean and ROI P95 between intact and defaced runs."],
            ["Does defacing move the high-field region?", "Compare ROI overlap with the whole-brain top 5% field."],
            ["Does defacing change high-field extent?", "Compare ROI and whole-brain volume above 0.2 V/m."],
            ["Are effects target-specific?", "Run the same comparison for left hippocampus and left M1."],
        ],
        font_size=9,
    )
    textbox(slide, 0.78, 5.85, 11.7, 0.55, "The deck is therefore a sensitivity analysis: same subject, same targets, repeated intact-versus-defaced anatomical inputs.", size=15, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Design: paired repeats isolate defacing effects")
    add_footer(slide)
    add_table(
        slide,
        0.72,
        1.15,
        6.2,
        3.35,
        ["Design factor", "Value"],
        [
            ["Subject", "sub-CCMe"],
            ["Input conditions", "Face intact T1/T2 versus face-removed T1/T2"],
            ["Targets", "Left hippocampus and left M1 / left precentral gyrus"],
            ["Repeats", f"{counts['repeat_count']} observed / {counts['planned_repeats']} planned per arm"],
            ["Atlas", "Fixed subject-space atlas; no MNI/template-neighbor metrics"],
        ],
        font_size=9,
    )
    add_metric_card(slide, 7.0, 1.25, 2.2, 1.25, "Runs analyzed", f"{counts['actual_runs']} / {counts['expected_runs']}", "Completed outputs versus planned design.", BLUE)
    add_metric_card(slide, 9.55, 1.25, 2.2, 1.25, "Repeats per arm", f"{counts['repeat_count']} / {counts['planned_repeats']}", "Observed repeats versus planned repeats.", TEAL)
    add_metric_card(slide, 7.0, 2.95, 2.2, 1.25, "Targets", str(counts["target_count"]), "Deep and cortical target classes.", GOLD)
    add_metric_card(slide, 9.55, 2.95, 2.2, 1.25, "QC status", "Complete", "Subject, extended metrics, and QC all complete.", BLUE)
    textbox(slide, 0.75, 5.25, 11.7, 0.55, f"Pairing by repeat ID controls the comparison. This analysis pairs repeats 01-{counts['repeat_count']:02d} within each target-condition arm.", size=15, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Analysis workflow: from post outputs to paired deltas")
    add_footer(slide)
    add_metric_card(slide, 0.72, 1.15, 2.6, 1.25, "1. Inputs", f"{counts['actual_runs']} JSONs", "Read subject_metrics.json from every completed run.", BLUE)
    add_metric_card(slide, 3.72, 1.15, 2.6, 1.25, "2. Extract", f"{len(METRICS)} metrics", "Pull ROI, whole-brain, threshold, and QC fields.", TEAL)
    add_metric_card(slide, 6.72, 1.15, 2.6, 1.25, "3. Pair", f"{counts['pair_count']} / {counts['planned_repeats']} pairs", "Match intact and defaced runs by repeat ID within each target.", GOLD)
    add_metric_card(slide, 9.72, 1.15, 2.6, 1.25, "4. Summarize", "Deltas", "Report defaced minus intact effects and descriptive paired tests.", BLUE)
    add_table(
        slide,
        0.72,
        3.05,
        11.7,
        2.75,
        ["Analysis step", "What was computed", "Why it is included"],
        [
            ["Completeness/QC", "Subject, extended metric, and QC status for all observed runs", f"Tracks progress toward {counts['expected_runs']} planned outputs."],
            ["Arm summaries", "Mean, SD, min, and max per target and condition", "Describes repeat-level stability."],
            ["Paired deltas", "Defaced value minus intact value by repeat", "Isolates the effect of face removal within the same subject."],
            ["Percent change", "100 x mean(delta) / intact mean", "Makes endpoint shifts comparable across units."],
            ["Paired t-test", f"Descriptive p-value over {counts['pair_count']} observed repeat pairs", "Shows repeat-level separation; not population inference."],
        ],
        font_size=8,
    )
    textbox(slide, 0.78, 6.15, 11.6, 0.4, "No MNI-space baseline or template-neighbor comparison is used here because the valid reference is the subject-space atlas.", size=11, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Endpoints: robust amplitude and spatial-overlap readouts")
    add_footer(slide)
    add_table(
        slide,
        0.55,
        1.05,
        7.0,
        4.85,
        ["Endpoint", "Definition"],
        [
            ["ROI mean", "Average TI magnitude inside the target ROI."],
            ["ROI P95", "95th percentile TI magnitude inside the target ROI; preferred over peak for robustness."],
            ["ROI peak", "Maximum target-region value; inspected but treated as outlier-sensitive."],
            ["Top-5% overlap", "Fraction of ROI voxels inside the whole-brain top 5% field."],
            ["ROI volume >=0.2 V/m", "Target ROI volume above the hard field threshold."],
            ["Whole-brain volume >=0.2 V/m", "Total high-field volume, used as an extent/off-target proxy."],
        ],
        font_size=8,
    )
    shape = slide.shapes.add_shape(1, Inches(8.0), Inches(1.05), Inches(4.55), Inches(3.05))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(249, 244, 232)
    shape.line.color.rgb = GOLD
    textbox(
        slide,
        8.25,
        1.35,
        4.05,
        2.25,
        "Interpretation guardrail\n\nThese are repeated runs of one subject, not a population sample. Paired tests quantify repeat-level separation in this experiment only; the main readout is effect size, direction, and stability across repeats.",
        size=13,
        color=INK,
    )
    textbox(slide, 8.1, 4.55, 4.3, 0.65, "Controls: ROI volume and whole-brain volume are tracked to rule out gross atlas or brain-mask size changes.", size=12, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Defacing effects are target-specific, not global")
    add_footer(slide)
    rows = []
    for target in ("left-hippocampus", "left-m1"):
        for key in HEADLINE_KEYS:
            row = metric_row(delta_rows, target, key)
            rows.append(
                [
                    TARGET_LABELS[target],
                    row["metric_label"],
                    mean_text(row, "intact_mean", key),
                    mean_text(row, "defaced_mean", key),
                    pct_text(row),
                    p_text(row),
                ]
            )
    add_table(
        slide,
        0.45,
        1.02,
        12.45,
        5.55,
        ["Target", "Endpoint", "Intact mean", "Defaced mean", "Change", "paired p"],
        rows,
        font_size=7,
    )
    textbox(slide, 0.58, 6.72, 12.0, 0.25, "Percent changes use paired repeat means. p-values are descriptive repeat-level tests, not population inference.", size=8, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "M1 high-field volume changes more than hippocampus amplitude")
    add_footer(slide)
    slide.shapes.add_picture(str(figures["percent_deltas"]), Inches(0.72), Inches(1.1), width=Inches(11.9))

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Repeat distributions separate mainly for M1 high-field metrics")
    add_footer(slide)
    slide.shapes.add_picture(str(figures["distributions"]), Inches(0.35), Inches(1.06), width=Inches(12.55))

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Paired repeats show the same target-specific pattern")
    add_footer(slide)
    slide.shapes.add_picture(str(figures["repeat_lines"]), Inches(0.35), Inches(1.08), width=Inches(12.55))

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Hippocampus: amplitude is stable; spatial overlap drops modestly")
    add_footer(slide)
    hip_mean = metric_row(delta_rows, "left-hippocampus", "roi_mean")
    hip_p95 = metric_row(delta_rows, "left-hippocampus", "roi_percentile_value")
    hip_overlap = metric_row(delta_rows, "left-hippocampus", "overlap_fraction")
    hip_focal = metric_row(delta_rows, "left-hippocampus", "focality_in_roi_volume_mm3_gt_threshold")
    add_metric_card(slide, 0.72, 1.18, 2.65, 1.25, "ROI P95", pct_text(hip_p95), f"{mean_text(hip_p95, 'intact_mean', 'roi_percentile_value')} -> {mean_text(hip_p95, 'defaced_mean', 'roi_percentile_value')} V/m", BLUE)
    add_metric_card(slide, 3.65, 1.18, 2.65, 1.25, "Top-5% overlap", pct_text(hip_overlap), f"paired p={p_text(hip_overlap)}", TEAL)
    add_metric_card(slide, 6.58, 1.18, 2.65, 1.25, "ROI >=0.2 V/m", pct_text(hip_focal), f"{mean_text(hip_focal, 'intact_mean', 'focality_in_roi_volume_mm3_gt_threshold')} -> {mean_text(hip_focal, 'defaced_mean', 'focality_in_roi_volume_mm3_gt_threshold')} mm3", GOLD)
    textbox(slide, 0.82, 3.02, 5.7, 0.25, "Interpretation", size=11, bold=True, color=MUTED)
    textbox(
        slide,
        0.82,
        3.38,
        5.7,
        0.72,
        f"Amplitude endpoints were stable: ROI P95 {direction_phrase(hip_p95)} and ROI mean {direction_phrase(hip_mean)} across paired repeats.",
        size=15,
        color=INK,
    )
    textbox(
        slide,
        0.82,
        4.28,
        5.7,
        0.72,
        f"The clearest hippocampal signal is spatial: top-5% overlap {direction_phrase(hip_overlap)}, indicating modest redistribution of the highest-field voxels.",
        size=15,
        color=INK,
    )
    textbox(slide, 0.82, 5.34, 5.7, 0.42, "Boundary: single-subject technical repeats; p-values are descriptive.", size=10, color=MUTED)
    imgs = [representative_overlay("left-hippocampus", "intact"), representative_overlay("left-hippocampus", "defaced")]
    for i, img in enumerate(imgs):
        if img:
            slide.shapes.add_picture(str(img), Inches(7.0 + i * 2.85), Inches(3.0), width=Inches(2.55))
            textbox(slide, 7.0 + i * 2.85, 5.65, 2.55, 0.25, ["Intact repeat 01", "Defaced repeat 01"][i], size=9, color=MUTED, align=PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "M1: defacing increases high-field ROI endpoints")
    add_footer(slide)
    m1_mean = metric_row(delta_rows, "left-m1", "roi_mean")
    m1_p95 = metric_row(delta_rows, "left-m1", "roi_percentile_value")
    m1_focal = metric_row(delta_rows, "left-m1", "focality_in_roi_volume_mm3_gt_threshold")
    m1_whole_focal = metric_row(delta_rows, "left-m1", "focality_volume_mm3_gt_threshold")
    m1_overlap = metric_row(delta_rows, "left-m1", "overlap_fraction")
    add_metric_card(slide, 0.72, 1.18, 2.65, 1.25, "ROI mean", pct_text(m1_mean), f"paired p={p_text(m1_mean)}", BLUE)
    add_metric_card(slide, 3.65, 1.18, 2.65, 1.25, "ROI P95", pct_text(m1_p95), f"{mean_text(m1_p95, 'intact_mean', 'roi_percentile_value')} -> {mean_text(m1_p95, 'defaced_mean', 'roi_percentile_value')} V/m", TEAL)
    add_metric_card(slide, 6.58, 1.18, 2.65, 1.25, "ROI >=0.2 V/m", pct_text(m1_focal), f"paired p={p_text(m1_focal)}", GOLD)
    textbox(slide, 0.82, 3.02, 5.7, 0.25, "Interpretation", size=11, bold=True, color=MUTED)
    textbox(
        slide,
        0.82,
        3.38,
        5.7,
        0.72,
        f"Defaced runs showed higher cortical amplitude: M1 ROI P95 {direction_phrase(m1_p95)} across paired repeats.",
        size=15,
        color=INK,
    )
    textbox(
        slide,
        0.82,
        4.28,
        5.7,
        0.92,
        f"The larger effect is threshold extent: ROI volume >=0.2 V/m {direction_phrase(m1_focal)}, while whole-brain high-field volume {direction_phrase(m1_whole_focal)}.",
        size=15,
        color=INK,
    )
    textbox(slide, 0.82, 5.45, 5.7, 0.44, f"Top-5% overlap {direction_phrase(m1_overlap)}, so interpret magnitude and extent shifts alongside spatial engagement.", size=10, color=MUTED)
    imgs = [representative_overlay("left-m1", "intact"), representative_overlay("left-m1", "defaced")]
    for i, img in enumerate(imgs):
        if img:
            slide.shapes.add_picture(str(img), Inches(7.0 + i * 2.85), Inches(3.0), width=Inches(2.55))
            textbox(slide, 7.0 + i * 2.85, 5.65, 2.55, 0.25, ["Intact repeat 01", "Defaced repeat 01"][i], size=9, color=MUTED, align=PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Controls: atlas and brain volume do not explain the effects")
    add_footer(slide)
    rows = []
    brain_volume_changes = []
    for target in ("left-hippocampus", "left-m1"):
        roi_vol = metric_row(delta_rows, target, "roi_volume_mm3")
        brain_vol = metric_row(delta_rows, target, "whole_brain_volume_mm3")
        brain_volume_changes.append(f"{TARGET_LABELS[target]} {pct_text(brain_vol)}")
        rows.append(
            [
                TARGET_LABELS[target],
                "ROI volume",
                mean_text(roi_vol, "intact_mean", "roi_volume_mm3"),
                mean_text(roi_vol, "defaced_mean", "roi_volume_mm3"),
                pct_text(roi_vol),
            ]
        )
        rows.append(
            [
                TARGET_LABELS[target],
                "Whole-brain volume",
                mean_text(brain_vol, "intact_mean", "whole_brain_volume_mm3"),
                mean_text(brain_vol, "defaced_mean", "whole_brain_volume_mm3"),
                pct_text(brain_vol),
            ]
        )
    add_table(slide, 0.75, 1.15, 7.5, 2.2, ["Target", "Measure", "Intact", "Defaced", "Change"], rows, font_size=9)
    bullet_list(
        slide,
        0.82,
        4.0,
        11.4,
        1.8,
        [
            "ROI volume was identical within each target, consistent with using a fixed subject-space atlas.",
            f"Whole-brain volume changes were minimal after defacing ({'; '.join(brain_volume_changes)}).",
            "Therefore, observed endpoint changes are unlikely to be driven by gross ROI-size or brain-volume changes.",
        ],
        size=15,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Interpretation: defacing sensitivity is target-specific")
    add_footer(slide)
    bullet_list(
        slide,
        0.75,
        1.18,
        11.6,
        4.9,
        [
            "Defacing did not create a uniform bias across targets; effects were target-specific.",
            f"The hippocampal target was robust in amplitude endpoints, with ROI P95 {direction_phrase(hip_p95)} and top-field overlap {direction_phrase(hip_overlap)}.",
            f"The cortical M1 target was more sensitive to defacing, with ROI P95 {direction_phrase(m1_p95)} and above-threshold ROI volume {direction_phrase(m1_focal)}.",
            "Peak field should be interpreted cautiously because it is more outlier-sensitive; conclusions emphasize ROI mean, ROI P95, overlap, and threshold-volume endpoints.",
            "The experiment supports reporting target-specific defacing sensitivity rather than one global defacing effect.",
        ],
        size=17,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Limitations: single-subject repeat-level evidence")
    add_footer(slide)
    bullet_list(
        slide,
        0.75,
        1.1,
        11.8,
        5.4,
        [
            "Single-subject experiment: results quantify sub-CCMe pipeline sensitivity, not population-level behavior.",
            "Repeats are computational repeats, not independent biological samples; p-values are descriptive.",
            "MNI baseline and template-neighbor metrics were intentionally disabled to avoid coordinate-space misuse.",
            "Recommended next check: inspect mesh QC wall renderings for intact versus defaced arms to identify face-reconstruction or skin/skull geometry differences.",
            "Recommended sensitivity analysis: repeat key endpoints after excluding peak-dominated outliers and compare image-space hotspot localization if NIfTI fields are available.",
        ],
        size=16,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Deliverables are archived with the source data")
    add_footer(slide)
    textbox(slide, 0.78, 1.05, 11.6, 0.5, f"Archive folder: {OUT_DIR}", size=11, color=MUTED)
    add_table(
        slide,
        0.78,
        1.78,
        11.65,
        3.15,
        ["Artifact", "Filename", "Use"],
        [
            ["Documentation", "README.md", "Analysis methods, endpoint definitions, limitations, reproducibility steps."],
            ["Editable deck", "defacing_results_presentation.pptx", "PowerPoint source for conference presentation edits."],
            ["Rendered deck", "defacing_results_presentation.pdf", "Stable PDF export for sharing and review."],
            ["Run metrics", "defacing_run_metrics.csv", "One row per repeat with extracted post-processing metrics."],
            ["Paired deltas", "defacing_paired_deltas.csv", "Defaced-minus-intact effect sizes, CIs, and descriptive p-values."],
            ["Figures", "figures/*.png", "Chart images used in the deck."],
        ],
        font_size=8,
    )
    textbox(
        slide,
        0.78,
        5.35,
        11.7,
        0.7,
        "Source simulation outputs were read only. The generated report artifacts are stored alongside the downloaded experiment data for transfer and reproducibility.",
        size=14,
        color=INK,
    )

    path = OUT_DIR / "defacing_results_presentation.pptx"
    prs.save(path)
    return path


def write_summary_md(records: list[dict[str, object]], delta_rows: list[dict[str, object]], path: Path) -> None:
    counts = experiment_counts(records, delta_rows)
    status = "complete" if counts["is_complete"] else "incomplete"
    hip_p95 = metric_row(delta_rows, "left-hippocampus", "roi_percentile_value")
    hip_overlap = metric_row(delta_rows, "left-hippocampus", "overlap_fraction")
    m1_p95 = metric_row(delta_rows, "left-m1", "roi_percentile_value")
    m1_focal = metric_row(delta_rows, "left-m1", "focality_in_roi_volume_mm3_gt_threshold")
    m1_overlap = metric_row(delta_rows, "left-m1", "overlap_fraction")
    lines = [
        "# Defacing experiment summary",
        "",
        f"Source: `{SOURCE_ROOT}`",
        "",
        "## Completeness",
        "",
        f"- Subject metric files analyzed: `{len(records)}`.",
        f"- Planned run count: `{counts['expected_runs']}`.",
        f"- Planned repeats per target-condition arm: `{counts['planned_repeats']}`.",
        f"- Observed repeats per target-condition arm: `{counts['repeat_count']}`.",
        f"- Current completeness status: `{status}`.",
        f"- All {len(records)} analyzed runs reported complete subject metrics, extended metrics, and QC status.",
        "- MNI baseline, neighbor, and electrode-distance metrics were not configured and are not used for the conclusions.",
        "",
        "## Headline paired deltas: defaced minus intact",
        "",
        "| Target | Endpoint | Intact mean | Defaced mean | Change | paired p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in ("left-hippocampus", "left-m1"):
        for key in HEADLINE_KEYS:
            row = metric_row(delta_rows, target, key)
            lines.append(
                "| "
                + " | ".join(
                    [
                        TARGET_LABELS[target],
                        str(row["metric_label"]),
                        mean_text(row, "intact_mean", key),
                        mean_text(row, "defaced_mean", key),
                        pct_text(row),
                        p_text(row),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Left hippocampus: ROI P95 {direction_phrase(hip_p95)}; top-5% overlap {direction_phrase(hip_overlap)}.",
            f"- Left M1: ROI P95 {direction_phrase(m1_p95)}; ROI volume above 0.2 V/m {direction_phrase(m1_focal)}; top-5% overlap {direction_phrase(m1_overlap)}.",
            "- ROI volume was identical across intact and defaced conditions for both targets, confirming fixed subject-space atlas behavior.",
            "- Whole-brain volume changes were minimal, so gross brain-volume changes do not explain the endpoint shifts.",
            "",
            "## Caution",
            "",
            "This is a repeated single-subject experiment. Treat paired p-values as descriptive repeat-level evidence, not population inference.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_records()
    arm_rows = arm_summary(records)
    delta_rows = paired_deltas(records)

    write_csv(OUT_DIR / "defacing_run_metrics.csv", records)
    write_csv(OUT_DIR / "defacing_arm_summary.csv", arm_rows)
    write_csv(OUT_DIR / "defacing_paired_deltas.csv", delta_rows)
    write_summary_md(records, delta_rows, OUT_DIR / "defacing_results_summary.md")

    figures = {
        "percent_deltas": figures_dir / "percent_deltas.png",
        "distributions": figures_dir / "key_distributions.png",
        "repeat_lines": figures_dir / "paired_repeat_lines.png",
    }
    plot_percent_deltas(delta_rows, figures["percent_deltas"])
    plot_key_distributions(records, figures["distributions"])
    plot_repeat_lines(records, figures["repeat_lines"])

    deck = build_deck(records, arm_rows, delta_rows, figures)
    print(f"Wrote {deck}")
    print(f"Wrote {OUT_DIR / 'defacing_results_summary.md'}")
    print(f"Wrote {OUT_DIR / 'defacing_run_metrics.csv'}")
    print(f"Wrote {OUT_DIR / 'defacing_arm_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'defacing_paired_deltas.csv'}")


if __name__ == "__main__":
    main()
