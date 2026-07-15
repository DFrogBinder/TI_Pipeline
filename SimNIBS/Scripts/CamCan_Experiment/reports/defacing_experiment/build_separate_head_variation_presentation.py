#!/usr/bin/env python3
"""Build the corrected defacing deck: analyze each head separately, then compare."""
from __future__ import annotations

import argparse
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


SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

WHITE = RGBColor(255, 255, 255)
INK = RGBColor(30, 37, 45)
MUTED = RGBColor(91, 102, 115)
LIGHT = RGBColor(246, 248, 250)
NAVY = RGBColor(11, 31, 53)
BLUE = RGBColor(31, 96, 145)
TEAL = RGBColor(26, 150, 156)
ORANGE = RGBColor(214, 126, 54)
GOLD = RGBColor(232, 174, 70)
RED = RGBColor(184, 70, 59)
GREEN = RGBColor(72, 143, 91)

ARM_NAMES = (
    "Left_Hippocampus_Intact",
    "Left_Hippocampus_Defaced",
    "Left_M1_Intact",
    "Left_M1_Defaced",
)

HEADS = {
    "sub-CCMe": {
        "label": "sub-CCMe",
        "description": "left skull trauma",
        "source_dirs": ("post_export_sub-CCMe", "post_export_40repeats"),
        "color": ORANGE,
    },
    "sub-IXI025": {
        "label": "sub-IXI025",
        "description": "healthy IXI head",
        "source_dirs": ("post_export_sub-IXI025",),
        "color": BLUE,
    },
}

TARGET_LABELS = {
    "left-hippocampus": "Left hippocampus",
    "left-m1": "Left M1",
}

CONDITION_LABELS = {
    "intact": "Face intact",
    "defaced": "Face removed",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    units: str
    decimals: int
    stored_as_fraction: bool = False


METRICS = [
    MetricSpec("roi_mean", "ROI mean field", "V/m", 3),
    MetricSpec("roi_percentile_value", "ROI P95 field", "V/m", 3),
    MetricSpec("overlap_fraction", "ROI in top 5% field", "%", 1, True),
    MetricSpec("focality_in_roi_volume_mm3_gt_threshold", "ROI volume >=0.2 V/m", "mm3", 0),
    MetricSpec("focality_volume_mm3_gt_threshold", "Whole-brain volume >=0.2 V/m", "mm3", 0),
    MetricSpec("roi_volume_mm3", "ROI volume", "mm3", 0),
    MetricSpec("whole_brain_volume_mm3", "Whole-brain volume", "mm3", 0),
    MetricSpec("csf_distance_mm", "Target-to-CSF distance", "mm", 2),
]

PRIMARY_METRICS = [
    "roi_mean",
    "roi_percentile_value",
    "overlap_fraction",
    "focality_in_roi_volume_mm3_gt_threshold",
    "focality_volume_mm3_gt_threshold",
]


def mpl(color: RGBColor) -> str:
    return "#{:02x}{:02x}{:02x}".format(*tuple(color))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return st.mean(vals) if vals else math.nan


def sd(values: Iterable[float]) -> float:
    vals = list(values)
    return st.stdev(vals) if len(vals) > 1 else 0.0


def sem(values: Iterable[float]) -> float:
    vals = list(values)
    return sd(vals) / math.sqrt(len(vals)) if len(vals) > 1 else math.nan


def cv_percent(values: Iterable[float]) -> float:
    vals = list(values)
    m = mean(vals)
    return 100 * sd(vals) / m if vals and m else math.nan


def metric_spec(key: str) -> MetricSpec:
    for spec in METRICS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def display_value(value: float, key: str) -> float:
    spec = metric_spec(key)
    return value * 100 if spec.stored_as_fraction else value


def fmt_value(value: float, key: str) -> str:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "n/a"
    spec = metric_spec(key)
    shown = display_value(value, key)
    if spec.decimals == 0:
        return f"{shown:,.0f}"
    return f"{shown:.{spec.decimals}f}"


def pct(value: float, decimals: int = 1) -> str:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "n/a"
    return f"{value:+.{decimals}f}%"


def p_text(value: float) -> str:
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "n/a"
    return f"{value:.3g}"


def numeric(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def resolve_head_root(data_root: Path, head: str) -> Path:
    for dirname in HEADS[head]["source_dirs"]:
        candidate = data_root / str(dirname)
        if candidate.is_dir() and all((candidate / arm).is_dir() for arm in ARM_NAMES):
            return candidate
    expected = ", ".join(str(data_root / str(d)) for d in HEADS[head]["source_dirs"])
    raise FileNotFoundError(f"Could not find a complete export root for {head}. Checked: {expected}")


def infer_arm(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        if part in ARM_NAMES:
            return part
    raise ValueError(f"Cannot infer arm from {path}")


def infer_target(arm: str) -> str:
    if "Hippocampus" in arm:
        return "left-hippocampus"
    if "M1" in arm:
        return "left-m1"
    raise ValueError(f"Cannot infer target from {arm}")


def infer_condition(arm: str) -> str:
    if "Defaced" in arm:
        return "defaced"
    if "Intact" in arm:
        return "intact"
    raise ValueError(f"Cannot infer condition from {arm}")


def infer_repeat(path: Path) -> int:
    match = re.search(r"Data_(\d+)", path.as_posix())
    if not match:
        raise ValueError(f"Cannot infer repeat from {path}")
    return int(match.group(1))


def load_records(data_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for head in HEADS:
        root = resolve_head_root(data_root, head)
        for path in sorted(root.rglob("subject_metrics.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            arm = infer_arm(path, root)
            target = infer_target(arm)
            condition = infer_condition(arm)
            target_roi = data["target_roi"]
            roi = data["rois"][target_roi]
            ext = data["extended_metrics"]
            rec: dict[str, object] = {
                "head": head,
                "head_label": HEADS[head]["label"],
                "head_description": HEADS[head]["description"],
                "source_root": str(root),
                "path": str(path),
                "arm": arm,
                "target": target,
                "condition": condition,
                "repeat": infer_repeat(path),
                "subject": data.get("subject", head),
                "target_roi": target_roi,
                "subject_status": data.get("subject_metrics_meta", {}).get("status"),
                "extended_status": data.get("extended_metrics_meta", {}).get("status"),
                "qc_status": data.get("qc_meta", {}).get("status"),
            }
            rec["whole_brain_volume_mm3"] = numeric(data.get("whole_brain_volume_mm3"))
            for key in (
                "roi_volume_mm3",
                "overlap_fraction",
                "roi_percentile_value",
                "focality_in_roi_volume_mm3_gt_threshold",
            ):
                rec[key] = numeric(roi.get(key))
            for key in (
                "roi_mean",
                "focality_volume_mm3_gt_threshold",
                "csf_distance_mm",
            ):
                rec[key] = numeric(ext.get(key))
            records.append(rec)
    return records


def records_by(
    records: list[dict[str, object]],
    *,
    head: str | None = None,
    target: str | None = None,
    condition: str | None = None,
) -> list[dict[str, object]]:
    out = records
    if head is not None:
        out = [r for r in out if r["head"] == head]
    if target is not None:
        out = [r for r in out if r["target"] == target]
    if condition is not None:
        out = [r for r in out if r["condition"] == condition]
    return out


def paired_delta_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head in HEADS:
        for target in TARGET_LABELS:
            intact = {int(r["repeat"]): r for r in records_by(records, head=head, target=target, condition="intact")}
            defaced = {int(r["repeat"]): r for r in records_by(records, head=head, target=target, condition="defaced")}
            repeats = sorted(set(intact) & set(defaced))
            for spec in METRICS:
                xs: list[float] = []
                ys: list[float] = []
                diffs: list[float] = []
                pct_diffs: list[float] = []
                for repeat in repeats:
                    x = intact[repeat].get(spec.key)
                    y = defaced[repeat].get(spec.key)
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not math.isnan(x) and not math.isnan(y):
                        x = float(x)
                        y = float(y)
                        diff = y - x
                        xs.append(x)
                        ys.append(y)
                        diffs.append(diff)
                        pct_diffs.append(100 * diff / x if x else math.nan)
                if not diffs:
                    continue
                intact_mean = mean(xs)
                defaced_mean = mean(ys)
                delta_mean = mean(diffs)
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
                        "head": head,
                        "target": target,
                        "metric": spec.key,
                        "metric_label": spec.label,
                        "units": spec.units,
                        "n_pairs": len(diffs),
                        "intact_mean": intact_mean,
                        "defaced_mean": defaced_mean,
                        "delta_mean": delta_mean,
                        "delta_sd": sd(diffs),
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "percent_delta": 100 * delta_mean / intact_mean if intact_mean else math.nan,
                        "percent_delta_sd": sd([v for v in pct_diffs if not math.isnan(v)]),
                        "paired_t_p": p_value,
                        "repeat_percent_deltas": json.dumps(pct_diffs),
                    }
                )
    return rows


def variation_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head in HEADS:
        for target in TARGET_LABELS:
            for condition in ("intact", "defaced"):
                subset = records_by(records, head=head, target=target, condition=condition)
                for spec in METRICS:
                    vals = [float(r[spec.key]) for r in subset if isinstance(r.get(spec.key), (int, float)) and not math.isnan(float(r[spec.key]))]
                    rows.append(
                        {
                            "head": head,
                            "target": target,
                            "condition": condition,
                            "metric": spec.key,
                            "metric_label": spec.label,
                            "units": spec.units,
                            "n": len(vals),
                            "mean": mean(vals),
                            "sd": sd(vals),
                            "cv_percent": cv_percent(vals),
                            "min": min(vals) if vals else math.nan,
                            "max": max(vals) if vals else math.nan,
                        }
                    )
    return rows


def delta_row(delta_rows: list[dict[str, object]], head: str, target: str, metric: str) -> dict[str, object]:
    for row in delta_rows:
        if row["head"] == head and row["target"] == target and row["metric"] == metric:
            return row
    raise KeyError((head, target, metric))


def variation_row(rows: list[dict[str, object]], head: str, target: str, condition: str, metric: str) -> dict[str, object]:
    for row in rows:
        if row["head"] == head and row["target"] == target and row["condition"] == condition and row["metric"] == metric:
            return row
    raise KeyError((head, target, condition, metric))


def significant(row: dict[str, object], alpha: float = 0.05) -> bool:
    p = row.get("paired_t_p")
    return isinstance(p, (int, float)) and not math.isnan(float(p)) and float(p) < alpha


def decision_rows(delta_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for head in HEADS:
        for target in TARGET_LABELS:
            mean_row = delta_row(delta_rows, head, target, "roi_mean")
            p95_row = delta_row(delta_rows, head, target, "roi_percentile_value")
            overlap = delta_row(delta_rows, head, target, "overlap_fraction")
            roi_extent = delta_row(delta_rows, head, target, "focality_in_roi_volume_mm3_gt_threshold")
            brain_extent = delta_row(delta_rows, head, target, "focality_volume_mm3_gt_threshold")
            amp_sig = significant(mean_row) or significant(p95_row)
            spatial_sig = significant(overlap) or significant(roi_extent) or significant(brain_extent)
            if head == "sub-CCMe" and target == "left-hippocampus":
                answer = "Partial: mean field and overlap changed; robust P95 amplitude was stable."
            elif head == "sub-CCMe" and target == "left-m1":
                answer = "Yes: M1 P95 and target high-field extent increased after defacing."
            elif head == "sub-IXI025" and target == "left-hippocampus":
                answer = "Yes: amplitude, overlap, and target high-field extent decreased."
            else:
                answer = "Yes, but small: amplitude and overlap decreased; target high-field extent did not separate."
            rows.append(
                {
                    "head": head,
                    "target": target,
                    "amplitude_repeat_level_significant": amp_sig,
                    "spatial_or_extent_repeat_level_significant": spatial_sig,
                    "roi_mean_percent_delta": mean_row["percent_delta"],
                    "roi_mean_p": mean_row["paired_t_p"],
                    "roi_p95_percent_delta": p95_row["percent_delta"],
                    "roi_p95_p": p95_row["paired_t_p"],
                    "overlap_percent_delta": overlap["percent_delta"],
                    "overlap_p": overlap["paired_t_p"],
                    "roi_extent_percent_delta": roi_extent["percent_delta"],
                    "roi_extent_p": roi_extent["paired_t_p"],
                    "whole_brain_extent_percent_delta": brain_extent["percent_delta"],
                    "whole_brain_extent_p": brain_extent["paired_t_p"],
                    "answer": answer,
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


def plot_head_effects(head: str, delta_rows: list[dict[str, object]], out: Path) -> None:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    sig: list[bool] = []
    for target in ("left-hippocampus", "left-m1"):
        for metric in PRIMARY_METRICS:
            row = delta_row(delta_rows, head, target, metric)
            labels.append(f"{TARGET_LABELS[target].replace('Left ', '')} | {metric_spec(metric).label.replace(' field', '').replace('ROI ', '')}")
            values.append(float(row["percent_delta"]))
            colors.append(mpl(HEADS[head]["color"]))
            sig.append(significant(row))
    y = list(range(len(labels)))[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.barh(y, values, color=colors, alpha=0.82)
    ax.axvline(0, color="#56606b", linewidth=1.0)
    for yi, val, is_sig in zip(y, values, sig):
        ha = "left" if val >= 0 else "right"
        dx = 0.15 if val >= 0 else -0.15
        star = " *" if is_sig else ""
        ax.text(val + dx, yi, f"{pct(val)}{star}", va="center", ha=ha, fontsize=9)
    limit = max(12, math.ceil(max(abs(v) for v in values) + 2))
    ax.set_xlim(-limit, limit)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Defaced minus intact mean change (%)")
    ax.set_title(f"{HEADS[head]['label']}: independent paired-repeat effects", loc="left", fontsize=13)
    ax.grid(axis="x", alpha=0.16)
    ax.text(0.01, -0.12, "* paired repeat-level p < 0.05", transform=ax.transAxes, fontsize=8, color="#59636f")
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_head_variability(head: str, variation: list[dict[str, object]], delta_rows: list[dict[str, object]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    ax = axes[0]
    labels: list[str] = []
    intact: list[float] = []
    defaced: list[float] = []
    for target in ("left-hippocampus", "left-m1"):
        for metric in ("roi_percentile_value", "overlap_fraction"):
            labels.append(f"{TARGET_LABELS[target].replace('Left ', '')} | {metric_spec(metric).label.replace(' field', '')}")
            intact.append(float(variation_row(variation, head, target, "intact", metric)["cv_percent"]))
            defaced.append(float(variation_row(variation, head, target, "defaced", metric)["cv_percent"]))
    y = list(range(len(labels)))[::-1]
    height = 0.34
    ax.barh([v + height / 2 for v in y], intact, height, color=mpl(BLUE), alpha=0.82, label="Intact")
    ax.barh([v - height / 2 for v in y], defaced, height, color=mpl(TEAL), alpha=0.82, label="Defaced")
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("Repeat CV (%)")
    ax.set_title("Within-condition repeat variability", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.16)

    ax = axes[1]
    labels = []
    vals = []
    for target in ("left-hippocampus", "left-m1"):
        for metric in ("roi_percentile_value", "overlap_fraction", "focality_in_roi_volume_mm3_gt_threshold"):
            row = delta_row(delta_rows, head, target, metric)
            labels.append(f"{TARGET_LABELS[target].replace('Left ', '')} | {metric_spec(metric).label.replace(' field', '').replace('ROI ', '')}")
            vals.append(float(row["percent_delta_sd"]))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, vals, color=mpl(HEADS[head]["color"]), alpha=0.82)
    ax.set_yticks(y, labels, fontsize=8.2)
    ax.set_xlabel("SD of paired percent change")
    ax.set_title("Defacing-effect variability across repeats", fontsize=11)
    ax.grid(axis="x", alpha=0.16)
    fig.suptitle(f"{HEADS[head]['label']}: repeat-level variation analysis", x=0.03, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_between_head(delta_rows: list[dict[str, object]], out: Path) -> None:
    rows: list[tuple[str, str, str]] = []
    for target in ("left-hippocampus", "left-m1"):
        for metric in ("roi_percentile_value", "overlap_fraction", "focality_in_roi_volume_mm3_gt_threshold"):
            rows.append((target, metric, f"{TARGET_LABELS[target]} | {metric_spec(metric).label}"))
    y = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    for head, offset in (("sub-CCMe", 0.15), ("sub-IXI025", -0.15)):
        xs = [float(delta_row(delta_rows, head, target, metric)["percent_delta"]) for target, metric, _ in rows]
        ys = [v + offset for v in y]
        ax.scatter(xs, ys, s=78, color=mpl(HEADS[head]["color"]), label=f"{HEADS[head]['label']} ({HEADS[head]['description']})", zorder=3)
        for x, yy in zip(xs, ys):
            ax.text(x + (0.16 if x >= 0 else -0.16), yy, pct(x), ha="left" if x >= 0 else "right", va="center", fontsize=8)
    ax.axvline(0, color="#56606b", linewidth=1.0)
    ax.set_yticks(y, [label for _, _, label in rows], fontsize=9)
    ax.set_xlabel("Defaced minus intact mean change (%)")
    ax.set_title("Secondary comparison: head-specific effects are not pooled", loc="left", fontsize=13)
    ax.set_xlim(-6.2, 12.2)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="x", alpha=0.16)
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def add_bg(slide, title: str | None = None) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    top = slide.shapes.add_shape(1, 0, 0, SLIDE_W, Inches(0.16))
    top.fill.solid()
    top.fill.fore_color.rgb = GOLD
    top.line.fill.background()
    if title:
        box = slide.shapes.add_textbox(Inches(0.55), Inches(0.33), Inches(12.1), Inches(0.48))
        p = box.text_frame.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(23)
        p.font.color.rgb = NAVY


def add_footer(slide) -> None:
    box = slide.shapes.add_textbox(Inches(0.55), Inches(7.13), Inches(12.2), Inches(0.24))
    p = box.text_frame.paragraphs[0]
    p.text = "Defacing experiment | heads analyzed separately | paired technical repeats"
    p.font.size = Pt(8)
    p.font.color.rgb = MUTED


def textbox(slide, x, y, w, h, text, *, size=15, bold=False, color=INK, align=None):
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


def bullet_list(slide, x, y, w, h, items: list[str], *, size=14, color=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = f"- {item}"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(7)
    return box


def metric_card(slide, x, y, w, h, label: str, value: str, subtitle: str, color: RGBColor):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = color
    shape.line.width = Pt(1.15)
    textbox(slide, x + 0.12, y + 0.10, w - 0.24, 0.26, label, size=9, bold=True, color=MUTED)
    textbox(slide, x + 0.12, y + 0.41, w - 0.24, 0.42, value, size=19, bold=True, color=color)
    textbox(slide, x + 0.12, y + 0.87, w - 0.24, h - 0.90, subtitle, size=8.2, color=INK)


def answer_card(slide, x, y, w, h, head: str, target: str, answer: str, color: RGBColor):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = color
    shape.line.width = Pt(1.25)
    textbox(slide, x + 0.16, y + 0.13, w - 0.32, 0.28, f"{head} | {target}", size=10, bold=True, color=color)
    textbox(slide, x + 0.16, y + 0.52, w - 0.32, h - 0.62, answer, size=12.5, color=INK)


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


def row_summary(delta_rows: list[dict[str, object]], head: str, target: str, metric: str) -> str:
    row = delta_row(delta_rows, head, target, metric)
    sig = "; p=" + p_text(float(row["paired_t_p"])) if significant(row) else "; p=" + p_text(float(row["paired_t_p"]))
    return f"{pct(float(row['percent_delta']))}{sig}"


def build_deck(
    records: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    variation: list[dict[str, object]],
    decisions: list[dict[str, object]],
    figures: dict[str, Path],
    out_dir: Path,
) -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    stripe = slide.shapes.add_shape(1, 0, 0, Inches(0.22), SLIDE_H)
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = GOLD
    stripe.line.fill.background()
    textbox(slide, 0.75, 0.75, 11.6, 0.75, "Does defacing change ROI E-fields?", size=34, bold=True, color=WHITE)
    textbox(slide, 0.78, 1.55, 11.3, 0.5, "Corrected analysis: run the variation analysis independently for each head, then compare the head-specific answers", size=17, color=RGBColor(214, 226, 238))
    textbox(slide, 0.78, 2.62, 11.25, 1.15, "Defacing affected the two heads differently. The healthy head showed consistent decreases after defacing; sub-CCMe showed a smaller hippocampal amplitude effect and an M1 high-field extent increase.", size=22, color=WHITE)
    textbox(slide, 0.80, 6.78, 11.8, 0.28, "320 simulations: 2 heads x 2 targets x 2 input conditions x 40 paired technical repeats.", size=9, color=RGBColor(190, 204, 218))

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Analysis rule: do not pool traumatic and healthy heads")
    add_footer(slide)
    metric_card(slide, 0.70, 1.05, 2.55, 1.25, "Head 1", "sub-CCMe", "Left skull trauma; analyzed independently.", ORANGE)
    metric_card(slide, 3.55, 1.05, 2.55, 1.25, "Head 2", "sub-IXI025", "Healthy IXI head; analyzed independently.", BLUE)
    metric_card(slide, 6.40, 1.05, 2.55, 1.25, "Pairing", "repeat N", "Defaced repeat N minus intact repeat N.", TEAL)
    metric_card(slide, 9.25, 1.05, 2.55, 1.25, "No pooling", "0 mixed tests", "Head comparison is secondary and descriptive.", GOLD)
    bullet_list(
        slide,
        0.85,
        3.00,
        11.6,
        2.7,
        [
            "The inferential unit in this experiment is a technical repeat within one head, not a biological subject.",
            "Each head receives its own paired defaced-versus-intact analysis for left hippocampus and left M1.",
            "The head-to-head slide compares the resulting effect sizes only after the per-head answers are established.",
            "This avoids interpreting the traumatic sub-CCMe anatomy as interchangeable with a healthy head.",
        ],
        size=15,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Analysis workflow and endpoints")
    add_footer(slide)
    metric_card(slide, 0.72, 1.08, 2.55, 1.22, "1. Read", "320 JSONs", "subject_metrics.json from each post output.", BLUE)
    metric_card(slide, 3.52, 1.08, 2.55, 1.22, "2. Pair", "40 pairs", "Within head and target only.", TEAL)
    metric_card(slide, 6.32, 1.08, 2.55, 1.22, "3. Test", "paired t", "Descriptive repeat-level separation.", GOLD)
    metric_card(slide, 9.12, 1.08, 2.55, 1.22, "4. Vary", "CV + SD", "Repeat variability and defacing-effect spread.", ORANGE)
    add_table(
        slide,
        0.70,
        3.02,
        11.9,
        2.85,
        ["Endpoint", "What it answers"],
        [
            ["ROI mean and ROI P95", "Did defacing change E-field amplitude in the target ROI?"],
            ["ROI in top 5% field", "Did defacing change how strongly the target overlaps the highest-field region?"],
            ["ROI volume >=0.2 V/m", "Did defacing change high-field extent inside the target ROI?"],
            ["Whole-brain volume >=0.2 V/m", "Did defacing change global high-field extent?"],
            ["ROI and whole-brain volume controls", "Do gross atlas or brain-volume changes explain the E-field shifts?"],
        ],
        font_size=9.6,
    )
    textbox(slide, 0.78, 6.25, 11.6, 0.42, "The word significant in this deck means repeat-level p < 0.05 within one head-target analysis; it is not population inference.", size=11, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Per-head answers are evaluated independently")
    add_footer(slide)
    decision_map = {(str(d["head"]), str(d["target"])): str(d["answer"]) for d in decisions}
    for x, y, head, target in (
        (0.72, 1.08, "sub-CCMe", "left-hippocampus"),
        (6.95, 1.08, "sub-CCMe", "left-m1"),
        (0.72, 3.58, "sub-IXI025", "left-hippocampus"),
        (6.95, 3.58, "sub-IXI025", "left-m1"),
    ):
        answer_card(
            slide,
            x,
            y,
            5.65,
            1.72,
            head,
            TARGET_LABELS[target],
            decision_map[(head, target)],
            HEADS[head]["color"],
        )
    bullet_list(
        slide,
        0.80,
        5.82,
        11.8,
        0.78,
        [
            "The healthy head answers yes for both ROIs, with decreases in amplitude and/or spatial engagement.",
            "sub-CCMe answers differ by target: hippocampal robust amplitude is stable, but M1 high-field extent increases.",
        ],
        size=12.7,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Per-head decision matrix")
    add_footer(slide)
    rows = []
    for decision in decisions:
        rows.append(
            [
                str(decision["head"]),
                TARGET_LABELS[str(decision["target"])],
                "Yes" if decision["amplitude_repeat_level_significant"] else "No clear",
                "Yes" if decision["spatial_or_extent_repeat_level_significant"] else "No clear",
                str(decision["answer"]),
            ]
        )
    add_table(
        slide,
        0.42,
        1.02,
        12.55,
        3.25,
        ["Head", "Target", "Amplitude changed?", "Spatial/extent changed?", "Answer"],
        rows,
        font_size=9.0,
    )
    bullet_list(
        slide,
        0.75,
        4.82,
        11.8,
        1.05,
        [
            "Significance is evaluated separately within each head-target analysis.",
            "The table is retained for exact wording; the previous slide gives the projected-readable decision summary.",
        ],
        size=13.5,
    )

    for head in ("sub-CCMe", "sub-IXI025"):
        slide = prs.slides.add_slide(blank)
        add_bg(slide, f"{HEADS[head]['label']}: paired defaced-versus-intact effects")
        add_footer(slide)
        slide.shapes.add_picture(str(figures[f"{head}_effects"]), Inches(0.60), Inches(0.98), width=Inches(12.0))
        textbox(slide, 0.74, 6.78, 11.9, 0.22, "Bars are defaced-minus-intact percent changes. Asterisk marks repeat-level paired p < 0.05 within this head only.", size=8.2, color=MUTED)

        slide = prs.slides.add_slide(blank)
        add_bg(slide, f"{HEADS[head]['label']}: repeat variability is analyzed within the head")
        add_footer(slide)
        slide.shapes.add_picture(str(figures[f"{head}_variation"]), Inches(0.68), Inches(1.05), width=Inches(11.9))
        textbox(slide, 0.75, 6.72, 11.6, 0.28, "CV describes within-condition repeat variability; paired-percent SD describes spread in the defacing effect across repeat pairs.", size=8.2, color=MUTED)

        slide = prs.slides.add_slide(blank)
        add_bg(slide, f"{HEADS[head]['label']}: independent ROI-level conclusions")
        add_footer(slide)
        textbox(slide, 0.52, 0.96, 5.8, 0.26, "Left hippocampus", size=12, bold=True, color=HEADS[head]["color"])
        rows = []
        for metric in ("roi_mean", "roi_percentile_value", "overlap_fraction", "focality_in_roi_volume_mm3_gt_threshold"):
            row = delta_row(delta_rows, head, "left-hippocampus", metric)
            rows.append(
                [
                    metric_spec(metric).label,
                    fmt_value(float(row["intact_mean"]), metric),
                    fmt_value(float(row["defaced_mean"]), metric),
                    pct(float(row["percent_delta"])),
                    p_text(float(row["paired_t_p"])),
                ]
            )
        add_table(slide, 0.52, 1.27, 12.25, 1.50, ["Endpoint", "Intact", "Defaced", "Change", "paired p"], rows, font_size=9.2)

        textbox(slide, 0.52, 3.08, 5.8, 0.26, "Left M1", size=12, bold=True, color=HEADS[head]["color"])
        rows = []
        for metric in ("roi_mean", "roi_percentile_value", "overlap_fraction", "focality_in_roi_volume_mm3_gt_threshold"):
            row = delta_row(delta_rows, head, "left-m1", metric)
            rows.append(
                [
                    metric_spec(metric).label,
                    fmt_value(float(row["intact_mean"]), metric),
                    fmt_value(float(row["defaced_mean"]), metric),
                    pct(float(row["percent_delta"])),
                    p_text(float(row["paired_t_p"])),
                ]
            )
        add_table(slide, 0.52, 3.39, 12.25, 1.50, ["Endpoint", "Intact", "Defaced", "Change", "paired p"], rows, font_size=9.2)
        if head == "sub-CCMe":
            bullets = [
                "Left hippocampus: robust ROI P95 amplitude did not separate; ROI mean and top-5% overlap decreased modestly.",
                "Left M1: ROI P95 and target volume above 0.2 V/m increased after defacing.",
                "Interpretation: for this traumatic head, defacing sensitivity is target-specific and cannot be summarized by one global bias.",
            ]
        else:
            bullets = [
                "Left hippocampus: amplitude, top-5% overlap, and high-field target extent decreased after defacing.",
                "Left M1: amplitude and top-5% overlap decreased; target volume above 0.2 V/m did not separate.",
                "Interpretation: for the healthy head, defacing produced coherent small-to-moderate reductions in ROI E-field endpoints.",
            ]
        bullet_list(slide, 0.70, 5.32, 11.9, 1.05, bullets, size=12.5)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Secondary comparison: head-specific effects differ")
    add_footer(slide)
    slide.shapes.add_picture(str(figures["between_head"]), Inches(0.62), Inches(1.00), width=Inches(12.0))
    textbox(slide, 0.75, 6.72, 11.7, 0.28, "This slide compares independent effect sizes after the per-head analyses. It is not a pooled head-level statistical test.", size=8.5, color=MUTED)

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Controls and mesh-quality boundary")
    add_footer(slide)
    rows = []
    for head in ("sub-CCMe", "sub-IXI025"):
        for target in ("left-hippocampus", "left-m1"):
            roi_volume = delta_row(delta_rows, head, target, "roi_volume_mm3")
            brain_volume = delta_row(delta_rows, head, target, "whole_brain_volume_mm3")
            rows.append(
                [
                    HEADS[head]["label"],
                    TARGET_LABELS[target],
                    pct(float(roi_volume["percent_delta"]), 2),
                    pct(float(brain_volume["percent_delta"]), 2),
                ]
            )
    add_table(slide, 0.75, 1.05, 8.8, 2.10, ["Head", "Target", "ROI volume", "Whole-brain volume"], rows, font_size=10.0)
    bullet_list(
        slide,
        0.85,
        3.75,
        11.6,
        2.25,
        [
            "ROI volume is unchanged for each head-target pair, consistent with fixed subject-space atlases.",
            "Whole-brain volume shifts are small, so the main E-field differences are not explained by gross mask volume.",
            "Local mesh-wall QC files were found for the sub-CCMe 160-run set only. I do not claim a healthy-versus-trauma mesh-quality comparison from the downloaded folder.",
            "To make mesh quality part of the decision, generate and download the same mesh-wall QC package for sub-IXI025, then compare both heads arm-by-arm.",
        ],
        size=13.2,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Decision guidance for downstream experiments")
    add_footer(slide)
    bullet_list(
        slide,
        0.75,
        1.08,
        11.7,
        4.65,
        [
            "Do not treat defaced and non-defaced anatomical inputs as interchangeable when precise ROI E-field values matter.",
            "For healthy anatomy, defacing produced repeat-level significant E-field changes in both ROIs, mostly reductions.",
            "For the traumatic sub-CCMe head, defacing effects were not globally predictable: hippocampal robust amplitude was stable, but M1 high-field extent increased.",
            "For future experiments, keep defacing status consistent within a study or explicitly include defacing status as a sensitivity factor.",
            "Mesh-quality decisions require matched mesh-wall QC for both heads; the current corrected deck answers the field-variability question completely and flags the missing mesh-QC comparison.",
        ],
        size=16,
    )

    slide = prs.slides.add_slide(blank)
    add_bg(slide, "Deliverables")
    add_footer(slide)
    textbox(slide, 0.75, 1.02, 11.8, 0.35, f"Output folder: {out_dir}", size=10, color=MUTED)
    add_table(
        slide,
        0.75,
        1.72,
        11.8,
        3.08,
        ["Artifact", "Filename", "Purpose"],
        [
            ["Editable deck", "defacing_separate_head_variation_analysis.pptx", "Corrected presentation source."],
            ["Rendered deck", "defacing_separate_head_variation_analysis.pdf", "Stable PDF for review/sharing."],
            ["Run metrics", "separate_head_run_metrics.csv", "One row per completed simulation."],
            ["Paired deltas", "separate_head_paired_deltas.csv", "Head-specific defaced-minus-intact tests."],
            ["Variation summary", "separate_head_variation_summary.csv", "Mean, SD, and CV per head-target-condition."],
            ["Decision matrix", "separate_head_decision_matrix.csv", "Per-head answer to the defacing question."],
            ["Narrative summary", "separate_head_variation_summary.md", "Methods, results, and interpretation boundary."],
        ],
        font_size=9.0,
    )
    textbox(slide, 0.80, 5.30, 11.5, 0.62, "The included builder script regenerates the package from `post_export_sub-CCMe` and `post_export_sub-IXI025` without pooling heads.", size=13, color=INK)

    out = out_dir / "defacing_separate_head_variation_analysis.pptx"
    prs.save(out)
    return out


def write_summary(path: Path, records: list[dict[str, object]], delta_rows: list[dict[str, object]], decisions: list[dict[str, object]]) -> None:
    lines = [
        "# Corrected separate-head defacing variation analysis",
        "",
        "## Question",
        "",
        "Does providing defaced MRI input images significantly change the E-field in the left hippocampus and left M1 when compared with non-defaced MRI input images?",
        "",
        "## Correction",
        "",
        "The traumatic `sub-CCMe` head and the healthy `sub-IXI025` head are analyzed separately. No statistical test pools repeats across the two heads. The head-to-head comparison is a secondary comparison of independent effect sizes.",
        "",
        "## Data analyzed",
        "",
        f"- Runs analyzed: `{len(records)}`.",
        "- Each head has 160 post-processed simulations: 2 targets x 2 conditions x 40 repeats.",
        "- Conditions: intact T1/T2 versus pydeface defaced T1/T2.",
        "- Targets: left hippocampus and left M1.",
        "",
        "## Per-head answer matrix",
        "",
        "| Head | Target | Amplitude changed? | Spatial/extent changed? | Answer |",
        "|---|---|---:|---:|---|",
    ]
    for row in decisions:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["head"]),
                    TARGET_LABELS[str(row["target"])],
                    "yes" if row["amplitude_repeat_level_significant"] else "no clear",
                    "yes" if row["spatial_or_extent_repeat_level_significant"] else "no clear",
                    str(row["answer"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Primary paired deltas", "", "| Head | Target | Endpoint | Intact | Defaced | Change | paired p |", "|---|---|---|---:|---:|---:|---:|"])
    for head in ("sub-CCMe", "sub-IXI025"):
        for target in ("left-hippocampus", "left-m1"):
            for metric in PRIMARY_METRICS:
                row = delta_row(delta_rows, head, target, metric)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            head,
                            TARGET_LABELS[target],
                            metric_spec(metric).label,
                            fmt_value(float(row["intact_mean"]), metric),
                            fmt_value(float(row["defaced_mean"]), metric),
                            pct(float(row["percent_delta"])),
                            p_text(float(row["paired_t_p"])),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `sub-IXI025` answers yes for both ROIs: defacing produced repeat-level significant decreases in amplitude and/or spatial engagement.",
            "- `sub-CCMe` is target-specific: hippocampal robust amplitude was stable, but M1 P95 and high-field target extent increased after defacing.",
            "- This supports using consistent defacing status within downstream studies, or treating defacing status as an explicit sensitivity factor.",
            "- Mesh-quality comparison is not claimed here because matched healthy-head mesh-wall QC was not present in the downloaded local folder.",
            "",
            "Paired p-values are descriptive technical-repeat statistics, not population inference.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build(data_root: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(data_root)
    delta_rows = paired_delta_rows(records)
    variation = variation_rows(records)
    decisions = decision_rows(delta_rows)

    write_csv(out_dir / "separate_head_run_metrics.csv", records)
    write_csv(out_dir / "separate_head_paired_deltas.csv", delta_rows)
    write_csv(out_dir / "separate_head_variation_summary.csv", variation)
    write_csv(out_dir / "separate_head_decision_matrix.csv", decisions)
    write_summary(out_dir / "separate_head_variation_summary.md", records, delta_rows, decisions)

    figures = {
        "sub-CCMe_effects": fig_dir / "sub_CCMe_effects.png",
        "sub-CCMe_variation": fig_dir / "sub_CCMe_variation.png",
        "sub-IXI025_effects": fig_dir / "sub_IXI025_effects.png",
        "sub-IXI025_variation": fig_dir / "sub_IXI025_variation.png",
        "between_head": fig_dir / "between_head_nonpooled_comparison.png",
    }
    for head in ("sub-CCMe", "sub-IXI025"):
        plot_head_effects(head, delta_rows, figures[f"{head}_effects"])
        plot_head_variability(head, variation, delta_rows, figures[f"{head}_variation"])
    plot_between_head(delta_rows, figures["between_head"])
    return build_deck(records, delta_rows, variation, decisions, figures, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/boyan/sandbox/Jake_Data/defacing_experiment"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/defacing_separate_head_variation_analysis"))
    args = parser.parse_args()
    deck = build(args.data_root.expanduser().resolve(), args.out_dir.expanduser().resolve())
    print(f"Wrote {deck}")


if __name__ == "__main__":
    main()
