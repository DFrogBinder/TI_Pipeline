"""Generate post-processing audit figures from subject_metrics.json files.

This module contains only static PNG/table generation. It intentionally does
not build presentation decks, and it avoids pandas/matplotlib so it can run in
minimal post-processing environments where compiled scientific wheels may vary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import textwrap
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.pipeline_layers import FIGURE_GENERATION_STAGE, subject_metrics_payload_analysis_complete


REPEAT_DATASET_PATTERN = re.compile(r"^(?P<roi_prefix>.+)_Data_(?P<repeat>\d+)$")

SUMMARY_METRICS = [
    "roi_peak",
    "roi_mean",
    "overlap_fraction",
    "focality_in_roi_voxels_gt_threshold",
    "focality_voxels_gt_threshold",
    "electrode_distance_mean_mm",
]

METRIC_LABELS = {
    "roi_peak": "ROI peak field",
    "roi_mean": "ROI mean field",
    "overlap_fraction": "Top-percentile overlap",
    "focality_in_roi_voxels_gt_threshold": "ROI voxels at/above threshold",
    "focality_voxels_gt_threshold": "Whole-brain voxels at/above threshold",
    "electrode_distance_mean_mm": "Mean electrode distance",
}

CANVAS_W = 2400
CANVAS_H = 1350

BG = "#F7F9FC"
PANEL = "#FFFFFF"
WHITE = "#FFFFFF"
INK = "#142033"
MUTED = "#64748B"
GRID = "#DDE6F2"
NAVY = "#17375E"
BLUE = "#2F6BA3"
TEAL = "#247C84"
GREEN = "#2F7D4E"
GOLD = "#D69235"
RED = "#C84C44"
PURPLE = "#8B3F74"
LIGHT_BLUE = "#EAF3FF"
LIGHT_GREEN = "#EAF7EF"
LIGHT_GOLD = "#FFF3DF"
LIGHT_RED = "#FDECEB"
LIGHT_PURPLE = "#F5EAF2"
LIGHT_TEAL = "#E7F6F7"

PALETTE = [
    (BLUE, LIGHT_BLUE),
    (GREEN, LIGHT_GREEN),
    (PURPLE, LIGHT_PURPLE),
    (TEAL, LIGHT_TEAL),
    (GOLD, LIGHT_GOLD),
    ("#52606D", "#EEF2F7"),
    ("#9A5A28", "#F5EEE8"),
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(58, bold=True)
FONT_H1 = load_font(44, bold=True)
FONT_H2 = load_font(34, bold=True)
FONT_H3 = load_font(26, bold=True)
FONT_BODY = load_font(24)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(15)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if text_size(draw, candidate, font)[0] <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    _, line_h = text_size(draw, "Ag", font)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = PANEL,
    outline: str = GRID,
    radius: int = 22,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    draw.rectangle((0, 0, CANVAS_W, 18), fill=NAVY)
    draw.rectangle((1640, 0, CANVAS_W, 18), fill=GOLD)
    draw.text((80, 58), title, font=FONT_TITLE, fill=INK)
    if subtitle:
        draw_wrapped_text(draw, subtitle, (84, 132), 2050, FONT_BODY, MUTED, line_gap=6)


def draw_footer(draw: ImageDraw.ImageDraw, text: str = "post-processing figures | subject_metrics.json") -> None:
    draw.text((80, 1294), text, font=FONT_TINY, fill=MUTED)


def fmt_int(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{int(round(float(value))):,}"


def fmt_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def mean(values: Iterable[float]) -> float | None:
    seq = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(seq) if seq else None


def median(values: Iterable[float]) -> float | None:
    seq = sorted(float(value) for value in values if math.isfinite(float(value)))
    return statistics.median(seq) if seq else None


def stdev(values: Iterable[float]) -> float | None:
    seq = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.stdev(seq) if len(seq) > 1 else 0.0 if seq else None


def quantile(values: Iterable[float], q: float) -> float | None:
    seq = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not seq:
        return None
    if len(seq) == 1:
        return seq[0]
    pos = (len(seq) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return seq[lo]
    frac = pos - lo
    return seq[lo] * (1 - frac) + seq[hi] * frac


def metric_stats(values: Iterable[float]) -> dict[str, float | None]:
    seq = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": float(len(seq)),
        "mean": mean(seq),
        "sd": stdev(seq),
        "median": median(seq),
        "min": min(seq) if seq else None,
        "q1": quantile(seq, 0.25),
        "q3": quantile(seq, 0.75),
        "max": max(seq) if seq else None,
    }


def _subject_metric_rows(
    records: list[dict[str, Any]],
    *,
    metric_key: str,
    output_prefix: str,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    by_roi_subject: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if not record.get("analysis_complete"):
            continue
        roi_label = str(record.get("roi_label") or "")
        subject = str(record.get("subject") or "")
        value = record.get(metric_key)
        if not roi_label or not subject or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        by_roi_subject[(roi_label, subject)].append(numeric * scale)

    rows: list[dict[str, Any]] = []
    for (roi_label, subject), values in sorted(by_roi_subject.items()):
        stats = metric_stats(values)
        rows.append(
            {
                "roi_label": roi_label,
                "subject": subject,
                "n_repeats": int(stats["n"] or 0),
                f"{output_prefix}_mean": stats["mean"],
                f"{output_prefix}_median": stats["median"],
                f"{output_prefix}_q1": stats["q1"],
                f"{output_prefix}_q3": stats["q3"],
                f"{output_prefix}_min": stats["min"],
                f"{output_prefix}_max": stats["max"],
            }
        )
    return rows


def _summary_from_subject_metric_rows(
    subject_rows: list[dict[str, Any]],
    *,
    output_prefix: str,
    summary_prefix: str,
) -> list[dict[str, Any]]:
    by_roi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        by_roi[str(row["roi_label"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    median_key = f"{output_prefix}_median"
    for roi_label, rows in sorted(by_roi.items()):
        medians = [
            float(row[median_key])
            for row in rows
            if isinstance(row.get(median_key), (int, float)) and math.isfinite(float(row[median_key]))
        ]
        repeat_counts = [float(row.get("n_repeats") or 0) for row in rows]
        stats = metric_stats(medians)
        summary_rows.append(
            {
                "roi_label": roi_label,
                "n_subjects": len(rows),
                "n_subject_repeat_rows": int(sum(repeat_counts)),
                "repeats_min": int(min(repeat_counts)) if repeat_counts else 0,
                "repeats_median": median(repeat_counts),
                "repeats_max": int(max(repeat_counts)) if repeat_counts else 0,
                f"population_median_{summary_prefix}": stats["median"],
                f"population_q1_{summary_prefix}": stats["q1"],
                f"population_q3_{summary_prefix}": stats["q3"],
                f"population_min_{summary_prefix}": stats["min"],
                f"population_max_{summary_prefix}": stats["max"],
            }
        )
    return summary_rows


def _mni_subject_dir_for_roi(roi_label: str, roi_prefix: str | None = None) -> str | None:
    text = f"{roi_label} {roi_prefix or ''}".casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", text)
    if "hippocampus" in normalized:
        return "MNI152-right-hippocampus" if "right" in normalized else "MNI152-left-hippocampus"
    if "thalamus" in normalized:
        return "MNI152-left-thalamus" if "left" in normalized else "MNI152-right-thalamus"
    if "front_middle" in normalized or "dlpc" in normalized:
        return "MNI152-left-dlpc" if "left" in normalized or "ctx_lh" in normalized else "MNI152-right-dlpc"
    if "precentral" in normalized or re.search(r"(^|_)m1($|_)", normalized):
        return "MNI152-right-m1" if "right" in normalized or "ctx_rh" in normalized else "MNI152-left-m1"
    return None


def _candidate_mni_roots(
    batch_root: Path | None,
    mni_baseline_post_root: str | Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if mni_baseline_post_root is not None:
        candidates.append(Path(mni_baseline_post_root).expanduser())
    if batch_root is not None:
        search_bases = [batch_root, *list(batch_root.parents)[:4]]
        for base in search_bases:
            candidates.append(base / "MNI152-data")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        zip_path = resolved.with_name(f"{resolved.name}-post.zip")
        if resolved in seen:
            continue
        if resolved.is_dir() or zip_path.is_file():
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _load_mni_post_metrics(mni_root: Path, mni_subject_dir: str) -> tuple[dict[str, Any], str] | None:
    relative = Path(mni_root.name) / mni_subject_dir / "anat" / "post" / "subject_metrics.json"
    unpacked = mni_root / mni_subject_dir / "anat" / "post" / "subject_metrics.json"
    if unpacked.is_file():
        return load_json(unpacked), str(unpacked)

    zip_path = mni_root.with_name(f"{mni_root.name}-post.zip")
    if not zip_path.is_file():
        return None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(str(relative)) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload, f"{zip_path}:{relative}"


def _mni_post_baseline_by_roi(
    records: list[dict[str, Any]],
    *,
    batch_root: Path | None,
    mni_baseline_post_root: str | Path | None,
) -> dict[str, tuple[float, str]]:
    identity_by_roi: dict[str, str | None] = {}
    for record in records:
        roi_label = str(record.get("roi_label") or "")
        if roi_label and roi_label not in identity_by_roi:
            identity_by_roi[roi_label] = str(record.get("roi_prefix") or "")

    output: dict[str, tuple[float, str]] = {}
    roots = _candidate_mni_roots(batch_root, mni_baseline_post_root)
    if not roots:
        return output

    for roi_label, roi_prefix in identity_by_roi.items():
        mni_subject_dir = _mni_subject_dir_for_roi(roi_label, roi_prefix)
        if not mni_subject_dir:
            continue
        for root in roots:
            loaded = _load_mni_post_metrics(root, mni_subject_dir)
            if loaded is None:
                continue
            payload, source = loaded
            extended = payload.get("extended_metrics")
            if not isinstance(extended, dict):
                continue
            value = safe_float(extended.get("roi_mean"))
            if value is not None:
                output[roi_label] = (value, source)
                break
    return output


def _stored_mni_baseline_by_roi(records: list[dict[str, Any]]) -> dict[str, tuple[float, str]]:
    values_by_roi: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not record.get("analysis_complete"):
            continue
        roi_label = str(record.get("roi_label") or "")
        value = record.get("mni_baseline_roi_mean")
        if not roi_label or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values_by_roi[roi_label].append(numeric)
    return {
        roi_label: (
            float(median(values) or 0.0),
            "subject_metrics.extended_metrics.mni_baseline_roi_mean",
        )
        for roi_label, values in values_by_roi.items()
        if values and median(values) is not None
    }


def _build_mni_transfer_summary(
    subject_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    batch_root: Path | None = None,
    mni_baseline_post_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    summary_rows = _summary_from_subject_metric_rows(
        subject_rows,
        output_prefix="roi_mean_v_per_m",
        summary_prefix="roi_mean_v_per_m",
    )
    baselines = _stored_mni_baseline_by_roi(records)
    baselines.update(
        _mni_post_baseline_by_roi(
            records,
            batch_root=batch_root,
            mni_baseline_post_root=mni_baseline_post_root,
        )
    )
    output: list[dict[str, Any]] = []
    for row in summary_rows:
        roi_label = str(row["roi_label"])
        baseline_entry = baselines.get(roi_label)
        population_median = row.get("population_median_roi_mean_v_per_m")
        if baseline_entry is None or not isinstance(population_median, (int, float)):
            continue
        baseline, source = baseline_entry
        row = dict(row)
        row["mni_baseline_roi_mean_v_per_m"] = baseline
        row["mni_minus_population_median_v_per_m"] = baseline - float(population_median)
        row["mni_baseline_source"] = source
        output.append(row)
    return output


def _rows_by_roi(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("roi_label") or "ROI")].append(row)
    return dict(sorted(grouped.items()))


def _finite_row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _draw_dashed_hline(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    *,
    fill: str,
    width: int = 4,
    dash: int = 24,
    gap: int = 14,
) -> None:
    x = x0
    while x < x1:
        draw.line((x, y, min(x + dash, x1), y), fill=fill, width=width)
        x += dash + gap


def _draw_violin_distribution(
    draw: ImageDraw.ImageDraw,
    values: list[float],
    *,
    center_x: int,
    max_width: int,
    y0: int,
    y1: int,
    min_value: float,
    max_value: float,
    color: str,
    light: str,
) -> None:
    if not values:
        return

    bins = min(26, max(8, len(values) // 3))
    span = max(max_value - min_value, 1e-9)
    counts = [0.0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, max(0, int(((value - min_value) / span) * bins)))
        counts[index] += 1.0
    for _ in range(2):
        counts = [
            (counts[max(0, idx - 1)] + counts[idx] * 2.0 + counts[min(bins - 1, idx + 1)]) / 4.0
            for idx in range(bins)
        ]
    max_count = max(counts) if counts else 1.0
    if max_count <= 0:
        max_count = 1.0

    right: list[tuple[int, int]] = []
    left: list[tuple[int, int]] = []
    for idx, count in enumerate(counts):
        center_value = min_value + ((idx + 0.5) / bins) * span
        y = y_scale(center_value, min_value, max_value, y0, y1)
        half_width = max(4, int(max_width * (count / max_count)))
        right.append((center_x + half_width, y))
        left.append((center_x - half_width, y))
    polygon = right + list(reversed(left))
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=light)
        draw.line(polygon + [polygon[0]], fill="#8A98A8", width=3)

    values_sorted = sorted(values)
    p5 = quantile(values_sorted, 0.05)
    q1 = quantile(values_sorted, 0.25)
    med = median(values_sorted)
    q3 = quantile(values_sorted, 0.75)
    p95 = quantile(values_sorted, 0.95)
    if p5 is not None and p95 is not None:
        draw.line(
            (
                center_x,
                y_scale(p5, min_value, max_value, y0, y1),
                center_x,
                y_scale(p95, min_value, max_value, y0, y1),
            ),
            fill=INK,
            width=4,
        )
    for value, width in [(q1, 210), (med, 260), (q3, 210)]:
        if value is None:
            continue
        y = y_scale(value, min_value, max_value, y0, y1)
        draw.line((center_x - width // 2, y, center_x + width // 2, y), fill=INK, width=5)

    for idx, value in enumerate(values):
        jitter = (((idx * 37) % 25) - 12) * 2
        y = y_scale(value, min_value, max_value, y0, y1)
        draw.ellipse((center_x + jitter - 5, y - 5, center_x + jitter + 5, y + 5), fill=color)


def _axis_ticks(min_value: float, max_value: float, count: int = 4) -> list[float]:
    if max_value <= min_value:
        return [min_value]
    return [min_value + (max_value - min_value) * idx / count for idx in range(count + 1)]


def _normalize_name_for_file(value: str | None) -> str:
    if not value:
        return "roi"
    normalized = "".join(char if char.isalnum() else "_" for char in value.strip().replace(" ", "_"))
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_") or "roi"


def _label_from_roi_prefix(value: str) -> str:
    return " ".join(part for part in value.replace("-", "_").split("_") if part)


def _short_label(value: str) -> str:
    words = [word for word in re.split(r"[\s_\-]+", value) if word]
    if not words:
        return "ROI"
    if len(words) == 1:
        return words[0][:6]
    return "".join(word[0].upper() for word in words[:3])


def _parse_repeat_value(value: str | int) -> int:
    return int(str(value).strip())


def _repeat_sort_key(value: str) -> tuple[int, str]:
    try:
        return (_parse_repeat_value(value), value)
    except ValueError:
        return (10**9, value)


def _canonical_expected_repeats(values: Sequence[str] | None, present_repeats: Iterable[str]) -> list[str]:
    if values is None:
        return sorted(set(present_repeats), key=_repeat_sort_key)
    return [str(value) for value in values]


def _selected_repeat_values(values: Sequence[str] | None) -> set[int] | None:
    if values is None:
        return None
    return {_parse_repeat_value(value) for value in values}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _read_batch_summary(batch_root: Path, summary_filename: str | None) -> dict[str, Any] | None:
    if not summary_filename:
        return None
    summary_path = Path(summary_filename).expanduser()
    if not summary_path.is_absolute():
        summary_path = batch_root / summary_path
    if not summary_path.is_file():
        return None
    try:
        return load_json(summary_path)
    except Exception:
        return None


def iter_dataset_dirs(
    batch_root: Path,
    *,
    dataset_glob: str,
    expected_repeats: Sequence[str] | None,
) -> list[tuple[Path, str, str]]:
    selected = _selected_repeat_values(expected_repeats)
    datasets: list[tuple[Path, str, str]] = []
    for path in batch_root.glob(dataset_glob):
        if not path.is_dir():
            continue
        match = REPEAT_DATASET_PATTERN.match(path.name)
        if not match:
            continue
        repeat = match.group("repeat")
        if selected is not None and _parse_repeat_value(repeat) not in selected:
            continue
        datasets.append((path.resolve(), match.group("roi_prefix"), repeat))
    return sorted(datasets, key=lambda item: (item[1].casefold(), _repeat_sort_key(item[2]), item[0].name))


def iter_subject_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith("sub-"))


def extract_roi_metrics(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    rois = payload.get("rois")
    if isinstance(rois, dict) and rois:
        target = payload.get("target_roi")
        if isinstance(target, str) and target in rois:
            value = rois.get(target)
            return target, value if isinstance(value, dict) else {}
        name = next(iter(rois))
        value = rois.get(name)
        return str(name), value if isinstance(value, dict) else {}
    target = payload.get("target_roi")
    return str(target) if target is not None else None, {}


def _failed_subjects_by_dataset(batch_summary: dict[str, Any] | None) -> dict[str, set[str]]:
    failed_by_dataset: dict[str, set[str]] = defaultdict(set)
    if not isinstance(batch_summary, dict):
        return failed_by_dataset
    for result in batch_summary.get("dataset_stage_results", []) or []:
        if not isinstance(result, dict):
            continue
        dataset_name = str(result.get("dataset_name", ""))
        subject_stage = result.get("stages", {}).get("subject_level", {})
        if not isinstance(subject_stage, dict):
            continue
        for item in subject_stage.get("failed_subjects", []) or []:
            if isinstance(item, list) and item:
                failed_by_dataset[dataset_name].add(str(item[0]))
            elif isinstance(item, str):
                failed_by_dataset[dataset_name].add(item)
    return failed_by_dataset


def _get_qc_threshold_reason(threshold_qc: dict[str, Any], key: str) -> str | None:
    value = threshold_qc.get(key)
    if isinstance(value, dict):
        reason = value.get("reason")
        return str(reason) if reason is not None else None
    return None


def collect_records(
    batch_root: Path,
    *,
    dataset_glob: str,
    expected_repeats: Sequence[str] | None,
    batch_summary: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    failed_by_dataset = _failed_subjects_by_dataset(batch_summary)
    present_repeats: list[str] = []

    for dataset_dir, roi_prefix, repeat in iter_dataset_dirs(
        batch_root,
        dataset_glob=dataset_glob,
        expected_repeats=expected_repeats,
    ):
        present_repeats.append(repeat)
        subject_dirs = iter_subject_dirs(dataset_dir)
        metric_files = 0
        complete_subjects: set[str] = set()
        nonblocking_count = 0
        fallback_roi_label = _label_from_roi_prefix(roi_prefix)
        dataset_roi_label = fallback_roi_label
        failed_subjects = failed_by_dataset.get(dataset_dir.name, set())
        record_start_index = len(records)

        for subject_dir in subject_dirs:
            metrics_path = subject_dir / "anat" / "post" / "subject_metrics.json"
            if not metrics_path.is_file():
                records.append(
                    {
                        "roi_prefix": roi_prefix,
                        "roi_label": fallback_roi_label,
                        "repeat": repeat,
                        "dataset": dataset_dir.name,
                        "subject": subject_dir.name,
                        "has_metrics": False,
                        "analysis_complete": False,
                        "missing_reason": "failed_subject"
                        if subject_dir.name in failed_subjects
                        else "missing_subject_metrics",
                    }
                )
                continue

            metric_files += 1
            payload = load_json(metrics_path)
            roi_name, roi_metrics = extract_roi_metrics(payload)
            roi_label = roi_name or fallback_roi_label
            dataset_roi_label = roi_label
            subject_meta = payload.get("subject_metrics_meta")
            if not isinstance(subject_meta, dict):
                subject_meta = {}
            extended_meta = payload.get("extended_metrics_meta")
            if not isinstance(extended_meta, dict):
                extended_meta = {}
            qc_meta = payload.get("qc_meta")
            if not isinstance(qc_meta, dict):
                qc_meta = {}
            extended = payload.get("extended_metrics")
            if not isinstance(extended, dict):
                extended = {}
            if not isinstance(roi_metrics, dict):
                roi_metrics = {}

            threshold_qc = roi_metrics.get("threshold_qc")
            if not isinstance(threshold_qc, dict):
                root_threshold_qc = payload.get("threshold_qc")
                threshold_qc = (
                    root_threshold_qc.get("rois", {}).get(roi_name, {})
                    if isinstance(root_threshold_qc, dict) and roi_name
                    else {}
                )
                if not isinstance(threshold_qc, dict):
                    threshold_qc = {}

            nonblocking = subject_meta.get("nonblocking_qc_checks") or []
            blocking = subject_meta.get("blocking_qc_checks") or []
            if not isinstance(nonblocking, list):
                nonblocking = []
            if not isinstance(blocking, list):
                blocking = []
            nonblocking_count += len(nonblocking)

            analysis_complete = subject_metrics_payload_analysis_complete(payload)
            if analysis_complete:
                complete_subjects.add(subject_dir.name)

            roi_threshold_voxels = safe_float(roi_metrics.get("focality_in_roi_voxels_gt_threshold"))
            electrode_file = metrics_path.parent / f"{_normalize_name_for_file(roi_name)}_electrode_distances.json"
            records.append(
                {
                    "roi_prefix": roi_prefix,
                    "roi_label": roi_label,
                    "repeat": repeat,
                    "dataset": dataset_dir.name,
                    "subject": payload.get("subject") or subject_dir.name,
                    "has_metrics": True,
                    "analysis_complete": analysis_complete,
                    "metrics_path": str(metrics_path),
                    "status": subject_meta.get("status"),
                    "extended_status": subject_meta.get("extended_metrics_status")
                    or extended_meta.get("status"),
                    "qc_status": subject_meta.get("qc_status") or qc_meta.get("status"),
                    "nonblocking_qc_checks": " ".join(str(item) for item in nonblocking),
                    "blocking_qc_checks": " ".join(str(item) for item in blocking),
                    "roi_name": roi_name,
                    "threshold_metric_reason": _get_qc_threshold_reason(threshold_qc, "metric_threshold"),
                    "threshold_overlay_reason": _get_qc_threshold_reason(threshold_qc, "overlay_threshold"),
                    "roi_peak": safe_float(extended.get("roi_peak")),
                    "roi_mean": safe_float(extended.get("roi_mean")),
                    "mni_baseline_roi_peak": safe_float(extended.get("mni_baseline_roi_peak")),
                    "mni_baseline_roi_mean": safe_float(extended.get("mni_baseline_roi_mean")),
                    "focality_voxels_gt_threshold": safe_float(extended.get("focality_voxels_gt_threshold")),
                    "focality_volume_mm3_gt_threshold": safe_float(
                        extended.get("focality_volume_mm3_gt_threshold")
                    ),
                    "electrode_distance_count": safe_float(extended.get("electrode_distance_count")),
                    "electrode_distance_mean_mm": safe_float(extended.get("electrode_distance_mean_mm")),
                    "electrode_distance_min_mm": safe_float(extended.get("electrode_distance_min_mm")),
                    "electrode_distance_max_mm": safe_float(extended.get("electrode_distance_max_mm")),
                    "electrode_distance_json": str(electrode_file) if electrode_file.is_file() else "",
                    "roi_voxels": safe_float(roi_metrics.get("roi_voxels")),
                    "overlap_fraction": safe_float(roi_metrics.get("overlap_fraction")),
                    "focality_in_roi_voxels_gt_threshold": roi_threshold_voxels,
                    "focality_in_roi_volume_mm3_gt_threshold": safe_float(
                        roi_metrics.get("focality_in_roi_volume_mm3_gt_threshold")
                    ),
                    "top_percentile_voxels": safe_float(payload.get("top_percentile_voxels")),
                    "zero_roi_threshold": bool(roi_threshold_voxels == 0),
                }
            )

        for record in records[record_start_index:]:
            if record.get("roi_label") == fallback_roi_label:
                record["roi_label"] = dataset_roi_label

        repeat_rows.append(
            {
                "roi_prefix": roi_prefix,
                "roi_label": dataset_roi_label,
                "repeat": repeat,
                "dataset": dataset_dir.name,
                "subject_dirs": len(subject_dirs),
                "metric_files": metric_files,
                "complete_subjects": len(complete_subjects),
                "missing_metrics": len(subject_dirs) - metric_files,
                "nonblocking_qc_events": nonblocking_count,
            }
        )

    return records, repeat_rows, present_repeats


def _style_lookup(roi_labels: Iterable[str]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for index, label in enumerate(roi_labels):
        color, light = PALETTE[index % len(PALETTE)]
        output[label] = {"color": color, "light": light, "short": _short_label(label)}
    return output


def _apply_styles(records: list[dict[str, Any]], repeat_rows: list[dict[str, Any]], styles: dict[str, dict[str, str]]) -> None:
    for row in records + repeat_rows:
        style = styles.get(str(row.get("roi_label")), styles.get(next(iter(styles), ""), {}))
        row["roi_color"] = style.get("color", BLUE)
        row["roi_light"] = style.get("light", LIGHT_BLUE)
        row["roi_short"] = style.get("short", _short_label(str(row.get("roi_label") or "ROI")))


def numeric_values(records: list[dict[str, Any]], roi_label: str, key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("roi_label") != roi_label or not record.get("analysis_complete"):
            continue
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def within_subject_cv(records: list[dict[str, Any]], roi_label: str, key: str) -> tuple[float | None, float | None]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("roi_label") != roi_label or not record.get("analysis_complete"):
            continue
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            by_subject[str(record.get("subject"))].append(float(value))

    cvs: list[float] = []
    for values in by_subject.values():
        if len(values) < 2:
            continue
        m = mean(values)
        s = stdev(values)
        if m is None or s is None or abs(m) < 1e-12:
            continue
        cvs.append((s / abs(m)) * 100.0)
    return mean(cvs), median(cvs)


def _repeatability_status_for_roi(batch_summary: dict[str, Any] | None, roi_label: str) -> dict[str, Any]:
    if not isinstance(batch_summary, dict):
        return {}
    for result in batch_summary.get("repeatability_results", []) or []:
        if not isinstance(result, dict):
            continue
        if str(result.get("roi_name")) == roi_label:
            return {
                "repeatability_status": result.get("status"),
                "repeatability_rows": result.get("rows_analysed"),
                "repeatability_dataset_count": result.get("dataset_count"),
            }
    return {}


def build_roi_summary(
    records: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    expected_repeats: Sequence[str],
    batch_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    labels = sorted({str(row.get("roi_label")) for row in records + repeat_rows if row.get("roi_label")})
    styles = _style_lookup(labels)
    _apply_styles(records, repeat_rows, styles)

    output: list[dict[str, Any]] = []
    for label in labels:
        style = styles[label]
        roi_records = [
            record
            for record in records
            if record.get("roi_label") == label and record.get("analysis_complete")
        ]
        all_roi_records = [record for record in records if record.get("roi_label") == label]
        rr = [row for row in repeat_rows if row.get("roi_label") == label]
        present_repeats = sorted({str(row["repeat"]) for row in rr}, key=_repeat_sort_key)

        complete_sets: list[set[str]] = []
        for repeat in present_repeats:
            complete_sets.append(
                {
                    str(record.get("subject"))
                    for record in roi_records
                    if record.get("repeat") == repeat
                }
            )
        complete_case = set.intersection(*complete_sets) if complete_sets else set()

        row: dict[str, Any] = {
            "roi_label": label,
            "roi_short": style["short"],
            "color": style["color"],
            "light": style["light"],
            "datasets_present": len(present_repeats),
            "expected_repeats": len(expected_repeats),
            "missing_repeats": " ".join(repeat for repeat in expected_repeats if repeat not in present_repeats),
            "subject_slots": len(all_roi_records),
            "metric_files": sum(1 for record in all_roi_records if record.get("has_metrics")),
            "complete_metrics": len(roi_records),
            "complete_case_subjects": len(complete_case),
            "missing_metrics": sum(1 for record in all_roi_records if not record.get("has_metrics")),
            "qc_complete": sum(1 for record in roi_records if record.get("qc_status") == "complete"),
            "qc_partial": sum(1 for record in roi_records if record.get("qc_status") == "partial"),
            "nonblocking_overlay_events": sum(
                1 for record in roi_records if "overlays" in str(record.get("nonblocking_qc_checks"))
            ),
            "zero_roi_threshold_rows": sum(1 for record in roi_records if record.get("zero_roi_threshold")),
            "nonzero_roi_threshold_rows": sum(
                1
                for record in roi_records
                if record.get("focality_in_roi_voxels_gt_threshold") is not None
                and float(record.get("focality_in_roi_voxels_gt_threshold")) > 0
            ),
            "electrode_count_four_rows": sum(
                1 for record in roi_records if record.get("electrode_distance_count") == 4
            ),
            "electrode_json_files": sum(1 for record in roi_records if record.get("electrode_distance_json")),
        }
        row.update(_repeatability_status_for_roi(batch_summary, label))
        for metric in SUMMARY_METRICS:
            values = numeric_values(records, label, metric)
            stats = metric_stats(values)
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
            cv_mean, cv_median = within_subject_cv(records, label, metric)
            row[f"{metric}_within_subject_cv_mean_pct"] = cv_mean
            row[f"{metric}_within_subject_cv_median_pct"] = cv_median
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def x_scale(value: float, min_value: float, max_value: float, x0: int, x1: int) -> int:
    if max_value <= min_value:
        return (x0 + x1) // 2
    return int(x0 + ((value - min_value) / (max_value - min_value)) * (x1 - x0))


def y_scale(value: float, min_value: float, max_value: float, y0: int, y1: int) -> int:
    if max_value <= min_value:
        return (y0 + y1) // 2
    return int(y1 - ((value - min_value) / (max_value - min_value)) * (y1 - y0))


def draw_barh(
    draw: ImageDraw.ImageDraw,
    label: str,
    value: float,
    max_value: float,
    x: int,
    y: int,
    w: int,
    h: int,
    color: str,
    value_label: str,
) -> None:
    if label:
        draw.text((x, y - 4), label, font=FONT_SMALL, fill=INK)
    bar_x = x + 290
    bar_w = max(w - 450, 40)
    draw.rounded_rectangle((bar_x, y, bar_x + bar_w, y + h), radius=h // 2, fill="#E8EEF7")
    filled = int(bar_w * min(max(value / max_value, 0.0), 1.0)) if max_value else 0
    draw.rounded_rectangle((bar_x, y, bar_x + filled, y + h), radius=h // 2, fill=color)
    draw.text((bar_x + bar_w + 22, y - 4), value_label, font=FONT_SMALL, fill=MUTED)


def _save(image: Image.Image, path: Path) -> Path:
    image.save(path, optimize=True)
    return path


def make_completion_matrix(
    roi_summary: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    expected_repeats: Sequence[str],
    figures_dir: Path,
) -> Path:
    path = figures_dir / "01_completion_matrix.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Processing coverage",
        "Subject metrics coverage by ROI and repeat. Counts show analysis-complete rows among present subject folders.",
    )

    row_lookup = {(row["roi_label"], row["repeat"]): row for row in repeat_rows}
    x0, y0 = 470, 300
    available_w = 1720
    cell_w = min(116, max(64, available_w // max(len(expected_repeats), 1)))
    cell_h = 84
    row_gap = min(150, max(96, 780 // max(len(roi_summary), 1)))
    draw.text((80, 232), "ROI", font=FONT_H3, fill=INK)
    draw.text((x0, 232), "Repeat", font=FONT_H3, fill=INK)
    for index, repeat in enumerate(expected_repeats):
        draw.text((x0 + index * cell_w + 24, 250), str(repeat), font=FONT_SMALL, fill=MUTED)

    for ridx, row in enumerate(roi_summary):
        y = y0 + ridx * row_gap
        draw.text((80, y + 22), str(row["roi_label"]), font=FONT_H2, fill=INK)
        draw.text(
            (80, y + 68),
            f"{int(row['complete_metrics']):,} complete metrics | {int(row['complete_case_subjects'])} complete-case subjects",
            font=FONT_SMALL,
            fill=MUTED,
        )
        for cidx, repeat in enumerate(expected_repeats):
            x = x0 + cidx * cell_w
            rep = row_lookup.get((row["roi_label"], repeat))
            if rep is None:
                fill, outline, text, sub = "#E7EAF0", "#CBD5E1", "NA", "missing"
            else:
                fill, outline = row["color"], row["color"]
                text = f"{int(rep['complete_subjects'])}"
                sub = f"/{int(rep['subject_dirs'])}"
            draw.rounded_rectangle((x, y, x + min(88, cell_w - 8), y + cell_h), radius=14, fill=fill, outline=outline, width=2)
            box_w = min(88, cell_w - 8)
            tw, _ = text_size(draw, text, FONT_H3)
            draw.text((x + box_w / 2 - tw / 2, y + 15), text, font=FONT_H3, fill=WHITE if fill != "#E7EAF0" else MUTED)
            sw, _ = text_size(draw, sub, FONT_TINY)
            draw.text((x + box_w / 2 - sw / 2, y + 52), sub, font=FONT_TINY, fill=WHITE if fill != "#E7EAF0" else MUTED)

    rounded_panel(draw, (80, 1090, 2320, 1238), fill=PANEL)
    draw.text((120, 1125), "Interpretation", font=FONT_H3, fill=INK)
    missing_total = sum(int(row["missing_metrics"]) for row in roi_summary)
    draw_wrapped_text(
        draw,
        f"Generated from {len(roi_summary)} ROI group(s). Missing metrics count subject folders without subject_metrics.json; partial overlay-only QC rows remain analysis-complete when scalar metrics are valid. Missing metrics: {missing_total:,}.",
        (120, 1166),
        2100,
        FONT_BODY,
        MUTED,
    )
    draw_footer(draw)
    return _save(image, path)


def make_status_bars(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "02_status_bars.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, "Subject-level completion", "Analysis-complete metrics retained for every available subject-run.")

    x = 120
    row_gap = min(178, max(120, 850 // max(len(roi_summary), 1)))
    y = 260
    for row in roi_summary:
        slots = int(row["subject_slots"])
        complete = int(row["complete_metrics"])
        missing = int(row["missing_metrics"])
        rounded_panel(draw, (x, y, 2280, y + 150), fill=PANEL)
        draw.text((x + 28, y + 28), str(row["roi_label"]), font=FONT_H3, fill=INK)
        draw_barh(
            draw,
            "",
            complete,
            max(slots, 1),
            x + 5,
            y + 78,
            1630,
            34,
            row["color"],
            f"{complete:,}/{slots:,} complete",
        )
        if missing:
            draw.text((1905, y + 26), f"{missing} missing", font=FONT_H3, fill=RED)
            draw.text((1907, y + 68), "metrics absent", font=FONT_SMALL, fill=MUTED)
        else:
            draw.text((1905, y + 40), "no missing", font=FONT_H3, fill=GREEN)
        y += row_gap

    rounded_panel(draw, (120, 1160, 2280, 1250), fill=LIGHT_GREEN, outline="#C9E9D3")
    draw.text((160, 1188), "Overlay-only QC partial status is non-blocking when extended scalar metrics are complete.", font=FONT_BODY, fill=INK)
    draw_footer(draw)
    return _save(image, path)


def make_metric_overview(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "03_field_metric_overview.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, "Field strength by target", "Mean and peak ROI fields are computed over analysis-complete subject-runs.")

    plot_x, plot_y, plot_w, plot_h = 160, 285, 2080, 780
    rounded_panel(draw, (plot_x - 40, plot_y - 40, plot_x + plot_w + 40, plot_y + plot_h + 100), fill=PANEL)
    max_value = max([float(row.get("roi_peak_mean") or 0) for row in roi_summary] + [0.01]) * 1.18
    axis_y0 = plot_y
    axis_y1 = plot_y + plot_h
    draw.line((plot_x, axis_y1, plot_x + plot_w, axis_y1), fill=GRID, width=3)
    draw.line((plot_x, axis_y0, plot_x, axis_y1), fill=GRID, width=3)
    tick_count = 5
    for tick_index in range(tick_count + 1):
        tick = max_value * tick_index / tick_count
        yy = y_scale(tick, 0, max_value, axis_y0, axis_y1)
        draw.line((plot_x - 10, yy, plot_x + plot_w, yy), fill="#EEF3F9", width=2)
        draw.text((plot_x - 82, yy - 12), f"{tick:.2f}", font=FONT_TINY, fill=MUTED)

    group_w = plot_w / max(len(roi_summary), 1)
    bar_w = 70
    for idx, row in enumerate(roi_summary):
        gx = int(plot_x + idx * group_w + group_w / 2)
        peak = float(row.get("roi_peak_mean") or 0)
        mean_v = float(row.get("roi_mean_mean") or 0)
        for offset, value, color in [
            (-bar_w // 2 - 8, peak, row["color"]),
            (bar_w // 2 + 8, mean_v, "#8EA2B8"),
        ]:
            x0 = gx + offset - bar_w // 2
            y0 = y_scale(value, 0, max_value, axis_y0, axis_y1)
            draw.rounded_rectangle((x0, y0, x0 + bar_w, axis_y1), radius=10, fill=color)
            text = f"{value:.3f}"
            tw, _ = text_size(draw, text, FONT_TINY)
            draw.text((x0 + bar_w / 2 - tw / 2, y0 - 30), text, font=FONT_TINY, fill=INK)
        label_lines = textwrap.wrap(str(row["roi_label"]), width=14)
        ly = axis_y1 + 28
        for line in label_lines:
            tw, _ = text_size(draw, line, FONT_SMALL)
            draw.text((gx - tw / 2, ly), line, font=FONT_SMALL, fill=INK)
            ly += 24

    draw.rectangle((plot_x + plot_w - 430, plot_y + 25, plot_x + plot_w - 398, plot_y + 45), fill=BLUE)
    draw.text((plot_x + plot_w - 380, plot_y + 20), "ROI peak", font=FONT_SMALL, fill=INK)
    draw.rectangle((plot_x + plot_w - 270, plot_y + 25, plot_x + plot_w - 238, plot_y + 45), fill="#8EA2B8")
    draw.text((plot_x + plot_w - 220, plot_y + 20), "ROI mean", font=FONT_SMALL, fill=INK)
    draw.text((160, 1125), "V/m", font=FONT_BODY, fill=MUTED)
    draw_footer(draw)
    return _save(image, path)


def make_threshold_support(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "04_threshold_support.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Threshold-support QC",
        "Rows with no ROI voxels above the hard threshold are retained as complete metrics and represented as zero.",
    )

    x0, y0 = 150, 285
    max_rows = max([int(row["complete_metrics"]) for row in roi_summary] + [1])
    row_gap = min(168, max(118, 850 // max(len(roi_summary), 1)))
    for idx, row in enumerate(roi_summary):
        y = y0 + idx * row_gap
        rounded_panel(draw, (x0 - 30, y - 24, 2250, y + 116), fill=PANEL)
        draw.text((x0, y), str(row["roi_label"]), font=FONT_H3, fill=INK)
        nonzero = int(row["nonzero_roi_threshold_rows"])
        zero = int(row["zero_roi_threshold_rows"])
        bar_x, bar_y, bar_w, bar_h = x0 + 420, y + 12, 1250, 46
        draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=20, fill="#E8EEF7")
        nz_w = int(bar_w * nonzero / max_rows)
        zero_w = int(bar_w * zero / max_rows)
        draw.rounded_rectangle((bar_x, bar_y, bar_x + nz_w, bar_y + bar_h), radius=20, fill=row["color"])
        if zero_w > 0:
            draw.rectangle((bar_x + nz_w, bar_y, bar_x + nz_w + zero_w, bar_y + bar_h), fill=RED)
        draw.text((bar_x + bar_w + 32, y - 2), f"{nonzero:,} nonzero", font=FONT_SMALL, fill=GREEN)
        draw.text((bar_x + bar_w + 32, y + 34), f"{zero:,} zero", font=FONT_SMALL, fill=RED if zero else MUTED)
        if row["nonblocking_overlay_events"]:
            draw.text(
                (bar_x + bar_w + 32, y + 70),
                f"{int(row['nonblocking_overlay_events'])} overlay-only QC",
                font=FONT_TINY,
                fill=PURPLE,
            )

    rounded_panel(draw, (150, 1164, 2250, 1248), fill=LIGHT_PURPLE, outline="#E8CDE0")
    total_zero = sum(int(row["zero_roi_threshold_rows"]) for row in roi_summary)
    draw.text((185, 1190), f"Zero-threshold rows retained in this figure set: {total_zero:,}.", font=FONT_BODY, fill=INK)
    draw_footer(draw)
    return _save(image, path)


def make_repeatability_cv(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "05_repeatability_cv.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, "Within-subject repeatability", "Coefficient of variation is computed across repeats for each subject, then averaged over subjects.")

    panels = [
        ("Field and coverage CV", ["roi_peak", "roi_mean", "overlap_fraction"], (120, 270, 2260, 735), 35.0),
        ("Threshold focality CV", ["focality_in_roi_voxels_gt_threshold"], (120, 820, 2260, 1195), 240.0),
    ]
    metric_colors = {
        "roi_peak": BLUE,
        "roi_mean": "#8EA2B8",
        "overlap_fraction": GOLD,
        "focality_in_roi_voxels_gt_threshold": RED,
    }

    for title, metrics, box, max_value in panels:
        x0, y0, x1, y1 = box
        rounded_panel(draw, box, fill=PANEL)
        draw.text((x0 + 30, y0 + 22), title, font=FONT_H3, fill=INK)
        plot_x0, plot_y0 = x0 + 80, y0 + 90
        plot_x1, plot_y1 = x1 - 60, y1 - 70
        draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=GRID, width=2)
        step = 10 if max_value <= 40 else 60
        for tick in range(0, int(max_value) + 1, step):
            yy = y_scale(tick, 0, max_value, plot_y0, plot_y1)
            draw.line((plot_x0, yy, plot_x1, yy), fill="#EEF3F9", width=1)
            draw.text((plot_x0 - 54, yy - 10), str(tick), font=FONT_TINY, fill=MUTED)
        group_w = (plot_x1 - plot_x0) / max(len(roi_summary), 1)
        bar_w = 28 if len(metrics) > 1 else 72
        for ridx, row in enumerate(roi_summary):
            gx = int(plot_x0 + ridx * group_w + group_w / 2)
            total_w = len(metrics) * (bar_w + 12) - 12
            for midx, metric in enumerate(metrics):
                value = row.get(f"{metric}_within_subject_cv_mean_pct")
                numeric = float(value) if value is not None else 0.0
                x = int(gx - total_w / 2 + midx * (bar_w + 12))
                yy = y_scale(min(numeric, max_value), 0, max_value, plot_y0, plot_y1)
                draw.rounded_rectangle((x, yy, x + bar_w, plot_y1), radius=8, fill=metric_colors[metric])
                label = f"{numeric:.1f}"
                tw, _ = text_size(draw, label, FONT_TINY)
                draw.text((x + bar_w / 2 - tw / 2, yy - 24), label, font=FONT_TINY, fill=INK)
            tw, _ = text_size(draw, str(row["roi_short"]), FONT_SMALL)
            draw.text((gx - tw / 2, plot_y1 + 22), str(row["roi_short"]), font=FONT_SMALL, fill=INK)
        lx = x1 - 840
        ly = y0 + 28
        for metric in metrics:
            draw.rectangle((lx, ly + 4, lx + 24, ly + 24), fill=metric_colors[metric])
            draw.text((lx + 34, ly), METRIC_LABELS[metric], font=FONT_TINY, fill=MUTED)
            lx += 300

    draw_footer(draw)
    return _save(image, path)


def make_electrode_summary(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "06_electrode_summary.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, "Electrode-distance metrics", "Mean target-to-electrode distance over analysis-complete subject-runs.")

    x0, y0, plot_w, plot_h = 170, 300, 2050, 620
    rounded_panel(draw, (120, 240, 2280, 1015), fill=PANEL)
    values = [float(row.get("electrode_distance_mean_mm_mean") or 0) for row in roi_summary]
    max_value = max(values + [1.0]) * 1.2
    axis_y1 = y0 + plot_h
    axis_y0 = y0
    draw.line((x0, axis_y1, x0 + plot_w, axis_y1), fill=GRID, width=3)
    step = max(10, int(max_value // 5) or 10)
    for tick in range(0, int(max_value) + step, step):
        yy = y_scale(tick, 0, max_value, axis_y0, axis_y1)
        draw.line((x0 - 8, yy, x0 + plot_w, yy), fill="#EEF3F9", width=1)
        draw.text((x0 - 60, yy - 10), str(tick), font=FONT_TINY, fill=MUTED)

    group_w = plot_w / max(len(roi_summary), 1)
    for idx, row in enumerate(roi_summary):
        value = float(row.get("electrode_distance_mean_mm_mean") or 0)
        gx = int(x0 + idx * group_w + group_w / 2)
        bw = 118
        yy = y_scale(value, 0, max_value, axis_y0, axis_y1)
        draw.rounded_rectangle((gx - bw // 2, yy, gx + bw // 2, axis_y1), radius=14, fill=row["color"])
        label = f"{value:.1f} mm"
        tw, _ = text_size(draw, label, FONT_SMALL)
        draw.text((gx - tw / 2, yy - 32), label, font=FONT_SMALL, fill=INK)
        tw, _ = text_size(draw, str(row["roi_short"]), FONT_SMALL)
        draw.text((gx - tw / 2, axis_y1 + 24), str(row["roi_short"]), font=FONT_SMALL, fill=INK)

    rounded_panel(draw, (120, 1065, 2280, 1218), fill=LIGHT_GREEN, outline="#C9E9D3")
    total_rows = sum(int(row["complete_metrics"]) for row in roi_summary)
    electrode_rows = sum(int(row["electrode_count_four_rows"]) for row in roi_summary)
    files = sum(int(row["electrode_json_files"]) for row in roi_summary)
    draw.text((170, 1104), f"{electrode_rows:,}/{total_rows:,} completed subject-runs have exactly four electrodes.", font=FONT_H3, fill=INK)
    draw.text((170, 1158), f"{files:,} per-subject electrode-distance JSON files were found.", font=FONT_BODY, fill=MUTED)
    draw_footer(draw)
    return _save(image, path)


def make_boxplot(records: list[dict[str, Any]], roi_summary: list[dict[str, Any]], metric: str, title: str, filename: str, figures_dir: Path) -> Path:
    path = figures_dir / filename
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, title, f"Distribution of {METRIC_LABELS[metric].lower()} over analysis-complete subject-runs.")

    x0, x1 = 620, 2190
    y0 = 310
    row_h = min(150, max(100, 700 // max(len(roi_summary), 1)))
    all_values: list[float] = []
    by_roi: list[tuple[dict[str, Any], dict[str, float | None]]] = []
    for row in roi_summary:
        values = numeric_values(records, str(row["roi_label"]), metric)
        all_values.extend(values)
        by_roi.append((row, metric_stats(values)))
    min_value = min(all_values) if all_values else 0.0
    max_value = max(all_values) if all_values else 1.0
    pad = (max_value - min_value) * 0.08 or 1.0
    min_value -= pad
    max_value += pad

    rounded_panel(draw, (90, 245, 2290, 1115), fill=PANEL)
    for tick_index in range(6):
        value = min_value + (max_value - min_value) * tick_index / 5
        xx = x_scale(value, min_value, max_value, x0, x1)
        draw.line((xx, y0 - 35, xx, y0 + row_h * len(by_roi) - 35), fill="#EEF3F9", width=2)
        label = f"{value:.2f}" if metric != "focality_in_roi_voxels_gt_threshold" else fmt_int(value)
        tw, _ = text_size(draw, label, FONT_TINY)
        draw.text((xx - tw / 2, y0 + row_h * len(by_roi) - 10), label, font=FONT_TINY, fill=MUTED)

    for idx, (row, summary) in enumerate(by_roi):
        y = y0 + idx * row_h
        draw.text((135, y + 18), str(row["roi_label"]), font=FONT_H3, fill=INK)
        values = [summary["min"], summary["q1"], summary["median"], summary["q3"], summary["max"]]
        if any(value is None for value in values):
            draw.text((x0, y + 22), "No numeric values", font=FONT_SMALL, fill=MUTED)
            continue
        mn, q1, med, q3, mx = [float(value) for value in values if value is not None]
        min_x = x_scale(mn, min_value, max_value, x0, x1)
        q1_x = x_scale(q1, min_value, max_value, x0, x1)
        med_x = x_scale(med, min_value, max_value, x0, x1)
        q3_x = x_scale(q3, min_value, max_value, x0, x1)
        max_x = x_scale(mx, min_value, max_value, x0, x1)
        mid_y = y + 42
        draw.line((min_x, mid_y, max_x, mid_y), fill=row["color"], width=6)
        draw.rounded_rectangle((q1_x, mid_y - 28, q3_x, mid_y + 28), radius=8, fill=row["color"], outline=row["color"])
        draw.line((med_x, mid_y - 34, med_x, mid_y + 34), fill=WHITE, width=5)
        label = f"median {med:.3f}" if metric != "focality_in_roi_voxels_gt_threshold" else f"median {fmt_int(med)}"
        draw.text((x1 - 280, y + 15), label, font=FONT_SMALL, fill=MUTED)

    draw.text((x0, 1148), METRIC_LABELS[metric], font=FONT_BODY, fill=MUTED)
    draw_footer(draw)
    return _save(image, path)


def make_aggregate_subject_variability(
    subject_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> Path:
    path = figures_dir / "10_aggregate_subject_variability.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Aggregate target coverage by ROI",
        "Each point is one subject: median top-percentile target coverage across repeats. Shaded bands show each subject's repeat IQR.",
    )

    grouped = _rows_by_roi(subject_rows)
    labels = list(grouped)
    styles = _style_lookup(labels)
    rounded_panel(draw, (80, 235, 2320, 1160), fill=PANEL)
    draw.text((140, 275), "Sorted subject medians", font=FONT_H3, fill=INK)
    draw.text((1600, 275), "Population distribution", font=FONT_H3, fill=INK)

    row_h = min(200, max(125, 760 // max(len(labels), 1)))
    y_start = 360
    curve_x0, curve_x1 = 430, 1450
    violin_x = 1875
    axis_max = 100.0

    for idx, label in enumerate(labels):
        rows = sorted(grouped[label], key=lambda row: float(row.get("overlap_percent_median") or 0))
        style = styles[label]
        y0 = y_start + idx * row_h + 12
        y1 = y_start + (idx + 1) * row_h - 18
        draw.text((125, y0 + 24), label, font=FONT_H3, fill=INK)
        draw.text((125, y0 + 64), f"{len(rows)} subjects", font=FONT_SMALL, fill=MUTED)

        if not rows:
            continue

        for tick in [0, 50, 100]:
            yy = y_scale(tick, 0.0, axis_max, y0, y1)
            draw.line((curve_x0 - 20, yy, 2155, yy), fill="#EEF3F9", width=2)
            draw.text((curve_x0 - 84, yy - 12), f"{tick:d}%", font=FONT_TINY, fill=MUTED)

        def row_x(position: int) -> int:
            if len(rows) == 1:
                return (curve_x0 + curve_x1) // 2
            return int(curve_x0 + position * (curve_x1 - curve_x0) / (len(rows) - 1))

        upper: list[tuple[int, int]] = []
        lower: list[tuple[int, int]] = []
        median_points: list[tuple[int, int]] = []
        for pos, row in enumerate(rows):
            x = row_x(pos)
            q1 = float(row.get("overlap_percent_q1") or 0)
            q3 = float(row.get("overlap_percent_q3") or 0)
            med = float(row.get("overlap_percent_median") or 0)
            upper.append((x, y_scale(q3, 0.0, axis_max, y0, y1)))
            lower.append((x, y_scale(q1, 0.0, axis_max, y0, y1)))
            median_points.append((x, y_scale(med, 0.0, axis_max, y0, y1)))
        if len(upper) >= 2:
            draw.polygon(upper + list(reversed(lower)), fill=style["light"])
        if len(median_points) >= 2:
            draw.line(median_points, fill=style["color"], width=5)
        else:
            x, y = median_points[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=style["color"])

        medians = _finite_row_values(rows, "overlap_percent_median")
        _draw_violin_distribution(
            draw,
            medians,
            center_x=violin_x,
            max_width=230,
            y0=y0,
            y1=y1,
            min_value=0.0,
            max_value=axis_max,
            color=style["color"],
            light=style["light"],
        )

    draw.text((curve_x0, 1115), "Subjects sorted by median coverage", font=FONT_SMALL, fill=MUTED)
    draw.text((1600, 1115), "Coverage inside target ROI (%)", font=FONT_SMALL, fill=MUTED)
    draw_footer(draw, "post-processing figures | aggregate subject medians across repeats")
    return _save(image, path)


def make_mni_transfer_distribution(
    subject_rows: list[dict[str, Any]],
    mni_summary: list[dict[str, Any]],
    figures_dir: Path,
) -> Path:
    path = figures_dir / "11_mni_transfer_distribution.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "MNI transfer gap",
        "Each point is one subject: median ROI mean field across repeats. The dashed red line is the MNI152 baseline when available.",
    )

    grouped = _rows_by_roi(subject_rows)
    summary_lookup = {str(row["roi_label"]): row for row in mni_summary}
    labels = [label for label in grouped if label in summary_lookup]
    styles = _style_lookup(labels)
    panel_boxes = [
        (120, 265, 1125, 695),
        (1275, 265, 2280, 695),
        (120, 780, 1125, 1210),
        (1275, 780, 2280, 1210),
    ]

    for idx, label in enumerate(labels[:4]):
        rows = grouped[label]
        values = _finite_row_values(rows, "roi_mean_v_per_m_median")
        if not values:
            continue
        summary = summary_lookup[label]
        baseline = float(summary["mni_baseline_roi_mean_v_per_m"])
        style = styles[label]
        x0, y0, x1, y1 = panel_boxes[idx]
        rounded_panel(draw, (x0, y0, x1, y1), fill=PANEL)
        draw.text((x0 + 36, y0 + 28), label, font=FONT_H3, fill=INK)
        plot_x0, plot_y0 = x0 + 90, y0 + 92
        plot_x1, plot_y1 = x1 - 70, y1 - 82
        min_value = min(min(values), baseline)
        max_value = max(max(values), baseline)
        pad = max((max_value - min_value) * 0.14, 0.015)
        min_value = max(0.0, min_value - pad)
        max_value += pad

        for tick in _axis_ticks(min_value, max_value, count=4):
            yy = y_scale(tick, min_value, max_value, plot_y0, plot_y1)
            draw.line((plot_x0, yy, plot_x1, yy), fill="#EEF3F9", width=2)
            draw.text((plot_x0 - 74, yy - 11), f"{tick:.2f}", font=FONT_TINY, fill=MUTED)
        draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=GRID, width=3)
        draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=GRID, width=3)

        baseline_y = y_scale(baseline, min_value, max_value, plot_y0, plot_y1)
        _draw_dashed_hline(draw, plot_x0, plot_x1, baseline_y, fill=RED)
        draw.text((plot_x1 - 205, baseline_y - 34), "MNI baseline", font=FONT_TINY, fill=RED)
        _draw_violin_distribution(
            draw,
            values,
            center_x=(plot_x0 + plot_x1) // 2,
            max_width=230,
            y0=plot_y0,
            y1=plot_y1,
            min_value=min_value,
            max_value=max_value,
            color=style["color"],
            light=style["light"],
        )
        draw.text((plot_x0, y1 - 52), "Mean field inside target ROI (V/m)", font=FONT_SMALL, fill=MUTED)

    if len(labels) > 4:
        draw.text((120, 1235), f"{len(labels) - 4} additional ROI(s) are included in CSV tables.", font=FONT_SMALL, fill=MUTED)
    draw_footer(draw, "post-processing figures | MNI baseline transfer distribution")
    return _save(image, path)


def make_threshold_edge_case_detail(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "08_threshold_edge_case_detail.png"
    row = max(roi_summary, key=lambda item: int(item.get("zero_roi_threshold_rows") or 0))
    total = int(row.get("complete_metrics") or 0)
    zero = int(row.get("zero_roi_threshold_rows") or 0)
    nonzero = int(row.get("nonzero_roi_threshold_rows") or 0)
    overlay = int(row.get("nonblocking_overlay_events") or 0)

    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "Threshold-zero edge case",
        "The ROI with the most zero-threshold rows is shown here as a direct check of retained zero-valued metrics.",
    )

    cards = [
        (f"{total:,}", "completed subject-runs", GREEN, LIGHT_GREEN),
        (f"{zero:,}", "zero ROI-threshold rows", RED, LIGHT_RED),
        (f"{nonzero:,}", "nonzero ROI-threshold rows", BLUE, LIGHT_BLUE),
        (f"{overlay:,}", "overlay-only QC events", PURPLE, LIGHT_PURPLE),
    ]
    x = 110
    for value, label, color, fill in cards:
        rounded_panel(draw, (x, 300, x + 500, 540), fill=fill, outline=color, radius=28, width=3)
        tw, _ = text_size(draw, value, FONT_TITLE)
        draw.text((x + 250 - tw / 2, 340), value, font=FONT_TITLE, fill=color)
        draw_wrapped_text(draw, label, (x + 55, 430), 390, FONT_BODY, INK)
        x += 560

    rounded_panel(draw, (190, 650, 2210, 1020), fill=PANEL)
    draw.text((240, 705), str(row["roi_label"]), font=FONT_H2, fill=INK)
    bullets = [
        "No subject-run is dropped solely because ROI focality above the hard threshold is zero.",
        "Overlay-generation QC can be non-blocking when scalar metrics are complete.",
        "Downstream statistics receive true zero values for threshold focality.",
    ]
    y = 770
    for bullet in bullets:
        draw.ellipse((250, y + 7, 270, y + 27), fill=PURPLE)
        y = draw_wrapped_text(draw, bullet, (292, y), 1760, FONT_BODY, MUTED) + 14

    rounded_panel(draw, (190, 1080, 2210, 1215), fill=LIGHT_PURPLE, outline="#E8CDE0")
    draw.text(
        (240, 1120),
        f"Mean ROI focality at/above threshold: {fmt_float(row.get('focality_in_roi_voxels_gt_threshold_mean'), 1)} voxels.",
        font=FONT_BODY,
        fill=INK,
    )
    draw_footer(draw)
    return _save(image, path)


def make_summary_table_figure(roi_summary: list[dict[str, Any]], figures_dir: Path) -> Path:
    path = figures_dir / "09_roi_summary_table.png"
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, "Per-ROI headline metrics", "Field and coverage metrics averaged over analysis-complete subject-runs.")

    x0, y0 = 80, 255
    col_w = [400, 270, 270, 310, 310, 310, 260]
    headers = ["ROI", "Rows", "ROI peak", "ROI mean", "Overlap", "ROI voxels >= threshold", "Elec dist"]
    rounded_panel(draw, (x0, y0, 2320, 1085), fill=PANEL)
    x = x0 + 28
    for width, header in zip(col_w, headers):
        draw.text((x, y0 + 35), header, font=FONT_SMALL, fill=MUTED)
        x += width
    draw.line((x0 + 20, y0 + 82, 2300, y0 + 82), fill=GRID, width=3)
    row_step = min(118, max(82, 760 // max(len(roi_summary), 1)))
    y = y0 + 110
    for row in roi_summary:
        x = x0 + 28
        values = [
            row["roi_label"],
            f"{int(row['complete_metrics']):,}",
            fmt_float(row.get("roi_peak_mean"), 3),
            fmt_float(row.get("roi_mean_mean"), 3),
            fmt_float(row.get("overlap_fraction_mean"), 3),
            fmt_int(row.get("focality_in_roi_voxels_gt_threshold_mean")),
            f"{float(row.get('electrode_distance_mean_mm_mean') or 0):.1f} mm",
        ]
        draw.rounded_rectangle((x0 + 18, y - 18, 2302, y + row_step - 40), radius=16, fill=row["light"])
        draw.rectangle((x0 + 18, y - 18, x0 + 28, y + row_step - 40), fill=row["color"])
        for width, value in zip(col_w, values):
            draw_wrapped_text(draw, str(value), (x, y + 3), width - 24, FONT_BODY, INK, line_gap=2)
            x += width
        y += row_step

    draw_footer(draw)
    return _save(image, path)


def generate_figures(
    records: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    roi_summary: list[dict[str, Any]],
    aggregate_subject_rows: list[dict[str, Any]],
    aggregate_mni_subject_rows: list[dict[str, Any]],
    aggregate_mni_summary: list[dict[str, Any]],
    *,
    expected_repeats: Sequence[str],
    figures_dir: Path,
) -> dict[str, Path]:
    figures = {
        "completion_matrix": make_completion_matrix(roi_summary, repeat_rows, expected_repeats, figures_dir),
        "status_bars": make_status_bars(roi_summary, figures_dir),
        "metric_overview": make_metric_overview(roi_summary, figures_dir),
        "threshold_support": make_threshold_support(roi_summary, figures_dir),
        "repeatability_cv": make_repeatability_cv(roi_summary, figures_dir),
        "electrode_summary": make_electrode_summary(roi_summary, figures_dir),
        "roi_peak_boxplot": make_boxplot(records, roi_summary, "roi_peak", "ROI peak field distribution", "07_roi_peak_distribution.png", figures_dir),
        "threshold_edge_case_detail": make_threshold_edge_case_detail(roi_summary, figures_dir),
        "summary_table": make_summary_table_figure(roi_summary, figures_dir),
    }
    if aggregate_subject_rows:
        figures["aggregate_subject_variability"] = make_aggregate_subject_variability(
            aggregate_subject_rows,
            figures_dir,
        )
    if aggregate_mni_subject_rows and aggregate_mni_summary:
        figures["aggregate_mni_transfer_distribution"] = make_mni_transfer_distribution(
            aggregate_mni_subject_rows,
            aggregate_mni_summary,
            figures_dir,
        )
    return figures


def run_figure_generation(
    *,
    batch_root: str | Path,
    output_dir: str | Path | None = None,
    dataset_glob: str = "*_Data_*",
    expected_repeats: Sequence[str] | None = None,
    batch_summary: dict[str, Any] | None = None,
    summary_filename: str | None = "post_processing_batch_summary.json",
    mni_baseline_post_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(batch_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Batch root directory not found: {root}")

    out_dir = Path(output_dir).expanduser() if output_dir is not None else root / "post_processing_figures"
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = batch_summary if batch_summary is not None else _read_batch_summary(root, summary_filename)
    records, repeat_rows, present_repeats = collect_records(
        root,
        dataset_glob=dataset_glob,
        expected_repeats=expected_repeats,
        batch_summary=summary_payload,
    )
    canonical_repeats = _canonical_expected_repeats(expected_repeats, present_repeats)
    roi_summary = build_roi_summary(records, repeat_rows, canonical_repeats, summary_payload)
    if not roi_summary:
        return {
            "stage": FIGURE_GENERATION_STAGE,
            "status": "skipped",
            "reason": f"No repeat datasets matched {dataset_glob!r} under {root}.",
            "output_dir": str(out_dir),
            "figure_count": 0,
            "figures": {},
            "tables": {},
        }

    long_csv = tables_dir / "subject_metrics_long.csv"
    roi_summary_csv = tables_dir / "roi_summary.csv"
    repeat_summary_csv = tables_dir / "repeat_summary.csv"
    aggregate_subject_csv = tables_dir / "aggregate_subject_variability_subjects.csv"
    aggregate_subject_summary_csv = tables_dir / "aggregate_subject_variability_summary.csv"
    aggregate_mni_subject_csv = tables_dir / "aggregate_mni_transfer_subjects.csv"
    aggregate_mni_summary_csv = tables_dir / "aggregate_mni_transfer_summary.csv"
    aggregate_subject_rows = _subject_metric_rows(
        records,
        metric_key="overlap_fraction",
        output_prefix="overlap_percent",
        scale=100.0,
    )
    aggregate_subject_summary = _summary_from_subject_metric_rows(
        aggregate_subject_rows,
        output_prefix="overlap_percent",
        summary_prefix="overlap_percent",
    )
    aggregate_mni_subject_rows = _subject_metric_rows(
        records,
        metric_key="roi_mean",
        output_prefix="roi_mean_v_per_m",
    )
    aggregate_mni_summary = _build_mni_transfer_summary(
        aggregate_mni_subject_rows,
        records,
        batch_root=root,
        mni_baseline_post_root=mni_baseline_post_root,
    )
    write_csv(long_csv, records)
    write_csv(roi_summary_csv, roi_summary)
    write_csv(repeat_summary_csv, repeat_rows)
    write_csv(aggregate_subject_csv, aggregate_subject_rows)
    write_csv(aggregate_subject_summary_csv, aggregate_subject_summary)
    if aggregate_mni_summary:
        write_csv(aggregate_mni_subject_csv, aggregate_mni_subject_rows)
        write_csv(aggregate_mni_summary_csv, aggregate_mni_summary)

    figures = generate_figures(
        records,
        repeat_rows,
        roi_summary,
        aggregate_subject_rows,
        aggregate_mni_subject_rows,
        aggregate_mni_summary,
        expected_repeats=canonical_repeats,
        figures_dir=figures_dir,
    )

    result = {
        "stage": FIGURE_GENERATION_STAGE,
        "status": "ok",
        "batch_root": str(root),
        "output_dir": str(out_dir),
        "figure_count": len(figures),
        "row_count": sum(1 for record in records if record.get("analysis_complete")),
        "roi_count": len(roi_summary),
        "expected_repeats": list(canonical_repeats),
        "figures": {name: str(path) for name, path in figures.items()},
        "tables": {
            "subject_metrics_long": str(long_csv),
            "roi_summary": str(roi_summary_csv),
            "repeat_summary": str(repeat_summary_csv),
            "aggregate_subject_variability_subjects": str(aggregate_subject_csv),
            "aggregate_subject_variability_summary": str(aggregate_subject_summary_csv),
        },
    }
    if aggregate_mni_summary:
        result["tables"].update(
            {
                "aggregate_mni_transfer_subjects": str(aggregate_mni_subject_csv),
                "aggregate_mni_transfer_summary": str(aggregate_mni_summary_csv),
            }
        )
    summary_path = out_dir / "figure_generation_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate post-processing PNG figures and CSV tables.")
    parser.add_argument("--batch-root", required=True, help="Parent directory containing *_Data_* repeat datasets.")
    parser.add_argument("--output-dir", default=None, help="Figure output directory. Default: <batch-root>/post_processing_figures.")
    parser.add_argument("--dataset-glob", default="*_Data_*", help="Glob used within batch-root to find repeat datasets.")
    parser.add_argument("--repeats", nargs="*", default=None, help="Optional repeat identifiers to include.")
    parser.add_argument(
        "--summary-filename",
        default="post_processing_batch_summary.json",
        help="Optional batch summary filename used to annotate failures. Empty string disables reading.",
    )
    parser.add_argument(
        "--mni-baseline-post-root",
        default=None,
        help=(
            "Optional MNI152-data root containing MNI152-*/anat/post/subject_metrics.json. "
            "When omitted, the figure stage auto-searches batch-root ancestors for MNI152-data "
            "or MNI152-data-post.zip, then falls back to per-subject stored MNI baseline metrics."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_figure_generation(
        batch_root=args.batch_root,
        output_dir=args.output_dir,
        dataset_glob=args.dataset_glob,
        expected_repeats=args.repeats,
        summary_filename=args.summary_filename or None,
        mni_baseline_post_root=args.mni_baseline_post_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
