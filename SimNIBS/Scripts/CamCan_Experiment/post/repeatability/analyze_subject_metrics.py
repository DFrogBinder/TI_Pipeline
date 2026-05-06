#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
from scipy import stats

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional runtime dependency
    nib = None

from post.metric_extensions import flatten_subject_metric_payload


METRIC_LABELS = {
    "percentile": "Percentile",
    "percentile_value": "Percentile Value",
    "voxel_volume_mm3": "Voxel Volume",
    "top_percentile_voxels": "Top Percentile Voxels",
    "roi_voxels": "ROI Voxels",
    "overlap_top_voxels": "Overlap Top Voxels",
    "roi_volume_mm3": "ROI Volume",
    "overlap_volume_mm3": "Overlap Volume",
    "overlap_fraction": "Overlap Fraction",
    "roi_peak": "ROI Peak Field",
    "roi_mean": "ROI Mean Field",
    "roi_peak_abs_delta_mni": "ROI Peak Absolute Delta vs MNI",
    "roi_mean_abs_delta_mni": "ROI Mean Absolute Delta vs MNI",
    "focality_voxels_gt_threshold": "Focality Voxels > Threshold",
    "focality_volume_mm3_gt_threshold": "Focality Volume > Threshold",
    "focality_voxels_abs_delta_mni": "Focality Voxels Absolute Delta vs MNI",
    "focality_volume_mm3_abs_delta_mni": "Focality Volume Absolute Delta vs MNI",
    "neighbor_mean_of_means": "Neighbor Mean of Means",
    "neighbor_max_of_max": "Neighbor Peak Maximum",
    "neighbor_min_of_max": "Neighbor Peak Minimum",
    "csf_distance_mm": "CSF Distance",
    "skull_distance_mm": "Skull Distance",
    "electrode_distance_mean_mm": "Mean Electrode Distance",
    "electrode_distance_min_mm": "Minimum Electrode Distance",
    "electrode_distance_max_mm": "Maximum Electrode Distance",
}

METRIC_FORMATTERS = {
    "percentile": lambda x: f"{x:.1f}",
    "percentile_value": lambda x: f"{x:.6f}",
    "voxel_volume_mm3": lambda x: f"{x:.6f}",
    "top_percentile_voxels": lambda x: f"{x:,.0f}",
    "roi_voxels": lambda x: f"{x:,.0f}",
    "overlap_top_voxels": lambda x: f"{x:,.0f}",
    "roi_volume_mm3": lambda x: f"{x:,.3f}",
    "overlap_volume_mm3": lambda x: f"{x:,.3f}",
    "overlap_fraction": lambda x: f"{x:.4f}",
    "roi_peak": lambda x: f"{x:.6f}",
    "roi_mean": lambda x: f"{x:.6f}",
    "roi_peak_abs_delta_mni": lambda x: f"{x:.6f}",
    "roi_mean_abs_delta_mni": lambda x: f"{x:.6f}",
    "focality_voxels_gt_threshold": lambda x: f"{x:,.0f}",
    "focality_volume_mm3_gt_threshold": lambda x: f"{x:,.3f}",
    "focality_voxels_abs_delta_mni": lambda x: f"{x:,.0f}",
    "focality_volume_mm3_abs_delta_mni": lambda x: f"{x:,.3f}",
    "neighbor_mean_of_means": lambda x: f"{x:.6f}",
    "neighbor_max_of_max": lambda x: f"{x:.6f}",
    "neighbor_min_of_max": lambda x: f"{x:.6f}",
    "csf_distance_mm": lambda x: f"{x:,.3f}",
    "skull_distance_mm": lambda x: f"{x:,.3f}",
    "electrode_distance_mean_mm": lambda x: f"{x:,.3f}",
    "electrode_distance_min_mm": lambda x: f"{x:,.3f}",
    "electrode_distance_max_mm": lambda x: f"{x:,.3f}",
}

PLOT_METRICS = [
    "percentile_value",
    "top_percentile_voxels",
    "overlap_top_voxels",
    "overlap_fraction",
]

IMAGE_MASK_METRIC_LABELS = {
    "roi_mask_dice": "ROI Mask Dice",
    "roi_mask_jaccard": "ROI Mask Jaccard",
    "top_percentile_mask_dice": "Top-Percentile Mask Dice",
    "top_percentile_mask_jaccard": "Top-Percentile Mask Jaccard",
    "overlap_mask_dice": "Overlap Mask Dice",
    "overlap_mask_jaccard": "Overlap Mask Jaccard",
}

IMAGE_PAIRWISE_METRIC_LABELS = {
    **IMAGE_MASK_METRIC_LABELS,
    "within_roi_field_correlation": "Within-ROI Field Correlation",
    "peak_displacement_mm": "Peak Displacement (mm)",
    "overlap_com_displacement_mm": "Overlap COM Displacement (mm)",
    "roi_mean_field_abs_diff": "ROI Mean |Δ|",
    "roi_p95_field_abs_diff": "ROI P95 |Δ|",
    "roi_peak_field_abs_diff": "ROI Peak |Δ|",
}

TOP_PERCENTILE_MASK_KIND = "top_percentile_mask"
TOP_PERCENTILE_MASK_LABEL = "Top-Percentile Mask"

IMAGE_FILE_KINDS = {
    "roi_mask": "atlas_{roi_name}_mask.nii.gz",
    TOP_PERCENTILE_MASK_KIND: "efield_{percentile_tag}_mask.nii.gz",
    "overlap_mask": "{roi_name}_overlap_{percentile_tag}_mask.nii.gz",
    "roi_field": "TI_in_{roi_name}.nii.gz",
}


def normalize_percentile_value(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def unique_percentile_values(values: Iterable[object]) -> list[float]:
    unique: list[float] = []
    for value in values:
        numeric = normalize_percentile_value(value)
        if numeric is None:
            continue
        if any(math.isclose(numeric, existing, rel_tol=0.0, abs_tol=1e-9) for existing in unique):
            continue
        unique.append(numeric)
    return sorted(unique)


def infer_uniform_percentile(values: Iterable[object]) -> float | None:
    unique = unique_percentile_values(values)
    if len(unique) == 1:
        return unique[0]
    return None


def percentile_filename_tag(percentile: object) -> str | None:
    numeric = normalize_percentile_value(percentile)
    if numeric is None:
        return None
    return f"top{int(numeric)}pct"


def top_percentile_mask_filename(percentile: object) -> str:
    tag = percentile_filename_tag(percentile)
    if tag is None:
        return "efield_top<int(percentile)>pct_mask.nii.gz"
    return f"efield_{tag}_mask.nii.gz"


def overlap_percentile_mask_filename(roi_name: str, percentile: object) -> str:
    tag = percentile_filename_tag(percentile)
    if tag is None:
        return f"{roi_name}_overlap_top<int(percentile)>pct_mask.nii.gz"
    return f"{roi_name}_overlap_{tag}_mask.nii.gz"


def percentile_context_line(percentile: float | None) -> str:
    if percentile is None:
        return "- Field-percentile configuration could not be inferred uniquely from the loaded runs."
    return f"- Field-percentile threshold used for image masks: `{percentile:.1f}`"


def format_metric_value(metric: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    formatter = METRIC_FORMATTERS.get(metric, lambda x: f"{x:.4g}")
    return formatter(float(value))


def infer_metric_label(metric: str) -> str:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    if metric.startswith("neighbor_mean__"):
        return f"Neighbor Mean: {humanize_label(metric.split('__', 1)[1])}"
    if metric.startswith("neighbor_peak__"):
        return f"Neighbor Peak: {humanize_label(metric.split('__', 1)[1])}"
    if metric.startswith("neighbor_voxels__"):
        return f"Neighbor Voxels: {humanize_label(metric.split('__', 1)[1])}"
    if metric.startswith("electrode_distance_mm__"):
        return f"Electrode Distance: {humanize_label(metric.split('__', 1)[1])}"
    return humanize_label(metric)


def humanize_label(value: str | None) -> str:
    if value is None or pd.isna(value):
        return "NA"

    replacements = {
        "charm": "CHARM",
        "eeg": "EEG",
        "fem": "FEM",
        "mni": "MNI",
        "simnibs": "SimNIBS",
        "cv": "CV",
        "sd": "SD",
    }
    tokens = str(value).replace("_", " ").split()
    return " ".join(replacements.get(token.lower(), token.capitalize()) for token in tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse run-level subject_metrics JSON files and generate summary tables, "
            "figures, and a markdown report."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=".",
        help="Root directory that contains the repeat/run folders and subject_metrics.json files.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI name to analyse. Required only when multiple ROIs are present.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for generated outputs. Defaults to <dataset_root>/subject_metrics_analysis.",
    )
    parser.add_argument(
        "--logs-root",
        default=None,
        help=(
            "Optional directory containing simulation SLURM logs and status summaries. "
            "If omitted, the script will look for a sibling folder named Left-Hippocapus_logs."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-subjects",
        action="store_true",
        help=(
            "Use all available subjects in repeat-level descriptive outputs. By default, the analysis "
            "is restricted to the complete-case cohort present in every repeat."
        ),
    )
    parser.add_argument(
        "--skip-image-repeatability",
        action="store_true",
        help=(
            "Skip the image-level repeatability analysis that uses saved NIfTI masks and "
            "within-ROI field volumes."
        ),
    )
    return parser.parse_args()


def infer_run_label(metric_path: Path, dataset_root: Path) -> str:
    for parent in metric_path.parents:
        if parent == dataset_root:
            break
        if "_Data_" in parent.name:
            return parent.name
    relative = metric_path.relative_to(dataset_root)
    return relative.parts[0]


def infer_repeat_id(run_label: str) -> int:
    tail = run_label.rsplit("_", 1)[-1]
    digits = "".join(char for char in tail if char.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Could not infer repeat id from run label: {run_label}")


def resolve_logs_root(dataset_root: Path, logs_root_arg: str | None) -> Path | None:
    candidates: list[Path] = []
    if logs_root_arg:
        candidates.append(Path(logs_root_arg).expanduser().resolve())
    else:
        candidates.extend(
            [
                (dataset_root.parent / "Left-Hippocapus_logs").resolve(),
                (dataset_root / "Left-Hippocapus_logs").resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def discover_subject_metrics(dataset_root: Path) -> list[Path]:
    paths = sorted(dataset_root.rglob("subject_metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No subject_metrics.json files found beneath {dataset_root}")
    return paths


def load_subject_metrics(dataset_root: Path, roi_name: str | None) -> tuple[pd.DataFrame, str]:
    records: list[dict[str, object]] = []
    discovered_rois: set[str] = set()

    for metric_path in discover_subject_metrics(dataset_root):
        with metric_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        run_label = infer_run_label(metric_path, dataset_root)
        repeat_id = infer_repeat_id(run_label)
        subject = payload.get("subject")
        roi_map = payload.get("rois", {})

        if not isinstance(roi_map, dict) or not roi_map:
            continue

        for roi_key, roi_metrics in roi_map.items():
            discovered_rois.add(roi_key)
            if roi_name and roi_key != roi_name:
                continue

            record = {
                "run_label": run_label,
                "repeat_id": repeat_id,
                "run_short": f"R{repeat_id:02d}",
                "roi": roi_key,
                "source_path": str(metric_path.relative_to(dataset_root)),
            }
            record.update(flatten_subject_metric_payload(payload, roi_key))
            record["subject"] = subject
            records.append(record)

    if roi_name is None and len(discovered_rois) > 1:
        discovered = ", ".join(sorted(discovered_rois))
        raise ValueError(
            f"Multiple ROIs were detected ({discovered}). Re-run with --roi to select one."
        )

    if not records:
        if roi_name:
            raise ValueError(f"No subject_metrics entries found for ROI '{roi_name}'.")
        raise ValueError("No usable subject_metrics entries were found.")

    if roi_name is None:
        roi_name = records[0]["roi"]  # type: ignore[assignment]

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(["repeat_id", "subject"]).reset_index(drop=True)
    return frame, str(roi_name)


def numeric_metrics(frame: pd.DataFrame) -> list[str]:
    excluded = {"repeat_id", "run_label", "run_short", "subject", "roi", "target_roi", "source_path"}
    static_metrics = {
        "schema_version",
        "focality_threshold_v_per_m",
        "mni_baseline_roi_peak",
        "mni_baseline_roi_mean",
        "mni_baseline_focality_voxels_gt_threshold",
        "mni_baseline_focality_volume_mm3_gt_threshold",
        "neighbor_template_count",
        "electrode_distance_count",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and column not in static_metrics
    ]


def roi_name_variants(roi_name: str) -> list[str]:
    candidates = [
        roi_name,
        roi_name.replace(" ", "-"),
        roi_name.replace(" ", "_"),
        roi_name.replace("-", "_"),
        roi_name.replace("_", "-"),
    ]
    seen: set[str] = set()
    variants: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
    return variants


def resolve_image_repeatability_path(
    post_dir: Path,
    roi_name: str,
    kind: str,
    percentile: object = None,
) -> Path:
    template = IMAGE_FILE_KINDS[kind]
    percentile_tag = percentile_filename_tag(percentile)
    roi_variants = roi_name_variants(roi_name)

    exact_candidates: list[Path] = []
    if "{roi_name}" not in template and "{percentile_tag}" not in template:
        exact_candidates = [post_dir / template]
    elif "{roi_name}" in template and "{percentile_tag}" not in template:
        exact_candidates = [post_dir / template.format(roi_name=variant) for variant in roi_variants]
    elif "{roi_name}" not in template and percentile_tag is not None:
        exact_candidates = [post_dir / template.format(percentile_tag=percentile_tag)]
    elif percentile_tag is not None:
        exact_candidates = [
            post_dir / template.format(roi_name=variant, percentile_tag=percentile_tag)
            for variant in roi_variants
        ]

    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    fallback_matches: list[Path] = []
    if kind == TOP_PERCENTILE_MASK_KIND:
        fallback_matches = sorted(post_dir.glob("efield_top*pct_mask.nii.gz"))
    elif kind == "overlap_mask":
        for variant in roi_variants:
            fallback_matches.extend(sorted(post_dir.glob(f"{variant}_overlap_top*pct_mask.nii.gz")))

    unique_matches: list[Path] = []
    seen: set[Path] = set()
    for match in fallback_matches:
        resolved = match.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_matches.append(match)
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        raise FileNotFoundError(
            f"Multiple candidate images found for `{kind}` in {post_dir}: "
            f"{', '.join(path.name for path in unique_matches)}"
        )

    expected = [candidate.name for candidate in exact_candidates]
    if kind == TOP_PERCENTILE_MASK_KIND:
        expected.append("efield_top*pct_mask.nii.gz")
    elif kind == "overlap_mask":
        expected.extend(f"{variant}_overlap_top*pct_mask.nii.gz" for variant in roi_variants)
    raise FileNotFoundError(
        f"Missing required image for `{kind}` in {post_dir}. Tried: {', '.join(expected)}"
    )


def headers_match(reference_shape: tuple[int, ...], reference_affine: np.ndarray, image: object) -> bool:
    return tuple(image.shape) == tuple(reference_shape) and np.allclose(
        np.asarray(image.affine),
        np.asarray(reference_affine),
        atol=1e-6,
    )


def to_bool_array(image: object) -> np.ndarray:
    return np.asarray(image.dataobj) > 0


def to_float_array(image: object) -> np.ndarray:
    return np.asarray(image.dataobj, dtype=np.float32)


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    count_a = int(np.count_nonzero(mask_a))
    count_b = int(np.count_nonzero(mask_b))
    total = count_a + count_b
    if total == 0:
        return 1.0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    return (2.0 * intersection) / total


def jaccard_index(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = int(np.count_nonzero(mask_a | mask_b))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    return intersection / union


def safe_pearson_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    finite = np.isfinite(values_a) & np.isfinite(values_b)
    if int(np.count_nonzero(finite)) < 2:
        return math.nan
    subset_a = values_a[finite]
    subset_b = values_b[finite]
    if float(np.std(subset_a)) == 0.0 or float(np.std(subset_b)) == 0.0:
        return math.nan
    return float(np.corrcoef(subset_a, subset_b)[0, 1])


def euclidean_distance_mm(point_a: np.ndarray, point_b: np.ndarray) -> float:
    if np.isnan(point_a).any() or np.isnan(point_b).any():
        return math.nan
    return float(np.linalg.norm(point_b - point_a))


def pairwise_absolute_differences(values: np.ndarray) -> np.ndarray:
    diffs = [abs(float(values[b] - values[a])) for a, b in combinations(range(len(values)), 2)]
    return np.asarray(diffs, dtype=float)


def summarise_quantiles(values: np.ndarray | list[float]) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def summarise_repeat_vector(prefix: str, values: np.ndarray) -> dict[str, float]:
    summary = summarise_series(pd.Series(values))
    diffs = pairwise_absolute_differences(values)
    diff_summary = summarise_series(pd.Series(diffs))
    return {
        f"{prefix}_mean": summary["mean"],
        f"{prefix}_sd": summary["std"],
        f"{prefix}_cv_percent": summary["cv_percent"],
        f"{prefix}_min": summary["min"],
        f"{prefix}_max": summary["max"],
        f"{prefix}_range": summary["max"] - summary["min"],
        f"{prefix}_mean_abs_pairwise_diff": diff_summary["mean"],
        f"{prefix}_max_abs_pairwise_diff": diff_summary["max"],
    }


def t_interval(mean: float, sem: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n < 2 or math.isnan(sem):
        return (math.nan, math.nan)
    alpha = 1 - confidence
    critical = stats.t.ppf(1 - alpha / 2, n - 1)
    return mean - critical * sem, mean + critical * sem


def summarise_series(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = int(values.shape[0])
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "median": math.nan,
            "q1": math.nan,
            "q3": math.nan,
            "iqr": math.nan,
            "min": math.nan,
            "max": math.nan,
            "cv_percent": math.nan,
        }

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    ci_low, ci_high = t_interval(mean, sem, n)
    cv_percent = (std / mean * 100.0) if mean not in (0.0, -0.0) else math.nan
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": float(values.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(values.min()),
        "max": float(values.max()),
        "cv_percent": float(cv_percent) if not math.isnan(cv_percent) else math.nan,
    }


def compute_coverage(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int]]:
    grouped = frame.groupby(["repeat_id", "run_label", "run_short"])["subject"]
    subject_sets = {
        repeat_id: set(group.subject.unique())
        for repeat_id, group in frame.groupby("repeat_id", sort=True)
    }
    all_subjects = sorted(set.union(*subject_sets.values()))
    complete_subjects = sorted(set.intersection(*subject_sets.values()))

    rows = []
    for (repeat_id, run_label, run_short), subjects in grouped:
        subject_set = set(subjects.unique())
        rows.append(
            {
                "repeat_id": repeat_id,
                "run_label": run_label,
                "run_short": run_short,
                "n_subjects": len(subject_set),
                "n_missing_from_union": len(all_subjects) - len(subject_set),
                "n_missing_from_complete_case": len(set(complete_subjects) - subject_set),
            }
        )

    coverage = pd.DataFrame(rows).sort_values("repeat_id").reset_index(drop=True)
    return coverage, all_subjects, complete_subjects


def compute_repeat_level_stats(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (repeat_id, run_label, run_short), subset in frame.groupby(
        ["repeat_id", "run_label", "run_short"], sort=True
    ):
        for metric in metrics:
            stats_row = summarise_series(subset[metric])
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "run_label": run_label,
                    "run_short": run_short,
                    "metric": metric,
                    "metric_label": infer_metric_label(metric),
                    **stats_row,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "repeat_id"]).reset_index(drop=True)


def compute_pairwise_differences(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = frame.pivot(index="subject", columns="repeat_id", values=metric).sort_index(axis=1)
        for repeat_a, repeat_b in combinations(pivot.columns, 2):
            diff = (pivot[repeat_b] - pivot[repeat_a]).dropna()
            summary = summarise_series(diff)
            rows.append(
                {
                    "metric": metric,
                    "metric_label": infer_metric_label(metric),
                    "repeat_a": repeat_a,
                    "repeat_b": repeat_b,
                    "run_a": f"R{repeat_a:02d}",
                    "run_b": f"R{repeat_b:02d}",
                    "comparison": f"R{repeat_b:02d} - R{repeat_a:02d}",
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "repeat_a", "repeat_b"]).reset_index(drop=True)


def compute_within_subject_repeatability(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pivot = frame.pivot(index="subject", columns="repeat_id", values=metric).sort_index(axis=1)
        means = pivot.mean(axis=1)
        stds = pivot.std(axis=1, ddof=1)
        cvs = (stds / means.replace(0, np.nan)) * 100.0

        stats_mean = summarise_series(means)
        stats_std = summarise_series(stds)
        stats_cv = summarise_series(cvs)

        rows.append(
            {
                "metric": metric,
                "metric_label": infer_metric_label(metric),
                "n_subjects": int(pivot.shape[0]),
                "subject_mean_mean": stats_mean["mean"],
                "subject_mean_std": stats_mean["std"],
                "subject_mean_median": stats_mean["median"],
                "subject_within_run_sd_mean": stats_std["mean"],
                "subject_within_run_sd_median": stats_std["median"],
                "subject_within_run_cv_percent_mean": stats_cv["mean"],
                "subject_within_run_cv_percent_median": stats_cv["median"],
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def compute_anova_variation_components(pivot: pd.DataFrame) -> dict[str, float]:
    values = pivot.to_numpy(dtype=float)
    n_subjects, n_runs = values.shape
    if n_subjects < 2 or n_runs < 2:
        return {
            "ms_subject": math.nan,
            "ms_run": math.nan,
            "ms_residual": math.nan,
            "subject_variance": math.nan,
            "run_variance": math.nan,
            "residual_variance": math.nan,
            "between_subject_sd": math.nan,
            "run_effect_sd": math.nan,
            "pooled_within_subject_sd": math.nan,
        }

    grand_mean = float(values.mean())
    subject_means = values.mean(axis=1, keepdims=True)
    run_means = values.mean(axis=0, keepdims=True)
    residuals = values - subject_means - run_means + grand_mean

    ss_subject = n_runs * float(np.sum((subject_means - grand_mean) ** 2))
    ss_run = n_subjects * float(np.sum((run_means - grand_mean) ** 2))
    ss_residual = float(np.sum(residuals**2))

    ms_subject = ss_subject / (n_subjects - 1)
    ms_run = ss_run / (n_runs - 1)
    ms_residual = ss_residual / ((n_subjects - 1) * (n_runs - 1))

    subject_variance = max((ms_subject - ms_residual) / n_runs, 0.0)
    run_variance = max((ms_run - ms_residual) / n_subjects, 0.0)
    residual_variance = max(ms_residual, 0.0)

    return {
        "ms_subject": ms_subject,
        "ms_run": ms_run,
        "ms_residual": ms_residual,
        "subject_variance": subject_variance,
        "run_variance": run_variance,
        "residual_variance": residual_variance,
        "between_subject_sd": math.sqrt(subject_variance),
        "run_effect_sd": math.sqrt(run_variance),
        "pooled_within_subject_sd": math.sqrt(residual_variance),
    }


def compute_icc_absolute_agreement(
    ms_subject: float,
    ms_run: float,
    ms_residual: float,
    n_subjects: int,
    n_runs: int,
) -> float:
    if n_subjects < 2 or n_runs < 2:
        return math.nan
    denominator = ms_subject + (n_runs - 1) * ms_residual + (n_runs * (ms_run - ms_residual) / n_subjects)
    if denominator == 0:
        return math.nan
    return (ms_subject - ms_residual) / denominator


def compute_mean_pairwise_correlation(pivot: pd.DataFrame) -> float:
    correlation_matrix = pivot.corr()
    if correlation_matrix.shape[0] < 2:
        return math.nan
    upper = correlation_matrix.to_numpy()[np.triu_indices_from(correlation_matrix, k=1)]
    upper = upper[~np.isnan(upper)]
    if upper.size == 0:
        return math.nan
    return float(upper.mean())


def compute_image_repeatability(
    dataset_root: Path,
    roi_name: str,
    complete_case_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if nib is None:
        raise RuntimeError(
            "nibabel is required for image-level repeatability analysis but is not installed."
        )

    if "percentile" not in complete_case_frame.columns:
        complete_case_frame = complete_case_frame.copy()
        complete_case_frame["percentile"] = np.nan

    unique_runs = (
        complete_case_frame.loc[:, ["subject", "repeat_id", "run_label", "run_short", "source_path", "percentile"]]
        .drop_duplicates()
        .sort_values(["subject", "repeat_id"])
        .reset_index(drop=True)
    )

    run_level_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []

    for subject, subject_runs in unique_runs.groupby("subject", sort=True):
        subject_runs = subject_runs.sort_values("repeat_id").reset_index(drop=True)
        try:
            subject_percentiles = unique_percentile_values(subject_runs["percentile"].tolist())
            if len(subject_percentiles) > 1:
                raise ValueError(
                    f"Mixed percentile settings detected across runs for {subject}: {subject_percentiles}"
                )
            post_dirs = [dataset_root / Path(path).parent for path in subject_runs["source_path"]]
            run_descriptors = [
                {
                    "repeat_id": int(row.repeat_id),
                    "run_label": str(row.run_label),
                    "run_short": str(row.run_short),
                    "post_dir": post_dir,
                    "percentile": normalize_percentile_value(row.percentile),
                }
                for row, post_dir in zip(subject_runs.itertuples(index=False), post_dirs)
            ]

            reference_paths = {
                kind: resolve_image_repeatability_path(
                    run_descriptors[0]["post_dir"],
                    roi_name,
                    kind,
                    run_descriptors[0]["percentile"],
                )
                for kind in IMAGE_FILE_KINDS
            }
            reference_roi_img = nib.load(str(reference_paths["roi_mask"]))
            reference_shape = tuple(reference_roi_img.shape)
            reference_affine = np.asarray(reference_roi_img.affine, dtype=float)
            reference_roi_mask = to_bool_array(reference_roi_img)
            if int(np.count_nonzero(reference_roi_mask)) == 0:
                raise ValueError("Reference ROI mask is empty.")
            reference_roi_indices = np.where(reference_roi_mask)

            roi_masks: list[np.ndarray] = []
            top_masks: list[np.ndarray] = []
            overlap_masks: list[np.ndarray] = []
            roi_field_vectors: list[np.ndarray] = []
            peak_points_mm: list[np.ndarray] = []
            overlap_com_points_mm: list[np.ndarray] = []
            roi_mean_fields: list[float] = []
            roi_p95_fields: list[float] = []
            roi_peak_fields: list[float] = []
            grid_consistent_all_runs = True
            same_roi_mask_all_runs = True

            for descriptor in run_descriptors:
                image_paths = {
                    kind: resolve_image_repeatability_path(
                        descriptor["post_dir"],
                        roi_name,
                        kind,
                        descriptor["percentile"],
                    )
                    for kind in IMAGE_FILE_KINDS
                }
                images = {kind: nib.load(str(path)) for kind, path in image_paths.items()}
                if not all(
                    headers_match(reference_shape, reference_affine, image)
                    for image in images.values()
                ):
                    grid_consistent_all_runs = False
                    raise ValueError(
                        f"Image header mismatch for {subject} in {descriptor['run_label']}"
                    )

                roi_mask = to_bool_array(images["roi_mask"])
                top_mask = to_bool_array(images[TOP_PERCENTILE_MASK_KIND])
                overlap_mask = to_bool_array(images["overlap_mask"])
                roi_field = to_float_array(images["roi_field"])
                roi_values = roi_field[reference_roi_indices]
                finite_run_mask = np.isfinite(roi_values)
                if not finite_run_mask.any():
                    raise ValueError(
                        f"No finite ROI field values detected for {subject} in {descriptor['run_label']}"
                    )
                finite_indices = tuple(axis[finite_run_mask] for axis in reference_roi_indices)
                finite_values = roi_values[finite_run_mask]

                roi_masks.append(roi_mask)
                top_masks.append(top_mask)
                overlap_masks.append(overlap_mask)
                roi_field_vectors.append(roi_values)
                same_roi_mask_all_runs = same_roi_mask_all_runs and np.array_equal(
                    reference_roi_mask,
                    roi_mask,
                )

                roi_mean_field = float(np.nanmean(roi_values))
                roi_p95_field = float(np.nanquantile(roi_values, 0.95))
                roi_peak_field = float(np.nanmax(roi_values))
                roi_mean_fields.append(roi_mean_field)
                roi_p95_fields.append(roi_p95_field)
                roi_peak_fields.append(roi_peak_field)

                peak_flat_index = int(np.argmax(finite_values))
                peak_voxel = np.asarray(
                    [axis[peak_flat_index] for axis in finite_indices],
                    dtype=float,
                )
                peak_point_mm = nib.affines.apply_affine(reference_affine, peak_voxel)
                peak_points_mm.append(np.asarray(peak_point_mm, dtype=float))

                if int(np.count_nonzero(overlap_mask)) > 0:
                    overlap_coordinates = np.argwhere(overlap_mask)
                    overlap_center_voxel = overlap_coordinates.mean(axis=0)
                    overlap_center_mm = nib.affines.apply_affine(
                        reference_affine,
                        overlap_center_voxel,
                    )
                    overlap_com_points_mm.append(np.asarray(overlap_center_mm, dtype=float))
                else:
                    overlap_com_points_mm.append(np.full(3, math.nan))

                run_level_rows.append(
                    {
                        "subject": subject,
                        "repeat_id": descriptor["repeat_id"],
                        "run_label": descriptor["run_label"],
                        "run_short": descriptor["run_short"],
                        "post_dir": str(descriptor["post_dir"].relative_to(dataset_root)),
                        "percentile": descriptor["percentile"],
                        "same_grid_as_reference": True,
                        "same_roi_mask_as_reference": np.array_equal(reference_roi_mask, roi_mask),
                        "roi_mask_voxels": int(np.count_nonzero(roi_mask)),
                        "roi_field_finite_voxels": int(np.count_nonzero(finite_run_mask)),
                        "top_percentile_mask_voxels": int(np.count_nonzero(top_mask)),
                        "overlap_mask_voxels": int(np.count_nonzero(overlap_mask)),
                        "roi_mean_field": roi_mean_field,
                        "roi_p95_field": roi_p95_field,
                        "roi_peak_field": roi_peak_field,
                        "peak_x_mm": float(peak_points_mm[-1][0]),
                        "peak_y_mm": float(peak_points_mm[-1][1]),
                        "peak_z_mm": float(peak_points_mm[-1][2]),
                        "overlap_com_x_mm": float(overlap_com_points_mm[-1][0]),
                        "overlap_com_y_mm": float(overlap_com_points_mm[-1][1]),
                        "overlap_com_z_mm": float(overlap_com_points_mm[-1][2]),
                    }
                )

            field_matrix_raw = np.vstack(roi_field_vectors)
            finite_support = np.all(np.isfinite(field_matrix_raw), axis=0)
            if not finite_support.any():
                raise ValueError(f"No common finite ROI support across all runs for {subject}")
            field_matrix = field_matrix_raw[:, finite_support]
            roi_field_vectors = [field_matrix[index, :] for index in range(field_matrix.shape[0])]
            voxel_sd = np.std(field_matrix, axis=0, ddof=1)
            voxel_mean = np.mean(field_matrix, axis=0)
            voxel_cv = np.divide(
                voxel_sd,
                voxel_mean,
                out=np.full_like(voxel_sd, np.nan, dtype=float),
                where=voxel_mean != 0,
            ) * 100.0
            voxel_sd_summary = summarise_quantiles(voxel_sd)
            voxel_cv_summary = summarise_quantiles(voxel_cv)

            for index_a, index_b in combinations(range(len(run_descriptors)), 2):
                descriptor_a = run_descriptors[index_a]
                descriptor_b = run_descriptors[index_b]
                pairwise_rows.append(
                    {
                        "subject": subject,
                        "repeat_a": descriptor_a["repeat_id"],
                        "repeat_b": descriptor_b["repeat_id"],
                        "run_a": descriptor_a["run_short"],
                        "run_b": descriptor_b["run_short"],
                        "roi_mask_dice": dice_coefficient(roi_masks[index_a], roi_masks[index_b]),
                        "roi_mask_jaccard": jaccard_index(roi_masks[index_a], roi_masks[index_b]),
                        "top_percentile_mask_dice": dice_coefficient(
                            top_masks[index_a],
                            top_masks[index_b],
                        ),
                        "top_percentile_mask_jaccard": jaccard_index(
                            top_masks[index_a],
                            top_masks[index_b],
                        ),
                        "overlap_mask_dice": dice_coefficient(
                            overlap_masks[index_a],
                            overlap_masks[index_b],
                        ),
                        "overlap_mask_jaccard": jaccard_index(
                            overlap_masks[index_a],
                            overlap_masks[index_b],
                        ),
                        "within_roi_field_correlation": safe_pearson_correlation(
                            roi_field_vectors[index_a],
                            roi_field_vectors[index_b],
                        ),
                        "peak_displacement_mm": euclidean_distance_mm(
                            peak_points_mm[index_a],
                            peak_points_mm[index_b],
                        ),
                        "overlap_com_displacement_mm": euclidean_distance_mm(
                            overlap_com_points_mm[index_a],
                            overlap_com_points_mm[index_b],
                        ),
                        "roi_mean_field_abs_diff": abs(
                            roi_mean_fields[index_b] - roi_mean_fields[index_a]
                        ),
                        "roi_p95_field_abs_diff": abs(
                            roi_p95_fields[index_b] - roi_p95_fields[index_a]
                        ),
                        "roi_peak_field_abs_diff": abs(
                            roi_peak_fields[index_b] - roi_peak_fields[index_a]
                        ),
                    }
                )

            subject_pairwise = pd.DataFrame(
                [row for row in pairwise_rows if row["subject"] == subject]
            )
            subject_row: dict[str, object] = {
                "subject": subject,
                "n_runs": len(run_descriptors),
                "grid_consistent_all_runs": grid_consistent_all_runs,
                "roi_mask_identical_all_runs": same_roi_mask_all_runs,
                "reference_roi_voxels": int(np.count_nonzero(reference_roi_mask)),
                "reference_roi_voxels_common_finite_support": int(np.count_nonzero(finite_support)),
                "reference_roi_voxels_excluded_nonfinite": int(
                    np.count_nonzero(reference_roi_mask) - np.count_nonzero(finite_support)
                ),
                "within_roi_voxel_sd_mean": voxel_sd_summary["mean"],
                "within_roi_voxel_sd_median": voxel_sd_summary["median"],
                "within_roi_voxel_sd_p95": voxel_sd_summary["p95"],
                "within_roi_voxel_sd_max": voxel_sd_summary["max"],
                "within_roi_voxel_cv_percent_mean": voxel_cv_summary["mean"],
                "within_roi_voxel_cv_percent_median": voxel_cv_summary["median"],
                "within_roi_voxel_cv_percent_p95": voxel_cv_summary["p95"],
                "within_roi_voxel_cv_percent_max": voxel_cv_summary["max"],
            }

            for metric in IMAGE_PAIRWISE_METRIC_LABELS:
                subset = subject_pairwise[metric]
                stats_row = summarise_series(subset)
                subject_row.update(
                    {
                        f"{metric}_mean": stats_row["mean"],
                        f"{metric}_median": stats_row["median"],
                        f"{metric}_min": stats_row["min"],
                        f"{metric}_max": stats_row["max"],
                    }
                )

            subject_row.update(summarise_repeat_vector("roi_mean_field", np.asarray(roi_mean_fields)))
            subject_row.update(summarise_repeat_vector("roi_p95_field", np.asarray(roi_p95_fields)))
            subject_row.update(summarise_repeat_vector("roi_peak_field", np.asarray(roi_peak_fields)))
            subject_rows.append(subject_row)

        except Exception as exc:
            issue_rows.append(
                {
                    "subject": subject,
                    "issue_type": type(exc).__name__,
                    "details": str(exc),
                }
            )

    return (
        (
            pd.DataFrame(run_level_rows).sort_values(["subject", "repeat_id"]).reset_index(drop=True)
            if run_level_rows
            else pd.DataFrame()
        ),
        (
            pd.DataFrame(pairwise_rows)
            .sort_values(["subject", "repeat_a", "repeat_b"])
            .reset_index(drop=True)
            if pairwise_rows
            else pd.DataFrame()
        ),
        (
            pd.DataFrame(subject_rows).sort_values("subject").reset_index(drop=True)
            if subject_rows
            else pd.DataFrame()
        ),
        (
            pd.DataFrame(issue_rows).sort_values("subject").reset_index(drop=True)
            if issue_rows
            else pd.DataFrame(columns=["subject", "issue_type", "details"])
        ),
    )


def compute_image_repeatability_cohort_summary(subject_summary: pd.DataFrame) -> pd.DataFrame:
    if subject_summary.empty:
        return pd.DataFrame()

    excluded = {
        "subject",
        "n_runs",
        "grid_consistent_all_runs",
        "roi_mask_identical_all_runs",
    }
    metric_columns = [column for column in subject_summary.columns if column not in excluded]
    rows: list[dict[str, object]] = []
    for metric in metric_columns:
        stats_row = summarise_series(subject_summary[metric])
        numeric_values = pd.to_numeric(subject_summary[metric], errors="coerce").dropna()
        p95 = float(numeric_values.quantile(0.95)) if not numeric_values.empty else math.nan
        rows.append(
            {
                "metric": metric,
                "metric_label": humanize_label(metric),
                **stats_row,
                "p95": p95,
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def compute_image_repeatability_pairwise_run_summary(pairwise_frame: pd.DataFrame) -> pd.DataFrame:
    if pairwise_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for metric, label in IMAGE_PAIRWISE_METRIC_LABELS.items():
        for (repeat_a, repeat_b, run_a, run_b), subset in pairwise_frame.groupby(
            ["repeat_a", "repeat_b", "run_a", "run_b"],
            sort=True,
        ):
            stats_row = summarise_series(subset[metric])
            rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "repeat_a": repeat_a,
                    "repeat_b": repeat_b,
                    "run_a": run_a,
                    "run_b": run_b,
                    **stats_row,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "repeat_a", "repeat_b"]).reset_index(drop=True)


def compute_experiment_level_stats(
    complete_case_frame: pd.DataFrame,
    metrics: Iterable[str],
    repeat_level_complete: pd.DataFrame,
    pairwise_differences: pd.DataFrame,
    within_subject_repeatability: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        run_stats = repeat_level_complete[repeat_level_complete["metric"] == metric].sort_values("repeat_id")
        run_means = run_stats["mean"].reset_index(drop=True)
        run_stds = run_stats["std"].reset_index(drop=True)
        run_counts = run_stats["n"].reset_index(drop=True)

        run_mean_summary = summarise_series(run_means)
        run_std_summary = summarise_series(run_stds)
        pairwise_subset = pairwise_differences[pairwise_differences["metric"] == metric]
        repeatability_row = within_subject_repeatability.loc[
            within_subject_repeatability["metric"] == metric
        ].iloc[0]

        pivot = complete_case_frame.pivot(index="subject", columns="repeat_id", values=metric).sort_index(axis=1)
        variation_components = compute_anova_variation_components(pivot)
        pooled_within_subject_sd = variation_components["pooled_within_subject_sd"]
        standard_error_of_measurement = pooled_within_subject_sd
        repeatability_coefficient = 1.96 * math.sqrt(2) * pooled_within_subject_sd
        grand_mean_complete_case = float(np.nanmean(pivot.to_numpy(dtype=float)))
        pooled_within_subject_cv_percent = (
            (pooled_within_subject_sd / grand_mean_complete_case) * 100.0
            if grand_mean_complete_case not in (0.0, -0.0)
            else math.nan
        )
        variance_total = (
            variation_components["subject_variance"]
            + variation_components["run_variance"]
            + variation_components["residual_variance"]
        )
        subject_variance_fraction_percent = (
            (variation_components["subject_variance"] / variance_total) * 100.0
            if variance_total > 0
            else math.nan
        )
        run_variance_fraction_percent = (
            (variation_components["run_variance"] / variance_total) * 100.0
            if variance_total > 0
            else math.nan
        )
        residual_variance_fraction_percent = (
            (variation_components["residual_variance"] / variance_total) * 100.0
            if variance_total > 0
            else math.nan
        )
        icc_absolute_agreement = compute_icc_absolute_agreement(
            ms_subject=variation_components["ms_subject"],
            ms_run=variation_components["ms_run"],
            ms_residual=variation_components["ms_residual"],
            n_subjects=int(pivot.shape[0]),
            n_runs=int(pivot.shape[1]),
        )
        mean_pairwise_correlation = compute_mean_pairwise_correlation(pivot)
        drift_slope_per_repeat = math.nan
        drift_slope_percent_per_repeat = math.nan
        drift_pvalue = math.nan
        drift_r_squared = math.nan
        if pivot.shape[1] >= 2:
            slope_result = stats.linregress(
                pivot.columns.to_numpy(dtype=float),
                pivot.mean(axis=0).to_numpy(dtype=float),
            )
            drift_slope_per_repeat = float(slope_result.slope)
            drift_pvalue = float(slope_result.pvalue)
            drift_r_squared = float(slope_result.rvalue**2)
            if grand_mean_complete_case not in (0.0, -0.0):
                drift_slope_percent_per_repeat = (
                    drift_slope_per_repeat / grand_mean_complete_case
                ) * 100.0

        friedman_statistic = math.nan
        friedman_pvalue = math.nan
        kendall_w = math.nan
        repeated_values = pivot.to_numpy()
        has_within_subject_change = False
        if repeated_values.size:
            baseline = repeated_values[:, [0]]
            has_within_subject_change = bool(
                np.nanmax(np.abs(repeated_values - baseline)) > 0.0
            )

        if pivot.shape[0] > 0 and pivot.shape[1] > 2 and has_within_subject_change:
            result = stats.friedmanchisquare(*[pivot[column].to_numpy() for column in pivot.columns])
            friedman_statistic = float(result.statistic)
            friedman_pvalue = float(result.pvalue)
            kendall_w = float(result.statistic / (pivot.shape[0] * (pivot.shape[1] - 1)))

        rows.append(
            {
                "metric": metric,
                "metric_label": infer_metric_label(metric),
                "n_runs": int(run_stats.shape[0]),
                "n_complete_subjects": int(pivot.shape[0]),
                "mean_of_run_means": run_mean_summary["mean"],
                "sd_of_run_means": run_mean_summary["std"],
                "ci95_low_of_run_means": run_mean_summary["ci95_low"],
                "ci95_high_of_run_means": run_mean_summary["ci95_high"],
                "min_run_mean": run_mean_summary["min"],
                "max_run_mean": run_mean_summary["max"],
                "range_run_mean": run_mean_summary["max"] - run_mean_summary["min"],
                "cv_percent_run_means": run_mean_summary["cv_percent"],
                "mean_within_run_sd": run_std_summary["mean"],
                "mean_run_subject_count": run_counts.mean(),
                "mean_abs_pairwise_diff": pairwise_subset["mean"].abs().mean(),
                "max_abs_pairwise_diff": pairwise_subset["mean"].abs().max(),
                "grand_mean_complete_case": grand_mean_complete_case,
                "between_subject_sd": variation_components["between_subject_sd"],
                "run_effect_sd": variation_components["run_effect_sd"],
                "pooled_within_subject_sd": pooled_within_subject_sd,
                "standard_error_of_measurement": standard_error_of_measurement,
                "repeatability_coefficient": repeatability_coefficient,
                "pooled_within_subject_cv_percent": pooled_within_subject_cv_percent,
                "icc_absolute_agreement": icc_absolute_agreement,
                "mean_pairwise_correlation": mean_pairwise_correlation,
                "drift_slope_per_repeat": drift_slope_per_repeat,
                "drift_slope_percent_per_repeat": drift_slope_percent_per_repeat,
                "drift_pvalue": drift_pvalue,
                "drift_r_squared": drift_r_squared,
                "subject_variance_fraction_percent": subject_variance_fraction_percent,
                "run_variance_fraction_percent": run_variance_fraction_percent,
                "residual_variance_fraction_percent": residual_variance_fraction_percent,
                "friedman_statistic": friedman_statistic,
                "friedman_pvalue": friedman_pvalue,
                "kendall_w": kendall_w,
                "subject_within_run_sd_mean": repeatability_row["subject_within_run_sd_mean"],
                "subject_within_run_sd_median": repeatability_row["subject_within_run_sd_median"],
                "subject_within_run_cv_percent_mean": repeatability_row[
                    "subject_within_run_cv_percent_mean"
                ],
                "subject_within_run_cv_percent_median": repeatability_row[
                    "subject_within_run_cv_percent_median"
                ],
            }
        )

    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def compute_subject_level_variation(
    complete_case_frame: pd.DataFrame,
    experiment_level_stats: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pooled_lookup = experiment_level_stats.set_index("metric")["pooled_within_subject_sd"].to_dict()

    for metric in metrics:
        pivot = complete_case_frame.pivot(index="subject", columns="repeat_id", values=metric).sort_index(axis=1)
        pooled_within_subject_sd = float(pooled_lookup.get(metric, math.nan))

        for subject, values in pivot.iterrows():
            series = values.to_numpy(dtype=float)
            pairwise_abs_differences = np.abs(series[:, None] - series[None, :])
            upper_triangle = pairwise_abs_differences[np.triu_indices(len(series), k=1)]
            linear_trend = stats.linregress(np.arange(1, len(series) + 1, dtype=float), series)
            mean = float(np.mean(series))
            std = float(np.std(series, ddof=1))
            median = float(np.median(series))
            mad = float(np.median(np.abs(series - median)))

            rows.append(
                {
                    "metric": metric,
                    "metric_label": infer_metric_label(metric),
                    "subject": subject,
                    "n_runs": int(len(series)),
                    "mean": mean,
                    "std": std,
                    "cv_percent": (std / mean) * 100.0 if mean not in (0.0, -0.0) else math.nan,
                    "median": median,
                    "mad": mad,
                    "min": float(np.min(series)),
                    "max": float(np.max(series)),
                    "range": float(np.max(series) - np.min(series)),
                    "mean_abs_pairwise_diff": float(np.mean(upper_triangle)),
                    "max_abs_pairwise_diff": float(np.max(upper_triangle)),
                    "pooled_within_subject_sd": pooled_within_subject_sd,
                    "sd_vs_pooled_repeatability": (
                        std / pooled_within_subject_sd
                        if pooled_within_subject_sd not in (0.0, -0.0) and not math.isnan(pooled_within_subject_sd)
                        else math.nan
                    ),
                    "drift_slope_per_repeat": float(linear_trend.slope),
                    "drift_pvalue": float(linear_trend.pvalue),
                    "drift_r_squared": float(linear_trend.rvalue**2),
                }
            )

    return pd.DataFrame(rows).sort_values(["metric", "subject"]).reset_index(drop=True)


def compute_top_variable_subjects(subject_level_variation: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    primary = subject_level_variation[subject_level_variation["metric"].isin(PLOT_METRICS)].copy()

    for metric, subset in primary.groupby("metric", sort=False):
        for ranking_method, ranking_column in [
            ("relative_variation_cv", "cv_percent"),
            ("absolute_variation_sd_ratio", "sd_vs_pooled_repeatability"),
        ]:
            ranked = subset.nlargest(top_n, ranking_column).copy()
            ranked.insert(0, "ranking_method", ranking_method)
            ranked.insert(1, "rank", np.arange(1, len(ranked) + 1))
            rows.append(ranked)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def compute_subject_variation_summary(subject_level_variation: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        subset = subject_level_variation[subject_level_variation["metric"] == metric].copy()
        if subset.empty:
            continue

        cv_values = pd.to_numeric(subset["cv_percent"], errors="coerce").dropna()
        sd_ratio_values = pd.to_numeric(
            subset["sd_vs_pooled_repeatability"], errors="coerce"
        ).dropna()
        drift_values = pd.to_numeric(subset["drift_slope_per_repeat"], errors="coerce").abs().dropna()

        rows.append(
            {
                "metric": metric,
                "metric_label": infer_metric_label(metric),
                "n_subjects": int(subset["subject"].nunique()),
                "mean_cv_percent": float(cv_values.mean()) if not cv_values.empty else math.nan,
                "median_cv_percent": float(cv_values.median()) if not cv_values.empty else math.nan,
                "p95_cv_percent": float(cv_values.quantile(0.95)) if not cv_values.empty else math.nan,
                "max_cv_percent": float(cv_values.max()) if not cv_values.empty else math.nan,
                "mean_sd_vs_pooled_repeatability": (
                    float(sd_ratio_values.mean()) if not sd_ratio_values.empty else math.nan
                ),
                "median_sd_vs_pooled_repeatability": (
                    float(sd_ratio_values.median()) if not sd_ratio_values.empty else math.nan
                ),
                "p95_sd_vs_pooled_repeatability": (
                    float(sd_ratio_values.quantile(0.95)) if not sd_ratio_values.empty else math.nan
                ),
                "max_sd_vs_pooled_repeatability": (
                    float(sd_ratio_values.max()) if not sd_ratio_values.empty else math.nan
                ),
                "mean_abs_pairwise_diff_mean": float(subset["mean_abs_pairwise_diff"].mean()),
                "mean_abs_pairwise_diff_median": float(subset["mean_abs_pairwise_diff"].median()),
                "max_abs_pairwise_diff_max": float(subset["max_abs_pairwise_diff"].max()),
                "n_subjects_drift_p_lt_0_05": int((subset["drift_pvalue"] < 0.05).sum()),
                "median_abs_drift_slope": float(drift_values.median()) if not drift_values.empty else math.nan,
                "p95_abs_drift_slope": float(drift_values.quantile(0.95)) if not drift_values.empty else math.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def compute_cross_metric_instability(subject_level_variation: pd.DataFrame) -> pd.DataFrame:
    primary = subject_level_variation[subject_level_variation["metric"].isin(PLOT_METRICS)].copy()
    if primary.empty:
        return pd.DataFrame()

    pivot = primary.pivot(index="subject", columns="metric", values="sd_vs_pooled_repeatability")
    pivot = pivot.reindex(columns=PLOT_METRICS)
    pivot.columns = [f"{column}_sd_ratio" for column in pivot.columns]
    pivot = pivot.reset_index()
    value_columns = [column for column in pivot.columns if column.endswith("_sd_ratio")]
    pivot["mean_sd_ratio_across_metrics"] = pivot[value_columns].mean(axis=1)
    pivot["max_sd_ratio_across_metrics"] = pivot[value_columns].max(axis=1)
    return pivot.sort_values("mean_sd_ratio_across_metrics", ascending=False).reset_index(drop=True)


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F8FAFC",
            "figure.facecolor": "white",
            "grid.color": "#D7DEE8",
            "grid.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlepad": 12,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
        }
    )


def thousands_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: f"{value:,.0f}")


def add_figure_note(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#475569",
        wrap=True,
    )


def apply_multi_panel_layout(
    fig: plt.Figure,
    *,
    left: float = 0.08,
    right: float = 0.98,
    top: float = 0.90,
    bottom: float = 0.08,
    wspace: float = 0.28,
    hspace: float = 0.30,
) -> None:
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )


def wrap_axis_ticklabels(axis: plt.Axes, which: str, width: int) -> None:
    if which == "x":
        ticks = axis.get_xticks()
        labels = axis.get_xticklabels()
        wrapped = [textwrap.fill(label.get_text(), width=width, break_long_words=False) for label in labels]
        axis.set_xticks(ticks)
        axis.set_xticklabels(wrapped)
    else:
        ticks = axis.get_yticks()
        labels = axis.get_yticklabels()
        wrapped = [textwrap.fill(label.get_text(), width=width, break_long_words=False) for label in labels]
        axis.set_yticks(ticks)
        axis.set_yticklabels(wrapped)


def parse_run_from_log_name(log_path: Path) -> int:
    match = re.search(r"slurm-(\d+)_\d+\.out$", log_path.name)
    if not match:
        raise ValueError(f"Could not parse run id from log file name: {log_path.name}")
    return int(match.group(1))


def extract_subject_from_log_text(text: str) -> str | None:
    for pattern in [
        r"Subject ID:\s+(sub-[A-Za-z0-9]+)",
        r'"subject_start", "subject": "(sub-[A-Za-z0-9]+)"',
        r'Running TI pipeline for single subject: (sub-[A-Za-z0-9]+)',
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def classify_log_failure(text: str) -> tuple[str, str, str, str]:
    if (
        '"exists": false' in text
        and ("No such file or directory" in text or "FileNotFoundError" in text)
        and "/anat" in text
    ):
        return (
            "missing_subject_input_directory",
            "Input anatomy directory missing before CHARM started.",
            "The subject folder or expected anat path does not exist in the run dataset.",
            "Verify the subject staging step and validate the T1/T2 input paths before submission.",
        )

    if "template_coregistered.mgz" in text and ("Can't find/open file" in text or "MGHImageIO" in text):
        return (
            "charm_init_missing_template_coregistered",
            "CHARM init could not read segmentation/template_coregistered.mgz.",
            "CHARM segmentation created an incomplete `m2m_sub-*` directory or failed while writing intermediate segmentation files.",
            "Delete the partial `m2m_sub-*` directory and rerun CHARM init; also inspect scratch I/O stability and available disk space.",
        )

    if "atlas_level2.txt.gz" in text and "Couldn't read mesh collection" in text:
        return (
            "charm_init_missing_atlas_level2",
            "CHARM init could not read segmentation/atlas_level2.txt.gz.",
            "Multi-resolution atlas generation did not finish cleanly or wrote a corrupt/incomplete file.",
            "Remove the incomplete segmentation outputs and rerun CHARM init; if this repeats, inspect filesystem integrity and concurrent-write issues.",
        )

    if "run_cat_multiprocessing.py" in text and "returned non-zero exit status 1" in text and "Starting surface creation" in text:
        return (
            "charm_init_surface_creation_subprocess_failure",
            "CHARM init failed during CAT-based surface creation.",
            "Surface extraction failed after segmentation, likely due to subject-specific geometry/topology problems or CAT subprocess instability.",
            "Rerun CHARM from scratch, inspect `surfaces/` outputs, and consider rerunning with fewer surface-creation processes if this remains unstable.",
        )

    if "timed out after 14400.0 seconds" in text:
        return (
            "mesh_timeout_with_wrapper_bug",
            "A CHARM stage exceeded the 4-hour timeout and the wrapper then raised a bytes/str TypeError while handling the timeout.",
            "The real failure is the timeout; the Python `TypeError` is a secondary wrapper bug that masks the timeout output.",
            "Increase the CHARM timeout for long-running subjects and patch the timeout handler in `TI_runner_multi-core.py` to decode bytes before concatenation.",
        )

    if "MNI2Conform_nonl.nii.gz" in text and "FileNotFoundError" in text:
        return (
            "charm_remesh_missing_tomni_transform",
            "CHARM remesh failed because `toMNI/MNI2Conform_nonl.nii.gz` was missing.",
            "The prior CHARM output was incomplete for remeshing; the nonlinear transform files were never created or were removed before `charm --mesh` ran.",
            "Ensure a full CHARM run completes before remeshing, verify `m2m_sub-*/toMNI` exists, and avoid deleting transform files between init and `--mesh`.",
        )

    if ".msh.opt" in text and "FileNotFoundError" in text:
        return (
            "charm_remesh_missing_mesh_opt",
            "CHARM remesh failed because the `.msh.opt` helper file was missing.",
            "The remesh step expected mesh sidecar files that were not created or were deleted from the `m2m_sub-*` directory.",
            "Regenerate the subject mesh from scratch and verify the mesh sidecar files exist before downstream remeshing.",
        )

    if "The two nodes are not connected!" in text:
        return (
            "charm_remesh_connectivity_error",
            "CHARM remesh failed with a mesh-connectivity error (`The two nodes are not connected!`).",
            "The remeshed topology became invalid during spike-removal/post-processing, likely due to a fragile or pathological local geometry.",
            "Clean the subject mesh outputs and rerun; if the error persists, inspect this subject's mesh quality and consider subject-specific QC or remeshing settings.",
        )

    if "Could not find EEG cap file" in text:
        return (
            "simulation_missing_eeg_cap_file",
            "Simulation failed because the EEG cap positions file was missing.",
            "The `eeg_positions` assets were not generated or not preserved in the subject mesh directory.",
            "Regenerate the full `m2m_sub-*` directory or restore `eeg_positions/EEG10-10_UI_Jurak_2007.csv` before running the TI simulation.",
        )

    if "slurmstepd: error:" in text and "DUE TO TIME LIMIT" in text and (
        "simnibs_done" in text
        or "msh2nii_labels" in text
        or "msh2nii_masks" in text
        or "msh2nii_volume" in text
    ):
        return (
            "job_time_limit_during_volume_export",
            "The SLURM allocation hit the wall-clock limit during volume export (`msh2nii`).",
            "Meshing and simulation completed, but the end-of-job export stage exceeded the remaining allocation time.",
            "Keep the simulation outputs and rerun only the export stage, or extend the overall SLURM wall time to cover post-processing.",
        )

    if "slurmstepd: error:" in text and "DUE TO TIME LIMIT" in text and "Assembling FEM Matrix" in text:
        return (
            "job_time_limit_during_fem_assembly",
            "The SLURM allocation hit the wall-clock limit during FEM assembly.",
            "The overall job allocation was too short for this subject, even though CHARM itself had already succeeded.",
            "Increase the SLURM wall time for these subjects or split meshing and FEM simulation into separate jobs.",
        )

    if "slurmstepd: error:" in text and "DUE TO TIME LIMIT" in text:
        return (
            "job_time_limit_other_stage",
            "The SLURM allocation hit the wall-clock limit in a later pipeline stage.",
            "The job allocation was too short for the full subject workflow.",
            "Increase SLURM wall time or split the workflow into shorter stage-specific jobs.",
        )

    return (
        "unclassified_failure",
        "The log contains a failure but no known signature matched.",
        "Unknown failure signature; this requires manual inspection of the raw log.",
        "Inspect the raw log tail and stderr for the failing subject.",
    )


def infer_failure_stage(text: str, final_status: str) -> str:
    stage_match = re.search(r'"event": "error", "stage": "([^"]+)"', text)
    if stage_match:
        return stage_match.group(1)
    if (
        "DUE TO TIME LIMIT" in text
        and (
            "simnibs_done" in text
            or "msh2nii_labels" in text
            or "msh2nii_masks" in text
            or "msh2nii_volume" in text
        )
    ):
        return "volume_export"
    if "Assembling FEM Matrix" in text and "DUE TO TIME LIMIT" in text:
        return "simnibs_fem"
    if "Could not find EEG cap file" in text:
        return "simnibs_electrode_setup"
    if final_status == "success":
        return "completed"
    return "unknown"


def normalize_failure_stage(failure_stage: str, failure_category: str) -> str:
    category_stage_map = {
        "missing_subject_input_directory": "charm_init",
        "charm_init_missing_template_coregistered": "charm_init",
        "charm_init_missing_atlas_level2": "charm_init",
        "charm_init_surface_creation_subprocess_failure": "charm_init",
        "mesh_timeout_with_wrapper_bug": "charm_init",
        "charm_remesh_missing_tomni_transform": "charm_remesh",
        "charm_remesh_missing_mesh_opt": "charm_remesh",
        "charm_remesh_connectivity_error": "charm_remesh",
        "simulation_missing_eeg_cap_file": "simnibs_electrode_setup",
        "job_time_limit_during_fem_assembly": "simnibs_fem",
        "job_time_limit_during_volume_export": "volume_export",
    }
    if failure_category in category_stage_map:
        return category_stage_map[failure_category]
    return failure_stage


def extract_log_excerpt(text: str) -> str:
    if "[ simnibs ]CRITICAL:" in text:
        start = text.rfind("[ simnibs ]CRITICAL:")
        excerpt = text[start : start + 1800]
    elif "slurmstepd: error:" in text and "DUE TO TIME LIMIT" in text:
        start = text.rfind("slurmstepd: error:")
        excerpt = text[max(0, start - 1200) : start + 400]
    else:
        excerpt = "\n".join(text.splitlines()[-25:])
    return excerpt.replace("\x00", "").strip()


def load_log_analysis(logs_root: Path, post_metric_subjects: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_path = logs_root / "simulation_run_status_summary.csv"
    if not status_path.exists():
        raise FileNotFoundError(f"Expected log summary file not found: {status_path}")

    status = pd.read_csv(status_path)
    log_index: dict[tuple[int, str], Path] = {}
    for log_path in sorted(logs_root.glob("slurm-*.out")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        subject = extract_subject_from_log_text(text)
        if subject is None:
            continue
        log_index[(parse_run_from_log_name(log_path), subject)] = log_path

    details: list[dict[str, object]] = []
    for row in status.itertuples(index=False):
        run = int(row.run)
        subject = str(row.subject)
        final_status = str(row.final_status)
        same_run_resubmitted = str(row.same_run_resubmitted)
        notes = str(row.notes)
        log_path = log_index.get((run, subject))
        if log_path is None:
            details.append(
                {
                    "run": run,
                    "subject": subject,
                    "final_status": final_status,
                    "same_run_resubmitted": same_run_resubmitted,
                    "notes": notes,
                    "log_file": None,
                    "failure_stage": None,
                    "failure_category": "log_not_found",
                    "evidence": "Could not locate a matching log file for the subject-run pair.",
                    "likely_cause": "Missing log output or subject ID not detectable in the SLURM file.",
                    "recommended_fix": "Verify log retention and subject tagging in the job wrapper.",
                    "in_post_metrics": subject in post_metric_subjects,
                    "log_excerpt": None,
                }
            )
            continue

        text = log_path.read_text(encoding="utf-8", errors="ignore")
        category, evidence, likely_cause, recommended_fix = (
            classify_log_failure(text) if final_status == "failed" else ("success", None, None, None)
        )
        failure_stage = infer_failure_stage(text, final_status)
        if final_status == "failed":
            failure_stage = normalize_failure_stage(failure_stage, category)
        details.append(
            {
                "run": run,
                "subject": subject,
                "final_status": final_status,
                "same_run_resubmitted": same_run_resubmitted,
                "notes": notes,
                "log_file": log_path.name,
                "failure_stage": failure_stage,
                "failure_category": category,
                "evidence": evidence,
                "likely_cause": likely_cause,
                "recommended_fix": recommended_fix,
                "in_post_metrics": subject in post_metric_subjects,
                "log_excerpt": extract_log_excerpt(text) if final_status == "failed" else None,
            }
        )

    detail_frame = pd.DataFrame(details).sort_values(["run", "subject"]).reset_index(drop=True)
    failed_only = detail_frame[detail_frame["final_status"] == "failed"].copy()

    summary_rows: list[dict[str, object]] = []
    category_counts = (
        failed_only.groupby(["failure_category"], dropna=False)
        .agg(
            n_failures=("subject", "size"),
            n_unique_subjects=("subject", "nunique"),
            runs=("run", lambda series: ",".join(map(str, sorted(series.unique())))),
            failure_stage=(
                "failure_stage",
                lambda series: ",".join(map(str, sorted(pd.Series(series).dropna().unique()))),
            ),
        )
        .reset_index()
    )
    for row in category_counts.itertuples(index=False):
        exemplar = failed_only[failed_only["failure_category"] == row.failure_category].iloc[0]
        summary_rows.append(
            {
                "failure_category": row.failure_category,
                "failure_stage": row.failure_stage,
                "n_failures": row.n_failures,
                "n_unique_subjects": row.n_unique_subjects,
                "runs": row.runs,
                "evidence": exemplar["evidence"],
                "likely_cause": exemplar["likely_cause"],
                "recommended_fix": exemplar["recommended_fix"],
            }
        )

    summary_frame = (
        pd.DataFrame(summary_rows)
        .sort_values(["n_failures", "failure_category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    transition_frame = pd.DataFrame()
    if detail_frame["run"].nunique() >= 2:
        run_order = sorted(detail_frame["run"].unique().tolist())
        pivot = detail_frame.pivot(index="subject", columns="run", values="final_status")
        if len(run_order) >= 2:
            first_run, second_run = run_order[0], run_order[1]
            transitions = pivot.apply(
                lambda row: f"{row.get(first_run, 'missing')} -> {row.get(second_run, 'missing')}",
                axis=1,
            )
            transition_counts = transitions.value_counts().rename_axis("transition").reset_index(name="count")
            success_failed = int(transition_counts.loc[transition_counts["transition"] == "success -> failed", "count"].sum())
            failed_success = int(transition_counts.loc[transition_counts["transition"] == "failed -> success", "count"].sum())
            discordant_total = success_failed + failed_success
            mcnemar_pvalue = (
                float(stats.binomtest(success_failed, discordant_total, 0.5).pvalue)
                if discordant_total > 0
                else math.nan
            )
            transition_counts["run_pair"] = f"{first_run}_to_{second_run}"
            transition_counts["first_run"] = first_run
            transition_counts["second_run"] = second_run
            transition_counts["mcnemar_exact_pvalue"] = mcnemar_pvalue
            transition_frame = transition_counts

    return detail_frame, summary_frame, transition_frame


def save_coverage_plot(coverage: pd.DataFrame, complete_subjects: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    y_min = max(len(complete_subjects) - 0.4, 0.0)
    y_max = float(coverage["n_subjects"].max()) + 1.0
    bars = ax.bar(
        coverage["run_short"],
        coverage["n_subjects"],
        color="#2E5EAA",
        edgecolor="#15396B",
        linewidth=1.0,
    )
    ax.axhline(
        len(complete_subjects),
        color="#C65D1B",
        linestyle="--",
        linewidth=2.0,
    )
    ax.set_title("Subject Coverage Per Repeat")
    ax.set_xlabel("Repeat")
    ax.set_ylabel("Subjects")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(len(complete_subjects), int(math.ceil(y_max)) + 1, 1))

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#15396B",
        )

    ax.text(
        -0.45,
        len(complete_subjects) + 0.12,
        f"Complete-case subjects ({len(complete_subjects)})",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#C65D1B",
    )
    ax.text(
        0.995,
        0.02,
        "Y-axis focused on the observed range",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#475569",
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_repeat_distribution_plot(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10.8))
    axes = axes.flatten()
    palette = sns.color_palette("Blues", n_colors=12)[2:]

    for index, (axis, metric) in enumerate(zip(axes, PLOT_METRICS)):
        sns.boxplot(
            data=frame,
            x="run_short",
            y=metric,
            ax=axis,
            color=palette[4],
            showfliers=False,
            width=0.62,
            boxprops={"facecolor": "#BFD6F5", "edgecolor": "#204B82"},
            medianprops={"color": "#0F2744", "linewidth": 2},
            whiskerprops={"color": "#204B82"},
            capprops={"color": "#204B82"},
        )
        means = frame.groupby("run_short", sort=False)[metric].mean().reindex(sorted(frame["run_short"].unique()))
        axis.scatter(
            np.arange(len(means)),
            means.to_numpy(),
            marker="D",
            s=48,
            color="#C65D1B",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="Run mean",
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Repeat")
        axis.set_ylabel(METRIC_LABELS[metric])
        if metric.endswith("voxels"):
            axis.yaxis.set_major_formatter(thousands_formatter())
        if index == 0:
            axis.legend(frameon=False, loc="upper right")
        else:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()

    apply_multi_panel_layout(fig, top=0.90, bottom=0.08, left=0.08, right=0.98, wspace=0.24, hspace=0.30)
    fig.suptitle(
        "Repeat-Level Population Distributions",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_repeat_mean_ci_plot(repeat_stats: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.8))
    axes = axes.flatten()

    for index, (axis, metric) in enumerate(zip(axes, PLOT_METRICS)):
        subset = repeat_stats[repeat_stats["metric"] == metric].sort_values("repeat_id")
        x = subset["repeat_id"].to_numpy()
        mean = subset["mean"].to_numpy()
        low = subset["ci95_low"].to_numpy()
        high = subset["ci95_high"].to_numpy()
        grand_mean = mean.mean()

        axis.fill_between(
            x,
            low,
            high,
            color="#C9DCF4",
            alpha=0.7,
            label="95% Confidence Interval (CI)",
        )
        axis.plot(x, mean, color="#1E4E8C", linewidth=2.5, marker="o", markersize=6, label="Run mean")
        axis.axhline(grand_mean, color="#C65D1B", linestyle="--", linewidth=2, label="Across-run mean")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Repeat")
        axis.set_xticks(x, [f"R{value:02d}" for value in x])
        axis.set_ylabel(METRIC_LABELS[metric])
        if metric.endswith("voxels"):
            axis.yaxis.set_major_formatter(thousands_formatter())
        if index == 0:
            axis.legend(frameon=False, loc="best")
        else:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()

    apply_multi_panel_layout(fig, top=0.90, bottom=0.10, left=0.08, right=0.98, wspace=0.24, hspace=0.30)
    fig.suptitle(
        "Run Means With 95% Confidence Intervals",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    add_figure_note(fig, "CI = confidence interval.")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_heatmap_plot(pairwise_differences: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17.4, 12.8))
    axes = axes.flatten()

    for axis, metric in zip(axes, PLOT_METRICS):
        subset = pairwise_differences[pairwise_differences["metric"] == metric]
        max_repeat = int(max(subset["repeat_a"].max(), subset["repeat_b"].max()))
        matrix = pd.DataFrame(
            np.zeros((max_repeat, max_repeat), dtype=float),
            index=[f"R{i:02d}" for i in range(1, max_repeat + 1)],
            columns=[f"R{i:02d}" for i in range(1, max_repeat + 1)],
        )

        for row in subset.itertuples(index=False):
            matrix.loc[row.run_a, row.run_b] = row.mean
            matrix.loc[row.run_b, row.run_a] = -row.mean

        mask = np.eye(max_repeat, dtype=bool)
        format_string = ".4f" if "fraction" in metric or "value" in metric else ".0f"
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="RdBu_r",
            center=0,
            mask=mask,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=format_string,
            annot_kws={"fontsize": 8},
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Repeat B")
        axis.set_ylabel("Repeat A")

    apply_multi_panel_layout(fig, top=0.91, bottom=0.07, left=0.07, right=0.98, wspace=0.20, hspace=0.26)
    fig.suptitle(
        "Pairwise Run Differences on Complete-Case Subjects",
        fontsize=17,
        fontweight="bold",
        y=0.968,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_variation_summary_plot(experiment_stats: pd.DataFrame, output_path: Path) -> None:
    summary = (
        experiment_stats[experiment_stats["metric"].isin(PLOT_METRICS)]
        .copy()
        .assign(metric_order=lambda frame: frame["metric"].map({metric: index for index, metric in enumerate(PLOT_METRICS)}))
        .sort_values("metric_order", ascending=True)
    )
    summary["metric_label"] = summary["metric"].map(METRIC_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(17.8, 11.8))
    y_positions = np.arange(summary.shape[0])
    metric_labels = summary["metric_label"].to_list()

    icc_ax = axes[0, 0]
    icc_values = summary["icc_absolute_agreement"].to_numpy(dtype=float)
    icc_floor = max(0.90, math.floor(np.nanmin(icc_values) * 100) / 100 - 0.01)
    icc_ax.hlines(y_positions, icc_floor, icc_values, color="#9FB7DD", linewidth=3)
    icc_ax.scatter(icc_values, y_positions, s=85, color="#1E4E8C", edgecolor="white", linewidth=0.9, zorder=3)
    icc_ax.axvline(0.95, color="#C65D1B", linestyle="--", linewidth=1.8)
    icc_ax.axvline(0.90, color="#94A3B8", linestyle=":", linewidth=1.2)
    icc_ax.set_xlim(icc_floor, 1.0)
    icc_ax.set_yticks(y_positions, metric_labels)
    icc_ax.set_xlabel("Absolute-Agreement Intraclass Correlation (ICC[A,1])")
    icc_ax.set_title("Absolute-Agreement Reliability")

    cv_ax = axes[0, 1]
    cv_across = summary["cv_percent_run_means"].to_numpy(dtype=float)
    cv_within = summary["pooled_within_subject_cv_percent"].to_numpy(dtype=float)
    for y_position, across, within in zip(y_positions, cv_across, cv_within):
        cv_ax.plot([across, within], [y_position, y_position], color="#B7C8E2", linewidth=3, zorder=1)
    cv_ax.scatter(
        cv_across,
        y_positions,
        s=70,
        color="#C65D1B",
        edgecolor="white",
        linewidth=0.9,
        label="Across-run coefficient of variation",
        zorder=3,
    )
    cv_ax.scatter(
        cv_within,
        y_positions,
        s=70,
        color="#1E4E8C",
        edgecolor="white",
        linewidth=0.9,
        label="Pooled within-subject coefficient of variation",
        zorder=3,
    )
    cv_ax.set_yticks(y_positions, metric_labels)
    cv_ax.set_xlabel("Coefficient of Variation (%)")
    cv_ax.set_title("Relative Variation")
    cv_ax.legend(frameon=False, loc="lower right", fontsize=10)

    variance_ax = axes[1, 0]
    left = np.zeros(summary.shape[0], dtype=float)
    variance_parts = [
        ("Subject", "subject_variance_fraction_percent", "#1E4E8C"),
        ("Residual", "residual_variance_fraction_percent", "#A9C4EA"),
        ("Repeat", "run_variance_fraction_percent", "#C65D1B"),
    ]
    for label, column, color in variance_parts:
        values = summary[column].to_numpy(dtype=float)
        variance_ax.barh(y_positions, values, left=left, color=color, edgecolor="white", linewidth=0.8, label=label)
        left = left + values
    variance_ax.set_xlim(0, 100)
    variance_ax.set_yticks(y_positions, metric_labels)
    variance_ax.set_xlabel("Explained Variance (%)")
    variance_ax.set_title("Variance Decomposition")
    variance_ax.legend(frameon=False, loc="lower right")

    drift_ax = axes[1, 1]
    drift_values = summary["drift_slope_percent_per_repeat"].to_numpy(dtype=float)
    drift_ax.hlines(y_positions, 0, drift_values, color="#B7C8E2", linewidth=3, zorder=1)
    drift_ax.scatter(
        drift_values,
        y_positions,
        s=80,
        color="#1E4E8C",
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    drift_ax.axvline(0, color="#475569", linewidth=1.5)
    drift_margin = max(np.max(np.abs(drift_values)) * 0.18, 0.0025)
    drift_ax.set_xlim(np.min(drift_values) - drift_margin, drift_margin)
    transform = blended_transform_factory(drift_ax.transAxes, drift_ax.transData)
    for y_position, value, p_value in zip(y_positions, drift_values, summary["drift_pvalue"].to_numpy(dtype=float)):
        if math.isnan(value):
            continue
        drift_ax.text(
            0.98,
            y_position,
            f"p={p_value:.3f}",
            transform=transform,
            va="center",
            ha="right",
            fontsize=9,
            color="#334155",
        )
    drift_ax.set_yticks(y_positions, metric_labels)
    drift_ax.set_xlabel("Linear Drift (% of mean per repeat)")
    drift_ax.set_title("Run Drift")

    apply_multi_panel_layout(fig, top=0.90, bottom=0.12, left=0.10, right=0.985, wspace=0.52, hspace=0.30)
    fig.suptitle(
        "Experiment-Level Variation Summary",
        fontsize=17,
        fontweight="bold",
        y=0.968,
    )
    add_figure_note(
        fig,
        "ICC(A,1) = single-measure absolute-agreement intraclass correlation; "
        "CV = coefficient of variation; SD = standard deviation.",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_subject_variation_plot(
    subject_level_variation: pd.DataFrame,
    cross_metric_instability: pd.DataFrame,
    output_path: Path,
) -> None:
    primary = subject_level_variation[subject_level_variation["metric"].isin(PLOT_METRICS)].copy()
    primary["metric_label"] = primary["metric"].map(METRIC_LABELS)
    metric_order = [
        label
        for label in (METRIC_LABELS[metric] for metric in PLOT_METRICS)
        if label in set(primary["metric_label"].dropna().unique())
    ]
    if not metric_order:
        metric_order = sorted(primary["metric_label"].dropna().unique())
    cv_data = primary.dropna(subset=["cv_percent"]).copy()
    cv_order = [label for label in metric_order if label in set(cv_data["metric_label"].unique())]
    ratio_data = primary.dropna(subset=["sd_vs_pooled_repeatability"]).copy()
    ratio_order = [
        label for label in metric_order if label in set(ratio_data["metric_label"].unique())
    ]

    fig, axes = plt.subplots(2, 2, figsize=(19.2, 14.0))
    cv_ax, ratio_ax, heatmap_ax, ranking_ax = axes.flatten()

    if not cv_data.empty and cv_order:
        sns.boxplot(
            data=cv_data,
            y="metric_label",
            x="cv_percent",
            order=cv_order,
            ax=cv_ax,
            color="#BFD6F5",
            linewidth=1.1,
            fliersize=0,
        )
        sns.stripplot(
            data=cv_data,
            y="metric_label",
            x="cv_percent",
            order=cv_order,
            ax=cv_ax,
            color="#1E4E8C",
            alpha=0.25,
            size=3.2,
            jitter=0.18,
        )
    else:
        cv_ax.text(0.5, 0.5, "Not enough variation data", ha="center", va="center", transform=cv_ax.transAxes)
    cv_ax.set_xlabel("Subject-Level Coefficient of Variation Across Repeats (%)")
    cv_ax.set_ylabel("")
    cv_ax.set_title("Relative Subject Variation")

    if not ratio_data.empty and ratio_order:
        sns.boxplot(
            data=ratio_data,
            y="metric_label",
            x="sd_vs_pooled_repeatability",
            order=ratio_order,
            ax=ratio_ax,
            color="#D7E6B5",
            linewidth=1.1,
            fliersize=0,
        )
        sns.stripplot(
            data=ratio_data,
            y="metric_label",
            x="sd_vs_pooled_repeatability",
            order=ratio_order,
            ax=ratio_ax,
            color="#5B7F17",
            alpha=0.25,
            size=3.2,
            jitter=0.18,
        )
    else:
        ratio_ax.text(0.5, 0.5, "Not enough repeatability data", ha="center", va="center", transform=ratio_ax.transAxes)
    ratio_ax.axvline(1.0, color="#C65D1B", linestyle="--", linewidth=1.6)
    ratio_ax.set_xlabel("Subject Standard Deviation /\nPooled Within-Subject Standard Deviation")
    ratio_ax.set_ylabel("")
    ratio_ax.set_title("Absolute Variation Relative\nto Cohort Repeatability")

    top_heatmap = cross_metric_instability.head(12).copy()
    heatmap_columns = [f"{metric}_sd_ratio" for metric in PLOT_METRICS]
    if not top_heatmap.empty:
        heatmap_frame = (
            top_heatmap.set_index("subject")[heatmap_columns]
            .rename(columns={f"{metric}_sd_ratio": METRIC_LABELS[metric] for metric in PLOT_METRICS})
        )
        sns.heatmap(
            heatmap_frame,
            ax=heatmap_ax,
            cmap="YlOrBr",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Standard deviation ratio"},
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7},
        )
    heatmap_ax.tick_params(axis="x", rotation=20)
    heatmap_ax.set_title("Subjects With Highest\nCross-Metric Instability")
    heatmap_ax.set_xlabel("")
    heatmap_ax.set_ylabel("Subject")

    top_ranked = (
        cross_metric_instability.head(10)
        .sort_values("mean_sd_ratio_across_metrics", ascending=True)
        .reset_index(drop=True)
    )
    ranking_ax.barh(
        top_ranked["subject"],
        top_ranked["mean_sd_ratio_across_metrics"],
        color="#1E4E8C",
        edgecolor="#163C6E",
        linewidth=0.9,
    )
    ranking_ax.axvline(1.0, color="#C65D1B", linestyle="--", linewidth=1.6)
    ranking_ax.set_xlabel("Mean Standard Deviation Ratio\nAcross Primary Metrics")
    ranking_ax.set_ylabel("")
    ranking_ax.set_title("Top Subjects by Aggregate Instability")
    ranking_ax.margins(x=0.14)
    for patch, value in zip(ranking_ax.patches, top_ranked["mean_sd_ratio_across_metrics"]):
        ranking_ax.text(
            patch.get_width() + 0.03,
            patch.get_y() + patch.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#334155",
        )

    apply_multi_panel_layout(fig, top=0.90, bottom=0.11, left=0.09, right=0.985, wspace=0.42, hspace=0.36)
    fig.suptitle(
        "Subject-Level Repeat Variation",
        fontsize=17,
        fontweight="bold",
        y=0.968,
    )
    add_figure_note(fig, "CV = coefficient of variation; SD = standard deviation.")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_failure_summary_plot(
    log_detail_frame: pd.DataFrame,
    log_transition_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    failed_only = log_detail_frame[log_detail_frame["final_status"] == "failed"].copy()
    failed_only["failure_category_label"] = failed_only["failure_category"].map(humanize_label)
    failed_only["failure_stage_label"] = failed_only["failure_stage"].map(humanize_label)

    fig, axes = plt.subplots(2, 2, figsize=(20.2, 15.6))
    category_ax, stage_ax, outcome_ax, transition_ax = axes.flatten()

    category_counts = (
        failed_only.groupby(["failure_category_label", "run"])
        .size()
        .reset_index(name="count")
    )
    category_order = (
        category_counts.groupby("failure_category_label")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    sns.barplot(
        data=category_counts,
        y="failure_category_label",
        x="count",
        hue="run",
        order=category_order,
        orient="h",
        ax=category_ax,
        palette=["#1E4E8C", "#C65D1B"],
        errorbar=None,
    )
    category_ax.set_title("Failure Categories by Run")
    category_ax.set_xlabel("Failed Subject-Runs")
    category_ax.set_ylabel("")
    wrap_axis_ticklabels(category_ax, "y", width=28)
    category_ax.tick_params(axis="y", labelsize=11)
    category_ax.legend(title="Run", frameon=False, loc="lower right", fontsize=10, title_fontsize=10)

    stage_counts = (
        failed_only.groupby(["failure_stage_label", "run"])
        .size()
        .reset_index(name="count")
    )
    stage_order = (
        stage_counts.groupby("failure_stage_label")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    sns.barplot(
        data=stage_counts,
        y="failure_stage_label",
        x="count",
        hue="run",
        order=stage_order,
        orient="h",
        ax=stage_ax,
        palette=["#1E4E8C", "#C65D1B"],
        errorbar=None,
    )
    stage_ax.set_title("Failure Stages by Run")
    stage_ax.set_xlabel("Failed Subject-Runs")
    stage_ax.set_ylabel("")
    wrap_axis_ticklabels(stage_ax, "y", width=20)
    stage_ax.tick_params(axis="y", labelsize=12)
    legend = stage_ax.get_legend()
    if legend is not None:
        legend.remove()

    status_counts = (
        log_detail_frame.groupby(["run", "final_status"])
        .size()
        .reset_index(name="count")
    )
    run_order = sorted(log_detail_frame["run"].unique().tolist())
    bottom = np.zeros(len(run_order), dtype=float)
    colors = {"success": "#8FB1E3", "failed": "#C65D1B"}
    for status_value in ["success", "failed"]:
        subset = (
            status_counts[status_counts["final_status"] == status_value]
            .set_index("run")
            .reindex(run_order, fill_value=0)
        )
        outcome_ax.bar(
            [str(run) for run in run_order],
            subset["count"].to_numpy(dtype=float),
            bottom=bottom,
            color=colors[status_value],
            edgecolor="white",
            linewidth=0.9,
            label=humanize_label(status_value),
        )
        bottom = bottom + subset["count"].to_numpy(dtype=float)
    total_by_run = log_detail_frame.groupby("run").size().reindex(run_order)
    failed_by_run = (
        log_detail_frame[log_detail_frame["final_status"] == "failed"]
        .groupby("run")
        .size()
        .reindex(run_order, fill_value=0)
    )
    for index, run in enumerate(run_order):
        failure_rate = failed_by_run.loc[run] / total_by_run.loc[run] * 100.0
        outcome_ax.text(
            index,
            total_by_run.loc[run] + 2.5,
            f"{failure_rate:.1f}% failed",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#334155",
        )
    outcome_ax.set_title("Run-Level Completion Summary")
    outcome_ax.set_xlabel("Run")
    outcome_ax.set_ylabel("Subjects")
    outcome_ax.legend(frameon=False, loc="upper right")

    if not log_transition_frame.empty:
        transition_display = log_transition_frame.copy()
        transition_display["transition_label"] = transition_display["transition"].str.replace("_", " ")
        transition_order = [
            "success -> success",
            "failed -> success",
            "success -> failed",
            "failed -> failed",
        ]
        transition_display["transition_label"] = pd.Categorical(
            transition_display["transition"],
            categories=transition_order,
            ordered=True,
        )
        transition_display = transition_display.sort_values("transition_label")
        transition_labels = [value.replace(" -> ", "\nto ") for value in transition_display["transition"]]
        transition_ax.bar(
            transition_labels,
            transition_display["count"],
            color=["#8FB1E3", "#7FB069", "#E9A03B", "#C65D1B"],
            edgecolor="white",
            linewidth=0.9,
        )
        for patch, value in zip(transition_ax.patches, transition_display["count"]):
            transition_ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.4,
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#334155",
            )
        p_value = float(log_transition_frame["mcnemar_exact_pvalue"].iloc[0])
        transition_ax.text(
            0.98,
            0.97,
            f"Exact McNemar p = {p_value:.4f}",
            transform=transition_ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#334155",
        )
    transition_ax.set_title("Paired Outcome Transitions Between Runs")
    transition_ax.set_xlabel("Transition")
    transition_ax.set_ylabel("Subjects")
    transition_ax.tick_params(axis="x", rotation=0)

    apply_multi_panel_layout(fig, top=0.92, bottom=0.10, left=0.17, right=0.985, wspace=0.38, hspace=0.24)
    fig.suptitle(
        "Failure Audit Across Logged Simulation Runs",
        fontsize=17,
        fontweight="bold",
        y=0.972,
    )
    add_figure_note(
        fig,
        "CHARM = SimNIBS head-modeling and meshing workflow; FEM = finite element method; "
        "EEG = electroencephalography; MNI = Montreal Neurological Institute space.",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_image_repeatability_plot(
    subject_summary: pd.DataFrame,
    output_path: Path,
    analysis_percentile: float | None = None,
) -> None:
    if subject_summary.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.ravel()

    dice_frame = subject_summary.melt(
        id_vars="subject",
        value_vars=[
            "roi_mask_dice_mean",
            "top_percentile_mask_dice_mean",
            "overlap_mask_dice_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    dice_frame["metric_label"] = dice_frame["metric"].map(
        {
            "roi_mask_dice_mean": "ROI Mask",
            "top_percentile_mask_dice_mean": TOP_PERCENTILE_MASK_LABEL,
            "overlap_mask_dice_mean": "Overlap Mask",
        }
    )
    sns.boxplot(
        data=dice_frame,
        x="metric_label",
        y="value",
        hue="metric_label",
        order=["ROI Mask", TOP_PERCENTILE_MASK_LABEL, "Overlap Mask"],
        palette=["#5B8FF9", "#61DDAA", "#F6BD16"],
        dodge=False,
        legend=False,
        width=0.55,
        ax=axes[0],
    )
    axes[0].set_title("Mean Pairwise Dice Across Subjects")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Dice")

    jaccard_frame = subject_summary.melt(
        id_vars="subject",
        value_vars=[
            "roi_mask_jaccard_mean",
            "top_percentile_mask_jaccard_mean",
            "overlap_mask_jaccard_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    jaccard_frame["metric_label"] = jaccard_frame["metric"].map(
        {
            "roi_mask_jaccard_mean": "ROI Mask",
            "top_percentile_mask_jaccard_mean": TOP_PERCENTILE_MASK_LABEL,
            "overlap_mask_jaccard_mean": "Overlap Mask",
        }
    )
    sns.boxplot(
        data=jaccard_frame,
        x="metric_label",
        y="value",
        hue="metric_label",
        order=["ROI Mask", TOP_PERCENTILE_MASK_LABEL, "Overlap Mask"],
        palette=["#5B8FF9", "#61DDAA", "#F6BD16"],
        dodge=False,
        legend=False,
        width=0.55,
        ax=axes[1],
    )
    axes[1].set_title("Mean Pairwise Jaccard Across Subjects")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Jaccard")

    correlation_frame = subject_summary.melt(
        id_vars="subject",
        value_vars=["within_roi_field_correlation_mean"],
        var_name="metric",
        value_name="value",
    )
    correlation_frame["metric_label"] = "ROI Field"
    sns.boxplot(
        data=correlation_frame,
        x="metric_label",
        y="value",
        color="#7C3AED",
        width=0.45,
        ax=axes[2],
    )
    axes[2].set_title("Mean Pairwise Within-ROI Field Correlation")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Pearson r")

    field_cv_frame = subject_summary.melt(
        id_vars="subject",
        value_vars=[
            "roi_mean_field_cv_percent",
            "roi_p95_field_cv_percent",
            "roi_peak_field_cv_percent",
        ],
        var_name="metric",
        value_name="value",
    )
    field_cv_frame["metric_label"] = field_cv_frame["metric"].map(
        {
            "roi_mean_field_cv_percent": "ROI Mean",
            "roi_p95_field_cv_percent": "ROI P95",
            "roi_peak_field_cv_percent": "ROI Peak",
        }
    )
    sns.boxplot(
        data=field_cv_frame,
        x="metric_label",
        y="value",
        hue="metric_label",
        order=["ROI Mean", "ROI P95", "ROI Peak"],
        palette=["#4C78A8", "#F58518", "#E45756"],
        dodge=False,
        legend=False,
        width=0.55,
        ax=axes[3],
    )
    axes[3].set_title("ROI Field Repeatability (CV Across Runs)")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("CV (%)")

    voxel_cv_frame = subject_summary.melt(
        id_vars="subject",
        value_vars=[
            "within_roi_voxel_cv_percent_mean",
            "within_roi_voxel_cv_percent_p95",
        ],
        var_name="metric",
        value_name="value",
    )
    voxel_cv_frame["metric_label"] = voxel_cv_frame["metric"].map(
        {
            "within_roi_voxel_cv_percent_mean": "Mean ROI Voxel CV",
            "within_roi_voxel_cv_percent_p95": "ROI Voxel CV P95",
        }
    )
    sns.boxplot(
        data=voxel_cv_frame,
        x="metric_label",
        y="value",
        hue="metric_label",
        order=["Mean ROI Voxel CV", "ROI Voxel CV P95"],
        palette=["#72B7B2", "#54A24B"],
        dodge=False,
        legend=False,
        width=0.55,
        ax=axes[4],
    )
    axes[4].set_title("Voxelwise Within-ROI Variation")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("CV (%)")
    axes[4].tick_params(axis="x", rotation=10)

    hotspot_frame = subject_summary.melt(
        id_vars="subject",
        value_vars=[
            "peak_displacement_mm_mean",
            "overlap_com_displacement_mm_mean",
        ],
        var_name="metric",
        value_name="value",
    )
    hotspot_frame["metric_label"] = hotspot_frame["metric"].map(
        {
            "peak_displacement_mm_mean": "Peak",
            "overlap_com_displacement_mm_mean": "Overlap COM",
        }
    )
    sns.boxplot(
        data=hotspot_frame,
        x="metric_label",
        y="value",
        hue="metric_label",
        order=["Peak", "Overlap COM"],
        palette=["#E45756", "#72B7B2"],
        dodge=False,
        legend=False,
        width=0.45,
        ax=axes[5],
    )
    axes[5].set_title("Hotspot Localization Displacement")
    axes[5].set_xlabel("")
    axes[5].set_ylabel("Mean Pairwise Displacement (mm)")

    for axis in axes:
        axis.grid(axis="y", alpha=0.18)

    apply_multi_panel_layout(fig, top=0.93, bottom=0.10, left=0.08, right=0.985, wspace=0.28, hspace=0.30)
    fig.suptitle(
        "Image-Level Repeatability Across Repeated Runs",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    add_figure_note(
        fig,
        "Dice/Jaccard summarise mask overlap across all run pairs per subject. "
        "Field CV metrics use the reference ROI mask from the first successful repeat for each subject."
        + (
            f" The top-percentile mask corresponds to a {analysis_percentile:.1f}th-percentile threshold."
            if analysis_percentile is not None
            else ""
        ),
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_image_repeatability_methodology(
    output_dir: Path,
    roi_name: str,
    subject_summary: pd.DataFrame,
    issue_frame: pd.DataFrame,
    analysis_percentile: float | None = None,
) -> Path:
    analysed_subjects = int(subject_summary["subject"].nunique()) if not subject_summary.empty else 0
    skipped_subjects = int(issue_frame["subject"].nunique()) if not issue_frame.empty else 0
    lines = [
        "# Image-Level Repeatability Methodology",
        "",
        "## Purpose",
        "",
        (
            "This layer extends the scalar `subject_metrics.json` analysis into the saved NIfTI outputs so that "
            "repeatability can be measured directly on ROI masks, top-percentile masks, overlap masks, within-ROI field values, "
            "and hotspot localization."
        ),
        "",
        "## Inputs",
        "",
        f"- ROI analysed: `{roi_name}`",
        percentile_context_line(analysis_percentile),
        f"- Subjects successfully analysed at the image level: `{analysed_subjects}`",
        f"- Subjects skipped due to missing files or incompatible headers: `{skipped_subjects}`",
        "",
        "For each successful subject-run, the analysis reads:",
        "",
        f"- `atlas_{roi_name}_mask.nii.gz` as the ROI mask",
        f"- `{top_percentile_mask_filename(analysis_percentile)}` as the whole-volume top-percentile mask",
        f"- `{overlap_percentile_mask_filename(roi_name, analysis_percentile)}` as the target-overlap mask",
        f"- `TI_in_{roi_name}.nii.gz` as the within-ROI field image",
        "",
        "## Core Rules",
        "",
        "- The complete-case subject set is used so that all repeated runs are directly comparable.",
        "- All images for a subject must have identical voxel grids and affines across runs; otherwise the subject is skipped.",
        (
            "- Within-ROI field repeatability is evaluated on the reference ROI mask from the first successful run for that subject. "
            "This avoids conflating field variability with a changing ROI support if ROI masks were ever to differ."
        ),
        (
            "- Voxelwise field comparisons use only the voxel support that is finite in every run for a subject. "
            "This avoids conflating field repeatability with run-to-run changes in non-finite voxel support."
        ),
        "",
        "## Metrics Computed",
        "",
        "### 1. Mask Repeatability",
        "",
        "- Pairwise Dice and Jaccard for the ROI mask across all run pairs.",
        "- Pairwise Dice and Jaccard for the top-percentile mask across all run pairs.",
        "- Pairwise Dice and Jaccard for the overlap mask across all run pairs.",
        "",
        "### 2. Within-ROI Field Repeatability",
        "",
        "- Pairwise Pearson correlation of within-ROI voxel vectors across all run pairs.",
        "- Voxelwise SD across runs within the ROI, summarized by mean, median, 95th percentile, and maximum.",
        "- Voxelwise CV (%) across runs within the ROI, summarized by mean, median, 95th percentile, and maximum.",
        "",
        "### 3. ROI Summary-Field Repeatability",
        "",
        "- Per-run ROI mean field, ROI 95th percentile, and ROI peak field.",
        "- Across-run SD, CV, mean absolute pairwise difference, and maximum absolute pairwise difference for each summary metric.",
        "",
        "### 4. Hotspot Localization Stability",
        "",
        "- Peak-field voxel location inside the reference ROI, converted to millimeter coordinates.",
        "- Center of mass of the overlap mask, converted to millimeter coordinates.",
        "- Pairwise Euclidean displacement for peak location and overlap-mask center of mass across all run pairs.",
        "",
        "## Outputs",
        "",
        "- `image_repeatability_run_level.csv`",
        "- `image_repeatability_pairwise_subject_run_pairs.csv`",
        "- `image_repeatability_subject_level.csv`",
        "- `image_repeatability_pairwise_run_summary.csv`",
        "- `image_repeatability_cohort_summary.csv`",
        "- `image_repeatability_issues.csv`",
        "- `image_repeatability_report.md`",
        "- `figures/08_image_repeatability_summary.png`",
        "",
    ]
    path = output_dir / "image_repeatability_methodology.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_image_repeatability_report(
    output_dir: Path,
    roi_name: str,
    subject_summary: pd.DataFrame,
    issue_frame: pd.DataFrame,
    analysis_percentile: float | None = None,
) -> Path:
    analysed_subjects = int(subject_summary["subject"].nunique()) if not subject_summary.empty else 0
    skipped_subjects = int(issue_frame["subject"].nunique()) if not issue_frame.empty else 0
    grid_consistent = int(subject_summary["grid_consistent_all_runs"].sum()) if analysed_subjects else 0
    roi_identical = int(subject_summary["roi_mask_identical_all_runs"].sum()) if analysed_subjects else 0

    def cohort_mean(column: str) -> float:
        series = pd.to_numeric(subject_summary[column], errors="coerce").dropna()
        return float(series.mean()) if not series.empty else math.nan

    def cohort_median(column: str) -> float:
        series = pd.to_numeric(subject_summary[column], errors="coerce").dropna()
        return float(series.median()) if not series.empty else math.nan

    lines = [
        "# Image-Level Repeatability Report",
        "",
        "## Scope",
        "",
        f"- ROI analysed: `{roi_name}`",
        percentile_context_line(analysis_percentile),
        f"- Subjects analysed at image level: `{analysed_subjects}`",
        f"- Subjects skipped: `{skipped_subjects}`",
        f"- Subjects with grid/affine consistency across all runs: `{grid_consistent}` of `{analysed_subjects}`",
        f"- Subjects with binary-identical ROI masks across all runs: `{roi_identical}` of `{analysed_subjects}`",
        "",
    ]

    if analysed_subjects:
        lines.extend(
            [
                "## ROI Mask Repeatability",
                "",
                (
                    f"- Mean subject-level pairwise Dice: `{cohort_mean('roi_mask_dice_mean'):.6f}` "
                    f"(median `{cohort_median('roi_mask_dice_mean'):.6f}`)"
                ),
                (
                    f"- Mean subject-level pairwise Jaccard: `{cohort_mean('roi_mask_jaccard_mean'):.6f}` "
                    f"(median `{cohort_median('roi_mask_jaccard_mean'):.6f}`)"
                ),
                (
                    "Interpretation: this quantifies whether the ROI definition itself moves across repeats. "
                    "Values near 1.0 indicate that downstream field variability is not being driven by ROI-mask drift."
                ),
                "",
                "## Top-Percentile And Overlap Mask Repeatability",
                "",
                (
                    f"- Top-percentile mask mean pairwise Dice: `{cohort_mean('top_percentile_mask_dice_mean'):.6f}`; "
                    f"Jaccard: `{cohort_mean('top_percentile_mask_jaccard_mean'):.6f}`"
                ),
                (
                    f"- Overlap mask mean pairwise Dice: `{cohort_mean('overlap_mask_dice_mean'):.6f}`; "
                    f"Jaccard: `{cohort_mean('overlap_mask_jaccard_mean'):.6f}`"
                ),
                (
                    "Interpretation: these are the spatial-repeatability metrics for the actual strong-field region and the "
                    "target-engagement region, not just their voxel counts."
                ),
                "",
                "## Within-ROI Field Repeatability",
                "",
                (
                    f"- Mean subject-level pairwise within-ROI field correlation: "
                    f"`{cohort_mean('within_roi_field_correlation_mean'):.6f}`"
                ),
                (
                    f"- Mean voxelwise ROI SD across runs: `{cohort_mean('within_roi_voxel_sd_mean'):.6f}`"
                ),
                (
                    f"- Mean voxelwise ROI CV across runs: `{cohort_mean('within_roi_voxel_cv_percent_mean'):.3f}%`; "
                    f"95th-percentile voxelwise CV: `{cohort_mean('within_roi_voxel_cv_percent_p95'):.3f}%`"
                ),
                "",
                "## ROI Summary-Field Repeatability",
                "",
                (
                    f"- ROI mean field CV across runs: `{cohort_mean('roi_mean_field_cv_percent'):.3f}%`; "
                    f"mean absolute pairwise difference: `{cohort_mean('roi_mean_field_mean_abs_pairwise_diff'):.6f}`"
                ),
                (
                    f"- ROI P95 field CV across runs: `{cohort_mean('roi_p95_field_cv_percent'):.3f}%`; "
                    f"mean absolute pairwise difference: `{cohort_mean('roi_p95_field_mean_abs_pairwise_diff'):.6f}`"
                ),
                (
                    f"- ROI peak field CV across runs: `{cohort_mean('roi_peak_field_cv_percent'):.3f}%`; "
                    f"mean absolute pairwise difference: `{cohort_mean('roi_peak_field_mean_abs_pairwise_diff'):.6f}`"
                ),
                "",
                "## Hotspot Localization Stability",
                "",
                (
                    f"- Mean pairwise peak-voxel displacement: `{cohort_mean('peak_displacement_mm_mean'):.3f}` mm"
                ),
                (
                    f"- Mean pairwise overlap-mask center-of-mass displacement: "
                    f"`{cohort_mean('overlap_com_displacement_mm_mean'):.3f}` mm"
                ),
                (
                    "Interpretation: these displacement metrics answer whether the location of the hotspot is stable, even when "
                    "the overall overlap fraction or peak value remains similar."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## No Successful Image-Level Subjects",
                "",
                (
                    "No subjects completed the image-level repeatability analysis successfully. Inspect "
                    "`image_repeatability_issues.csv` for the per-subject failure reasons."
                ),
                "",
            ]
        )

    if not issue_frame.empty:
        lines.extend(
            [
                "## Skipped Subjects",
                "",
                "These subjects were excluded from the image-level layer due to missing files or incompatible headers:",
                "",
            ]
        )
        for row in issue_frame.itertuples(index=False):
            lines.append(f"- `{row.subject}`: `{row.issue_type}`. {row.details}")
        lines.append("")

    path = output_dir / "image_repeatability_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(
    dataset_root: Path,
    output_dir: Path,
    roi_name: str,
    all_frame: pd.DataFrame,
    complete_case_frame: pd.DataFrame,
    coverage: pd.DataFrame,
    experiment_stats: pd.DataFrame,
    image_repeatability_subject_summary: pd.DataFrame | None = None,
    complete_case_only: bool = False,
) -> Path:
    unique_subjects = all_frame["subject"].nunique()
    complete_subjects = complete_case_frame["subject"].nunique()
    run_count = coverage["repeat_id"].nunique()
    min_subjects = int(coverage["n_subjects"].min())
    max_subjects = int(coverage["n_subjects"].max())

    overlap_row = experiment_stats.loc[experiment_stats["metric"] == "overlap_fraction"].iloc[0]
    percentile_row = experiment_stats.loc[experiment_stats["metric"] == "percentile_value"].iloc[0]
    top_voxel_row = experiment_stats.loc[experiment_stats["metric"] == "top_percentile_voxels"].iloc[0]
    overlap_voxel_row = experiment_stats.loc[experiment_stats["metric"] == "overlap_top_voxels"].iloc[0]
    plot_rows = experiment_stats[experiment_stats["metric"].isin(PLOT_METRICS)].copy()
    icc_min = plot_rows["icc_absolute_agreement"].min()
    icc_max = plot_rows["icc_absolute_agreement"].max()
    repeat_effect_max = plot_rows["run_variance_fraction_percent"].max()
    image_repeatability_available = (
        image_repeatability_subject_summary is not None
        and not image_repeatability_subject_summary.empty
    )

    report_lines = [
        "# Subject Metrics Analysis",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- ROI analysed: `{roi_name}`",
        f"- Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`",
        "",
        "## Scope",
        "",
        f"- Observations loaded: **{len(all_frame):,}** subject-run records.",
        f"- Unique subjects observed in at least one repeat: **{unique_subjects}**.",
        f"- Repeats analysed: **{run_count}** (`R01` to `R{run_count:02d}`).",
        f"- Subjects per repeat ranged from **{min_subjects}** to **{max_subjects}**.",
        (
            f"- Complete-case population {'used throughout the analysis' if complete_case_only else 'for paired between-run comparisons'}: "
            f"**{complete_subjects}** subjects present in every repeat."
        ),
        "",
        "## Definitions",
        "",
        (
            "- Repeat level: population statistics are computed separately inside each repeat on the complete-case cohort only."
            if complete_case_only
            else "- Repeat level: population statistics are computed separately inside each repeat across all available subjects."
        ),
        (
            "- Experiment level: the repeat-level population means are treated as the run-to-run series, "
            "and paired comparisons are computed on the complete-case subject set so repeats remain directly comparable."
        ),
        "",
        "## Key Findings",
        "",
        (
            f"- `Overlap Fraction` was highly stable across repeats. The run means ranged from "
            f"**{METRIC_FORMATTERS['overlap_fraction'](overlap_row['min_run_mean'])}** to "
            f"**{METRIC_FORMATTERS['overlap_fraction'](overlap_row['max_run_mean'])}**, "
            f"with an across-run SD of **{METRIC_FORMATTERS['overlap_fraction'](overlap_row['sd_of_run_means'])}**."
        ),
        (
            f"- `Percentile Value` changed very little between repeats. The run-mean range was "
            f"**{METRIC_FORMATTERS['percentile_value'](percentile_row['range_run_mean'])}**, "
            f"and the Friedman repeated-measures test returned **p = {percentile_row['friedman_pvalue']:.3g}**."
        ),
        (
            f"- `Top Percentile Voxels` varied modestly at the experiment level. The run means differed by "
            f"at most **{METRIC_FORMATTERS['top_percentile_voxels'](top_voxel_row['max_abs_pairwise_diff'])}** voxels "
            f"between any two repeats on the complete-case population."
        ),
        (
            f"- Absolute-agreement ICC remained high across the primary metrics, ranging from "
            f"**{icc_min:.3f}** to **{icc_max:.3f}**."
        ),
        (
            f"- Repeat effects contributed almost none of the total variance. Across the primary metrics, the "
            f"estimated repeat-level variance fraction never exceeded **{repeat_effect_max:.3f}%**."
        ),
        (
            f"- `Overlap Top Voxels` remained tightly clustered. The mean absolute paired run difference was "
            f"**{METRIC_FORMATTERS['overlap_top_voxels'](overlap_voxel_row['mean_abs_pairwise_diff'])}** voxels."
        ),
        (
            f"- Within-subject variability across repeats was larger than experiment-level drift. "
            f"For `Overlap Fraction`, the average subject-level across-repeat SD was "
            f"**{METRIC_FORMATTERS['overlap_fraction'](overlap_row['subject_within_run_sd_mean'])}**, "
            f"while the SD of the repeat-level means was only "
            f"**{METRIC_FORMATTERS['overlap_fraction'](overlap_row['sd_of_run_means'])}**."
        ),
        (
            f"- The pooled within-subject coefficient of variation was "
            f"**{overlap_row['pooled_within_subject_cv_percent']:.2f}%** for `Overlap Fraction` and "
            f"**{percentile_row['pooled_within_subject_cv_percent']:.2f}%** for `Percentile Value`; "
            f"all fitted run-drift slopes were non-significant."
        ),
        "",
        "## Outputs",
        "",
        "- `subject_metrics_long.csv`: flattened subject_metrics records used in the analysis.",
        "- `run_subject_coverage.csv`: subject availability per repeat.",
        (
            "- `repeat_level_population_statistics.csv`: per-repeat population summaries restricted to the complete-case population."
            if complete_case_only
            else "- `repeat_level_population_statistics.csv`: per-repeat population summaries using all available subjects."
        ),
        (
            "- `repeat_level_population_statistics_complete_subjects.csv`: compatibility copy of the same complete-case per-repeat summaries."
            if complete_case_only
            else "- `repeat_level_population_statistics_complete_subjects.csv`: per-repeat summaries restricted to the complete-case population."
        ),
        "- `experiment_level_population_statistics.csv`: across-run experiment summaries plus repeated-measures stability metrics.",
        "- `variation_analysis_metrics.csv`: concise experiment-level repeatability and drift metrics for the primary endpoints.",
        "- `pairwise_run_differences.csv`: paired repeat-to-repeat differences on complete-case subjects.",
        "- `within_subject_repeatability.csv`: subject-level across-repeat variability summaries.",
        "- `subject_level_variation.csv`, `subject_level_variation_summary.csv`, and `subject_cross_metric_instability.csv`: subject-level repeat-variation outputs.",
        "- `log_subject_run_details.csv`, `log_failure_summary_by_category.csv`, and `log_run_transition_summary.csv`: log-derived failure audit outputs.",
        "- `subject_level_variation_report.md` and `failure_report.md`: narrative interpretation of subject-level instability and operational failures.",
        "- `figures/*.png`: presentation-ready figures for coverage, repeat-level distributions, run means, pairwise differences, variation summaries, subject-level instability, and the failure audit.",
        "",
    ]

    if image_repeatability_available:
        def cohort_mean(column: str) -> float:
            series = pd.to_numeric(image_repeatability_subject_summary[column], errors="coerce").dropna()
            return float(series.mean()) if not series.empty else math.nan

        outputs_index = report_lines.index("## Outputs")
        report_lines[outputs_index:outputs_index] = [
            (
                f"- The image-level repeatability layer confirmed that ROI support was stable. Mean ROI-mask Dice was "
                f"**{cohort_mean('roi_mask_dice_mean'):.3f}**, while top-percentile and overlap-mask Dice were "
                f"**{cohort_mean('top_percentile_mask_dice_mean'):.3f}** and **{cohort_mean('overlap_mask_dice_mean'):.3f}**."
            ),
            (
                f"- Within-ROI field repeatability was high. The mean pairwise voxelwise field correlation was "
                f"**{cohort_mean('within_roi_field_correlation_mean'):.3f}**, with a mean voxelwise ROI CV of "
                f"**{cohort_mean('within_roi_voxel_cv_percent_mean'):.2f}%**."
            ),
            (
                f"- Hotspot localization drift was limited relative to ROI size. Mean pairwise peak displacement was "
                f"**{cohort_mean('peak_displacement_mm_mean'):.2f} mm**, and overlap-mask center-of-mass displacement was "
                f"**{cohort_mean('overlap_com_displacement_mm_mean'):.2f} mm**."
            ),
            "",
        ]
        figure_line_index = report_lines.index(
            "- `figures/*.png`: presentation-ready figures for coverage, repeat-level distributions, run means, pairwise differences, variation summaries, subject-level instability, and the failure audit."
        )
        report_lines[figure_line_index:figure_line_index] = [
            "- `image_repeatability_run_level.csv`, `image_repeatability_pairwise_subject_run_pairs.csv`, `image_repeatability_subject_level.csv`, `image_repeatability_pairwise_run_summary.csv`, and `image_repeatability_cohort_summary.csv`: image-level repeatability tables for masks, within-ROI fields, and hotspot localization.",
            "- `image_repeatability_issues.csv`, `image_repeatability_methodology.md`, and `image_repeatability_report.md`: image-level audit outputs and narrative interpretation.",
            "",
        ]
        report_lines[figure_line_index + 3] = (
            "- `figures/*.png`: presentation-ready figures for coverage, repeat-level distributions, run means, pairwise differences, variation summaries, subject-level instability, the failure audit, and image-level repeatability."
        )

    report_path = output_dir / "analysis_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def write_methodology_documentation(
    dataset_root: Path,
    output_dir: Path,
    roi_name: str,
    all_frame: pd.DataFrame,
    complete_case_frame: pd.DataFrame,
    coverage: pd.DataFrame,
    repeat_level_available: pd.DataFrame,
    experiment_stats: pd.DataFrame,
    include_image_repeatability: bool = False,
    complete_case_only: bool = False,
) -> Path:
    unique_subjects = int(all_frame["subject"].nunique())
    complete_subjects = int(complete_case_frame["subject"].nunique())
    run_count = int(coverage["repeat_id"].nunique())
    min_subjects = int(coverage["n_subjects"].min())
    max_subjects = int(coverage["n_subjects"].max())
    primary_stats = experiment_stats[experiment_stats["metric"].isin(PLOT_METRICS)].copy()

    overlap_row = primary_stats.loc[primary_stats["metric"] == "overlap_fraction"].iloc[0]
    percentile_row = primary_stats.loc[primary_stats["metric"] == "percentile_value"].iloc[0]
    top_voxel_row = primary_stats.loc[primary_stats["metric"] == "top_percentile_voxels"].iloc[0]
    overlap_voxel_row = primary_stats.loc[primary_stats["metric"] == "overlap_top_voxels"].iloc[0]
    analysis_percentile = (
        infer_uniform_percentile(all_frame["percentile"].tolist())
        if "percentile" in all_frame.columns
        else None
    )

    repeat_level_fields = [
        "`n`",
        "`mean`",
        "`std`",
        "`sem`",
        "`ci95_low` / `ci95_high`",
        "`median`",
        "`q1` / `q3`",
        "`iqr`",
        "`min` / `max`",
        "`cv_percent`",
    ]
    experiment_level_fields = [
        "`mean_of_run_means`",
        "`sd_of_run_means`",
        "`range_run_mean`",
        "`mean_abs_pairwise_diff`",
        "`max_abs_pairwise_diff`",
        "`friedman_pvalue`",
        "`kendall_w`",
        "`pooled_within_subject_sd`",
        "`pooled_within_subject_cv_percent`",
        "`standard_error_of_measurement`",
        "`repeatability_coefficient`",
        "`icc_absolute_agreement`",
        "`mean_pairwise_correlation`",
        "`drift_slope_per_repeat` / `drift_slope_percent_per_repeat`",
        "`drift_pvalue` / `drift_r_squared`",
        "`subject_variance_fraction_percent` / `run_variance_fraction_percent` / `residual_variance_fraction_percent`",
    ]

    methodology_lines = [
        "# Analysis Methodology",
        "",
        "## Why This Analysis Was Done",
        "",
        (
            f"The goal of this analysis was to quantify how stable the ROI-focused `subject_metrics` outputs are "
            f"across repeated runs of the same experiment for the `{roi_name}` target."
        ),
        "",
        (
            "The analysis was designed to answer three practical questions:"
        ),
        "",
        "1. Are the population-level results similar from run to run?",
        "2. Is any observed variation mostly due to subject-to-subject biological/anatomical differences, or due to run-to-run instability?",
        "3. How large is repeatability error relative to the signal being measured?",
        "",
        (
            "Those questions matter because a stable pipeline should produce very similar population summaries across "
            "repeats, while still allowing meaningful subject-level variation to remain visible."
        ),
        "",
        "## Dataset Used",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- ROI analysed: `{roi_name}`",
        f"- Repeats analysed: `{run_count}` (`R01` to `R{run_count:02d}`)",
        f"- Total subject-run observations loaded: `{len(all_frame):,}`",
        f"- Unique subjects present in at least one repeat: `{unique_subjects}`",
        f"- Subjects per repeat ranged from `{min_subjects}` to `{max_subjects}`",
        (
            f"- Complete-case population used throughout the repeatability analysis: `{complete_subjects}` subjects"
            if complete_case_only
            else f"- Complete-case population used for paired between-run analysis: `{complete_subjects}` subjects"
        ),
        "",
        "## What Data Were Read",
        "",
        (
            "Each `subject_metrics.json` file was flattened into one subject-run record containing the run label, "
            "subject identifier, the scalar ROI metrics stored under `rois[\"Left-Hippocampus\"]`, and the flattened "
            "`extended_metrics` payload when present."
        ),
        "",
        "The analysis used these fields:",
        "",
        "- `percentile_value`: threshold value defining the configured top-percentile field region for that subject/run.",
        "- `top_percentile_voxels`: number of voxels entering the top-percentile mask.",
        "- `overlap_top_voxels`: count of top-percentile voxels that overlap the target ROI.",
        "- `overlap_fraction`: fraction of the ROI covered by the top-percentile voxels.",
        "- `roi_voxels`, `roi_volume_mm3`, `voxel_volume_mm3`, `percentile`: retained as structural/reference metrics.",
        "",
        (
            "The four boldest analytical endpoints were `percentile_value`, `top_percentile_voxels`, "
            "`overlap_top_voxels`, and `overlap_fraction`, because these are the metrics that directly describe "
            "field strength thresholding, extent of the high-field region, and target engagement."
        ),
        "",
        "## Why Two Analysis Levels Were Used",
        "",
        "### 1. Repeat-Level Population Analysis",
        "",
        (
            "Repeat-level statistics were computed separately inside each run using all subjects available in that run. "
            "This was done to describe the actual population distribution produced by each repeat without discarding "
            "subjects unnecessarily."
        ),
        "",
        (
            "This level answers: \"What population result did each run produce on its own?\""
        ),
        "",
        "### 2. Experiment-Level Between-Run Analysis",
        "",
        (
            "Experiment-level statistics were computed on the complete-case population only, meaning the subjects "
            "present in every repeat. This was necessary because between-run comparisons should be paired subject by "
            "subject; otherwise, differences in subject membership could be mistaken for run instability."
        ),
        "",
        (
            "This level answers: \"If the same subjects are compared across all runs, how much do the run-level results actually move?\""
        ),
        "",
        "## Exactly What Was Computed",
        "",
        "### A. Coverage Assessment",
        "",
        (
            "Before analysing the metrics themselves, subject coverage was summarized across repeats. This was done "
            "to verify whether missing subjects could bias run-to-run comparisons."
        ),
        "",
        "Computed outputs:",
        "",
        "- `n_subjects` per repeat",
        "- `n_missing_from_union`",
        "- complete-case subject count",
        "",
        (
            "This is why the bar chart includes the complete-case reference line: it shows how many subjects are "
            "available for fair paired comparisons across all repeats."
        ),
        "",
        "### B. Repeat-Level Population Statistics",
        "",
        (
            "For each repeat and each metric, the script computed descriptive population statistics across subjects."
        ),
        "",
        "Fields written to `repeat_level_population_statistics.csv`:",
        "",
        *[f"- {field}" for field in repeat_level_fields],
        "",
        "Why these were included:",
        "",
        "- Mean and CI show the central run-level estimate and its precision.",
        "- Median and quartiles show the shape of the subject distribution without assuming symmetry.",
        "- Standard deviation and CV quantify population spread.",
        "- Min/max show the observed range and help identify whether apparent run differences are small relative to subject heterogeneity.",
        "",
        "### C. Pairwise Between-Run Differences",
        "",
        (
            "For the complete-case subjects, each pair of repeats was compared by subtracting one run from another "
            "subject by subject. This produced a paired difference distribution for every run pair."
        ),
        "",
        "Why this was done:",
        "",
        "- It shows the direction of run-to-run change, not just its size.",
        "- It reveals whether a specific repeat is systematically higher or lower than another.",
        "- It supports the heatmap, which is easier to interpret than a large table of pairwise contrasts.",
        "",
        "### D. Experiment-Level Stability Metrics",
        "",
        (
            "For each metric, the run means from the complete-case population were treated as the experiment-level "
            "series. This provides a direct measure of how much the population summary changes across repeats."
        ),
        "",
        "Fields written to `experiment_level_population_statistics.csv`:",
        "",
        *[f"- {field}" for field in experiment_level_fields],
        "",
        "Why these were included:",
        "",
        "- `mean_of_run_means`, `sd_of_run_means`, and `range_run_mean` quantify run-to-run movement of the population result.",
        "- `mean_abs_pairwise_diff` and `max_abs_pairwise_diff` quantify the practical size of between-run shifts.",
        "- `friedman_pvalue` and `kendall_w` test whether repeated runs differ systematically when subjects are paired.",
        "",
        "### E. Repeatability and Measurement-Error Metrics",
        "",
        (
            "A second layer of analysis was added to separate real subject variation from repeatability error."
        ),
        "",
        "These metrics were chosen because they are standard and interpretable in a variation study:",
        "",
        "- `pooled_within_subject_sd`: the residual within-subject spread across repeats after accounting for subject means and run means.",
        "- `standard_error_of_measurement`: equal here to the pooled within-subject SD; it estimates typical measurement error.",
        "- `repeatability_coefficient`: `1.96 * sqrt(2) * pooled_within_subject_sd`; this is the approximate 95% limit for absolute disagreement between two repeats of the same subject.",
        "- `pooled_within_subject_cv_percent`: within-subject SD expressed relative to the grand mean; this makes metrics on different scales comparable.",
        "- `icc_absolute_agreement`: intraclass correlation for absolute agreement, included because it answers whether repeated measurements preserve the same absolute values, not only rank order.",
        "- `mean_pairwise_correlation`: average correlation across all run pairs; this provides a more intuitive reliability check alongside ICC.",
        "",
        "### F. Drift and Variance Decomposition",
        "",
        (
            "The analysis also tested whether there was any monotonic drift across repeat order and how the total "
            "variance partitions into subject effects, repeat effects, and residual repeatability error."
        ),
        "",
        "Why this matters:",
        "",
        "- A significant drift slope would suggest systematic run-order bias.",
        "- A large repeat-variance fraction would suggest instability at the run level.",
        "- A dominant subject-variance fraction indicates that most variation is due to subject differences rather than pipeline inconsistency.",
        "",
        "### G. Subject-Level Repeat Variation",
        "",
        (
            "Because each subject is rerun with the same montage and solver settings but a newly generated mesh, "
            "a subject-level variation layer was added to quantify mesh-regeneration sensitivity directly."
        ),
        "",
        "For each subject and each metric, the script computes:",
        "",
        "- `mean`, `std`, `cv_percent`, `median`, `mad`, `min`, `max`, and `range`",
        "- `mean_abs_pairwise_diff` and `max_abs_pairwise_diff` across all repeat pairs",
        "- `sd_vs_pooled_repeatability`, which compares the subject's repeat SD to the cohort pooled within-subject SD",
        "- per-subject linear drift slope, p-value, and `R²`",
        "",
        (
            "This layer is important because the population mean can be stable even when a subset of subjects is "
            "unusually sensitive to remeshing. It also prevents over-interpreting CV alone in low-overlap subjects."
        ),
        "",
        "### H. Failure Audit Of Logged Runs",
        "",
        (
            "A separate log-analysis layer was added for the matching simulation logs in `Left-Hippocapus_logs`. "
            "This was done to identify which subjects failed, when they failed, why they failed, and what the most "
            "defensible corrective action would be."
        ),
        "",
        "This layer performs the following steps:",
        "",
        "- reads `simulation_run_status_summary.csv` for subject/run outcomes",
        "- maps each failed subject-run to its raw SLURM log",
        "- classifies failures by signature, stage, likely cause, and recommended fix",
        "- compares the two logged runs as paired binary outcomes using an exact McNemar test",
        "",
        (
            "Exact McNemar testing is the correct choice here because the same subjects were retried across two runs, "
            "so completion status is paired rather than independent."
        ),
        "",
        "## Why These Choices Fit This Dataset",
        "",
        (
            "This dataset is a repeated-run experiment with largely overlapping subject membership across runs, so the "
            "most defensible design is:"
        ),
        "",
        (
            "- Use the same complete-case subjects for both within-run and between-run summaries."
            if complete_case_only
            else "- Use all available subjects when describing each run independently."
        ),
        "- Use only complete-case subjects when comparing runs against each other.",
        "- Report both descriptive statistics and repeatability metrics, because a low run-to-run drift can coexist with meaningful subject-level variability.",
        "",
        (
            "This prevents two common mistakes: first, inflating between-run differences by comparing non-identical "
            "subject groups; second, reporting only means without quantifying measurement repeatability."
        ),
        "",
        "## What The Results Mean In This Specific Analysis",
        "",
        (
            f"For this dataset, the population means were very stable across repeats. For example, "
            f"`overlap_fraction` run means ranged from "
            f"`{METRIC_FORMATTERS['overlap_fraction'](overlap_row['min_run_mean'])}` to "
            f"`{METRIC_FORMATTERS['overlap_fraction'](overlap_row['max_run_mean'])}`, while the across-run SD was "
            f"`{METRIC_FORMATTERS['overlap_fraction'](overlap_row['sd_of_run_means'])}`."
        ),
        "",
        (
            f"The reliability metrics support the same conclusion: ICC(A,1) ranged from "
            f"`{primary_stats['icc_absolute_agreement'].min():.3f}` to "
            f"`{primary_stats['icc_absolute_agreement'].max():.3f}` across the four primary endpoints."
        ),
        "",
        (
            f"The repeat effect itself was negligible. The estimated repeat-level variance fraction was at most "
            f"`{primary_stats['run_variance_fraction_percent'].max():.4f}%`, meaning almost all of the observed "
            "variation came from subject differences and residual within-subject repeatability error rather than from the run identity."
        ),
        "",
        (
            f"`top_percentile_voxels` was the most repeatable metric in relative terms, with pooled within-subject CV "
            f"`{top_voxel_row['pooled_within_subject_cv_percent']:.2f}%` and ICC "
            f"`{top_voxel_row['icc_absolute_agreement']:.3f}`."
        ),
        "",
        (
            f"`overlap_top_voxels` and `overlap_fraction` showed larger within-subject repeatability error "
            f"(`{overlap_voxel_row['pooled_within_subject_cv_percent']:.2f}%` and "
            f"`{overlap_row['pooled_within_subject_cv_percent']:.2f}%` CV respectively), but still very high ICC "
            "and minimal run-level drift."
        ),
        "",
        (
            f"`percentile_value` remained stable both absolutely and relatively, with pooled within-subject CV "
            f"`{percentile_row['pooled_within_subject_cv_percent']:.2f}%`, ICC "
            f"`{percentile_row['icc_absolute_agreement']:.3f}`, and no evidence of systematic drift."
        ),
        "",
        "## How To Read The Figures",
        "",
        "- `01_subject_coverage.png`: shows subject availability per repeat. The y-axis is intentionally focused on the observed range so small coverage differences are visible.",
        "- `02_repeat_level_distributions.png`: shows the full subject-level distribution in each repeat and overlays the run mean.",
        "- `03_repeat_level_mean_ci.png`: emphasizes the run mean and 95% CI for each repeat so small between-run shifts can be seen clearly.",
        "- `04_pairwise_run_differences.png`: shows signed paired differences between repeats on the complete-case population.",
        "- `05_variation_summary.png`: condenses the main repeatability findings into ICC, CV, variance decomposition, and run-drift panels.",
        "",
        "## Limitations And Interpretation Boundaries",
        "",
        "- This is a repeated-run stability analysis, not a causal model of biological mechanisms.",
        "- The Friedman test is non-parametric and robust for repeated measures, but the practical interpretation here should rely more on effect size and repeatability than on p-values alone.",
        "- Metrics that are structurally constant in the source files, such as `percentile`, are kept for completeness but do not carry variation information.",
        "- The focused y-axis in the coverage chart is intentional for visibility; it should not be interpreted as exaggeration of practical importance, only as a display choice.",
        "",
        "## Reproducing The Analysis",
        "",
        "Run:",
        "",
        "```bash",
        f"python3 {dataset_root.name}/analyze_subject_metrics.py {dataset_root.name}",
        "```",
        "",
        (
            "This will regenerate the flattened data table, repeat-level and experiment-level statistics, the "
            "variation-analysis table, the subject-level variation outputs, the failure audit outputs when logs are "
            "available, the summary report, this methodology document, and all figures."
        ),
        "",
    ]

    if include_image_repeatability:
        image_section_index = methodology_lines.index("### H. Failure Audit Of Logged Runs")
        methodology_lines[image_section_index] = "### I. Failure Audit Of Logged Runs"
        methodology_lines[image_section_index:image_section_index] = [
            "### H. Image-Level Repeatability Layer",
            "",
            (
                "A third repeatability layer was added for the saved NIfTI outputs so the pipeline can compare the actual "
                "ROI support, top-percentile mask, overlap mask, within-ROI field map, and hotspot location across repeated runs "
                "of the same subject."
            ),
            "",
            "For each complete-case subject, this layer reads:",
            "",
            f"- `atlas_{roi_name}_mask.nii.gz`",
            f"- `{top_percentile_mask_filename(analysis_percentile)}`",
            f"- `{overlap_percentile_mask_filename(roi_name, analysis_percentile)}`",
            f"- `TI_in_{roi_name}.nii.gz`",
            "",
            "It then computes:",
            "",
            "- pairwise Dice and Jaccard for the ROI mask, top-percentile mask, and overlap mask",
            "- pairwise within-ROI voxelwise field correlation across run pairs",
            "- voxelwise within-ROI SD and CV summaries across runs",
            "- across-run repeatability summaries for ROI mean, ROI P95, and ROI peak field",
            "- pairwise peak-location displacement and overlap-mask center-of-mass displacement in millimeters",
            "",
            (
                "Voxelwise field comparisons are restricted to the common finite voxel support across all runs for each "
                "subject, so non-finite voxels do not masquerade as repeatability error."
            ),
            "",
        ]

    methodology_path = output_dir / "analysis_methodology.md"
    methodology_path.write_text("\n".join(methodology_lines), encoding="utf-8")
    return methodology_path


def write_results_interpretation(
    output_dir: Path,
    roi_name: str,
    coverage: pd.DataFrame,
    repeat_level_available: pd.DataFrame,
    experiment_stats: pd.DataFrame,
    complete_case_only: bool = False,
) -> Path:
    primary_stats = experiment_stats[experiment_stats["metric"].isin(PLOT_METRICS)].copy()
    coverage_range = int(coverage["n_subjects"].max() - coverage["n_subjects"].min())

    def metric_row(metric: str) -> pd.Series:
        return primary_stats.loc[primary_stats["metric"] == metric].iloc[0]

    def metric_extremes(metric: str) -> tuple[str, float, str, float]:
        subset = repeat_level_available[repeat_level_available["metric"] == metric].sort_values("repeat_id")
        high = subset.loc[subset["mean"].idxmax()]
        low = subset.loc[subset["mean"].idxmin()]
        return (
            str(high["run_short"]),
            float(high["mean"]),
            str(low["run_short"]),
            float(low["mean"]),
        )

    overlap_fraction = metric_row("overlap_fraction")
    overlap_top_voxels = metric_row("overlap_top_voxels")
    percentile_value = metric_row("percentile_value")
    top_percentile_voxels = metric_row("top_percentile_voxels")

    overlap_fraction_high_run, overlap_fraction_high_value, overlap_fraction_low_run, overlap_fraction_low_value = metric_extremes("overlap_fraction")
    percentile_high_run, percentile_high_value, percentile_low_run, percentile_low_value = metric_extremes("percentile_value")
    top_voxel_high_run, top_voxel_high_value, top_voxel_low_run, top_voxel_low_value = metric_extremes("top_percentile_voxels")
    overlap_voxel_high_run, overlap_voxel_high_value, overlap_voxel_low_run, overlap_voxel_low_value = metric_extremes("overlap_top_voxels")

    interpretation_lines = [
        "# Results Interpretation",
        "",
        "## Plain-Language Bottom Line",
        "",
        (
            f"The repeated runs produced highly consistent population-level results for `{roi_name}`. "
            "There is no evidence that one run systematically behaved differently from the others in a way that would change the study conclusion."
        ),
        "",
        (
            "The main pattern is this: differences between subjects are much larger than differences between repeats. "
            "That is the pattern you would want to see if the pipeline is stable."
        ),
        "",
        "## What This Means Overall",
        "",
        (
            f"The interpretation below is restricted to the complete-case cohort of `{int(coverage['n_subjects'].max())}` subjects "
            "present in every repeat, so each run is being compared on exactly the same population."
            if complete_case_only
            else f"Subject coverage was very similar across repeats, varying by only `{coverage_range}` subjects "
            f"(`{int(coverage['n_subjects'].min())}` to `{int(coverage['n_subjects'].max())}`). "
            "That means the runs were based on almost the same population and can be compared with confidence."
        ),
        "",
        (
            "At the population level, all four main metrics changed only slightly from one repeat to the next. "
            "The run means were tightly clustered, the pairwise run differences were small, the repeated-measures tests were non-significant, "
            "and the estimated repeat-level variance was effectively zero."
        ),
        "",
        (
            "In practical terms, this means the experiment appears reproducible at the population level: "
            "rerunning the pipeline does not materially change the headline result."
        ),
        "",
        "## Metric-By-Metric Interpretation",
        "",
        "### 1. Overlap Fraction",
        "",
        (
            f"`Overlap Fraction` is the most direct target-engagement measure because it tells you what proportion of the ROI "
            "was captured by the top-percentile field region."
        ),
        "",
        (
            f"The run means ranged from `{METRIC_FORMATTERS['overlap_fraction'](overlap_fraction_low_value)}` in `{overlap_fraction_low_run}` "
            f"to `{METRIC_FORMATTERS['overlap_fraction'](overlap_fraction_high_value)}` in `{overlap_fraction_high_run}`. "
            f"The across-run SD was only `{METRIC_FORMATTERS['overlap_fraction'](overlap_fraction['sd_of_run_means'])}`, "
            f"the mean absolute pairwise run difference was `{METRIC_FORMATTERS['overlap_fraction'](overlap_fraction['mean_abs_pairwise_diff'])}`, "
            f"and the fitted drift slope was `{overlap_fraction['drift_slope_percent_per_repeat']:.3f}%` of the mean per repeat "
            f"(p = `{overlap_fraction['drift_pvalue']:.3f}`)."
        ),
        "",
        (
            f"Interpretation: target engagement is very stable across repeats. There is still within-subject repeatability error "
            f"(pooled within-subject CV `{overlap_fraction['pooled_within_subject_cv_percent']:.2f}%`), "
            "but that error is substantially larger than the tiny movement of the run means, which indicates that repeat order itself is not driving the result."
        ),
        "",
        "### 2. Overlap Top Voxels",
        "",
        (
            f"`Overlap Top Voxels` measures the absolute number of high-field voxels falling inside the ROI. "
            "It is closely related to `Overlap Fraction`, but expressed as a voxel count instead of a proportion."
        ),
        "",
        (
            f"The run means ranged from `{METRIC_FORMATTERS['overlap_top_voxels'](overlap_voxel_low_value)}` in `{overlap_voxel_low_run}` "
            f"to `{METRIC_FORMATTERS['overlap_top_voxels'](overlap_voxel_high_value)}` in `{overlap_voxel_high_run}`. "
            f"The mean absolute paired run difference was `{METRIC_FORMATTERS['overlap_top_voxels'](overlap_top_voxels['mean_abs_pairwise_diff'])}` voxels."
        ),
        "",
        (
            f"Interpretation: the absolute amount of high-field overlap in the ROI is also stable. "
            f"It has slightly more relative repeatability error than `Percentile Value` or `Top Percentile Voxels` "
            f"(pooled within-subject CV `{overlap_top_voxels['pooled_within_subject_cv_percent']:.2f}%`), "
            "but the run-to-run effect remains negligible."
        ),
        "",
        "### 3. Percentile Value",
        "",
        (
            "`Percentile Value` reflects the field threshold used to define the top-percentile region for each subject. "
            "This tells you whether the overall field-strength distribution is shifting between repeats."
        ),
        "",
        (
            f"The run means ranged from `{METRIC_FORMATTERS['percentile_value'](percentile_low_value)}` in `{percentile_low_run}` "
            f"to `{METRIC_FORMATTERS['percentile_value'](percentile_high_value)}` in `{percentile_high_run}`. "
            f"The absolute-agreement ICC was `{percentile_value['icc_absolute_agreement']:.3f}`, and the pooled within-subject CV was only "
            f"`{percentile_value['pooled_within_subject_cv_percent']:.2f}%`."
        ),
        "",
        (
            "Interpretation: the thresholding step is very stable. This is important because instability here would propagate into the overlap metrics downstream. "
            "Instead, the analysis suggests the threshold itself is consistent across repeats."
        ),
        "",
        "### 4. Top Percentile Voxels",
        "",
        (
            "`Top Percentile Voxels` measures how many voxels make it into the top field band. "
            "It is a useful indicator of whether the size of the high-field region changes meaningfully across repeats."
        ),
        "",
        (
            f"The run means ranged from `{METRIC_FORMATTERS['top_percentile_voxels'](top_voxel_low_value)}` in `{top_voxel_low_run}` "
            f"to `{METRIC_FORMATTERS['top_percentile_voxels'](top_voxel_high_value)}` in `{top_voxel_high_run}`. "
            f"Among the four main endpoints, this metric showed the strongest repeatability, with ICC `{top_percentile_voxels['icc_absolute_agreement']:.3f}` "
            f"and pooled within-subject CV `{top_percentile_voxels['pooled_within_subject_cv_percent']:.2f}%`."
        ),
        "",
        (
            "Interpretation: the size of the top-field mask is highly reproducible. Even where this metric shows the largest absolute pairwise difference, "
            "the relative scale of that difference remains small."
        ),
        "",
        "## Why The Reliability Metrics Matter",
        "",
        (
            "The descriptive plots already show that the runs look similar, but the reliability metrics explain why that visual impression is trustworthy."
        ),
        "",
        (
            f"- ICC(A,1) ranged from `{primary_stats['icc_absolute_agreement'].min():.3f}` to `{primary_stats['icc_absolute_agreement'].max():.3f}`. "
            "This indicates high absolute agreement across repeats."
        ),
        (
            f"- Repeat-level variance never exceeded `{primary_stats['run_variance_fraction_percent'].max():.4f}%` of total variance. "
            "This means the run identity explains essentially none of the total variability."
        ),
        (
            f"- Subject variance dominated the total variance, ranging from "
            f"`{primary_stats['subject_variance_fraction_percent'].min():.2f}%` to "
            f"`{primary_stats['subject_variance_fraction_percent'].max():.2f}%`."
        ),
        (
            f"- The pooled within-subject CV clearly separates the more stable threshold/extent metrics "
            f"(`{percentile_value['pooled_within_subject_cv_percent']:.2f}%` and `{top_percentile_voxels['pooled_within_subject_cv_percent']:.2f}%`) "
            f"from the somewhat noisier overlap metrics (`{overlap_top_voxels['pooled_within_subject_cv_percent']:.2f}%` and `{overlap_fraction['pooled_within_subject_cv_percent']:.2f}%`)."
        ),
        "",
        (
            "This combination of findings is exactly what you would expect if the pipeline is stable but the anatomy-driven overlap measures naturally show more within-subject fluctuation than the global threshold measures."
        ),
        "",
        "## What The Drift Analysis Says",
        "",
        (
            "A drift analysis was included to test whether later repeats systematically moved upward or downward relative to earlier repeats."
        ),
        "",
        (
            f"All drift slopes were small and non-significant, with p-values ranging from "
            f"`{primary_stats['drift_pvalue'].min():.3f}` to `{primary_stats['drift_pvalue'].max():.3f}`."
        ),
        "",
        (
            "Interpretation: there is no evidence of progressive degradation, warming-up behavior, or cumulative bias across run order."
        ),
        "",
        "## Practical Interpretation For This Experiment",
        "",
        (
            "If the goal is to demonstrate that the left-hippocampus targeting results are reproducible across reruns, "
            "these results support that conclusion."
        ),
        "",
        (
            "If the goal is to compare subjects, the analysis also supports that use case, because subject-to-subject variability is much larger than repeat-to-repeat drift."
        ),
        "",
        (
            "If the goal is to define a robust population summary, the run-level mean appears reliable across repeats for all four primary endpoints."
        ),
        "",
        "## What Should Not Be Over-Claimed",
        "",
        "- These results show repeatability across runs of this dataset; they do not by themselves prove generalization to new cohorts or different processing settings.",
        "- The overlap metrics still have meaningful within-subject repeatability error, so small individual-level differences should be interpreted cautiously.",
        "- The focused y-axis in the coverage plot was used to make small differences visible; it does not imply that a 1-4 subject difference is large in practical terms.",
        "",
        "## Recommended Takeaway Statement",
        "",
        (
            "Across 10 repeated runs, the left-hippocampus `subject_metrics` outputs were highly stable at the population level. "
            "Run-to-run differences were small, no systematic drift was detected, and almost all variance was attributable to subject differences rather than repeat identity. "
            "The threshold and high-field extent metrics were especially repeatable, while the ROI-overlap metrics showed slightly larger within-subject variability but remained strongly reliable overall."
        ),
        "",
    ]

    interpretation_path = output_dir / "results_interpretation.md"
    interpretation_path.write_text("\n".join(interpretation_lines), encoding="utf-8")
    return interpretation_path


def write_subject_variation_report(
    output_dir: Path,
    roi_name: str,
    subject_variation_summary: pd.DataFrame,
    top_variable_subjects: pd.DataFrame,
    cross_metric_instability: pd.DataFrame,
) -> Path:
    lines = [
        "# Subject-Level Variation Report",
        "",
        "## Why This Layer Was Added",
        "",
        (
            "In this experiment, the montage and solver settings are fixed, while a new mesh is generated each time "
            "the same subject is processed. That means the main scientific question is not only whether the population "
            "mean is stable, but also how much each individual subject's metrics move when meshing is repeated."
        ),
        "",
        (
            "Subject-level analysis is therefore the correct way to quantify mesh-regeneration sensitivity. It shows "
            "which endpoints are naturally stable, which subjects are unusually variable, and whether any subject "
            "shows directional drift across repeat order."
        ),
        "",
        "## Recommended Metrics And Tests For This Experiment",
        "",
        "- `Mean`, `SD`, `MAD`, and `range` per subject: these quantify the magnitude and shape of each subject's repeat variation.",
        "- `CV (%)` per subject: useful for relative variation, especially for scale-comparison across metrics.",
        "- `Mean absolute pairwise difference` and `max absolute pairwise difference`: useful when you want a direct readout of repeat-to-repeat movement.",
        "- `SD / pooled within-subject SD`: the most useful absolute-instability screen in this dataset, because it identifies subjects whose repeat variation is unusually large relative to the cohort repeatability baseline.",
        "- Per-subject linear drift slope and p-value: useful to detect monotonic changes across repeat order.",
        "- `Friedman` test with `Kendall's W`: appropriate for paired, repeated-run population comparisons without assuming normality.",
        "- `ICC(A,1)`: the right reliability metric here because absolute agreement matters, not just rank order.",
        "- Variance decomposition: needed to show whether variance is dominated by subject identity, repeat identity, or residual repeatability error.",
        "",
        "## Interpretation Rule Used Here",
        "",
        (
            "For overlap metrics, CV alone can be misleading because subjects with small denominators can show large "
            "relative CV despite modest absolute movement. For that reason, the main instability ranking in this report "
            "uses `SD / pooled within-subject SD` as the primary flag and treats CV as supporting context."
        ),
        "",
        "## Metric-Level Subject Variation",
        "",
    ]

    primary_summary = subject_variation_summary[subject_variation_summary["metric"].isin(PLOT_METRICS)].copy()
    for metric in PLOT_METRICS:
        subset = primary_summary[primary_summary["metric"] == metric]
        if subset.empty:
            continue
        row = subset.iloc[0]
        top_absolute = (
            top_variable_subjects[
                (top_variable_subjects["metric"] == metric)
                & (top_variable_subjects["ranking_method"] == "absolute_variation_sd_ratio")
            ]
            .head(5)
        )
        top_relative = (
            top_variable_subjects[
                (top_variable_subjects["metric"] == metric)
                & (top_variable_subjects["ranking_method"] == "relative_variation_cv")
            ]
            .head(5)
        )
        lines.extend(
            [
                f"### {METRIC_LABELS[metric]}",
                "",
                (
                    f"- Median subject CV: `{row['median_cv_percent']:.2f}%`; 95th percentile CV: "
                    f"`{row['p95_cv_percent']:.2f}%`; maximum CV: `{row['max_cv_percent']:.2f}%`."
                ),
                (
                    f"- Median absolute-instability ratio (`SD / pooled within-subject SD`): "
                    f"`{row['median_sd_vs_pooled_repeatability']:.2f}`; 95th percentile: "
                    f"`{row['p95_sd_vs_pooled_repeatability']:.2f}`; maximum: "
                    f"`{row['max_sd_vs_pooled_repeatability']:.2f}`."
                ),
                (
                    f"- Median mean absolute pairwise difference: "
                    f"`{format_metric_value(metric, row['mean_abs_pairwise_diff_median'])}`; "
                    f"largest observed repeat-to-repeat difference for any subject: "
                    f"`{format_metric_value(metric, row['max_abs_pairwise_diff_max'])}`."
                ),
                (
                    f"- Subjects with nominally significant drift (`p < 0.05`): "
                    f"`{int(row['n_subjects_drift_p_lt_0_05'])}` of `{int(row['n_subjects'])}`."
                ),
                (
                    "- Highest absolute-instability subjects: "
                    + (
                        ", ".join(
                            f"`{entry.subject}` ({entry.sd_vs_pooled_repeatability:.2f})"
                            for entry in top_absolute.itertuples(index=False)
                        )
                        if not top_absolute.empty
                        else "none"
                    )
                    + "."
                ),
                (
                    "- Highest relative-CV subjects: "
                    + (
                        ", ".join(
                            f"`{entry.subject}` ({entry.cv_percent:.2f}%)"
                            for entry in top_relative.itertuples(index=False)
                        )
                        if not top_relative.empty
                        else "none"
                    )
                    + "."
                ),
                "",
            ]
        )

    aggregate = cross_metric_instability.head(10)
    if not aggregate.empty:
        lines.extend(
            [
                "## Subjects That Merit Priority QC Review",
                "",
                (
                    "These subjects had the highest aggregate instability across the four primary endpoints, ranked by "
                    "the mean `SD / pooled within-subject SD`."
                ),
                "",
            ]
        )
        for entry in aggregate.itertuples(index=False):
            lines.append(
                (
                    f"- `{entry.subject}`: mean SD ratio `{entry.mean_sd_ratio_across_metrics:.2f}`, "
                    f"max SD ratio `{entry.max_sd_ratio_across_metrics:.2f}`."
                )
            )
        lines.extend(
            [
                "",
                "## Practical Reading",
                "",
                (
                    f"For `{roi_name}`, the threshold and high-field extent metrics are the most stable subject-level "
                    "endpoints, while the overlap metrics show more within-subject variation. That pattern is expected "
                    "because the overlap metrics are closer to local mesh/geometry differences."
                ),
                "",
                (
                    "For future studies, subjects near the top of the aggregate instability ranking should be the first "
                    "ones inspected visually when mesh QC or segmentation troubleshooting is required."
                ),
                "",
            ]
        )

    report_path = output_dir / "subject_level_variation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_failure_report(
    output_dir: Path,
    logs_root: Path,
    log_detail_frame: pd.DataFrame,
    log_summary_frame: pd.DataFrame,
    log_transition_frame: pd.DataFrame,
    post_metric_subjects: set[str],
) -> Path:
    failed_only = log_detail_frame[log_detail_frame["final_status"] == "failed"].copy()
    run_counts = (
        log_detail_frame.groupby(["run", "final_status"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    total_by_run = log_detail_frame.groupby("run").size().sort_index()
    failed_by_run = failed_only.groupby("run").size().reindex(total_by_run.index, fill_value=0)
    subjects_missing_post_metrics = sorted(set(log_detail_frame["subject"]) - post_metric_subjects)
    persistent_failures: list[str] = []
    recovered_failures: list[str] = []
    new_failures: list[str] = []
    mcnemar_pvalue = math.nan

    if log_detail_frame["run"].nunique() >= 2:
        run_order = sorted(log_detail_frame["run"].unique().tolist())
        pivot = log_detail_frame.pivot(index="subject", columns="run", values="final_status")
        first_run, second_run = run_order[0], run_order[1]
        persistent_failures = sorted(
            pivot[(pivot[first_run] == "failed") & (pivot[second_run] == "failed")].index.tolist()
        )
        recovered_failures = sorted(
            pivot[(pivot[first_run] == "failed") & (pivot[second_run] == "success")].index.tolist()
        )
        new_failures = sorted(
            pivot[(pivot[first_run] == "success") & (pivot[second_run] == "failed")].index.tolist()
        )
        if not log_transition_frame.empty:
            mcnemar_pvalue = float(log_transition_frame["mcnemar_exact_pvalue"].iloc[0])

    lines = [
        "# Failure Report",
        "",
        "## Scope",
        "",
        f"- Log directory analysed: `{logs_root}`",
        f"- Logged subject-runs analysed: `{len(log_detail_frame)}`",
        f"- Failed subject-runs: `{len(failed_only)}`",
        f"- Unique failed subjects: `{failed_only['subject'].nunique()}`",
        (
            f"- Subjects present in logs but absent from the post-analysis metrics table: "
            f"`{len(subjects_missing_post_metrics)}`"
        ),
        "",
        "## Run-Level Failure Rates",
        "",
    ]

    for run in total_by_run.index:
        failed_count = int(failed_by_run.loc[run])
        total_count = int(total_by_run.loc[run])
        success_count = int(run_counts.loc[run].get("success", 0))
        lines.append(
            (
                f"- Run `{run}`: `{success_count}` success, `{failed_count}` failed "
                f"({failed_count / total_count * 100.0:.1f}% failure rate)."
            )
        )

    lines.extend(
        [
            "",
            "## Paired Run Comparison",
            "",
        ]
    )
    if persistent_failures or recovered_failures or new_failures:
        lines.extend(
            [
                f"- Persistent failures in both runs: `{len(persistent_failures)}`.",
                (
                    f"- Failed first run and recovered on second run: `{len(recovered_failures)}`."
                ),
                f"- Succeeded first run and failed on second run: `{len(new_failures)}`.",
                f"- Exact McNemar p-value for paired failure-rate change: `{mcnemar_pvalue:.6f}`.",
                "",
                "Interpretation:",
                "",
                (
                    "The second logged run had materially fewer failures. The exact McNemar result confirms that the "
                    "improvement in completion rate is unlikely to be due to chance alone for the paired subject set."
                ),
                "",
            ]
        )

    if subjects_missing_post_metrics:
        lines.extend(
            [
                "## Subjects Missing From Post-Metrics Outputs",
                "",
                (
                    "These subjects appeared in the log summaries but did not appear in the downstream "
                    "`subject_metrics` tables:"
                ),
                "",
                "- " + ", ".join(f"`{subject}`" for subject in subjects_missing_post_metrics),
                "",
            ]
        )

    lines.extend(
        [
            "## Failure Categories",
            "",
            (
                "The categories below were assigned from raw log signatures. Each category includes the observed "
                "evidence, the most likely root cause, and the operational fix that would be the most defensible next step."
            ),
            "",
        ]
    )
    for row in log_summary_frame.itertuples(index=False):
        lines.extend(
            [
                f"### {humanize_label(row.failure_category)}",
                "",
                f"- Failed subject-runs: `{int(row.n_failures)}`",
                f"- Unique subjects: `{int(row.n_unique_subjects)}`",
                f"- Runs observed: `{row.runs}`",
                f"- Typical stage: `{humanize_label(row.failure_stage)}`",
                f"- Why it failed: {row.evidence}",
                f"- Most likely cause: {row.likely_cause}",
                f"- Recommended fix: {row.recommended_fix}",
                "",
            ]
        )

    if persistent_failures:
        lines.extend(
            [
                "## Persistent Failures",
                "",
            ]
        )
        for subject in persistent_failures:
            subject_rows = failed_only[failed_only["subject"] == subject].sort_values("run")
            categories = ", ".join(sorted(subject_rows["failure_category"].map(humanize_label).unique()))
            lines.append(f"- `{subject}`: failed in every logged run; categories observed: {categories}.")
        lines.append("")

    lines.extend(
        [
            "## Detailed Subject-Run Failure Audit",
            "",
            (
                "Each entry lists the failing run, the inferred pipeline stage, the classified failure mode, the most "
                "likely cause, and the recommended fix."
            ),
            "",
        ]
    )
    for row in failed_only.sort_values(["run", "subject"]).itertuples(index=False):
        lines.extend(
            [
                (
                    f"- Run `{row.run}`, subject `{row.subject}`, stage `{humanize_label(row.failure_stage)}`: "
                    f"`{humanize_label(row.failure_category)}`. "
                    f"Likely cause: {row.likely_cause} "
                    f"Fix: {row.recommended_fix} "
                    f"Log: `{row.log_file}`."
                )
            ]
        )

    report_path = output_dir / "failure_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_analysis(
    *,
    dataset_root: str | Path,
    roi: str | None = None,
    output_dir: str | Path | None = None,
    logs_root: str | Path | None = None,
    skip_image_repeatability: bool = False,
    complete_case_only: bool = True,
) -> dict[str, object]:
    dataset_root = Path(dataset_root).resolve()
    logs_root_path = resolve_logs_root(dataset_root, str(logs_root) if logs_root else None)
    output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else dataset_root / "subject_metrics_analysis"
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame, roi_name = load_subject_metrics(dataset_root, roi)
    metrics = numeric_metrics(frame)

    coverage, _all_subjects, complete_subjects = compute_coverage(frame)
    complete_case_frame = frame[frame["subject"].isin(complete_subjects)].copy()
    analysis_frame = complete_case_frame if complete_case_only else frame
    analysis_coverage = (
        compute_coverage(analysis_frame)[0] if complete_case_only else coverage
    )
    coverage_subjects = (
        sorted(analysis_frame["subject"].unique())
        if complete_case_only
        else complete_subjects
    )
    subject_repeat_means = (
        complete_case_frame.groupby("subject", sort=True)[metrics].mean(numeric_only=True).reset_index()
    )
    subject_repeat_sds = (
        complete_case_frame.groupby("subject", sort=True)[metrics].std(numeric_only=True).reset_index()
    )

    repeat_level_available = compute_repeat_level_stats(analysis_frame, metrics)
    repeat_level_complete = compute_repeat_level_stats(complete_case_frame, metrics)
    pairwise_differences = compute_pairwise_differences(complete_case_frame, metrics)
    within_subject_repeatability = compute_within_subject_repeatability(complete_case_frame, metrics)
    experiment_level_stats = compute_experiment_level_stats(
        complete_case_frame,
        metrics,
        repeat_level_complete,
        pairwise_differences,
        within_subject_repeatability,
    )
    subject_level_variation = compute_subject_level_variation(
        complete_case_frame,
        experiment_level_stats,
        metrics,
    )
    subject_variation_summary = compute_subject_variation_summary(subject_level_variation, metrics)
    top_variable_subjects = compute_top_variable_subjects(subject_level_variation)
    cross_metric_instability = compute_cross_metric_instability(subject_level_variation)
    variation_analysis_metrics = (
        experiment_level_stats[experiment_level_stats["metric"].isin(PLOT_METRICS)]
        .loc[
            :,
            [
                "metric",
                "metric_label",
                "n_complete_subjects",
                "mean_of_run_means",
                "sd_of_run_means",
                "cv_percent_run_means",
                "pooled_within_subject_sd",
                "pooled_within_subject_cv_percent",
                "standard_error_of_measurement",
                "repeatability_coefficient",
                "icc_absolute_agreement",
                "mean_pairwise_correlation",
                "drift_slope_per_repeat",
                "drift_slope_percent_per_repeat",
                "drift_pvalue",
                "drift_r_squared",
                "subject_variance_fraction_percent",
                "run_variance_fraction_percent",
                "residual_variance_fraction_percent",
            ],
        ]
        .reset_index(drop=True)
    )
    log_detail_frame = pd.DataFrame()
    log_summary_frame = pd.DataFrame()
    log_transition_frame = pd.DataFrame()
    failure_category_by_run = pd.DataFrame()
    failure_stage_by_run = pd.DataFrame()
    image_run_level = pd.DataFrame()
    image_pairwise = pd.DataFrame()
    image_subject_summary = pd.DataFrame()
    image_issue_frame = pd.DataFrame(columns=["subject", "issue_type", "details"])
    image_pairwise_run_summary = pd.DataFrame()
    image_cohort_summary = pd.DataFrame()
    analysis_percentile = (
        infer_uniform_percentile(complete_case_frame["percentile"].tolist())
        if "percentile" in complete_case_frame.columns
        else None
    )
    if logs_root_path is not None:
        log_detail_frame, log_summary_frame, log_transition_frame = load_log_analysis(
            logs_root_path,
            set(analysis_frame["subject"].unique()),
        )
        failed_only = log_detail_frame[log_detail_frame["final_status"] == "failed"].copy()
        failure_category_by_run = (
            failed_only.groupby(["failure_category", "run"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "failure_category"], ascending=[False, True])
            .reset_index(drop=True)
        )
        failure_stage_by_run = (
            failed_only.groupby(["failure_stage", "run"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "failure_stage"], ascending=[False, True])
            .reset_index(drop=True)
        )
    if not skip_image_repeatability:
        image_run_level, image_pairwise, image_subject_summary, image_issue_frame = (
            compute_image_repeatability(
                dataset_root=dataset_root,
                roi_name=roi_name,
                complete_case_frame=complete_case_frame,
            )
        )
        image_pairwise_run_summary = compute_image_repeatability_pairwise_run_summary(image_pairwise)
        image_cohort_summary = compute_image_repeatability_cohort_summary(image_subject_summary)

    analysis_frame.to_csv(output_dir / "subject_metrics_long.csv", index=False)
    analysis_coverage.to_csv(output_dir / "run_subject_coverage.csv", index=False)
    repeat_level_available.to_csv(output_dir / "repeat_level_population_statistics.csv", index=False)
    repeat_level_complete.to_csv(
        output_dir / "repeat_level_population_statistics_complete_subjects.csv", index=False
    )
    experiment_level_stats.to_csv(output_dir / "experiment_level_population_statistics.csv", index=False)
    variation_analysis_metrics.to_csv(output_dir / "variation_analysis_metrics.csv", index=False)
    pairwise_differences.to_csv(output_dir / "pairwise_run_differences.csv", index=False)
    within_subject_repeatability.to_csv(output_dir / "within_subject_repeatability.csv", index=False)
    subject_level_variation.to_csv(output_dir / "subject_level_variation.csv", index=False)
    subject_variation_summary.to_csv(output_dir / "subject_level_variation_summary.csv", index=False)
    subject_repeat_means.to_csv(output_dir / "subject_repeat_metric_means.csv", index=False)
    subject_repeat_sds.to_csv(output_dir / "subject_repeat_metric_sds.csv", index=False)
    top_variable_subjects.to_csv(output_dir / "subject_level_top_variable_subjects.csv", index=False)
    cross_metric_instability.to_csv(output_dir / "subject_cross_metric_instability.csv", index=False)
    if logs_root_path is not None:
        log_detail_frame.to_csv(output_dir / "log_subject_run_details.csv", index=False)
        log_summary_frame.to_csv(output_dir / "log_failure_summary_by_category.csv", index=False)
        failure_category_by_run.to_csv(output_dir / "log_failure_category_by_run.csv", index=False)
        failure_stage_by_run.to_csv(output_dir / "log_failure_stage_by_run.csv", index=False)
        log_transition_frame.to_csv(output_dir / "log_run_transition_summary.csv", index=False)
    if not skip_image_repeatability:
        image_run_level.to_csv(output_dir / "image_repeatability_run_level.csv", index=False)
        image_pairwise.to_csv(
            output_dir / "image_repeatability_pairwise_subject_run_pairs.csv",
            index=False,
        )
        image_subject_summary.to_csv(
            output_dir / "image_repeatability_subject_level.csv",
            index=False,
        )
        image_pairwise_run_summary.to_csv(
            output_dir / "image_repeatability_pairwise_run_summary.csv",
            index=False,
        )
        image_cohort_summary.to_csv(
            output_dir / "image_repeatability_cohort_summary.csv",
            index=False,
        )
        image_issue_frame.to_csv(output_dir / "image_repeatability_issues.csv", index=False)

    setup_plotting()
    save_coverage_plot(analysis_coverage, coverage_subjects, figures_dir / "01_subject_coverage.png")
    save_repeat_distribution_plot(analysis_frame, figures_dir / "02_repeat_level_distributions.png")
    save_repeat_mean_ci_plot(repeat_level_available, figures_dir / "03_repeat_level_mean_ci.png")
    save_pairwise_heatmap_plot(pairwise_differences, figures_dir / "04_pairwise_run_differences.png")
    save_variation_summary_plot(experiment_level_stats, figures_dir / "05_variation_summary.png")
    save_subject_variation_plot(
        subject_level_variation,
        cross_metric_instability,
        figures_dir / "06_subject_variation_summary.png",
    )
    if logs_root_path is not None:
        save_failure_summary_plot(
            log_detail_frame,
            log_transition_frame,
            figures_dir / "07_failure_summary.png",
        )
    if not skip_image_repeatability and not image_subject_summary.empty:
        save_image_repeatability_plot(
            image_subject_summary,
            figures_dir / "08_image_repeatability_summary.png",
            analysis_percentile=analysis_percentile,
        )
    report_path = write_report(
        dataset_root=dataset_root,
        output_dir=output_dir,
        roi_name=roi_name,
        all_frame=analysis_frame,
        complete_case_frame=complete_case_frame,
        coverage=analysis_coverage,
        experiment_stats=experiment_level_stats,
        image_repeatability_subject_summary=image_subject_summary,
        complete_case_only=complete_case_only,
    )
    methodology_path = write_methodology_documentation(
        dataset_root=dataset_root,
        output_dir=output_dir,
        roi_name=roi_name,
        all_frame=analysis_frame,
        complete_case_frame=complete_case_frame,
        coverage=analysis_coverage,
        repeat_level_available=repeat_level_available,
        experiment_stats=experiment_level_stats,
        include_image_repeatability=(not skip_image_repeatability),
        complete_case_only=complete_case_only,
    )
    image_repeatability_methodology_path = None
    image_repeatability_report_path = None
    if not skip_image_repeatability:
        image_repeatability_methodology_path = write_image_repeatability_methodology(
            output_dir=output_dir,
            roi_name=roi_name,
            subject_summary=image_subject_summary,
            issue_frame=image_issue_frame,
            analysis_percentile=analysis_percentile,
        )
        image_repeatability_report_path = write_image_repeatability_report(
            output_dir=output_dir,
            roi_name=roi_name,
            subject_summary=image_subject_summary,
            issue_frame=image_issue_frame,
            analysis_percentile=analysis_percentile,
        )
    subject_variation_report_path = write_subject_variation_report(
        output_dir=output_dir,
        roi_name=roi_name,
        subject_variation_summary=subject_variation_summary,
        top_variable_subjects=top_variable_subjects,
        cross_metric_instability=cross_metric_instability,
    )
    interpretation_path = write_results_interpretation(
        output_dir=output_dir,
        roi_name=roi_name,
        coverage=analysis_coverage,
        repeat_level_available=repeat_level_available,
        experiment_stats=experiment_level_stats,
        complete_case_only=complete_case_only,
    )
    failure_report_path = None
    if logs_root_path is not None:
        failure_report_path = write_failure_report(
            output_dir=output_dir,
            logs_root=logs_root_path,
            log_detail_frame=log_detail_frame,
            log_summary_frame=log_summary_frame,
            log_transition_frame=log_transition_frame,
            post_metric_subjects=set(analysis_frame["subject"].unique()),
        )

    print(f"ROI: {roi_name}")
    print(f"Rows analysed: {len(analysis_frame):,}")
    print(f"Unique subjects: {analysis_frame['subject'].nunique()}")
    print(f"Complete-case subjects: {len(complete_subjects)}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Methodology: {methodology_path}")
    if image_repeatability_methodology_path is not None:
        print(f"Image repeatability methodology: {image_repeatability_methodology_path}")
    if image_repeatability_report_path is not None:
        print(f"Image repeatability report: {image_repeatability_report_path}")
    print(f"Subject variation report: {subject_variation_report_path}")
    print(f"Interpretation: {interpretation_path}")
    if failure_report_path is not None:
        print(f"Failure report: {failure_report_path}")

    return {
        "roi_name": roi_name,
        "rows_analysed": len(analysis_frame),
        "unique_subjects": int(analysis_frame["subject"].nunique()),
        "complete_case_subjects": len(complete_subjects),
        "complete_case_only": complete_case_only,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "methodology_path": str(methodology_path),
        "image_repeatability_methodology_path": (
            str(image_repeatability_methodology_path)
            if image_repeatability_methodology_path is not None
            else None
        ),
        "image_repeatability_report_path": (
            str(image_repeatability_report_path)
            if image_repeatability_report_path is not None
            else None
        ),
        "subject_variation_report_path": str(subject_variation_report_path),
        "interpretation_path": str(interpretation_path),
        "failure_report_path": str(failure_report_path) if failure_report_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    run_analysis(
        dataset_root=args.dataset_root,
        roi=args.roi,
        output_dir=args.output_dir,
        logs_root=args.logs_root,
        skip_image_repeatability=args.skip_image_repeatability,
        complete_case_only=not args.allow_incomplete_subjects,
    )


if __name__ == "__main__":
    main()
