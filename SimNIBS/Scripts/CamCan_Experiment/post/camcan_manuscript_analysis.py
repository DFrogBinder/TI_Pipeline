"""Optimizer-matched CamCan manuscript metrics and ten-repeat aggregation.

This module is intentionally independent of the visualization-heavy legacy
post-processing pass.  It reads the existing whole-brain TI NIfTI and the
subject-space atlas, reconstructs the parcel-clipped spherical target used by
the optimizer, writes one small resumable JSON record per simulation, and
builds publication-ready tables and figures from arithmetic means across the
ten remeshing repeats. Full anatomical-parcel metrics are retained with an
``anatomical_`` prefix as a secondary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.post_functions import roi_masks_on_ti_grid  # noqa: E402
from post.optimizer_target_roi import (  # noqa: E402
    RADIUS_CAP_MM,
    RADIUS_STEP_MM,
    ROI_DEFINITION_SCHEMA_VERSION,
    START_RADIUS_MM,
    TARGET_VOLUME_MM3_BY_ROI,
    build_optimizer_target_roi,
    flatten_roi_metadata,
)
from utils.roi_registry import (  # noqa: E402
    match_fastsurfer_roi_from_directory,
    resolve_fastsurfer_roi_label_ids,
)
from utils.ti_utils import load_ti_as_scalar, vol_mm3  # noqa: E402


ANALYSIS_SCHEMA_VERSION = 3
METRIC_MARKER_FILENAME = "optimizer_matched_metrics.json"
DEFAULT_THRESHOLDS_V_PER_M = (0.20, 0.18, 0.15)
DEFAULT_TOP_PERCENTILE = 95.0
DEFAULT_ROBUST_MAX_PERCENTILE = 99.9
DEFAULT_UPPER_TAIL_FRACTION = 0.01
ROI_ORDER = (
    "Left_Hippocampus",
    "Left_M1",
    "Right_DLPC",
    "Right_Thalamus",
)
MNI_BASELINE_NAMES = {
    "Left_Hippocampus": "MNI152-left-hippocampus",
    "Left_M1": "MNI152-left-m1",
    "Right_DLPC": "MNI152-right-dlpc",
    "Right_Thalamus": "MNI152-right-thalamus",
}


def _threshold_slug(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator / denominator * 100.0)


def _localization_percent(numerator: int, denominator: int) -> float:
    """Return 0% when no whole-brain voxels satisfy the threshold.

    In that case there is no suprathreshold stimulation to localize in the
    target. Encoding the result as zero keeps all ten repeats in the arithmetic
    mean instead of silently dropping the repeat as an undefined 0/0 ratio.
    """

    if denominator <= 0:
        return 0.0
    return float(numerator / denominator * 100.0)


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, percentile))


def _upper_tail_median(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return math.nan
    cutoff = float(np.percentile(values, (1.0 - fraction) * 100.0))
    tail = values[values >= cutoff]
    return float(np.median(tail)) if tail.size else math.nan


def _metric_depends_on_roi_scope(name: str) -> bool:
    """Return whether changing target ROI changes the metric's value."""

    if name.startswith("whole_brain_"):
        return False
    if name in {
        "voxel_volume_mm3",
        "top_5_percent_threshold_v_per_m",
        "top_5_percent_voxels",
        "top_5_percent_whole_brain_volume_mm3",
    }:
        return False
    return True


def manuscript_metric_names(
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS_V_PER_M,
    *,
    include_anatomical_secondary: bool = True,
) -> list[str]:
    names = [
        "roi_min_v_per_m",
        "roi_median_v_per_m",
        "roi_robust_max_p99_9_v_per_m",
        "roi_upper_1_percent_median_v_per_m",
        "whole_brain_median_v_per_m",
        "whole_brain_robust_max_p99_9_v_per_m",
        "whole_brain_upper_1_percent_median_v_per_m",
        "off_target_median_v_per_m",
        "off_target_robust_max_p99_9_v_per_m",
        "off_target_upper_1_percent_median_v_per_m",
        "top_5_percent_threshold_v_per_m",
        "top_5_percent_whole_brain_volume_mm3",
        "top_5_percent_target_volume_mm3",
        "top_5_percent_target_coverage_percent",
        "top_5_percent_localization_percent_in_roi",
    ]
    for threshold in thresholds:
        slug = _threshold_slug(threshold)
        names.extend(
            [
                f"target_coverage_voxels_ge_{slug}",
                f"target_coverage_volume_mm3_ge_{slug}",
                f"target_coverage_percent_ge_{slug}",
                f"whole_brain_coverage_voxels_ge_{slug}",
                f"whole_brain_coverage_volume_mm3_ge_{slug}",
                f"whole_brain_coverage_percent_ge_{slug}",
                f"off_target_coverage_voxels_ge_{slug}",
                f"off_target_coverage_volume_mm3_ge_{slug}",
                f"off_target_coverage_percent_ge_{slug}",
                f"threshold_localization_percent_in_roi_ge_{slug}",
            ]
        )
    if include_anatomical_secondary:
        names.extend(
            f"anatomical_{name}"
            for name in tuple(names)
            if _metric_depends_on_roi_scope(name)
        )
    return names


def compute_manuscript_metrics(
    *,
    ti_img: nib.spatialimages.SpatialImage,
    ti_data: np.ndarray,
    roi_mask: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS_V_PER_M,
    top_percentile: float = DEFAULT_TOP_PERCENTILE,
    robust_max_percentile: float = DEFAULT_ROBUST_MAX_PERCENTILE,
    upper_tail_fraction: float = DEFAULT_UPPER_TAIL_FRACTION,
) -> dict[str, float | int]:
    """Compute effectiveness, spread, localization, and robust intensity metrics.

    The supplied ROI is the denominator for target coverage. ROI voxels with
    non-finite TI values remain in the denominator and count as unstimulated.
    """

    if ti_data.shape != roi_mask.shape:
        raise ValueError(
            f"TI data shape {ti_data.shape} does not match ROI mask shape {roi_mask.shape}."
        )
    if not 0.0 < upper_tail_fraction < 1.0:
        raise ValueError("upper_tail_fraction must be strictly between 0 and 1.")
    if not 0.0 < top_percentile < 100.0:
        raise ValueError("top_percentile must be strictly between 0 and 100.")

    finite = np.isfinite(ti_data)
    if not np.any(finite):
        raise ValueError("TI image contains no finite whole-brain voxels.")
    roi = np.asarray(roi_mask, dtype=bool)
    if not np.any(roi):
        raise ValueError("Target ROI mask is empty.")

    roi_finite = roi & finite
    off_target = (~roi) & finite
    whole_values = np.asarray(ti_data[finite], dtype=np.float64)
    roi_values = np.asarray(ti_data[roi_finite], dtype=np.float64)
    off_target_values = np.asarray(ti_data[off_target], dtype=np.float64)
    voxel_volume = float(vol_mm3(ti_img))

    roi_voxels = int(np.count_nonzero(roi))
    roi_finite_voxels = int(np.count_nonzero(roi_finite))
    whole_brain_voxels = int(np.count_nonzero(finite))
    off_target_voxels = int(np.count_nonzero(off_target))

    metrics: dict[str, float | int] = {
        "roi_voxels": roi_voxels,
        "roi_finite_voxels": roi_finite_voxels,
        "roi_nonfinite_voxels": roi_voxels - roi_finite_voxels,
        "whole_brain_voxels": whole_brain_voxels,
        "off_target_voxels": off_target_voxels,
        "voxel_volume_mm3": voxel_volume,
        "roi_min_v_per_m": float(np.min(roi_values)) if roi_values.size else math.nan,
        "roi_median_v_per_m": float(np.median(roi_values)) if roi_values.size else math.nan,
        "roi_robust_max_p99_9_v_per_m": _safe_percentile(
            roi_values, robust_max_percentile
        ),
        "roi_upper_1_percent_median_v_per_m": _upper_tail_median(
            roi_values, upper_tail_fraction
        ),
        "whole_brain_median_v_per_m": float(np.median(whole_values)),
        "whole_brain_robust_max_p99_9_v_per_m": _safe_percentile(
            whole_values, robust_max_percentile
        ),
        "whole_brain_upper_1_percent_median_v_per_m": _upper_tail_median(
            whole_values, upper_tail_fraction
        ),
        "off_target_median_v_per_m": (
            float(np.median(off_target_values)) if off_target_values.size else math.nan
        ),
        "off_target_robust_max_p99_9_v_per_m": _safe_percentile(
            off_target_values, robust_max_percentile
        ),
        "off_target_upper_1_percent_median_v_per_m": _upper_tail_median(
            off_target_values, upper_tail_fraction
        ),
    }

    top_threshold = float(np.percentile(whole_values, top_percentile))
    top_mask = finite & (ti_data >= top_threshold)
    top_voxels = int(np.count_nonzero(top_mask))
    top_in_roi = int(np.count_nonzero(top_mask & roi))
    top_fraction_label = int(round(100.0 - top_percentile))
    metrics.update(
        {
            f"top_{top_fraction_label}_percent_threshold_v_per_m": top_threshold,
            f"top_{top_fraction_label}_percent_voxels": top_voxels,
            f"top_{top_fraction_label}_percent_target_voxels": top_in_roi,
            f"top_{top_fraction_label}_percent_whole_brain_volume_mm3": (
                top_voxels * voxel_volume
            ),
            f"top_{top_fraction_label}_percent_target_volume_mm3": (
                top_in_roi * voxel_volume
            ),
            f"top_{top_fraction_label}_percent_target_coverage_percent": _safe_percent(
                top_in_roi, roi_voxels
            ),
            f"top_{top_fraction_label}_percent_localization_percent_in_roi": _safe_percent(
                top_in_roi, top_voxels
            ),
        }
    )

    for threshold in thresholds:
        threshold = float(threshold)
        slug = _threshold_slug(threshold)
        above = finite & (ti_data >= threshold)
        target_above = above & roi
        off_target_above = above & (~roi)
        target_count = int(np.count_nonzero(target_above))
        whole_count = int(np.count_nonzero(above))
        off_target_count = int(np.count_nonzero(off_target_above))
        metrics.update(
            {
                f"target_coverage_voxels_ge_{slug}": target_count,
                f"target_coverage_volume_mm3_ge_{slug}": target_count * voxel_volume,
                f"target_coverage_percent_ge_{slug}": _safe_percent(
                    target_count, roi_voxels
                ),
                f"whole_brain_coverage_voxels_ge_{slug}": whole_count,
                f"whole_brain_coverage_volume_mm3_ge_{slug}": whole_count
                * voxel_volume,
                f"whole_brain_coverage_percent_ge_{slug}": _safe_percent(
                    whole_count, whole_brain_voxels
                ),
                f"off_target_coverage_voxels_ge_{slug}": off_target_count,
                f"off_target_coverage_volume_mm3_ge_{slug}": off_target_count
                * voxel_volume,
                f"off_target_coverage_percent_ge_{slug}": _safe_percent(
                    off_target_count, off_target_voxels
                ),
                f"threshold_localization_percent_in_roi_ge_{slug}": _localization_percent(
                    target_count, whole_count
                ),
            }
        )
    return metrics


def compute_optimizer_matched_metric_bundle(
    *,
    ti_img: nib.spatialimages.SpatialImage,
    ti_data: np.ndarray,
    anatomical_roi_mask: np.ndarray,
    roi: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS_V_PER_M,
    top_percentile: float = DEFAULT_TOP_PERCENTILE,
    robust_max_percentile: float = DEFAULT_ROBUST_MAX_PERCENTILE,
    upper_tail_fraction: float = DEFAULT_UPPER_TAIL_FRACTION,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Compute primary optimizer-target and secondary anatomical-parcel metrics."""

    optimizer_roi = build_optimizer_target_roi(
        anatomical_mask=anatomical_roi_mask,
        reference_img=ti_img,
        roi=roi,
    )
    common = {
        "ti_img": ti_img,
        "ti_data": ti_data,
        "thresholds": thresholds,
        "top_percentile": top_percentile,
        "robust_max_percentile": robust_max_percentile,
        "upper_tail_fraction": upper_tail_fraction,
    }
    primary = compute_manuscript_metrics(
        roi_mask=optimizer_roi.mask,
        **common,
    )
    anatomical = compute_manuscript_metrics(
        roi_mask=np.asarray(anatomical_roi_mask, dtype=bool),
        **common,
    )
    metrics = {
        **primary,
        **{
            f"anatomical_{key}": value
            for key, value in anatomical.items()
            if _metric_depends_on_roi_scope(key)
        },
    }
    return metrics, optimizer_roi.metadata


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _analysis_fingerprint(
    *,
    ti_path: Path,
    atlas_path: Path,
    canonical_roi: str,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
) -> str:
    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "ti": _file_identity(ti_path),
        "atlas": _file_identity(atlas_path),
        "canonical_roi": canonical_roi,
        "roi_definition_schema_version": ROI_DEFINITION_SCHEMA_VERSION,
        "roi_definition": {
            "target_volume_mm3": TARGET_VOLUME_MM3_BY_ROI[
                canonical_roi_to_dataset_roi(canonical_roi)
            ],
            "start_radius_mm": START_RADIUS_MM,
            "radius_step_mm": RADIUS_STEP_MM,
            "radius_cap_mm": RADIUS_CAP_MM,
        },
        "thresholds_v_per_m": [float(value) for value in thresholds],
        "top_percentile": float(top_percentile),
        "robust_max_percentile": float(robust_max_percentile),
        "upper_tail_fraction": float(upper_tail_fraction),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_roi_to_dataset_roi(canonical_roi: str) -> str:
    """Resolve a canonical atlas ROI name to the four-ROI dataset label."""

    matches = [
        roi
        for roi in ROI_ORDER
        if match_fastsurfer_roi_from_directory(f"{roi}_Data_01").canonical_name
        == canonical_roi
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Canonical ROI {canonical_roi!r} does not map uniquely to ROI_ORDER."
        )
    return matches[0]


def _resolve_atlas(atlas_root: Path, subject: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = atlas_root / f"{subject}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing subject-space atlas: {atlas_root}/{subject}.nii[.gz]")


def validate_atlas_rois(atlas_path: Path) -> dict[str, Any]:
    """Fail early if an atlas lacks any label required by the four-ROI study."""

    image = nib.load(str(atlas_path))
    present = set(np.unique(np.asanyarray(image.dataobj)).astype(np.int64).tolist())
    roi_labels: dict[str, list[int]] = {}
    missing: dict[str, list[int]] = {}
    for roi in ROI_ORDER:
        canonical = match_fastsurfer_roi_from_directory(
            f"{roi}_Data_01"
        ).canonical_name
        expected = list(resolve_fastsurfer_roi_label_ids(canonical))
        roi_labels[roi] = expected
        absent = [label for label in expected if label not in present]
        if absent:
            missing[roi] = absent
    if missing:
        raise ValueError(
            f"Atlas {atlas_path} lacks required ROI label IDs: {missing}. "
            "Use the Destrieux/aparc.a2009s+aseg atlas used by the CamCan analysis."
        )
    return {
        "status": "complete",
        "atlas": str(atlas_path.resolve()),
        "roi_label_ids": roi_labels,
    }


def _load_existing_complete(path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        isinstance(payload, dict)
        and payload.get("status") == "complete"
        and payload.get("analysis_schema_version") == ANALYSIS_SCHEMA_VERSION
        and payload.get("config_fingerprint") == fingerprint
    ):
        return payload
    return None


@dataclass(frozen=True)
class ExtractionTask:
    dataset_root: str
    subject: str
    roi: str
    repeat: str
    canonical_roi: str
    atlas_root: str
    thresholds: tuple[float, ...]
    top_percentile: float
    robust_max_percentile: float
    upper_tail_fraction: float
    force: bool


def _extract_subject(task: ExtractionTask) -> dict[str, Any]:
    dataset_root = Path(task.dataset_root)
    ti_path = (
        dataset_root
        / task.subject
        / "anat"
        / "SimNIBS"
        / "ti_brain_only.nii.gz"
    )
    if not ti_path.is_file():
        raise FileNotFoundError(f"Missing whole-brain TI field: {ti_path}")
    atlas_path = _resolve_atlas(Path(task.atlas_root), task.subject)
    output_path = (
        dataset_root
        / task.subject
        / "anat"
        / "post"
        / METRIC_MARKER_FILENAME
    )
    fingerprint = _analysis_fingerprint(
        ti_path=ti_path,
        atlas_path=atlas_path,
        canonical_roi=task.canonical_roi,
        thresholds=task.thresholds,
        top_percentile=task.top_percentile,
        robust_max_percentile=task.robust_max_percentile,
        upper_tail_fraction=task.upper_tail_fraction,
    )
    if not task.force:
        existing = _load_existing_complete(output_path, fingerprint)
        if existing is not None:
            return {
                "subject": task.subject,
                "status": "skipped",
                "output": str(output_path),
            }

    ti_img = nib.load(str(ti_path))
    ti_data = load_ti_as_scalar(ti_img)
    roi_masks, _ = roi_masks_on_ti_grid(
        ti_img,
        atlas_mode="fastsurfer",
        subject=task.subject,
        fastsurfer_atlas_path=str(atlas_path),
        roi_names=[task.canonical_roi],
    )
    roi_mask = roi_masks.get(task.canonical_roi)
    if roi_mask is None and len(roi_masks) == 1:
        roi_mask = next(iter(roi_masks.values()))
    if roi_mask is None:
        raise ValueError(
            f"ROI '{task.canonical_roi}' was not returned for {task.subject}."
        )
    metrics, roi_definition = compute_optimizer_matched_metric_bundle(
        ti_img=ti_img,
        ti_data=ti_data,
        anatomical_roi_mask=roi_mask,
        roi=task.roi,
        thresholds=task.thresholds,
        top_percentile=task.top_percentile,
        robust_max_percentile=task.robust_max_percentile,
        upper_tail_fraction=task.upper_tail_fraction,
    )
    payload = _json_ready(
        {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "status": "complete",
            "config_fingerprint": fingerprint,
            "subject": task.subject,
            "roi": task.roi,
            "repeat": task.repeat,
            "canonical_roi": task.canonical_roi,
            "ti_path": str(ti_path.resolve()),
            "atlas_path": str(atlas_path.resolve()),
            "roi_definition": roi_definition,
            "definitions": {
                "thresholds_v_per_m": list(task.thresholds),
                "threshold_comparator": ">=",
                "target_coverage_denominator": (
                    "all optimizer-matched parcel-clipped spherical target voxels; "
                    "non-finite target voxels count as unstimulated"
                ),
                "off_target_coverage_denominator": (
                    "finite whole-brain voxels outside the optimizer-matched target ROI"
                ),
                "whole_brain_coverage_denominator": "all finite whole-brain voxels",
                "threshold_localization_denominator": (
                    "all finite whole-brain voxels at or above threshold"
                ),
                "zero_suprathreshold_localization_policy": (
                    "0% when no finite whole-brain voxels meet the threshold"
                ),
                "top_percentile": task.top_percentile,
                "robust_max_percentile": task.robust_max_percentile,
                "upper_tail_fraction": task.upper_tail_fraction,
                "primary_roi": (
                    "MakeROIs.m-equivalent parcel-clipped sphere centred on the "
                    "anatomical parcel volume centroid"
                ),
                "secondary_roi": (
                    "full anatomical atlas parcel; metrics use the anatomical_ prefix"
                ),
            },
            "metrics": metrics,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(
        f".json.tmp-{os.getpid()}"
    )
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    return {
        "subject": task.subject,
        "status": "complete",
        "output": str(output_path),
    }


def _read_subjects(path: Path) -> list[str]:
    subjects = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not subjects:
        raise ValueError(f"Subject file is empty: {path}")
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Subject file contains duplicates: {path}")
    return subjects


def extract_dataset(
    *,
    dataset_root: Path,
    roi: str,
    repeat: str,
    subjects_file: Path,
    atlas_root: Path,
    summary_path: Path,
    workers: int,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
    force: bool,
) -> dict[str, Any]:
    subjects = _read_subjects(subjects_file)
    canonical_roi = match_fastsurfer_roi_from_directory(dataset_root).canonical_name
    tasks = [
        ExtractionTask(
            dataset_root=str(dataset_root),
            subject=subject,
            roi=roi,
            repeat=repeat,
            canonical_roi=canonical_roi,
            atlas_root=str(atlas_root),
            thresholds=tuple(float(value) for value in thresholds),
            top_percentile=float(top_percentile),
            robust_max_percentile=float(robust_max_percentile),
            upper_tail_fraction=float(upper_tail_fraction),
            force=force,
        )
        for subject in subjects
    ]
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=min(max(1, workers), len(tasks)),
        mp_context=get_context("spawn"),
    ) as pool:
        future_map = {pool.submit(_extract_subject, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "subject": task.subject,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "skipped")
    }
    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete" if not errors and len(results) == len(subjects) else "incomplete",
        "roi": roi,
        "canonical_roi": canonical_roi,
        "repeat": repeat,
        "dataset_root": str(dataset_root.resolve()),
        "subjects": len(subjects),
        "complete": counts["complete"],
        "skipped": counts["skipped"],
        "errors": errors,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] != "complete":
        raise RuntimeError(
            f"Manuscript metric extraction failed for {len(errors)} of "
            f"{len(subjects)} subjects; see {summary_path}."
        )
    return payload


def _flatten_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "subject": payload["subject"],
        "roi": payload["roi"],
        "repeat": str(payload["repeat"]).zfill(2),
        "canonical_roi": payload["canonical_roi"],
        "config_fingerprint": payload["config_fingerprint"],
    }
    roi_definition = payload.get("roi_definition")
    if not isinstance(roi_definition, Mapping):
        raise ValueError("Manuscript metrics payload has no ROI-definition object.")
    record.update(flatten_roi_metadata(dict(roi_definition)))
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Manuscript metrics payload has no metrics object.")
    record.update(metrics)
    return record


def _find_baseline_ti(root: Path) -> Path:
    direct = root / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
    if direct.is_file():
        return direct
    candidates = sorted(root.glob("*/anat/SimNIBS/ti_brain_only.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No MNI baseline TI NIfTI found below {root}.")
    return candidates[0]


def _compute_mni_record(
    *,
    roi: str,
    baseline_parent: Path,
    mni_atlas_path: Path,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
) -> dict[str, Any]:
    baseline_root = baseline_parent / MNI_BASELINE_NAMES[roi]
    ti_path = _find_baseline_ti(baseline_root)
    ti_img = nib.load(str(ti_path))
    ti_data = load_ti_as_scalar(ti_img)
    canonical_roi = match_fastsurfer_roi_from_directory(f"{roi}_Data_01").canonical_name
    roi_masks, _ = roi_masks_on_ti_grid(
        ti_img,
        atlas_mode="fastsurfer",
        subject="MNI152",
        fastsurfer_atlas_path=str(mni_atlas_path),
        roi_names=[canonical_roi],
    )
    roi_mask = roi_masks.get(canonical_roi)
    if roi_mask is None and len(roi_masks) == 1:
        roi_mask = next(iter(roi_masks.values()))
    if roi_mask is None:
        raise ValueError(f"MNI ROI '{canonical_roi}' was not returned for {roi}.")
    metrics, roi_definition = compute_optimizer_matched_metric_bundle(
        ti_img=ti_img,
        ti_data=ti_data,
        anatomical_roi_mask=roi_mask,
        roi=roi,
        thresholds=thresholds,
        top_percentile=top_percentile,
        robust_max_percentile=robust_max_percentile,
        upper_tail_fraction=upper_tail_fraction,
    )
    return {
        "subject": "MNI152",
        "roi": roi,
        **flatten_roi_metadata(roi_definition),
        **metrics,
    }


METRIC_METADATA: dict[str, tuple[str, str, str]] = {
    "roi_min_v_per_m": (
        "Minimum optimizer-target TI field",
        "V/m",
        "Minimum over finite voxels in the optimizer-matched parcel-clipped sphere.",
    ),
    "roi_median_v_per_m": (
        "Median optimizer-target TI field",
        "V/m",
        "Median over finite voxels in the optimizer-matched parcel-clipped sphere.",
    ),
    "roi_robust_max_p99_9_v_per_m": (
        "Robust maximum optimizer-target TI field (P99.9)",
        "V/m",
        "99.9th percentile over finite optimizer-target voxels; primary robust maximum.",
    ),
    "roi_upper_1_percent_median_v_per_m": (
        "Median of upper 1% optimizer-target TI field",
        "V/m",
        "Median among optimizer-target values at or above P99; sensitivity robust maximum.",
    ),
    "whole_brain_median_v_per_m": (
        "Median whole-brain TI field",
        "V/m",
        "Median over finite whole-brain voxels.",
    ),
    "whole_brain_robust_max_p99_9_v_per_m": (
        "Robust maximum whole-brain TI field (P99.9)",
        "V/m",
        "99.9th percentile over finite whole-brain voxels.",
    ),
    "whole_brain_upper_1_percent_median_v_per_m": (
        "Median of upper 1% whole-brain TI field",
        "V/m",
        "Median among whole-brain values at or above P99.",
    ),
    "off_target_median_v_per_m": (
        "Median off-target TI field",
        "V/m",
        "Median over finite whole-brain voxels outside the target ROI.",
    ),
    "off_target_robust_max_p99_9_v_per_m": (
        "Robust maximum off-target TI field (P99.9)",
        "V/m",
        "99.9th percentile outside the target ROI.",
    ),
    "off_target_upper_1_percent_median_v_per_m": (
        "Median of upper 1% off-target TI field",
        "V/m",
        "Median among off-target values at or above off-target P99.",
    ),
    "top_5_percent_target_coverage_percent": (
        "Target coverage by whole-brain top 5% field",
        "%",
        "Optimizer-target voxels in the whole-brain top 5%, divided by all "
        "optimizer-target voxels.",
    ),
    "top_5_percent_whole_brain_volume_mm3": (
        "Whole-brain top-5% field volume",
        "mm³",
        "Physical volume occupied by values at or above the whole-brain 95th percentile.",
    ),
    "top_5_percent_target_volume_mm3": (
        "Target volume in whole-brain top 5% field",
        "mm³",
        "Physical optimizer-target volume occupied by values at or above the "
        "whole-brain 95th percentile.",
    ),
    "top_5_percent_localization_percent_in_roi": (
        "Localization of whole-brain top 5% field in target",
        "%",
        "Whole-brain top-5% voxels in the target ROI, divided by all top-5% voxels.",
    ),
}
for _threshold in DEFAULT_THRESHOLDS_V_PER_M:
    _slug = _threshold_slug(_threshold)
    _label = f"{_threshold:.2f} V/m"
    METRIC_METADATA.update(
        {
            f"target_coverage_percent_ge_{_slug}": (
                f"Target coverage ≥ {_label}",
                "%",
                "Suprathreshold optimizer-target voxels divided by all "
                "optimizer-target voxels.",
            ),
            f"target_coverage_volume_mm3_ge_{_slug}": (
                f"Target volume ≥ {_label}",
                "mm³",
                "Physical target-ROI volume at or above threshold.",
            ),
            f"whole_brain_coverage_percent_ge_{_slug}": (
                f"Whole-brain coverage ≥ {_label}",
                "%",
                "Suprathreshold finite voxels divided by all finite whole-brain voxels.",
            ),
            f"whole_brain_coverage_volume_mm3_ge_{_slug}": (
                f"Whole-brain volume ≥ {_label}",
                "mm³",
                "Physical finite whole-brain volume at or above threshold.",
            ),
            f"off_target_coverage_percent_ge_{_slug}": (
                f"Off-target coverage ≥ {_label}",
                "%",
                "Suprathreshold finite voxels outside the ROI divided by all finite off-target voxels.",
            ),
            f"off_target_coverage_volume_mm3_ge_{_slug}": (
                f"Off-target volume ≥ {_label}",
                "mm³",
                "Physical finite volume outside the target ROI at or above threshold.",
            ),
            f"threshold_localization_percent_in_roi_ge_{_slug}": (
                f"Suprathreshold localization in target ≥ {_label}",
                "%",
                "Suprathreshold target voxels divided by all suprathreshold whole-brain voxels.",
            ),
        }
    )

# Retain the full anatomical-parcel analysis as a clearly labelled secondary
# result. This permits direct comparison with the earlier schema while the
# unprefixed metrics match the ROI used for optimization.
for _metric, (_label, _unit, _definition) in tuple(METRIC_METADATA.items()):
    if not _metric_depends_on_roi_scope(_metric):
        continue
    METRIC_METADATA[f"anatomical_{_metric}"] = (
        f"Full anatomical parcel: {_label}",
        _unit,
        f"Secondary analysis using the full anatomical parcel instead of the "
        f"optimizer-matched sphere; otherwise the calculation matches {_metric}.",
    )

MAIN_METRICS = (
    "roi_min_v_per_m",
    "roi_median_v_per_m",
    "roi_robust_max_p99_9_v_per_m",
    "target_coverage_percent_ge_0p2",
    "off_target_coverage_percent_ge_0p2",
    "threshold_localization_percent_in_roi_ge_0p2",
    "target_coverage_percent_ge_0p18",
    "off_target_coverage_percent_ge_0p18",
    "whole_brain_coverage_percent_ge_0p18",
    "threshold_localization_percent_in_roi_ge_0p18",
    "top_5_percent_target_coverage_percent",
    "top_5_percent_localization_percent_in_roi",
)


def _summary_row(
    *,
    roi: str,
    metric: str,
    values: pd.Series,
    mni_value: float,
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    label, unit, definition = METRIC_METADATA.get(
        metric, (metric, "", "See repeat-level metric definition.")
    )
    return {
        "roi": roi,
        "metric": metric,
        "metric_label": label,
        "unit": unit,
        "subject_n": int(numeric.size),
        "subject_mean": float(numeric.mean()),
        "subject_sd": float(numeric.std(ddof=1)),
        "subject_median": float(numeric.median()),
        "subject_q1": q1,
        "subject_q3": q3,
        "subject_iqr": q3 - q1,
        "subject_min": float(numeric.min()),
        "subject_max": float(numeric.max()),
        "mni_baseline": float(mni_value),
        "definition": definition,
    }


def _write_effectiveness_spread_figure(
    *,
    subject_frame: pd.DataFrame,
    mni_frame: pd.DataFrame,
    threshold: float,
    spread_scope: str,
    output_base: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    slug = _threshold_slug(threshold)
    x_metric = f"target_coverage_percent_ge_{slug}"
    if spread_scope == "off_target":
        y_metric = f"off_target_coverage_percent_ge_{slug}"
        y_label = f"Off-target volume ≥ {threshold:.2f} V/m (%)"
    elif spread_scope == "whole_brain":
        y_metric = f"whole_brain_coverage_percent_ge_{slug}"
        y_label = f"Whole-brain volume ≥ {threshold:.2f} V/m (%)"
    else:
        raise ValueError(f"Unknown spread scope: {spread_scope}")

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1), sharex=True, sharey=True)
    for axis, roi in zip(axes, ROI_ORDER):
        rows = subject_frame.loc[subject_frame["roi"] == roi]
        baseline = mni_frame.loc[mni_frame["roi"] == roi].iloc[0]
        axis.scatter(
            rows[x_metric],
            rows[y_metric],
            marker="x",
            s=32,
            linewidths=1.1,
            color="#2878B5",
            alpha=0.72,
        )
        axis.scatter(
            [baseline[x_metric]],
            [baseline[y_metric]],
            marker="o",
            s=70,
            linewidths=1.8,
            facecolors="white",
            edgecolors="#D55E00",
            zorder=5,
        )
        axis.set_title(roi.replace("_", " "), fontsize=10.5)
        axis.set_xlim(-2.0, 102.0)
        axis.set_ylim(-2.0, 102.0)
        axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_xlabel(
            f"Optimizer-matched target coverage ≥ {threshold:.2f} V/m (%)"
        )
    axes[0].set_ylabel(y_label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color="#2878B5",
            markersize=7,
            label="CamCan subject (mean of 10 repeats)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#D55E00",
            markeredgewidth=1.6,
            markersize=7,
            label="MNI152 baseline",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_analysis_outputs(
    *,
    repeat_frame: pd.DataFrame,
    mni_frame: pd.DataFrame,
    out_dir: Path,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
    expected_subjects: set[str] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required_identity_columns = {"subject", "roi", "repeat", "canonical_roi"}
    missing_identity_columns = required_identity_columns.difference(repeat_frame.columns)
    if missing_identity_columns:
        raise RuntimeError(
            "Repeat-level metric table is missing identity columns: "
            f"{sorted(missing_identity_columns)}"
        )
    if repeat_frame.duplicated(["subject", "roi", "repeat"]).any():
        raise RuntimeError("Repeat-level metric table contains duplicate subject/ROI/repeat rows.")
    if set(repeat_frame["roi"]) != set(ROI_ORDER):
        raise RuntimeError("Repeat-level ROI set does not match the four planned ROIs.")
    if expected_subjects is not None and set(repeat_frame["subject"]) != expected_subjects:
        raise RuntimeError("Collected subject set does not match the cohort subject file.")
    if set(mni_frame["roi"]) != set(ROI_ORDER) or len(mni_frame) != len(ROI_ORDER):
        raise RuntimeError("MNI baseline table must contain exactly one row for each planned ROI.")

    metric_columns = [
        name
        for name in manuscript_metric_names(thresholds)
        if name in repeat_frame.columns
    ]
    missing_metrics = set(manuscript_metric_names(thresholds)).difference(metric_columns)
    if missing_metrics:
        raise RuntimeError(
            "Repeat-level metric table is missing manuscript metrics: "
            f"{sorted(missing_metrics)}"
        )
    missing_mni_metrics = set(metric_columns).difference(mni_frame.columns)
    if missing_mni_metrics:
        raise RuntimeError(
            "MNI baseline table is missing manuscript metrics: "
            f"{sorted(missing_mni_metrics)}"
        )
    finite_metric_values = np.isfinite(
        repeat_frame[metric_columns].to_numpy(dtype=float, copy=False)
    )
    if not finite_metric_values.all():
        bad_columns = repeat_frame[metric_columns].columns[
            ~finite_metric_values.all(axis=0)
        ].tolist()
        raise RuntimeError(
            "Repeat-level manuscript metrics contain non-finite values; "
            "refusing an aggregation that could silently omit repeats. "
            f"Affected metrics: {bad_columns}"
        )
    finite_mni_values = np.isfinite(
        mni_frame[metric_columns].to_numpy(dtype=float, copy=False)
    )
    if not finite_mni_values.all():
        bad_columns = mni_frame[metric_columns].columns[
            ~finite_mni_values.all(axis=0)
        ].tolist()
        raise RuntimeError(
            "MNI baseline manuscript metrics contain non-finite values. "
            f"Affected metrics: {bad_columns}"
        )

    group_columns = ["subject", "roi", "canonical_roi"]
    roi_definition_columns = [
        column
        for column in repeat_frame.columns
        if column.startswith("optimizer_roi_")
        and pd.api.types.is_numeric_dtype(repeat_frame[column])
    ]
    means = (
        repeat_frame.groupby(group_columns, sort=False)[
            metric_columns + roi_definition_columns
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    repeat_sds = (
        repeat_frame.groupby(group_columns, sort=False)[metric_columns]
        .std(ddof=1, numeric_only=True)
        .add_suffix("__repeat_sd")
        .reset_index()
    )
    repeat_counts = (
        repeat_frame.groupby(group_columns, sort=False)
        .size()
        .rename("repeat_count")
        .reset_index()
    )
    subject_frame = means.merge(repeat_sds, on=group_columns).merge(
        repeat_counts, on=group_columns
    )
    if not (subject_frame["repeat_count"] == 10).all():
        raise RuntimeError("At least one subject/ROI does not have exactly ten repeats.")

    summary_rows: list[dict[str, Any]] = []
    for roi in ROI_ORDER:
        subject_rows = subject_frame.loc[subject_frame["roi"] == roi]
        mni_row = mni_frame.loc[mni_frame["roi"] == roi].iloc[0]
        for metric in metric_columns:
            if metric not in METRIC_METADATA:
                continue
            summary_rows.append(
                _summary_row(
                    roi=roi,
                    metric=metric,
                    values=subject_rows[metric],
                    mni_value=float(mni_row[metric]),
                )
            )
    supplementary = pd.DataFrame(summary_rows)
    main_long = supplementary.loc[
        supplementary["metric"].isin(MAIN_METRICS)
    ].copy()
    main_long["formatted_mean_sd"] = main_long.apply(
        lambda row: f"{row['subject_mean']:.3f} ± {row['subject_sd']:.3f}",
        axis=1,
    )
    main_long["_roi_order"] = main_long["roi"].map(
        {roi: index for index, roi in enumerate(ROI_ORDER)}
    )
    main_long["_metric_order"] = main_long["metric"].map(
        {metric: index for index, metric in enumerate(MAIN_METRICS)}
    )
    main_long = main_long.sort_values(
        ["_metric_order", "_roi_order"]
    ).drop(columns=["_roi_order", "_metric_order"])
    formatted_rows: list[dict[str, Any]] = []
    for metric in MAIN_METRICS:
        metric_rows = main_long.loc[main_long["metric"] == metric]
        if metric_rows.empty:
            continue
        first = metric_rows.iloc[0]
        formatted_row: dict[str, Any] = {
            "metric": metric,
            "metric_label": first["metric_label"],
            "unit": first["unit"],
        }
        for roi in ROI_ORDER:
            roi_row = metric_rows.loc[metric_rows["roi"] == roi].iloc[0]
            formatted_row[roi] = roi_row["formatted_mean_sd"]
            formatted_row[f"{roi}__MNI152"] = roi_row["mni_baseline"]
        formatted_rows.append(formatted_row)
    main_formatted = pd.DataFrame(formatted_rows)

    dictionary_rows = []
    for metric, (label, unit, definition) in METRIC_METADATA.items():
        dictionary_rows.append(
            {
                "metric": metric,
                "label": label,
                "unit": unit,
                "definition": definition,
                "main_table": metric in MAIN_METRICS,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    repeat_frame.to_csv(out_dir / "repeat_level_metrics.csv", index=False)
    subject_frame.to_csv(out_dir / "subject_level_repeat_mean_metrics.csv", index=False)
    subject_frame[
        group_columns + ["repeat_count"] + roi_definition_columns
    ].to_csv(out_dir / "table_optimizer_roi_definitions.csv", index=False)
    mni_frame.to_csv(out_dir / "mni152_baseline_metrics.csv", index=False)
    main_long.to_csv(out_dir / "table_main_long.csv", index=False)
    main_formatted.to_csv(out_dir / "table_main_formatted.csv", index=False)
    supplementary.to_csv(
        out_dir / "table_supplementary_descriptive_statistics.csv", index=False
    )
    pd.DataFrame(dictionary_rows).to_csv(out_dir / "metric_dictionary.csv", index=False)

    figures_dir = out_dir / "figures"
    for threshold in thresholds:
        slug = _threshold_slug(threshold)
        _write_effectiveness_spread_figure(
            subject_frame=subject_frame,
            mni_frame=mni_frame,
            threshold=threshold,
            spread_scope="off_target",
            output_base=figures_dir
            / f"effectiveness_vs_off_target_coverage_ge_{slug}",
        )
        _write_effectiveness_spread_figure(
            subject_frame=subject_frame,
            mni_frame=mni_frame,
            threshold=threshold,
            spread_scope="whole_brain",
            output_base=figures_dir
            / f"effectiveness_vs_whole_brain_coverage_ge_{slug}",
        )

    manifest: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "aggregation": (
            "Each metric is calculated independently per repeat, then arithmetic-mean "
            "aggregated across the ten repeats for each subject and ROI."
        ),
        "subjects": int(repeat_frame["subject"].nunique()),
        "rois": list(ROI_ORDER),
        "repeats_per_subject_roi": 10,
        "repeat_level_records": len(repeat_frame),
        "subject_level_records": len(subject_frame),
        "mni_baselines": len(mni_frame),
        "thresholds_v_per_m": [float(value) for value in thresholds],
        "top_percentile": float(top_percentile),
        "robust_max_percentile": float(robust_max_percentile),
        "upper_tail_fraction": float(upper_tail_fraction),
        "primary_robust_maximum": "99.9th percentile (P99.9)",
        "robust_maximum_sensitivity": "median of values in the upper 1%",
        "primary_roi_definition": (
            "MakeROIs.m-equivalent sphere centred on the anatomical parcel "
            "volume centroid and clipped to that parcel"
        ),
        "secondary_roi_definition": (
            "full anatomical atlas parcel; metrics carry the anatomical_ prefix"
        ),
        "roi_definition_schema_version": ROI_DEFINITION_SCHEMA_VERSION,
        "roi_target_volumes_mm3": TARGET_VOLUME_MM3_BY_ROI,
        "roi_sphere_start_radius_mm": START_RADIUS_MM,
        "roi_sphere_radius_step_mm": RADIUS_STEP_MM,
        "roi_sphere_radius_cap_mm": RADIUS_CAP_MM,
        "roi_distance_comparator": "<",
        "roi_representation_note": (
            "The tetrahedral MakeROIs.m construction is reproduced on the "
            "voxelized subject-space atlas/TI grid. Constant NIfTI voxel volume "
            "makes the world-coordinate voxel-centre mean volume weighted."
        ),
        "nonfinite_roi_policy": "counted as unstimulated in target coverage denominator",
        "zero_suprathreshold_localization_policy": (
            "0% when no finite whole-brain voxels meet the threshold"
        ),
        "cross_roi_inference": False,
        "individualized_optimization_included": False,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    manifest["outputs"] = sorted(
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def collect_analysis(
    *,
    study_root: Path,
    subjects_file: Path,
    mni_atlas_path: Path,
    mni_baseline_parent: Path,
    out_dir: Path,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
) -> dict[str, Any]:
    subjects = _read_subjects(subjects_file)
    expected_subjects = set(subjects)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for roi in ROI_ORDER:
        for repeat_number in range(1, 11):
            repeat = f"{repeat_number:02d}"
            dataset_root = study_root / "runs" / f"{roi}_Runs" / f"{roi}_Data_{repeat}"
            for subject in subjects:
                path = (
                    dataset_root
                    / subject
                    / "anat"
                    / "post"
                    / METRIC_MARKER_FILENAME
                )
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    missing.append(str(path))
                    continue
                if payload.get("status") != "complete":
                    missing.append(str(path))
                    continue
                records.append(_flatten_payload(payload))

    expected_records = len(subjects) * len(ROI_ORDER) * 10
    if missing or len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} complete manuscript metric records; "
            f"found {len(records)} with {len(missing)} missing/incomplete."
        )
    repeat_frame = pd.DataFrame(records)
    if set(repeat_frame["subject"]) != expected_subjects:
        raise RuntimeError("Collected subject set does not match the cohort subject file.")
    repeat_frame, repair_counts = _repair_zero_denominator_localization(
        repeat_frame,
        thresholds,
    )
    metric_columns = manuscript_metric_names(thresholds)
    finite_metric_values = np.isfinite(
        repeat_frame[metric_columns].to_numpy(dtype=float, copy=False)
    )
    if not finite_metric_values.all():
        bad_columns = repeat_frame[metric_columns].columns[
            ~finite_metric_values.all(axis=0)
        ].tolist()
        raise RuntimeError(
            "Repeat-level manuscript metrics contain non-finite values; "
            "refusing an aggregation that could silently omit repeats. "
            f"Affected metrics: {bad_columns}"
        )

    mni_records = [
        _compute_mni_record(
            roi=roi,
            baseline_parent=mni_baseline_parent,
            mni_atlas_path=mni_atlas_path,
            thresholds=thresholds,
            top_percentile=top_percentile,
            robust_max_percentile=robust_max_percentile,
            upper_tail_fraction=upper_tail_fraction,
        )
        for roi in ROI_ORDER
    ]
    mni_frame = pd.DataFrame(mni_records)
    return _write_analysis_outputs(
        repeat_frame=repeat_frame,
        mni_frame=mni_frame,
        out_dir=out_dir,
        thresholds=thresholds,
        top_percentile=top_percentile,
        robust_max_percentile=robust_max_percentile,
        upper_tail_fraction=upper_tail_fraction,
        expected_subjects=expected_subjects,
        manifest_extra={
            "execution_mode": "full_image_metric_extraction_and_aggregation",
            "zero_denominator_localization_values_repaired": sum(
                repair_counts.values()
            ),
            "zero_denominator_localization_repairs_by_metric": repair_counts,
        },
    )


def _repair_zero_denominator_localization(
    repeat_frame: pd.DataFrame,
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Repair only the legacy 0/0 localization values that have a defined policy."""

    repaired = repeat_frame.copy()
    repair_counts: dict[str, int] = {}
    for threshold in thresholds:
        slug = _threshold_slug(threshold)
        localization = f"threshold_localization_percent_in_roi_ge_{slug}"
        whole_count = f"whole_brain_coverage_voxels_ge_{slug}"
        target_count = f"target_coverage_voxels_ge_{slug}"
        required = {localization, whole_count, target_count}
        missing = required.difference(repaired.columns)
        if missing:
            raise RuntimeError(
                f"Cannot apply the zero-denominator policy; missing columns: {sorted(missing)}"
            )

        localization_values = pd.to_numeric(repaired[localization], errors="coerce")
        whole_values = pd.to_numeric(repaired[whole_count], errors="coerce")
        target_values = pd.to_numeric(repaired[target_count], errors="coerce")
        invalid = ~np.isfinite(localization_values.to_numpy(dtype=float, copy=False))
        repairable = invalid & (whole_values == 0).to_numpy() & (target_values == 0).to_numpy()
        unrepairable = invalid & ~repairable
        if np.any(unrepairable):
            bad_rows = repaired.loc[
                unrepairable, ["subject", "roi", "repeat", localization, whole_count, target_count]
            ]
            raise RuntimeError(
                "Non-finite localization values were found outside the defined 0/0 case; "
                f"first affected rows: {bad_rows.head(5).to_dict(orient='records')}"
            )
        repaired.loc[repairable, localization] = 0.0
        repair_counts[localization] = int(np.count_nonzero(repairable))
    return repaired, repair_counts


def reaggregate_existing_analysis(
    *,
    input_dir: Path,
    out_dir: Path,
    thresholds: Sequence[float],
    top_percentile: float,
    robust_max_percentile: float,
    upper_tail_fraction: float,
    subjects_file: Path | None = None,
) -> dict[str, Any]:
    """Rebuild tables and figures from existing repeat-level CSV metrics.

    This mode intentionally performs no NIfTI, atlas, or simulation reads. It
    only accepts repeat-level metrics already extracted with the current
    optimizer-matched schema. Earlier full-anatomical-parcel CSVs cannot be
    converted to the new ROI definition and are rejected.
    """

    repeat_path = input_dir / "repeat_level_metrics.csv"
    mni_path = input_dir / "mni152_baseline_metrics.csv"
    if not repeat_path.is_file():
        raise FileNotFoundError(f"Missing repeat-level metric table: {repeat_path}")
    if not mni_path.is_file():
        raise FileNotFoundError(f"Missing MNI baseline metric table: {mni_path}")
    if input_dir.resolve() == out_dir.resolve():
        raise RuntimeError(
            "Refusing to overwrite the source results in place; choose a new --out-dir."
        )
    source_manifest_path = input_dir / "analysis_manifest.json"
    if not source_manifest_path.is_file():
        raise RuntimeError(
            "CSV-only reaggregation requires a source analysis_manifest.json."
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise RuntimeError(
            "CSV-only reaggregation cannot convert an earlier anatomical-ROI "
            "schema to the optimizer-matched ROI. Run image extraction instead."
        )
    source_manifest_sha256 = _sha256_file(source_manifest_path)

    repeat_frame = pd.read_csv(repeat_path)
    mni_frame = pd.read_csv(mni_path)
    repaired_frame, repair_counts = _repair_zero_denominator_localization(
        repeat_frame, thresholds
    )
    expected_subjects = (
        set(_read_subjects(subjects_file)) if subjects_file is not None else None
    )
    repeat_sha256 = _sha256_file(repeat_path)
    mni_sha256 = _sha256_file(mni_path)
    total_repairs = int(sum(repair_counts.values()))

    return _write_analysis_outputs(
        repeat_frame=repaired_frame,
        mni_frame=mni_frame,
        out_dir=out_dir,
        thresholds=thresholds,
        top_percentile=top_percentile,
        robust_max_percentile=robust_max_percentile,
        upper_tail_fraction=upper_tail_fraction,
        expected_subjects=expected_subjects,
        manifest_extra={
            "execution_mode": "csv_only_reaggregation",
            "image_metric_extraction_rerun": False,
            "source_results_dir": str(input_dir.resolve()),
            "source_repeat_level_metrics_sha256": repeat_sha256,
            "source_mni152_baseline_metrics_sha256": mni_sha256,
            "source_analysis_manifest_sha256": source_manifest_sha256,
            "zero_denominator_localization_values_repaired": total_repairs,
            "zero_denominator_localization_repairs_by_metric": repair_counts,
        },
    )


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Thresholds must be a comma-separated list of numbers."
        ) from exc
    if not thresholds or any(item < 0 for item in thresholds):
        raise argparse.ArgumentTypeError("At least one non-negative threshold is required.")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-dataset")
    extract.add_argument("--dataset-root", type=Path, required=True)
    extract.add_argument("--roi", required=True, choices=ROI_ORDER)
    extract.add_argument("--repeat", required=True)
    extract.add_argument("--subjects-file", type=Path, required=True)
    extract.add_argument("--atlas-root", type=Path, required=True)
    extract.add_argument("--summary", type=Path, required=True)
    extract.add_argument("--workers", type=int, default=12)
    extract.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS_V_PER_M,
    )
    extract.add_argument("--top-percentile", type=float, default=DEFAULT_TOP_PERCENTILE)
    extract.add_argument(
        "--robust-max-percentile",
        type=float,
        default=DEFAULT_ROBUST_MAX_PERCENTILE,
    )
    extract.add_argument(
        "--upper-tail-fraction",
        type=float,
        default=DEFAULT_UPPER_TAIL_FRACTION,
    )
    extract.add_argument("--force", action="store_true")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--study-root", type=Path, required=True)
    collect.add_argument("--subjects-file", type=Path, required=True)
    collect.add_argument("--mni-atlas", type=Path, required=True)
    collect.add_argument("--mni-baseline-parent", type=Path, required=True)
    collect.add_argument("--out-dir", type=Path, required=True)
    collect.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS_V_PER_M,
    )
    collect.add_argument("--top-percentile", type=float, default=DEFAULT_TOP_PERCENTILE)
    collect.add_argument(
        "--robust-max-percentile",
        type=float,
        default=DEFAULT_ROBUST_MAX_PERCENTILE,
    )
    collect.add_argument(
        "--upper-tail-fraction",
        type=float,
        default=DEFAULT_UPPER_TAIL_FRACTION,
    )

    reaggregate = subparsers.add_parser(
        "reaggregate-existing",
        help=(
            "Rebuild tables and figures from current-schema repeat-level CSVs "
            "without reading NIfTIs or atlases."
        ),
    )
    reaggregate.add_argument("--input-dir", type=Path, required=True)
    reaggregate.add_argument("--out-dir", type=Path, required=True)
    reaggregate.add_argument("--subjects-file", type=Path)
    reaggregate.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS_V_PER_M,
    )
    reaggregate.add_argument(
        "--top-percentile", type=float, default=DEFAULT_TOP_PERCENTILE
    )
    reaggregate.add_argument(
        "--robust-max-percentile",
        type=float,
        default=DEFAULT_ROBUST_MAX_PERCENTILE,
    )
    reaggregate.add_argument(
        "--upper-tail-fraction",
        type=float,
        default=DEFAULT_UPPER_TAIL_FRACTION,
    )

    validate_atlas = subparsers.add_parser(
        "validate-atlas",
        help="Confirm that an atlas contains all labels required by the four ROIs.",
    )
    validate_atlas.add_argument("--atlas", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract-dataset":
        result = extract_dataset(
            dataset_root=args.dataset_root,
            roi=args.roi,
            repeat=str(args.repeat).zfill(2),
            subjects_file=args.subjects_file,
            atlas_root=args.atlas_root,
            summary_path=args.summary,
            workers=args.workers,
            thresholds=args.thresholds,
            top_percentile=args.top_percentile,
            robust_max_percentile=args.robust_max_percentile,
            upper_tail_fraction=args.upper_tail_fraction,
            force=args.force,
        )
    elif args.command == "collect":
        result = collect_analysis(
            study_root=args.study_root,
            subjects_file=args.subjects_file,
            mni_atlas_path=args.mni_atlas,
            mni_baseline_parent=args.mni_baseline_parent,
            out_dir=args.out_dir,
            thresholds=args.thresholds,
            top_percentile=args.top_percentile,
            robust_max_percentile=args.robust_max_percentile,
            upper_tail_fraction=args.upper_tail_fraction,
        )
    elif args.command == "reaggregate-existing":
        result = reaggregate_existing_analysis(
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            subjects_file=args.subjects_file,
            thresholds=args.thresholds,
            top_percentile=args.top_percentile,
            robust_max_percentile=args.robust_max_percentile,
            upper_tail_fraction=args.upper_tail_fraction,
        )
    else:
        result = validate_atlas_rois(args.atlas)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
