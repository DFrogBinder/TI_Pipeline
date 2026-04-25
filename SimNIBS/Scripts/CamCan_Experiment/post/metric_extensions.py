from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt

from post.post_functions import roi_masks_on_ti_grid
from utils.roi_registry import FASTSURFER_DKT_LABELS, resolve_fastsurfer_roi_name
from utils.ti_utils import load_ti_as_scalar, resample_atlas_to_ti_grid, vol_mm3


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _normalize_baseline_metric_name(metric_name: str) -> str:
    aliases = {
        "focality_voxels": "focality_voxels_gt_threshold",
        "focality_volume_mm3": "focality_volume_mm3_gt_threshold",
    }
    return aliases.get(metric_name, metric_name)


def _metric_slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


@lru_cache(maxsize=None)
def _canonical_label_ids(roi_name: str) -> Tuple[int, ...]:
    canonical = resolve_fastsurfer_roi_name(roi_name).canonical_name
    label_ids = tuple(
        sorted(label_id for label_id, label_name in FASTSURFER_DKT_LABELS.items() if label_name == canonical)
    )
    if not label_ids:
        raise ValueError(f"Could not resolve FastSurfer label ids for ROI '{roi_name}'.")
    return label_ids


def _compute_core_field_metrics(
    *,
    ti_img: nib.Nifti1Image,
    ti_data: np.ndarray,
    roi_mask: np.ndarray,
    finite_mask: np.ndarray,
    focality_threshold: float,
) -> Dict[str, float]:
    voxel_volume_mm3 = vol_mm3(ti_img)
    roi_vals = ti_data[roi_mask & finite_mask]
    roi_peak = float(np.max(roi_vals)) if roi_vals.size else math.nan
    roi_mean = float(np.mean(roi_vals)) if roi_vals.size else math.nan

    focality_mask = finite_mask & (ti_data > focality_threshold)
    focality_voxels = int(np.sum(focality_mask))
    focality_volume_mm3 = float(focality_voxels * voxel_volume_mm3)
    return {
        "roi_peak": roi_peak,
        "roi_mean": roi_mean,
        "focality_voxels_gt_threshold": float(focality_voxels),
        "focality_volume_mm3_gt_threshold": focality_volume_mm3,
    }


def _baseline_subject_root_candidates(root: Path) -> List[Tuple[Path, str]]:
    candidates: List[Tuple[Path, str]] = []
    direct_ti = root / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
    if direct_ti.is_file():
        candidates.append((root, root.name))

    for subject_name in ("MNI152", "sub-mni152", "mni152"):
        subject_root = root / subject_name
        ti_path = subject_root / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
        if ti_path.is_file():
            candidates.append((subject_root, subject_name))

    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            ti_path = child / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
            if ti_path.is_file():
                candidates.append((child, child.name))

    unique: List[Tuple[Path, str]] = []
    seen: set[Path] = set()
    for subject_root, subject_name in candidates:
        resolved = subject_root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((resolved, subject_name))
    return unique


@lru_cache(maxsize=None)
def load_mni_baseline_metrics(
    baseline_root_value: Optional[str],
    roi_name: str,
    mni_fixed_atlas_path: Optional[str],
    focality_threshold: float,
) -> Dict[str, float]:
    if not baseline_root_value or not mni_fixed_atlas_path:
        return {}

    baseline_root = Path(baseline_root_value).expanduser()
    atlas_path = Path(mni_fixed_atlas_path).expanduser()
    if not baseline_root.exists() or not atlas_path.is_file():
        return {}

    for subject_root, subject_name in _baseline_subject_root_candidates(baseline_root):
        ti_path = subject_root / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
        if not ti_path.is_file():
            continue

        ti_img = nib.load(str(ti_path))
        ti_data = load_ti_as_scalar(ti_img)
        finite_mask = np.isfinite(ti_data)
        roi_masks, _ = roi_masks_on_ti_grid(
            ti_img,
            atlas_mode="fastsurfer",
            subject=subject_name,
            fastsurfer_atlas_path=str(atlas_path),
            roi_names=[roi_name],
        )
        roi_mask = roi_masks.get(roi_name)
        if roi_mask is None:
            continue
        return _compute_core_field_metrics(
            ti_img=ti_img,
            ti_data=ti_data,
            roi_mask=roi_mask,
            finite_mask=finite_mask,
            focality_threshold=focality_threshold,
        )

    return {}


@lru_cache(maxsize=None)
def _fixed_neighbor_template(mni_atlas_path: str, roi_name: str, dilation_iter: int) -> Tuple[Tuple[int, str], ...]:
    atlas_img = nib.load(mni_atlas_path)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
    roi_ids = set(_canonical_label_ids(roi_name))
    roi_mask = np.isin(atlas_data, tuple(roi_ids))
    if not np.any(roi_mask):
        raise ValueError(
            f"ROI '{roi_name}' was not found in fixed MNI atlas template '{mni_atlas_path}'."
        )

    dilated = binary_dilation(roi_mask, iterations=dilation_iter)
    border = dilated & (~roi_mask)
    neighbor_ids = set(int(value) for value in np.unique(atlas_data[border]))
    neighbor_ids.discard(0)
    neighbor_ids -= roi_ids
    return tuple(
        (label_id, FASTSURFER_DKT_LABELS.get(label_id, f"Label-{label_id}"))
        for label_id in sorted(neighbor_ids)
    )


def fixed_neighbor_template(
    mni_atlas_path: Optional[str],
    roi_name: str,
    dilation_iter: int,
) -> List[Dict[str, Any]]:
    if not mni_atlas_path:
        return []
    return [
        {"label_id": label_id, "label_name": label_name}
        for label_id, label_name in _fixed_neighbor_template(
            str(Path(mni_atlas_path).expanduser()),
            roi_name,
            dilation_iter,
        )
    ]


def _roi_centroid_world(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return None
    center_ijk = ijk.mean(axis=0)
    return np.asarray(nib.affines.apply_affine(affine, center_ijk), dtype=float)


def _roi_centroid_ijk(mask: np.ndarray) -> Optional[np.ndarray]:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return None
    return ijk.mean(axis=0)


def _distance_to_tissue(
    tissue_mask: np.ndarray,
    roi_centroid_ijk: np.ndarray,
    zooms: Tuple[float, float, float],
) -> float:
    inv = ~tissue_mask
    dist_map = distance_transform_edt(inv, sampling=zooms)
    idx = np.round(roi_centroid_ijk).astype(int)
    idx = np.clip(idx, [0, 0, 0], np.array(dist_map.shape) - 1)
    return float(dist_map[tuple(idx)])


def _load_electrode_centers(path: Path) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    df = pd.read_csv(path)
    required = {"subject", "electrode", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"Electrode CSV missing required columns: {sorted(required)}")

    by_subject: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for _, row in df.iterrows():
        subj = str(row["subject"])
        coord = np.array([row["x"], row["y"], row["z"]], dtype=float)
        by_subject.setdefault(subj, []).append((str(row["electrode"]), coord))
    return by_subject


def _read_eeg_positions(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return {}

    try:
        df = pd.read_csv(path)
        cols = {column.lower(): column for column in df.columns}
        if {"name", "x", "y", "z"}.issubset(cols):
            out = {}
            for _, row in df.iterrows():
                out[str(row[cols["name"]])] = np.array(
                    [row[cols["x"]], row[cols["y"]], row[cols["z"]]],
                    dtype=float,
                )
            return out
    except Exception:
        pass

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            out[parts[0]] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        except ValueError:
            continue
    return out


def resolve_electrode_centers(
    *,
    root_dir: str,
    subject: str,
    electrode_csv: Optional[str],
    electrode_names: Optional[Sequence[str]],
    eeg_positions_path_template: Optional[str],
) -> List[Tuple[str, np.ndarray]]:
    if electrode_csv:
        path = Path(electrode_csv).expanduser()
        if path.is_file():
            return _load_electrode_centers(path).get(subject, [])

    if not electrode_names:
        return []

    if eeg_positions_path_template:
        eeg_path = Path(
            eeg_positions_path_template.format(root=root_dir, subject=subject)
        ).expanduser()
    else:
        eeg_path = Path(root_dir) / subject / "anat" / f"m2m_{subject}" / "eeg_positions.csv"

    positions = _read_eeg_positions(eeg_path)
    return [
        (name, positions[name])
        for name in electrode_names
        if name in positions
    ]


def _neighbor_stats(
    neighbor_template: Sequence[Dict[str, Any]],
    atlas_data: np.ndarray,
    ti_data: np.ndarray,
    finite: np.ndarray,
    voxel_volume_mm3: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in neighbor_template:
        label_id = int(entry["label_id"])
        mask = atlas_data == label_id
        vals = ti_data[mask & finite]
        if vals.size == 0:
            row = {
                "label_id": label_id,
                "label_name": entry["label_name"],
                "voxels": 0,
                "volume_mm3": 0.0,
                "mean": math.nan,
                "max": math.nan,
            }
        else:
            row = {
                "label_id": label_id,
                "label_name": entry["label_name"],
                "voxels": int(mask.sum()),
                "volume_mm3": float(mask.sum() * voxel_volume_mm3),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
            }
        rows.append(row)
    return rows


def summarise_electrode_distances(
    electrode_distances: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    values = np.array(
        [
            _safe_float(entry.get("distance_mm"))
            for entry in electrode_distances
            if math.isfinite(_safe_float(entry.get("distance_mm")))
        ],
        dtype=float,
    )
    if values.size == 0:
        return {
            "electrode_distance_count": 0,
            "electrode_distance_mean_mm": math.nan,
            "electrode_distance_min_mm": math.nan,
            "electrode_distance_max_mm": math.nan,
        }
    return {
        "electrode_distance_count": int(values.size),
        "electrode_distance_mean_mm": float(np.mean(values)),
        "electrode_distance_min_mm": float(np.min(values)),
        "electrode_distance_max_mm": float(np.max(values)),
    }


def compute_extended_subject_metrics(
    *,
    root_dir: str,
    subject: str,
    roi_name: str,
    ti_img: nib.Nifti1Image,
    ti_data: np.ndarray,
    roi_mask: np.ndarray,
    finite_mask: np.ndarray,
    subject_fastsurfer_atlas_path: Optional[str],
    mni_baseline_root: Optional[str],
    mni_fixed_atlas_path: Optional[str],
    focality_threshold: float,
    neighbor_dilation_iter: int,
    csf_labels: Optional[Sequence[int]],
    skull_labels: Optional[Sequence[int]],
    electrode_csv: Optional[str],
    electrode_names: Optional[Sequence[str]],
    eeg_positions_path_template: Optional[str],
) -> Dict[str, Any]:
    core_metrics = _compute_core_field_metrics(
        ti_img=ti_img,
        ti_data=ti_data,
        roi_mask=roi_mask,
        finite_mask=finite_mask,
        focality_threshold=focality_threshold,
    )
    voxel_volume_mm3 = vol_mm3(ti_img)
    roi_peak = _safe_float(core_metrics.get("roi_peak"))
    roi_mean = _safe_float(core_metrics.get("roi_mean"))
    focality_voxels_value = _safe_float(core_metrics.get("focality_voxels_gt_threshold"))
    focality_voxels = int(focality_voxels_value) if math.isfinite(focality_voxels_value) else 0
    focality_volume_mm3 = _safe_float(core_metrics.get("focality_volume_mm3_gt_threshold"))

    baseline = load_mni_baseline_metrics(
        mni_baseline_root,
        roi_name,
        mni_fixed_atlas_path,
        focality_threshold,
    )
    baseline_peak = _safe_float(baseline.get("roi_peak"))
    baseline_mean = _safe_float(baseline.get("roi_mean"))
    baseline_focality_voxels = _safe_float(baseline.get("focality_voxels_gt_threshold"))
    baseline_focality_volume = _safe_float(baseline.get("focality_volume_mm3_gt_threshold"))

    metrics: Dict[str, Any] = {
        "roi_peak": roi_peak,
        "roi_mean": roi_mean,
        "roi_peak_abs_delta_mni": abs(roi_peak - baseline_peak) if math.isfinite(baseline_peak) and math.isfinite(roi_peak) else math.nan,
        "roi_mean_abs_delta_mni": abs(roi_mean - baseline_mean) if math.isfinite(baseline_mean) and math.isfinite(roi_mean) else math.nan,
        "mni_baseline_roi_peak": baseline_peak,
        "mni_baseline_roi_mean": baseline_mean,
        "focality_threshold_v_per_m": float(focality_threshold),
        "focality_voxels_gt_threshold": focality_voxels,
        "focality_volume_mm3_gt_threshold": focality_volume_mm3,
        "focality_voxels_abs_delta_mni": abs(focality_voxels - baseline_focality_voxels) if math.isfinite(baseline_focality_voxels) else math.nan,
        "focality_volume_mm3_abs_delta_mni": abs(focality_volume_mm3 - baseline_focality_volume) if math.isfinite(baseline_focality_volume) else math.nan,
        "mni_baseline_focality_voxels_gt_threshold": baseline_focality_voxels,
        "mni_baseline_focality_volume_mm3_gt_threshold": baseline_focality_volume,
    }

    neighbor_template = fixed_neighbor_template(
        mni_fixed_atlas_path,
        roi_name,
        neighbor_dilation_iter,
    )
    metrics["neighbor_template_count"] = len(neighbor_template)

    atlas_data = None
    if subject_fastsurfer_atlas_path and Path(subject_fastsurfer_atlas_path).is_file():
        atlas_img = resample_atlas_to_ti_grid(nib.load(subject_fastsurfer_atlas_path), ti_img)
        atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)

    neighbor_rows: List[Dict[str, Any]] = []
    if atlas_data is not None and neighbor_template:
        neighbor_rows = _neighbor_stats(
            neighbor_template,
            atlas_data,
            ti_data,
            finite_mask,
            voxel_volume_mm3,
        )

    metrics["neighbors"] = neighbor_rows
    if neighbor_rows:
        mean_values = np.array(
            [row["mean"] for row in neighbor_rows if math.isfinite(_safe_float(row["mean"]))],
            dtype=float,
        )
        max_values = np.array(
            [row["max"] for row in neighbor_rows if math.isfinite(_safe_float(row["max"]))],
            dtype=float,
        )
        metrics["neighbor_mean_of_means"] = float(np.mean(mean_values)) if mean_values.size else math.nan
        metrics["neighbor_max_of_max"] = float(np.max(max_values)) if max_values.size else math.nan
        metrics["neighbor_min_of_max"] = float(np.min(max_values)) if max_values.size else math.nan
    else:
        metrics["neighbor_mean_of_means"] = math.nan
        metrics["neighbor_max_of_max"] = math.nan
        metrics["neighbor_min_of_max"] = math.nan

    for row in neighbor_rows:
        slug = _metric_slug(str(row["label_name"]))
        metrics[f"neighbor_mean__{slug}"] = _safe_float(row.get("mean"))
        metrics[f"neighbor_peak__{slug}"] = _safe_float(row.get("max"))
        metrics[f"neighbor_voxels__{slug}"] = _safe_float(row.get("voxels"))

    roi_centroid_ijk = _roi_centroid_ijk(roi_mask)
    roi_centroid_xyz = _roi_centroid_world(roi_mask, ti_img.affine)
    metrics["roi_centroid_x"] = float(roi_centroid_xyz[0]) if roi_centroid_xyz is not None else math.nan
    metrics["roi_centroid_y"] = float(roi_centroid_xyz[1]) if roi_centroid_xyz is not None else math.nan
    metrics["roi_centroid_z"] = float(roi_centroid_xyz[2]) if roi_centroid_xyz is not None else math.nan

    csf_dist = math.nan
    skull_dist = math.nan
    if atlas_data is not None and roi_centroid_ijk is not None:
        zooms = ti_img.header.get_zooms()[:3]
        if csf_labels:
            csf_mask = np.isin(atlas_data, list(csf_labels))
            if np.any(csf_mask):
                csf_dist = _distance_to_tissue(csf_mask, roi_centroid_ijk, zooms)
        if skull_labels:
            skull_mask = np.isin(atlas_data, list(skull_labels))
            if np.any(skull_mask):
                skull_dist = _distance_to_tissue(skull_mask, roi_centroid_ijk, zooms)
    metrics["csf_distance_mm"] = csf_dist
    metrics["skull_distance_mm"] = skull_dist

    electrode_entries: List[Dict[str, Any]] = []
    if roi_centroid_xyz is not None:
        centers = resolve_electrode_centers(
            root_dir=root_dir,
            subject=subject,
            electrode_csv=electrode_csv,
            electrode_names=electrode_names,
            eeg_positions_path_template=eeg_positions_path_template,
        )
        for name, coord in centers:
            electrode_entries.append(
                {
                    "electrode": name,
                    "distance_mm": float(np.linalg.norm(coord - roi_centroid_xyz)),
                }
            )
    metrics["electrode_distances"] = electrode_entries
    metrics.update(summarise_electrode_distances(electrode_entries))

    return metrics


def flatten_subject_metric_payload(payload: Dict[str, Any], roi_key: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "subject": payload.get("subject"),
        "target_roi": payload.get("target_roi"),
        "percentile": payload.get("percentile"),
        "percentile_value": payload.get("percentile_value"),
        "voxel_volume_mm3": payload.get("voxel_volume_mm3"),
        "top_percentile_voxels": payload.get("top_percentile_voxels"),
    }
    roi_metrics = payload.get("rois", {}).get(roi_key, {})
    if isinstance(roi_metrics, dict):
        record.update(
            {
                "roi_voxels": roi_metrics.get("roi_voxels"),
                "overlap_top_voxels": roi_metrics.get("overlap_top_voxels"),
                "roi_volume_mm3": roi_metrics.get("roi_volume_mm3"),
                "overlap_volume_mm3": roi_metrics.get("overlap_volume_mm3"),
                "overlap_fraction": roi_metrics.get("overlap_fraction"),
            }
        )

    extended = payload.get("extended_metrics", {})
    if isinstance(extended, dict):
        for key, value in extended.items():
            if key in {"neighbors", "electrode_distances"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                record[key] = value

        for row in extended.get("neighbors", []):
            if not isinstance(row, dict):
                continue
            slug = _metric_slug(str(row.get("label_name", row.get("label_id", "neighbor"))))
            record[f"neighbor_mean__{slug}"] = row.get("mean")
            record[f"neighbor_peak__{slug}"] = row.get("max")
            record[f"neighbor_voxels__{slug}"] = row.get("voxels")

        for row in extended.get("electrode_distances", []):
            if not isinstance(row, dict):
                continue
            slug = _metric_slug(str(row.get("electrode", "electrode")))
            record[f"electrode_distance_mm__{slug}"] = row.get("distance_mm")

    return record
