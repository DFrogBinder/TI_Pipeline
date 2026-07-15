from __future__ import annotations

import hashlib
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

from post.eeg_positions import read_eeg_positions

from utils.roi_registry import (
    FASTSURFER_DKT_LABELS,
    resolve_fastsurfer_roi_label_ids,
)
from utils.ti_utils import (
    load_ti_as_scalar,
    normalize_roi_name,
    resample_atlas_to_ti_grid,
    vol_mm3,
)


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


EXTENDED_METRIC_SCHEMA_VERSION = 6
WHOLE_BRAIN_COVERAGE_THRESHOLD_V_PER_M = 0.2
WHOLE_BRAIN_COVERAGE_METRIC_KEYS = (
    "whole_brain_coverage_threshold_v_per_m",
    "whole_brain_coverage_voxels_ge_threshold",
    "whole_brain_coverage_volume_mm3_ge_threshold",
    "whole_brain_coverage_percent_ge_threshold",
)
EXTENDED_METRIC_LIST_FIELDS = ("neighbors", "electrode_distances")
EXTENDED_METRIC_SCALAR_FIELDS = (
    "roi_peak",
    "roi_mean",
    "roi_peak_abs_delta_mni",
    "roi_mean_abs_delta_mni",
    "mni_baseline_roi_peak",
    "mni_baseline_roi_mean",
    "focality_threshold_v_per_m",
    "focality_voxels_gt_threshold",
    "focality_volume_mm3_gt_threshold",
    "focality_percent_of_whole_brain_gt_threshold",
    "focality_voxels_abs_delta_mni",
    "focality_volume_mm3_abs_delta_mni",
    "mni_baseline_focality_voxels_gt_threshold",
    "mni_baseline_focality_volume_mm3_gt_threshold",
    *WHOLE_BRAIN_COVERAGE_METRIC_KEYS,
    "neighbor_template_count",
    "neighbor_mean_of_means",
    "neighbor_max_of_max",
    "neighbor_min_of_max",
    "roi_centroid_x",
    "roi_centroid_y",
    "roi_centroid_z",
    "csf_distance_mm",
    "skull_distance_mm",
    "electrode_distance_count",
    "electrode_distance_mean_mm",
    "electrode_distance_min_mm",
    "electrode_distance_max_mm",
)
EXTENDED_METRIC_FIELDS = EXTENDED_METRIC_SCALAR_FIELDS + EXTENDED_METRIC_LIST_FIELDS
ROI_INTENSITY_METRIC_KEYS = ("roi_peak", "roi_mean")
BASELINE_METRIC_KEYS = (
    "roi_peak_abs_delta_mni",
    "roi_mean_abs_delta_mni",
    "mni_baseline_roi_peak",
    "mni_baseline_roi_mean",
    "mni_baseline_focality_voxels_gt_threshold",
    "mni_baseline_focality_volume_mm3_gt_threshold",
    "focality_voxels_abs_delta_mni",
    "focality_volume_mm3_abs_delta_mni",
)
FOCALITY_METRIC_KEYS = (
    "focality_threshold_v_per_m",
    "focality_voxels_gt_threshold",
    "focality_volume_mm3_gt_threshold",
    "focality_percent_of_whole_brain_gt_threshold",
)
NEIGHBOR_METRIC_KEYS = (
    "neighbor_template_count",
    "neighbors",
    "neighbor_mean_of_means",
    "neighbor_max_of_max",
    "neighbor_min_of_max",
)
CENTROID_METRIC_KEYS = ("roi_centroid_x", "roi_centroid_y", "roi_centroid_z")
ANATOMY_DISTANCE_METRIC_KEYS = ("csf_distance_mm", "skull_distance_mm")
ELECTRODE_METRIC_KEYS = (
    "electrode_distances",
    "electrode_distance_count",
    "electrode_distance_mean_mm",
    "electrode_distance_min_mm",
    "electrode_distance_max_mm",
)


def build_extended_metrics_scaffold() -> Dict[str, Any]:
    metrics = {key: None for key in EXTENDED_METRIC_SCALAR_FIELDS}
    metrics.update({key: [] for key in EXTENDED_METRIC_LIST_FIELDS})
    return metrics


def build_extended_metric_status_scaffold() -> Dict[str, str]:
    return {key: "pending" for key in EXTENDED_METRIC_FIELDS}


def build_extended_metric_message_scaffold() -> Dict[str, Optional[str]]:
    return {key: None for key in EXTENDED_METRIC_FIELDS}


def json_ready_metric_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_ready_metric_value(entry) for entry in value]
    if isinstance(value, dict):
        return {str(key): json_ready_metric_value(entry) for key, entry in value.items()}
    return value


def extended_metrics_config_fingerprint(
    *,
    root_dir: str,
    subject: str,
    ti_path: Optional[str],
    atlas_mode: str,
    fastsurfer_root: Optional[str],
    subject_fastsurfer_atlas_path: Optional[str],
    roi_name: str,
    percentile: float,
    region_percentile: float,
    focality_threshold: float,
    mni_baseline_root: Optional[str],
    mni_fixed_atlas_path: Optional[str],
    neighbor_dilation_iter: int,
    write_neighbor_visualization: bool,
    csf_labels: Optional[Sequence[int]],
    skull_labels: Optional[Sequence[int]],
    electrode_csv: Optional[str],
    electrode_dataset_dir: Optional[str],
    electrode_names: Optional[Sequence[str]],
    eeg_positions_path_template: Optional[str],
) -> str:
    def _norm_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return str(Path(value).expanduser().resolve())

    payload = {
        "root_dir": _norm_path(root_dir),
        "extended_metric_schema_version": EXTENDED_METRIC_SCHEMA_VERSION,
        "subject": subject,
        "ti_path": _norm_path(ti_path),
        "atlas_mode": atlas_mode,
        "fastsurfer_root": _norm_path(fastsurfer_root),
        "subject_fastsurfer_atlas_path": _norm_path(subject_fastsurfer_atlas_path),
        "roi_name": roi_name,
        "percentile": float(percentile),
        "region_percentile": float(region_percentile),
        "focality_threshold": float(focality_threshold),
        "mni_baseline_root": _norm_path(mni_baseline_root),
        "mni_fixed_atlas_path": _norm_path(mni_fixed_atlas_path),
        "neighbor_dilation_iter": int(neighbor_dilation_iter),
        "write_neighbor_visualization": bool(write_neighbor_visualization),
        "csf_labels": sorted(int(value) for value in csf_labels) if csf_labels else None,
        "skull_labels": sorted(int(value) for value in skull_labels) if skull_labels else None,
        "electrode_csv": _norm_path(electrode_csv),
        "electrode_dataset_dir": _norm_path(electrode_dataset_dir),
        "electrode_names": list(electrode_names) if electrode_names else None,
        "eeg_positions_path_template": eeg_positions_path_template,
    }
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=None)
def _canonical_label_ids(roi_name: str) -> Tuple[int, ...]:
    label_ids = resolve_fastsurfer_roi_label_ids(roi_name)
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

    focality_mask = finite_mask & (ti_data >= focality_threshold)
    focality_voxels = int(np.sum(focality_mask))
    whole_brain_voxels = int(np.sum(finite_mask))
    focality_volume_mm3 = float(focality_voxels * voxel_volume_mm3)
    return {
        "roi_peak": roi_peak,
        "roi_mean": roi_mean,
        "focality_voxels_gt_threshold": float(focality_voxels),
        "focality_volume_mm3_gt_threshold": focality_volume_mm3,
        "focality_percent_of_whole_brain_gt_threshold": (
            float((focality_voxels / whole_brain_voxels) * 100.0)
            if whole_brain_voxels
            else math.nan
        ),
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
    if not baseline_root.exists():
        raise FileNotFoundError(
            f"MNI baseline root not found: {baseline_root}. "
            "Set mni_baseline_root to a directory containing an MNI subject with "
            "anat/SimNIBS/ti_brain_only.nii.gz."
        )
    if not atlas_path.is_file():
        raise FileNotFoundError(
            f"Fixed MNI FastSurfer atlas not found: {atlas_path}. "
            "Set mni_fixed_atlas_path to the atlas used to define the baseline ROI mask."
        )

    candidates = _baseline_subject_root_candidates(baseline_root)
    if not candidates:
        raise FileNotFoundError(
            f"No MNI baseline TI file found below {baseline_root}. Expected either "
            "anat/SimNIBS/ti_brain_only.nii.gz directly below the root, or below a "
            "subject directory such as MNI152/anat/SimNIBS/ti_brain_only.nii.gz."
        )

    for subject_root, subject_name in candidates:
        ti_path = subject_root / "anat" / "SimNIBS" / "ti_brain_only.nii.gz"
        if not ti_path.is_file():
            continue

        from post.post_functions import roi_masks_on_ti_grid

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
        if not np.any(roi_mask):
            raise ValueError(
                f"MNI baseline ROI '{roi_name}' resolved to an empty mask using atlas '{atlas_path}'."
            )
        return _compute_core_field_metrics(
            ti_img=ti_img,
            ti_data=ti_data,
            roi_mask=roi_mask,
            finite_mask=finite_mask,
            focality_threshold=focality_threshold,
        )

    raise ValueError(
        f"Could not build MNI baseline metrics for ROI '{roi_name}' from root "
        f"'{baseline_root}' and atlas '{atlas_path}'."
    )


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


def build_fixed_neighbor_masks(
    *,
    mni_fixed_atlas_path: Optional[str],
    roi_name: str,
    dilation_iter: int,
    subject_atlas_data: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Build subject-space masks for the exact fixed-template neighbors used by
    ``compute_neighbor_metrics``.

    The neighbor labels are selected in fixed MNI atlas space, then projected
    into the subject by reusing the same label ids in the subject FastSurfer
    atlas on the TI grid. The ROI labels are explicitly excluded so composite
    ROIs do not bleed into the neighbor visualization.
    """
    if subject_atlas_data is None:
        raise ValueError("Subject atlas data are required to build neighbor visualization masks.")

    neighbor_template = fixed_neighbor_template(
        mni_fixed_atlas_path,
        roi_name,
        dilation_iter,
    )
    neighbor_ids = tuple(int(row["label_id"]) for row in neighbor_template)
    atlas_data = np.asarray(subject_atlas_data).astype(np.int32, copy=False)
    roi_ids = set(_canonical_label_ids(roi_name))
    neighbor_mask = np.isin(atlas_data, neighbor_ids)
    if roi_ids:
        neighbor_mask &= ~np.isin(atlas_data, tuple(roi_ids))

    categorical_mask = np.where(neighbor_mask, atlas_data, 0).astype(np.int32, copy=False)
    return {
        "neighbor_template": neighbor_template,
        "neighbor_label_ids": list(neighbor_ids),
        "neighbor_union_mask": neighbor_mask.astype(bool, copy=False),
        "neighbor_categorical_mask": categorical_mask,
    }


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


def _electrode_dataset_roi_keys(roi_name: Optional[str]) -> List[str]:
    if not roi_name:
        return []

    normalized = normalize_roi_name(roi_name)
    snake = re.sub(r"[^0-9a-zA-Z]+", "_", roi_name.strip().lower()).strip("_")
    compact = snake.replace("_", "")
    candidates = [normalized, snake, normalized.replace("_", "-"), snake.replace("_", "-")]

    if "hippocampus" in snake:
        if snake.startswith("left") or "_left" in snake or "lh_" in snake:
            candidates.append("left-hippocampus")
        if snake.startswith("right") or "_right" in snake or "rh_" in snake:
            candidates.append("right-hippocampus")
    if "thalamus" in snake:
        if snake.startswith("right") or "_right" in snake or "rh_" in snake:
            candidates.append("right-thalamus")
        if snake.startswith("left") or "_left" in snake or "lh_" in snake:
            candidates.append("left-thalamus")
    if "pallidum" in snake:
        if snake.startswith("right") or "_right" in snake or "rh_" in snake:
            candidates.append("right-pallidum")
        if snake.startswith("left") or "_left" in snake or "lh_" in snake:
            candidates.append("left-pallidum")
    if "precentral" in snake or compact in {"leftm1", "lhm1", "m1left", "rightm1", "rhm1", "m1right"}:
        if snake.startswith(("ctx_lh", "ctx-lh", "left", "lh")) or "left" in snake:
            candidates.append("left-m1")
        if snake.startswith(("ctx_rh", "ctx-rh", "right", "rh")) or "right" in snake:
            candidates.append("right-m1")
    if "front_middle" in snake or "dlpfc" in snake or "dlpc" in snake:
        if snake.startswith(("ctx_rh", "ctx-rh", "right", "rh")) or "right" in snake:
            candidates.append("right-dlpc")
        if snake.startswith(("ctx_lh", "ctx-lh", "left", "lh")) or "left" in snake:
            candidates.append("left-dlpc")

    out: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _load_electrode_centers_from_dataset_dir(
    *,
    path: Path,
    subject: str,
    roi_name: Optional[str],
) -> List[Tuple[str, np.ndarray]]:
    root = path.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Electrode dataset directory not found: {root}")

    candidates: List[Path] = []
    for roi_key in _electrode_dataset_roi_keys(roi_name):
        roi_dir = root / roi_key
        candidates.extend(
            [
                roi_dir / subject / "electrodes.csv",
                roi_dir / f"{subject}.csv",
                roi_dir / "electrode_centers.csv",
            ]
        )
    candidates.extend(
        [
            root / subject / "electrodes.csv",
            root / f"{subject}.csv",
            root / "electrode_centers.csv",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        centers = _load_electrode_centers(candidate).get(subject, [])
        if centers:
            return centers
    return []


def _read_eeg_positions(path: Path) -> Dict[str, np.ndarray]:
    return read_eeg_positions(path)


def resolve_electrode_centers(
    *,
    root_dir: str,
    subject: str,
    roi_name: Optional[str],
    electrode_csv: Optional[str],
    electrode_dataset_dir: Optional[str],
    electrode_names: Optional[Sequence[str]],
    eeg_positions_path_template: Optional[str],
) -> List[Tuple[str, np.ndarray]]:
    if electrode_csv:
        path = Path(electrode_csv).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Electrode CSV not found: {path}. Required columns are "
                "subject,electrode,x,y,z, with x/y/z in millimetres in the same "
                "world coordinate frame as the TI image."
        )
        return _load_electrode_centers(path).get(subject, [])

    if electrode_dataset_dir:
        return _load_electrode_centers_from_dataset_dir(
            path=Path(electrode_dataset_dir),
            subject=subject,
            roi_name=roi_name,
        )

    if not electrode_names:
        return []

    if eeg_positions_path_template:
        eeg_path = Path(
            eeg_positions_path_template.format(root=root_dir, subject=subject)
        ).expanduser()
    else:
        eeg_path = Path(root_dir) / subject / "anat" / f"m2m_{subject}" / "eeg_positions.csv"

    positions = _read_eeg_positions(eeg_path)
    missing = [name for name in electrode_names if name not in positions]
    if missing:
        raise ValueError(
            f"EEG positions file {eeg_path} is missing requested electrode(s): "
            + ", ".join(missing)
        )
    return [(name, positions[name]) for name in electrode_names]


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


def load_subject_fastsurfer_atlas_data(
    ti_img: nib.Nifti1Image,
    subject_fastsurfer_atlas_path: Optional[str],
) -> Optional[np.ndarray]:
    if not subject_fastsurfer_atlas_path:
        return None
    atlas_path = Path(subject_fastsurfer_atlas_path).expanduser()
    if not atlas_path.is_file():
        raise FileNotFoundError(f"FastSurfer atlas not found: {atlas_path}")
    atlas_img = resample_atlas_to_ti_grid(nib.load(str(atlas_path)), ti_img)
    return np.asarray(atlas_img.dataobj).astype(np.int32)


def compute_roi_intensity_metrics(
    *,
    ti_img: nib.Nifti1Image,
    ti_data: np.ndarray,
    roi_mask: np.ndarray,
    finite_mask: np.ndarray,
) -> Dict[str, float]:
    roi_vals = ti_data[roi_mask & finite_mask]
    roi_peak = float(np.max(roi_vals)) if roi_vals.size else math.nan
    roi_mean = float(np.mean(roi_vals)) if roi_vals.size else math.nan
    return {
        "roi_peak": roi_peak,
        "roi_mean": roi_mean,
    }


def compute_focality_metrics(
    *,
    ti_img: nib.Nifti1Image,
    ti_data: np.ndarray,
    finite_mask: np.ndarray,
    focality_threshold: float,
) -> Dict[str, float]:
    voxel_volume_mm3 = vol_mm3(ti_img)
    focality_mask = finite_mask & (ti_data >= focality_threshold)
    focality_voxels = int(np.sum(focality_mask))
    whole_brain_voxels = int(np.sum(finite_mask))
    return {
        "focality_threshold_v_per_m": float(focality_threshold),
        "focality_voxels_gt_threshold": focality_voxels,
        "focality_volume_mm3_gt_threshold": float(focality_voxels * voxel_volume_mm3),
        "focality_percent_of_whole_brain_gt_threshold": (
            float((focality_voxels / whole_brain_voxels) * 100.0)
            if whole_brain_voxels
            else math.nan
        ),
    }


def compute_whole_brain_coverage_metrics(
    *,
    ti_img: nib.Nifti1Image,
    ti_data: np.ndarray,
    finite_mask: np.ndarray,
    threshold: float = WHOLE_BRAIN_COVERAGE_THRESHOLD_V_PER_M,
) -> Dict[str, float]:
    voxel_volume_mm3 = vol_mm3(ti_img)
    coverage_mask = finite_mask & (ti_data >= threshold)
    coverage_voxels = int(np.sum(coverage_mask))
    whole_brain_voxels = int(np.sum(finite_mask))
    return {
        "whole_brain_coverage_threshold_v_per_m": float(threshold),
        "whole_brain_coverage_voxels_ge_threshold": coverage_voxels,
        "whole_brain_coverage_volume_mm3_ge_threshold": float(coverage_voxels * voxel_volume_mm3),
        "whole_brain_coverage_percent_ge_threshold": (
            float((coverage_voxels / whole_brain_voxels) * 100.0)
            if whole_brain_voxels
            else math.nan
        ),
    }


def compute_baseline_delta_metrics(
    *,
    roi_name: str,
    mni_baseline_root: Optional[str],
    mni_fixed_atlas_path: Optional[str],
    focality_threshold: float,
    roi_peak: Any,
    roi_mean: Any,
    focality_voxels_gt_threshold: Any,
    focality_volume_mm3_gt_threshold: Any,
) -> Dict[str, float]:
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
    roi_peak_value = _safe_float(roi_peak)
    roi_mean_value = _safe_float(roi_mean)
    focality_voxels_value = _safe_float(focality_voxels_gt_threshold)
    focality_volume_value = _safe_float(focality_volume_mm3_gt_threshold)
    return {
        "roi_peak_abs_delta_mni": (
            abs(roi_peak_value - baseline_peak)
            if math.isfinite(roi_peak_value) and math.isfinite(baseline_peak)
            else math.nan
        ),
        "roi_mean_abs_delta_mni": (
            abs(roi_mean_value - baseline_mean)
            if math.isfinite(roi_mean_value) and math.isfinite(baseline_mean)
            else math.nan
        ),
        "mni_baseline_roi_peak": baseline_peak,
        "mni_baseline_roi_mean": baseline_mean,
        "focality_voxels_abs_delta_mni": (
            abs(focality_voxels_value - baseline_focality_voxels)
            if math.isfinite(focality_voxels_value) and math.isfinite(baseline_focality_voxels)
            else math.nan
        ),
        "focality_volume_mm3_abs_delta_mni": (
            abs(focality_volume_value - baseline_focality_volume)
            if math.isfinite(focality_volume_value) and math.isfinite(baseline_focality_volume)
            else math.nan
        ),
        "mni_baseline_focality_voxels_gt_threshold": baseline_focality_voxels,
        "mni_baseline_focality_volume_mm3_gt_threshold": baseline_focality_volume,
    }


def compute_neighbor_metrics(
    *,
    mni_fixed_atlas_path: Optional[str],
    roi_name: str,
    dilation_iter: int,
    subject_atlas_data: Optional[np.ndarray],
    ti_data: np.ndarray,
    finite_mask: np.ndarray,
    voxel_volume_mm3: float,
) -> Dict[str, Any]:
    neighbor_template = fixed_neighbor_template(
        mni_fixed_atlas_path,
        roi_name,
        dilation_iter,
    )
    metrics: Dict[str, Any] = {
        "neighbor_template_count": len(neighbor_template),
        "neighbors": [],
        "neighbor_mean_of_means": math.nan,
        "neighbor_max_of_max": math.nan,
        "neighbor_min_of_max": math.nan,
    }
    if subject_atlas_data is None or not neighbor_template:
        return metrics

    neighbor_rows = _neighbor_stats(
        neighbor_template,
        subject_atlas_data,
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

    for row in neighbor_rows:
        slug = _metric_slug(str(row["label_name"]))
        metrics[f"neighbor_mean__{slug}"] = _safe_float(row.get("mean"))
        metrics[f"neighbor_peak__{slug}"] = _safe_float(row.get("max"))
        metrics[f"neighbor_voxels__{slug}"] = _safe_float(row.get("voxels"))
    return metrics


def compute_centroid_metrics(
    roi_mask: np.ndarray,
    affine: np.ndarray,
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    roi_centroid_ijk = _roi_centroid_ijk(roi_mask)
    roi_centroid_xyz = _roi_centroid_world(roi_mask, affine)
    metrics = {
        "roi_centroid_x": float(roi_centroid_xyz[0]) if roi_centroid_xyz is not None else math.nan,
        "roi_centroid_y": float(roi_centroid_xyz[1]) if roi_centroid_xyz is not None else math.nan,
        "roi_centroid_z": float(roi_centroid_xyz[2]) if roi_centroid_xyz is not None else math.nan,
    }
    return metrics, roi_centroid_ijk, roi_centroid_xyz


def compute_anatomy_distance_metrics(
    *,
    atlas_data: Optional[np.ndarray],
    roi_centroid_ijk: Optional[np.ndarray],
    zooms: Tuple[float, float, float],
    csf_labels: Optional[Sequence[int]],
    skull_labels: Optional[Sequence[int]],
) -> Dict[str, float]:
    csf_dist = math.nan
    skull_dist = math.nan
    if atlas_data is not None and roi_centroid_ijk is not None:
        if csf_labels:
            csf_mask = np.isin(atlas_data, list(csf_labels))
            if np.any(csf_mask):
                csf_dist = _distance_to_tissue(csf_mask, roi_centroid_ijk, zooms)
        if skull_labels:
            skull_mask = np.isin(atlas_data, list(skull_labels))
            if np.any(skull_mask):
                skull_dist = _distance_to_tissue(skull_mask, roi_centroid_ijk, zooms)
    return {
        "csf_distance_mm": csf_dist,
        "skull_distance_mm": skull_dist,
    }


def compute_electrode_distance_metrics(
    *,
    root_dir: str,
    subject: str,
    roi_name: Optional[str],
    roi_centroid_xyz: Optional[np.ndarray],
    electrode_csv: Optional[str],
    electrode_dataset_dir: Optional[str],
    electrode_names: Optional[Sequence[str]],
    eeg_positions_path_template: Optional[str],
) -> Dict[str, Any]:
    electrode_entries: List[Dict[str, Any]] = []
    if roi_centroid_xyz is not None:
        centers = resolve_electrode_centers(
            root_dir=root_dir,
            subject=subject,
            roi_name=roi_name,
            electrode_csv=electrode_csv,
            electrode_dataset_dir=electrode_dataset_dir,
            electrode_names=electrode_names,
            eeg_positions_path_template=eeg_positions_path_template,
        )
        if (electrode_csv or electrode_dataset_dir or electrode_names) and not centers:
            raise ValueError(
                f"No electrode centres were found for subject '{subject}'. "
                "For electrode_csv, provide rows with columns subject,electrode,x,y,z. "
                "For electrode_dataset_dir, provide <roi>/<subject>/electrodes.csv "
                "or <roi>/electrode_centers.csv with the same columns. "
                "For electrode_names, ensure the names exist in eeg_positions.csv or in "
                "the configured eeg_positions_path_template."
            )
        for name, coord in centers:
            electrode_entries.append(
                {
                    "electrode": name,
                    "distance_mm": float(np.linalg.norm(coord - roi_centroid_xyz)),
                }
            )

    metrics: Dict[str, Any] = {"electrode_distances": electrode_entries}
    metrics.update(summarise_electrode_distances(electrode_entries))
    return metrics


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
    electrode_dataset_dir: Optional[str],
    electrode_names: Optional[Sequence[str]],
    eeg_positions_path_template: Optional[str],
) -> Dict[str, Any]:
    voxel_volume_mm3 = vol_mm3(ti_img)
    metrics: Dict[str, Any] = {}
    metrics.update(
        compute_roi_intensity_metrics(
            ti_img=ti_img,
            ti_data=ti_data,
            roi_mask=roi_mask,
            finite_mask=finite_mask,
        )
    )
    metrics.update(
        compute_focality_metrics(
            ti_img=ti_img,
            ti_data=ti_data,
            finite_mask=finite_mask,
            focality_threshold=focality_threshold,
        )
    )
    metrics.update(
        compute_baseline_delta_metrics(
            roi_name=roi_name,
            mni_baseline_root=mni_baseline_root,
            mni_fixed_atlas_path=mni_fixed_atlas_path,
            focality_threshold=focality_threshold,
            roi_peak=metrics.get("roi_peak"),
            roi_mean=metrics.get("roi_mean"),
            focality_voxels_gt_threshold=metrics.get("focality_voxels_gt_threshold"),
            focality_volume_mm3_gt_threshold=metrics.get("focality_volume_mm3_gt_threshold"),
        )
    )

    atlas_data = load_subject_fastsurfer_atlas_data(ti_img, subject_fastsurfer_atlas_path)
    metrics.update(
        compute_neighbor_metrics(
            mni_fixed_atlas_path=mni_fixed_atlas_path,
            roi_name=roi_name,
            dilation_iter=neighbor_dilation_iter,
            subject_atlas_data=atlas_data,
            ti_data=ti_data,
            finite_mask=finite_mask,
            voxel_volume_mm3=voxel_volume_mm3,
        )
    )
    centroid_metrics, roi_centroid_ijk, roi_centroid_xyz = compute_centroid_metrics(roi_mask, ti_img.affine)
    metrics.update(centroid_metrics)
    metrics.update(
        compute_anatomy_distance_metrics(
            atlas_data=atlas_data,
            roi_centroid_ijk=roi_centroid_ijk,
            zooms=ti_img.header.get_zooms()[:3],
            csf_labels=csf_labels,
            skull_labels=skull_labels,
        )
    )
    metrics.update(
        compute_electrode_distance_metrics(
            root_dir=root_dir,
            subject=subject,
            roi_name=roi_name,
            roi_centroid_xyz=roi_centroid_xyz,
            electrode_csv=electrode_csv,
            electrode_dataset_dir=electrode_dataset_dir,
            electrode_names=electrode_names,
            eeg_positions_path_template=eeg_positions_path_template,
        )
    )

    return metrics


def flatten_subject_metric_payload(payload: Dict[str, Any], roi_key: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "subject": payload.get("subject"),
        "target_roi": payload.get("target_roi"),
        "percentile": payload.get("percentile"),
        "percentile_value": payload.get("percentile_value"),
        "voxel_volume_mm3": payload.get("voxel_volume_mm3"),
        "whole_brain_voxels": payload.get("whole_brain_voxels"),
        "whole_brain_volume_mm3": payload.get("whole_brain_volume_mm3"),
        "top_percentile_voxels": payload.get("top_percentile_voxels"),
        "top_percentile_percent_of_whole_brain": payload.get("top_percentile_percent_of_whole_brain"),
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
                "roi_percentile_value": roi_metrics.get("roi_percentile_value"),
                "roi_percent_of_whole_brain": roi_metrics.get("roi_percent_of_whole_brain"),
                "overlap_top_percent_of_whole_brain": roi_metrics.get("overlap_top_percent_of_whole_brain"),
                "focality_in_roi_voxels_gt_threshold": roi_metrics.get("focality_in_roi_voxels_gt_threshold"),
                "focality_in_roi_volume_mm3_gt_threshold": roi_metrics.get("focality_in_roi_volume_mm3_gt_threshold"),
                "focality_in_roi_percent_of_whole_brain_gt_threshold": roi_metrics.get(
                    "focality_in_roi_percent_of_whole_brain_gt_threshold"
                ),
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
