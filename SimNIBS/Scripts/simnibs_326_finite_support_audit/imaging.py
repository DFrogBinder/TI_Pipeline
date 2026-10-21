"""Read-only NIfTI-stage audit using the exact optimizer-matched ROI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .audit import classify_voxel_support


SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
CAMCAN_ROOT = SCRIPTS_ROOT / "CamCan_Experiment"
for _candidate in (SCRIPTS_ROOT, CAMCAN_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from CamCan_Experiment.post.optimizer_target_roi import (  # noqa: E402
    build_optimizer_target_roi,
)


def _scalar_data(image: nib.spatialimages.SpatialImage, *, label: str) -> np.ndarray:
    data = np.asarray(image.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{label} must be scalar 3D data; observed {data.shape}")
    return np.asarray(data, dtype=np.float64)


def _same_grid(left: nib.spatialimages.SpatialImage, right: nib.spatialimages.SpatialImage) -> bool:
    return left.shape[:3] == right.shape[:3] and bool(
        np.allclose(left.affine, right.affine, rtol=0.0, atol=1e-5)
    )


def audit_nifti_stages(
    *,
    raw_field_path: Path,
    masked_field_path: Path,
    tissue_labels_path: Path,
    atlas_path: Path,
    roi: str,
    atlas_labels: tuple[int, ...],
    atlas_image: nib.spatialimages.SpatialImage | None = None,
    roi_cache: dict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any], bool]] | None = None,
) -> dict[str, Any]:
    """Partition ROI support loss among interpolation, masking, and consistency."""

    raw_img = nib.load(str(raw_field_path))
    masked_img = nib.load(str(masked_field_path))
    labels_native = nib.load(str(tissue_labels_path))
    atlas_native = atlas_image if atlas_image is not None else nib.load(str(atlas_path))
    if not _same_grid(raw_img, masked_img):
        raise ValueError(
            "Raw and final fields differ in grid; the final file cannot be audited "
            "against its purported source without an undocumented interpolation"
        )

    labels_resampled = not _same_grid(labels_native, raw_img)
    atlas_resampled = not _same_grid(atlas_native, raw_img)
    labels_img = (
        resample_from_to(labels_native, raw_img, order=0)
        if labels_resampled
        else labels_native
    )
    raw = _scalar_data(raw_img, label="raw field")
    masked = _scalar_data(masked_img, label="masked field")
    tissue_labels = np.rint(_scalar_data(labels_img, label="tissue labels")).astype(
        np.int32,
        copy=False,
    )
    cache_key = (
        tuple(int(value) for value in raw_img.shape[:3]),
        tuple(np.asarray(raw_img.affine, dtype=float).round(7).ravel()),
        roi,
        atlas_labels,
    )
    cached = None if roi_cache is None else roi_cache.get(cache_key)
    if cached is None:
        atlas_img = (
            resample_from_to(atlas_native, raw_img, order=0)
            if atlas_resampled
            else atlas_native
        )
        atlas = np.rint(_scalar_data(atlas_img, label="atlas")).astype(
            np.int32, copy=False
        )
        missing_labels = [label for label in atlas_labels if not np.any(atlas == label)]
        if missing_labels:
            raise ValueError(f"Atlas lacks required labels {missing_labels}: {atlas_path}")
        anatomical_mask = np.isin(atlas, atlas_labels)
        optimizer_roi = build_optimizer_target_roi(
            anatomical_mask=anatomical_mask,
            reference_img=raw_img,
            roi=roi,
        )
        roi_mask = optimizer_roi.mask
        roi_metadata = optimizer_roi.metadata
        if roi_cache is not None:
            roi_cache[cache_key] = (roi_mask, roi_metadata, atlas_resampled)
    else:
        roi_mask, roi_metadata, atlas_resampled = cached
    support = classify_voxel_support(
        roi_mask=roi_mask,
        raw_field=raw,
        masked_field=masked,
        tissue_labels=tissue_labels,
    )
    roi_voxels = int(support["roi_voxels"])
    support["finite_roi_fraction"] = (
        float(support["final_finite_voxels"]) / roi_voxels if roi_voxels else 0.0
    )
    return {
        "support": support,
        "roi_metadata": roi_metadata,
        "grid": {
            "shape": [int(value) for value in raw_img.shape[:3]],
            "affine": np.asarray(raw_img.affine, dtype=float).round(8).tolist(),
            "raw_masked_same_grid": True,
            "tissue_labels_resampled_nearest": labels_resampled,
            "atlas_resampled_nearest": atlas_resampled,
        },
        "diagnostic": {
            "reference_img": raw_img,
            "raw_field": raw,
            "masked_field": masked,
            "tissue_labels": tissue_labels,
            "roi_mask": roi_mask,
            "raw_nonfinite_mask": roi_mask & ~np.isfinite(raw),
            "mask_excluded_mask": (
                roi_mask
                & np.isfinite(raw)
                & ~np.isfinite(masked)
                & ~np.isin(tissue_labels, (1, 2))
            ),
            "unexplained_mask": (
                roi_mask
                & np.isfinite(raw)
                & ~np.isfinite(masked)
                & np.isin(tissue_labels, (1, 2))
            ),
            "inconsistent_mask": (
                (
                    roi_mask
                    & np.isfinite(raw)
                    & ~np.isfinite(masked)
                    & np.isin(tissue_labels, (1, 2))
                )
                | (roi_mask & np.isfinite(masked) & ~np.isfinite(raw))
                | (roi_mask & np.isfinite(masked) & ~np.isin(tissue_labels, (1, 2)))
                | (
                    roi_mask
                    & np.isfinite(masked)
                    & np.isfinite(raw)
                    & np.isin(tissue_labels, (1, 2))
                    & ~np.isclose(raw, masked, rtol=1e-6, atol=1e-8)
                )
            ),
        },
    }
