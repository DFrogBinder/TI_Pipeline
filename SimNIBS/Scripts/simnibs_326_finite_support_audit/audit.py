#!/usr/bin/env python3
"""Audit where optimizer-ROI field support is lost without modifying sources."""

from __future__ import annotations

from typing import Any

import numpy as np


BRAIN_TISSUE_LABELS = (1, 2)


def classify_voxel_support(
    *,
    roi_mask: np.ndarray,
    raw_field: np.ndarray,
    masked_field: np.ndarray,
    tissue_labels: np.ndarray,
) -> dict[str, Any]:
    """Partition final ROI support loss by the stage at which it appears."""

    roi = np.asarray(roi_mask, dtype=bool)
    raw = np.asarray(raw_field, dtype=float)
    masked = np.asarray(masked_field, dtype=float)
    labels = np.asarray(tissue_labels)
    shapes = {roi.shape, raw.shape, masked.shape, labels.shape}
    if len(shapes) != 1:
        raise ValueError(f"Support arrays must share one shape; observed {sorted(shapes)}")

    raw_finite = np.isfinite(raw)
    final_finite = np.isfinite(masked)
    brain_mask = np.isin(labels, BRAIN_TISSUE_LABELS)
    finite_value_matches = np.isclose(
        raw,
        masked,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=False,
    )
    roi_labels, roi_label_counts = np.unique(labels[roi], return_counts=True)

    return {
        "roi_voxels": int(np.count_nonzero(roi)),
        "final_finite_voxels": int(np.count_nonzero(roi & final_finite)),
        "raw_nonfinite_voxels": int(np.count_nonzero(roi & ~raw_finite)),
        "mask_excluded_voxels": int(
            np.count_nonzero(roi & raw_finite & ~final_finite & ~brain_mask)
        ),
        "unexplained_nonfinite_voxels": int(
            np.count_nonzero(roi & raw_finite & ~final_finite & brain_mask)
        ),
        "finite_outside_brain_mask_voxels": int(
            np.count_nonzero(roi & final_finite & ~brain_mask)
        ),
        "final_finite_raw_nonfinite_voxels": int(
            np.count_nonzero(roi & final_finite & ~raw_finite)
        ),
        "finite_value_mismatch_voxels": int(
            np.count_nonzero(
                roi & final_finite & raw_finite & brain_mask & ~finite_value_matches
            )
        ),
        "tissue_label_counts": {
            str(int(label)): int(count)
            for label, count in zip(roi_labels, roi_label_counts, strict=True)
        },
    }


def classify_run_cause(
    *,
    mesh_nonfinite_voxels: int,
    support: dict[str, Any],
) -> str:
    """Name the earliest observed boundary that can explain lost ROI support."""

    if int(mesh_nonfinite_voxels) > 0:
        return "mesh_field_nonfinite"
    unexplained = int(support["unexplained_nonfinite_voxels"])
    finite_outside = int(support["finite_outside_brain_mask_voxels"])
    final_from_nonfinite = int(support.get("final_finite_raw_nonfinite_voxels", 0))
    value_mismatch = int(support.get("finite_value_mismatch_voxels", 0))
    if unexplained > 0 or finite_outside > 0 or final_from_nonfinite > 0 or value_mismatch > 0:
        return "masked_output_inconsistent"
    raw_nonfinite = int(support["raw_nonfinite_voxels"])
    mask_excluded = int(support["mask_excluded_voxels"])
    if raw_nonfinite > 0 and mask_excluded > 0:
        return "mixed_interpolation_and_mask_exclusion"
    if raw_nonfinite > 0:
        return "interpolation_nonfinite"
    if mask_excluded > 0:
        return "brain_mask_exclusion"
    return "complete_support"
