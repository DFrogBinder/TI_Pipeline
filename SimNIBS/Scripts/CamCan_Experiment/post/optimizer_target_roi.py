"""Construct the optimizer-matched CamCan target ROI on a NIfTI grid.

The supervisor's MATLAB ``MakeROIs.m`` implementation constructs each target
from the volumetric centre of its anatomical FreeSurfer parcel.  A sphere is
grown from 3 mm in 0.01 mm increments until the parcel-clipped volume reaches
100 mm3 for cortical targets or 200 mm3 for subcortical targets, with a
10 mm radius cap.

The optimization inputs are tetrahedral meshes, whereas the CamCan
post-processing inputs are voxelized TI fields and subject-space atlas
NIfTIs.  On a regular NIfTI grid, the arithmetic mean of the world-space voxel
centres is the volume-weighted centroid because every voxel has the same
physical volume.  This module therefore implements the direct voxel-grid
equivalent of ``MakeROIs.m`` and records enough provenance to audit the
conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import nibabel as nib
import numpy as np

from utils.ti_utils import vol_mm3


ROI_DEFINITION_SCHEMA_VERSION = 1
START_RADIUS_MM = 3.0
RADIUS_STEP_MM = 0.01
RADIUS_CAP_MM = 10.0

CORTICAL_ROIS = frozenset({"Left_M1", "Right_DLPC"})
SUBCORTICAL_ROIS = frozenset({"Left_Hippocampus", "Right_Thalamus"})
TARGET_VOLUME_MM3_BY_ROI = {
    **{roi: 100.0 for roi in CORTICAL_ROIS},
    **{roi: 200.0 for roi in SUBCORTICAL_ROIS},
}


@dataclass(frozen=True)
class OptimizerTargetROI:
    """Mask and construction provenance for one optimizer-matched target."""

    mask: np.ndarray
    metadata: dict[str, Any]


def _world_coordinates(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (np.asarray(indices, dtype=np.float64), np.ones(len(indices)))
    )
    return (homogeneous @ np.asarray(affine, dtype=np.float64).T)[:, :3]


def optimizer_target_volume_mm3(roi: str) -> float:
    try:
        return TARGET_VOLUME_MM3_BY_ROI[roi]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CamCan ROI {roi!r}; expected one of "
            f"{sorted(TARGET_VOLUME_MM3_BY_ROI)}."
        ) from exc


def build_optimizer_target_roi(
    *,
    anatomical_mask: np.ndarray,
    reference_img: nib.spatialimages.SpatialImage,
    roi: str,
    start_radius_mm: float = START_RADIUS_MM,
    radius_step_mm: float = RADIUS_STEP_MM,
    radius_cap_mm: float = RADIUS_CAP_MM,
) -> OptimizerTargetROI:
    """Return the parcel-clipped spherical ROI used by the optimizer.

    This mirrors the loop in ``MakeROIs.m``.  The strict distance comparator
    is ``distance < radius`` and the largest evaluated radius is one step
    below the cap, just as in the MATLAB ``while SphereRadius < 10`` loop.
    """

    anatomical = np.asarray(anatomical_mask, dtype=bool)
    if anatomical.shape != tuple(reference_img.shape[:3]):
        raise ValueError(
            f"Anatomical mask shape {anatomical.shape} does not match reference "
            f"image shape {tuple(reference_img.shape[:3])}."
        )
    if not np.any(anatomical):
        raise ValueError(f"Anatomical parcel is empty for {roi}.")
    if start_radius_mm <= 0 or radius_step_mm <= 0:
        raise ValueError("Sphere start radius and radius step must be positive.")
    if radius_cap_mm <= start_radius_mm:
        raise ValueError("Sphere radius cap must exceed the start radius.")

    requested_volume = optimizer_target_volume_mm3(roi)
    voxel_volume = float(vol_mm3(reference_img))
    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError(f"Invalid NIfTI voxel volume: {voxel_volume!r}.")

    parcel_indices = np.argwhere(anatomical)
    parcel_world = _world_coordinates(parcel_indices, reference_img.affine)
    centre_world = parcel_world.mean(axis=0)
    centre_voxel = nib.affines.apply_affine(
        np.linalg.inv(reference_img.affine), centre_world
    )
    distances = np.linalg.norm(parcel_world - centre_world, axis=1)

    # Integer steps avoid accumulated floating-point drift while preserving
    # MATLAB's evaluated radii: 3.00, 3.01, ..., 9.99 mm.
    maximum_step = int(
        np.ceil((radius_cap_mm - start_radius_mm) / radius_step_mm)
    )
    selected = np.zeros(len(parcel_indices), dtype=bool)
    selected_radius = float(start_radius_mm)
    achieved_volume = 0.0
    target_reached = False
    for step in range(maximum_step):
        radius = float(start_radius_mm + step * radius_step_mm)
        if radius >= radius_cap_mm:
            break
        inside = distances < radius
        volume = float(np.count_nonzero(inside) * voxel_volume)
        selected = inside
        selected_radius = radius
        achieved_volume = volume
        if volume >= requested_volume:
            target_reached = True
            break

    target_mask = np.zeros(anatomical.shape, dtype=bool)
    chosen_indices = parcel_indices[selected]
    if len(chosen_indices):
        target_mask[tuple(chosen_indices.T)] = True
    if not np.any(target_mask):
        raise ValueError(
            f"Optimizer-matched target ROI is empty for {roi} at "
            f"{selected_radius:.2f} mm."
        )
    if np.any(target_mask & ~anatomical):
        raise AssertionError("Optimizer target escaped its anatomical parcel.")

    metadata: dict[str, Any] = {
        "roi_definition_schema_version": ROI_DEFINITION_SCHEMA_VERSION,
        "method": (
            "voxel-grid equivalent of MakeROIs.m: sphere centred on the "
            "anatomical parcel volume centroid and clipped to that parcel"
        ),
        "source_representation": "subject-space atlas resampled to TI NIfTI grid",
        "centroid_weighting": (
            "world-space voxel-centre mean; equivalent to volume weighting "
            "because NIfTI voxel volume is constant"
        ),
        "roi": roi,
        "roi_class": "cortical" if roi in CORTICAL_ROIS else "subcortical",
        "anatomical_parcel_voxels": int(np.count_nonzero(anatomical)),
        "anatomical_parcel_volume_mm3": float(
            np.count_nonzero(anatomical) * voxel_volume
        ),
        "centre_world_x_mm": float(centre_world[0]),
        "centre_world_y_mm": float(centre_world[1]),
        "centre_world_z_mm": float(centre_world[2]),
        "centre_voxel_i": float(centre_voxel[0]),
        "centre_voxel_j": float(centre_voxel[1]),
        "centre_voxel_k": float(centre_voxel[2]),
        "requested_volume_mm3": requested_volume,
        "achieved_volume_mm3": achieved_volume,
        "target_voxels": int(np.count_nonzero(target_mask)),
        "voxel_volume_mm3": voxel_volume,
        "radius_mm": selected_radius,
        "start_radius_mm": float(start_radius_mm),
        "radius_step_mm": float(radius_step_mm),
        "radius_cap_mm": float(radius_cap_mm),
        "distance_comparator": "<",
        "target_volume_reached": target_reached,
        "clipped_to_anatomical_parcel": True,
    }
    return OptimizerTargetROI(mask=target_mask, metadata=metadata)


def flatten_roi_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Prefix scalar ROI-construction metadata for CSV output."""

    return {
        f"optimizer_roi_{key}": value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }
