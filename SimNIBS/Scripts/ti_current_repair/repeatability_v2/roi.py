"""Volume-aware target and under-anode ROI construction on tetrahedra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .precision import require_float64


TARGET_VOLUMES_MM3 = {"left_hippocampus": 200.0, "right_m1": 100.0}


@dataclass(frozen=True)
class TetrahedralROI:
    mask: np.ndarray
    provenance: dict[str, Any]


def _validate_geometry(
    centres: object, volumes: object, membership: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre_array = require_float64(centres, "tetrahedron centres")
    volume_array = require_float64(volumes, "tetrahedron volumes")
    mask = np.asarray(membership, dtype=bool)
    if centre_array.ndim != 2 or centre_array.shape[1] != 3:
        raise ValueError("tetrahedron centres must have shape N x 3")
    if volume_array.shape != (len(centre_array),) or mask.shape != volume_array.shape:
        raise ValueError("tetrahedron geometry and membership shapes differ")
    if not np.isfinite(centre_array).all() or not np.isfinite(volume_array).all():
        raise ValueError("tetrahedron geometry must be finite")
    if np.any(volume_array <= 0):
        raise ValueError("tetrahedron volumes must be positive")
    if not np.any(mask):
        raise ValueError("ROI source membership is empty")
    return centre_array, volume_array, mask


def _grow_sphere(
    *,
    centres: np.ndarray,
    volumes: np.ndarray,
    allowed: np.ndarray,
    centre: np.ndarray,
    requested_volume_mm3: float,
    start_radius_mm: float,
    radius_step_mm: float,
    radius_cap_mm: float,
) -> tuple[np.ndarray, float, float, bool]:
    distances = np.linalg.norm(centres - centre, axis=1)
    steps = int(np.ceil((radius_cap_mm - start_radius_mm) / radius_step_mm))
    selected = np.zeros(len(centres), dtype=bool)
    achieved = 0.0
    radius = float(start_radius_mm)
    reached = False
    for step in range(steps):
        radius = float(start_radius_mm + step * radius_step_mm)
        if radius >= radius_cap_mm:
            break
        selected = allowed & (distances < radius)
        achieved = float(np.sum(volumes[selected], dtype=np.float64))
        if achieved >= requested_volume_mm3:
            reached = True
            break
    if not np.any(selected):
        raise ValueError("ROI sphere has no tetrahedral support")
    return selected, radius, achieved, reached


def build_parcel_clipped_roi(
    *,
    centres: object,
    volumes: object,
    parcel_mask: object,
    target: str,
    start_radius_mm: float = 3.0,
    radius_step_mm: float = 0.01,
    radius_cap_mm: float = 10.0,
) -> TetrahedralROI:
    centre_array, volume_array, parcel = _validate_geometry(
        centres, volumes, parcel_mask
    )
    try:
        requested = TARGET_VOLUMES_MM3[target]
    except KeyError as error:
        raise ValueError(f"Unsupported target: {target}") from error
    parcel_volumes = volume_array[parcel]
    centroid = np.average(
        centre_array[parcel], axis=0, weights=parcel_volumes
    ).astype(np.float64)
    selected, radius, achieved, reached = _grow_sphere(
        centres=centre_array,
        volumes=volume_array,
        allowed=parcel,
        centre=centroid,
        requested_volume_mm3=requested,
        start_radius_mm=start_radius_mm,
        radius_step_mm=radius_step_mm,
        radius_cap_mm=radius_cap_mm,
    )
    return TetrahedralROI(
        mask=selected,
        provenance={
            "schema_version": 2,
            "method": "volume_weighted_parcel_centroid_strict_sphere",
            "target": target,
            "requested_volume_mm3": requested,
            "achieved_volume_mm3": achieved,
            "radius_mm": radius,
            "distance_comparator": "<",
            "clipped_to_anatomical_parcel": True,
            "target_volume_reached": reached,
            "centroid_world_mm": centroid.tolist(),
        },
    )


def build_under_anode_roi(
    *,
    electrode_name: str,
    surface_nodes: object,
    surface_triangles: object,
    tetra_centres: object,
    tetra_volumes: object,
    cortical_gm_mask: object,
    target_volume_mm3: float = 100.0,
    start_radius_mm: float = 3.0,
    radius_step_mm: float = 0.01,
    radius_cap_mm: float = 30.0,
) -> TetrahedralROI:
    centres, volumes, gm = _validate_geometry(
        tetra_centres, tetra_volumes, cortical_gm_mask
    )
    nodes = require_float64(surface_nodes, "electrode surface nodes")
    triangles = np.asarray(surface_triangles, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("electrode surface nodes must have shape N x 3")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("electrode surface triangles must have shape M x 3")
    if len(triangles) == 0 or np.any(triangles < 0) or np.any(triangles >= len(nodes)):
        raise ValueError("electrode surface triangles are empty or invalid")
    vertices = nodes[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    areas = np.linalg.norm(cross, axis=1) * np.float64(0.5)
    if not np.isfinite(areas).all() or float(np.sum(areas)) <= 0:
        raise ValueError("electrode stimulation surface has zero area")
    triangle_centres = np.mean(vertices, axis=1, dtype=np.float64)
    surface_centre = np.average(triangle_centres, axis=0, weights=areas).astype(
        np.float64
    )
    gm_indices = np.flatnonzero(gm)
    nearest = gm_indices[
        int(np.argmin(np.linalg.norm(centres[gm] - surface_centre, axis=1)))
    ]
    projected = centres[nearest]
    selected, radius, achieved, reached = _grow_sphere(
        centres=centres,
        volumes=volumes,
        allowed=gm,
        centre=projected,
        requested_volume_mm3=float(target_volume_mm3),
        start_radius_mm=start_radius_mm,
        radius_step_mm=radius_step_mm,
        radius_cap_mm=radius_cap_mm,
    )
    return TetrahedralROI(
        mask=selected,
        provenance={
            "schema_version": 2,
            "method": "area_weighted_anode_surface_to_nearest_cortical_gm_sphere",
            "electrode_name": electrode_name,
            "surface_area_mm2": float(np.sum(areas, dtype=np.float64)),
            "surface_centre_world_mm": surface_centre.tolist(),
            "projected_centre_world_mm": projected.tolist(),
            "projected_to_cortical_gm": True,
            "gm_only": True,
            "requested_volume_mm3": float(target_volume_mm3),
            "achieved_volume_mm3": achieved,
            "radius_mm": radius,
            "distance_comparator": "<",
            "target_volume_reached": reached,
        },
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    threshold = probability * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def volume_weighted_metrics(
    *, values: object, volumes: object, roi_mask: object
) -> dict[str, float | int]:
    value_array = require_float64(values, "analytical field")
    volume_array = require_float64(volumes, "tetrahedron volumes")
    mask = np.asarray(roi_mask, dtype=bool)
    if value_array.ndim != 1:
        raise ValueError("analytical scalar field must be one-dimensional")
    if value_array.shape != volume_array.shape or mask.shape != value_array.shape:
        raise ValueError("field, volume, and ROI shapes differ")
    selected_volumes = volume_array[mask]
    achieved = float(np.sum(selected_volumes, dtype=np.float64))
    finite = mask & np.isfinite(value_array) & np.isfinite(volume_array)
    if not np.any(finite):
        raise ValueError("ROI has no finite support")
    finite_values = value_array[finite]
    finite_volumes = volume_array[finite]
    finite_volume = float(np.sum(finite_volumes, dtype=np.float64))
    return {
        "mean": float(
            np.sum(finite_values * finite_volumes, dtype=np.float64) / finite_volume
        ),
        "median": _weighted_quantile(finite_values, finite_volumes, 0.5),
        "p95": _weighted_quantile(finite_values, finite_volumes, 0.95),
        "achieved_roi_volume_mm3": achieved,
        "finite_volume_mm3": finite_volume,
        "finite_volume_fraction": finite_volume / achieved,
        "finite_tetrahedra": int(np.count_nonzero(finite)),
        "nonfinite_tetrahedra": int(np.count_nonzero(mask & ~finite)),
    }

