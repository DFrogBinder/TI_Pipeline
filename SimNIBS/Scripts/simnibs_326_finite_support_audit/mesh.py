"""Direct finite-value audit of the retained SimNIBS element field."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def summarize_mesh_field(
    *,
    values: np.ndarray,
    element_centres_world: np.ndarray,
    element_types: np.ndarray,
    roi_mask: np.ndarray,
    roi_affine: np.ndarray,
) -> dict[str, Any]:
    """Summarize TImax tetrahedra whose barycentres fall in the voxel ROI."""

    field = np.asarray(values)
    if field.ndim == 2 and field.shape[1] == 1:
        field = field[:, 0]
    if field.ndim != 1:
        raise ValueError(f"Expected scalar mesh field; observed {field.shape}")
    centres = np.asarray(element_centres_world, dtype=float)
    types = np.asarray(element_types)
    if centres.shape != (field.size, 3) or types.shape != (field.size,):
        raise ValueError("Mesh values, centres, and element types must align")

    voxels = nib.affines.apply_affine(np.linalg.inv(roi_affine), centres)
    indices = np.rint(voxels).astype(np.int64)
    shape = np.asarray(roi_mask.shape, dtype=np.int64)
    in_grid = np.all((indices >= 0) & (indices < shape), axis=1)
    in_roi = np.zeros(field.size, dtype=bool)
    valid_indices = indices[in_grid]
    in_roi[in_grid] = np.asarray(roi_mask, dtype=bool)[tuple(valid_indices.T)]
    tetra = types == 4
    selected = tetra & in_roi
    finite = np.isfinite(field)
    selected_values = field[selected & finite].astype(float, copy=False)
    return {
        "mesh_field_elements": int(field.size),
        "mesh_field_nonfinite_elements": int(np.count_nonzero(~finite)),
        "mesh_tetrahedra": int(np.count_nonzero(tetra)),
        "mesh_roi_elements": int(np.count_nonzero(selected)),
        "mesh_roi_finite_elements": int(np.count_nonzero(selected & finite)),
        "mesh_roi_nonfinite_elements": int(np.count_nonzero(selected & ~finite)),
        "mesh_roi_min_v_per_m": (
            float(np.min(selected_values)) if selected_values.size else None
        ),
        "mesh_roi_max_v_per_m": (
            float(np.max(selected_values)) if selected_values.size else None
        ),
    }


def _mesh_element_centres(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(mesh.nodes.node_coord, dtype=float)
    node_numbers = np.asarray(mesh.elm.node_number_list, dtype=np.int64)
    element_types = np.asarray(mesh.elm.elm_type, dtype=np.int64).reshape(-1)
    if node_numbers.shape[0] != element_types.size:
        raise ValueError("Mesh topology arrays do not align")
    centres = np.empty((node_numbers.shape[0], 3), dtype=float)
    for index, row in enumerate(node_numbers):
        valid = row[row > 0] - 1
        if valid.size == 0 or np.any(valid >= nodes.shape[0]):
            raise ValueError(f"Invalid mesh node indices for element {index}")
        centres[index] = nodes[valid].mean(axis=0)
    return centres, element_types


def audit_mesh_field(
    mesh_path: Path,
    *,
    roi_mask: np.ndarray,
    roi_affine: np.ndarray,
    field_name: str = "TImax",
) -> dict[str, Any]:
    """Read one mesh with the active SimNIBS module and audit its field."""

    import simnibs  # type: ignore

    simnibs_version = str(getattr(simnibs, "__version__", "unknown"))
    if not simnibs_version.startswith("3.2.6"):
        raise RuntimeError(
            f"Direct v3 mesh audit requires SimNIBS 3.2.6; observed {simnibs_version}"
        )
    try:
        from simnibs import mesh_io  # type: ignore
    except ImportError:
        from simnibs.msh import mesh_io  # type: ignore

    mesh = mesh_io.read_msh(str(mesh_path))
    try:
        values = np.asarray(mesh.field[field_name].value)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"Mesh lacks element field {field_name!r}: {mesh_path}") from exc
    centres, element_types = _mesh_element_centres(mesh)
    summary = summarize_mesh_field(
        values=values,
        element_centres_world=centres,
        element_types=element_types,
        roi_mask=roi_mask,
        roi_affine=roi_affine,
    )
    summary.update(
        {
            "mesh_path": str(mesh_path),
            "mesh_size_bytes": mesh_path.stat().st_size,
            "mesh_field_name": field_name,
            "simnibs_version": simnibs_version,
        }
    )
    return summary
