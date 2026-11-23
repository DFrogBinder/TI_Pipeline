"""Version-tolerant native SimNIBS mesh readers for authoritative fields."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .extraction import NativeMeshField
from .geometry import geometry_fingerprint
from .precision import require_float64


def _simnibs_reader(path: str):
    try:
        from simnibs import mesh_io  # type: ignore
    except ImportError:
        from simnibs.msh import mesh_io  # type: ignore
    return mesh_io.read_msh(path)


def load_native_element_field(
    path: Path,
    field_name: str,
    *,
    mesh_reader: Callable[[str], object] | None = None,
) -> NativeMeshField:
    """Load float64 tetrahedral ElementData from v3.2.6, v4.0.1, or v4.6."""

    if path.suffix.lower() != ".msh":
        raise ValueError(f"Authoritative native field must be .msh: {path}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe native mesh: {path}")
    mesh = (mesh_reader or _simnibs_reader)(str(path))
    try:
        nodes = require_float64(mesh.nodes.node_coord, "mesh nodes")
        element_types = np.asarray(mesh.elm.elm_type)
        all_connectivity = np.asarray(mesh.elm.node_number_list)
        all_tags = np.asarray(mesh.elm.tag1)
        data = mesh.field[field_name]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Native mesh lacks required field {field_name!r}: {path}") from error
    tetra_indices = np.flatnonzero(element_types == 4)
    if len(tetra_indices) == 0:
        raise ValueError(f"Native mesh contains no tetrahedra: {path}")
    tetrahedra = np.asarray(all_connectivity[tetra_indices, :4], dtype=np.int64) - 1
    if np.any(tetrahedra < 0) or np.any(tetrahedra >= len(nodes)):
        raise ValueError(f"Native mesh has invalid tetrahedral connectivity: {path}")
    tags = np.asarray(all_tags[tetra_indices], dtype=np.int64)
    raw_values = require_float64(data.value, f"native {field_name} field")
    if len(raw_values) == len(element_types):
        values = np.ascontiguousarray(raw_values[tetra_indices])
    elif len(raw_values) == len(tetra_indices):
        values = np.ascontiguousarray(raw_values)
    else:
        raise ValueError(
            f"Field {field_name!r} has {len(raw_values)} values for "
            f"{len(element_types)} elements and {len(tetra_indices)} tetrahedra"
        )
    vertices = nodes[tetrahedra]
    centres = np.mean(vertices, axis=1, dtype=np.float64)
    determinants = np.linalg.det(
        np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=1,
        )
    )
    volumes = np.asarray(np.abs(determinants) / np.float64(6.0), dtype=np.float64)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError(f"Native mesh contains non-positive tetrahedron volumes: {path}")
    return NativeMeshField(
        path=path,
        field_name=field_name,
        values=values,
        tetra_centres=np.ascontiguousarray(centres, dtype=np.float64),
        tetra_volumes=np.ascontiguousarray(volumes, dtype=np.float64),
        geometry_sha256=geometry_fingerprint(nodes, tetrahedra, tags),
        stored_dtype=str(raw_values.dtype),
        tetra_tags=np.ascontiguousarray(tags, dtype=np.int64),
    )


def load_tagged_surface(
    path: Path,
    surface_tag: int,
    *,
    mesh_reader: Callable[[str], object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact nodes/triangles for one electrode stimulation surface."""

    if path.suffix.lower() != ".msh" or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe electrode geometry mesh: {path}")
    mesh = (mesh_reader or _simnibs_reader)(str(path))
    nodes = require_float64(mesh.nodes.node_coord, "mesh nodes")
    element_types = np.asarray(mesh.elm.elm_type)
    tags = np.asarray(mesh.elm.tag1)
    connectivity = np.asarray(mesh.elm.node_number_list)
    selected = (element_types == 2) & (tags == int(surface_tag))
    triangles_global = np.asarray(connectivity[selected, :3], dtype=np.int64) - 1
    if len(triangles_global) == 0:
        raise ValueError(f"No triangles carry electrode surface tag {surface_tag}")
    used = np.unique(triangles_global)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return (
        np.ascontiguousarray(nodes[used], dtype=np.float64),
        np.ascontiguousarray(remap[triangles_global], dtype=np.int64),
    )

