from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry_qc import SurfaceArrays


def _faces_from_pyvista(surface) -> np.ndarray:
    raw = np.asarray(surface.faces, dtype=np.int64)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    faces = []
    idx = 0
    while idx < raw.size:
        n = int(raw[idx])
        verts = raw[idx + 1 : idx + 1 + n]
        if n == 3:
            faces.append(verts)
        elif n > 3:
            for j in range(1, n - 1):
                faces.append([verts[0], verts[j], verts[j + 1]])
        idx += n + 1
    return np.asarray(faces, dtype=np.int64)


def _from_pyvista_dataset(dataset) -> SurfaceArrays:
    surface = dataset.extract_surface().triangulate()
    return SurfaceArrays(points=np.asarray(surface.points, dtype=float), faces=_faces_from_pyvista(surface))


def _from_meshio_mesh(mesh) -> SurfaceArrays:
    triangles = []
    for cell_block in mesh.cells:
        data = np.asarray(cell_block.data, dtype=np.int64)
        if cell_block.type == "triangle":
            triangles.append(data[:, :3])
        elif cell_block.type == "quad":
            triangles.append(data[:, [0, 1, 2]])
            triangles.append(data[:, [0, 2, 3]])
    faces = np.vstack(triangles) if triangles else np.empty((0, 3), dtype=np.int64)
    return SurfaceArrays(points=np.asarray(mesh.points, dtype=float), faces=faces)


def load_surface_arrays(path: Path) -> SurfaceArrays:
    path = Path(path)
    pyvista_error: Exception | None = None
    try:
        import pyvista as pv

        return _from_pyvista_dataset(pv.read(str(path)))
    except Exception as exc:
        pyvista_error = exc

    meshio_error: Exception | None = None
    try:
        import meshio

        return _from_meshio_mesh(meshio.read(str(path)))
    except Exception as exc:
        meshio_error = exc

    try:
        from simnibs import mesh_io

        mesh = mesh_io.read_msh(str(path))
        points = np.asarray(mesh.nodes.node_coord, dtype=float)
        triangles = np.asarray(mesh.elm.node_number_list[:, :3], dtype=np.int64) - 1
        return SurfaceArrays(points=points, faces=triangles)
    except Exception as simnibs_error:
        raise RuntimeError(
            f"Could not load {path}; pyvista={pyvista_error}; meshio={meshio_error}; "
            f"simnibs={simnibs_error}"
        ) from simnibs_error

