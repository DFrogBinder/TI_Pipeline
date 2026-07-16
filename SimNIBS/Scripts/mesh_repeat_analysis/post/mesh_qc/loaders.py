from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .geometry_qc import SurfaceArrays


TISSUE_TAG_NAMES = {
    1: "white_matter",
    2: "gray_matter",
    3: "csf",
    4: "bone",
    5: "scalp",
    6: "eye_balls",
    7: "compact_bone",
    8: "spongy_bone",
    9: "blood",
    10: "muscle",
    11: "cartilage",
    12: "fat",
}
TISSUE_DISPLAY_NAMES = {
    1: "White matter",
    2: "Gray matter",
    3: "CSF",
    4: "Bone",
    5: "Scalp",
    6: "Eye balls",
    7: "Compact bone",
    8: "Spongy bone",
    9: "Blood",
    10: "Muscle",
    11: "Cartilage",
    12: "Fat",
}


@dataclass(frozen=True)
class TissueSurface:
    tag: int
    name: str
    slug: str
    surface: SurfaceArrays


def tissue_name(tag: int) -> str:
    return TISSUE_TAG_NAMES.get(int(tag), f"tissue_{int(tag)}")


def tissue_display_name(tag: int) -> str:
    return TISSUE_DISPLAY_NAMES.get(int(tag), tissue_name(tag).replace("_", " ").title())


def tissue_slug(tag: int) -> str:
    return f"tag_{int(tag):02d}_{tissue_name(tag)}"


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
    tetra_faces = []
    triangles = []
    for cell_block in mesh.cells:
        data = np.asarray(cell_block.data, dtype=np.int64)
        if cell_block.type in {"tetra", "tetra10"}:
            tetra = data[:, :4]
            tetra_faces.append(tetra[:, [0, 2, 1]])
            tetra_faces.append(tetra[:, [0, 1, 3]])
            tetra_faces.append(tetra[:, [1, 2, 3]])
            tetra_faces.append(tetra[:, [2, 0, 3]])
        elif cell_block.type == "triangle":
            triangles.append(data[:, :3])
        elif cell_block.type == "quad":
            triangles.append(data[:, [0, 1, 2]])
            triangles.append(data[:, [0, 2, 3]])

    if tetra_faces:
        all_faces = np.vstack(tetra_faces)
        sorted_faces = np.sort(all_faces, axis=1)
        _, first_indices, counts = np.unique(
            sorted_faces,
            axis=0,
            return_index=True,
            return_counts=True,
        )
        faces = all_faces[first_indices[counts == 1]]
    else:
        faces = np.vstack(triangles) if triangles else np.empty((0, 3), dtype=np.int64)
    return SurfaceArrays(points=np.asarray(mesh.points, dtype=float), faces=faces)


def _compact_surface(points: np.ndarray, faces: np.ndarray, *, one_based: bool = False) -> SurfaceArrays:
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return SurfaceArrays(
            points=np.empty((0, 3), dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
        )
    faces = faces.reshape(-1, 3)
    if one_based:
        faces = faces - 1
    valid = np.all((faces >= 0) & (faces < len(points)), axis=1)
    faces = faces[valid]
    if faces.size == 0:
        return SurfaceArrays(
            points=np.empty((0, 3), dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
        )
    used_nodes = np.unique(faces)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used_nodes] = np.arange(len(used_nodes), dtype=np.int64)
    return SurfaceArrays(points=points[used_nodes], faces=remap[faces])


def _tetrahedral_boundary(tetrahedra: np.ndarray) -> np.ndarray:
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    if tetrahedra.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    tetrahedra = tetrahedra.reshape(-1, tetrahedra.shape[-1])[:, :4]
    faces = np.vstack(
        (
            tetrahedra[:, [0, 2, 1]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 3, 2]],
            tetrahedra[:, [1, 2, 3]],
        )
    )
    _, first_indices, counts = np.unique(
        np.sort(faces, axis=1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    return faces[first_indices[counts == 1]]


def _iter_tissue_surfaces_from_simnibs_mesh(mesh) -> Iterator[TissueSurface]:
    points = np.asarray(mesh.nodes.node_coord, dtype=float)
    element_types = np.asarray(mesh.elm.elm_type, dtype=np.int64)
    tags = np.asarray(mesh.elm.tag1, dtype=np.int64)
    element_numbers = np.asarray(mesh.elm.elm_number, dtype=np.int64)
    volume_tags = sorted(int(tag) for tag in np.unique(tags[element_types == 4]) if int(tag) > 0)

    for tag in volume_tags:
        tetrahedron_numbers = element_numbers[(element_types == 4) & (tags == tag)]
        faces = mesh.elm.get_outside_faces(tetrahedron_numbers)
        surface = _compact_surface(points, faces, one_based=True)
        if surface.faces.size == 0:
            continue
        yield TissueSurface(
            tag=tag,
            name=tissue_display_name(tag),
            slug=tissue_slug(tag),
            surface=surface,
        )


def _meshio_physical_tags(mesh, block_index: int, cell_count: int) -> np.ndarray:
    cell_data = getattr(mesh, "cell_data", {}) or {}
    physical_data = cell_data.get("gmsh:physical", [])
    if block_index >= len(physical_data):
        return np.zeros(cell_count, dtype=np.int64)
    tags = np.asarray(physical_data[block_index], dtype=np.int64).reshape(-1)
    if len(tags) != cell_count:
        raise ValueError(
            "meshio gmsh:physical cell data does not match its cell block: "
            f"block={block_index}, cells={cell_count}, tags={len(tags)}"
        )
    return tags


def _iter_tissue_surfaces_from_meshio_mesh(mesh) -> Iterator[TissueSurface]:
    tagged_tetrahedra: dict[int, list[np.ndarray]] = {}
    for block_index, cell_block in enumerate(mesh.cells):
        if cell_block.type not in {"tetra", "tetra10"}:
            continue
        data = np.asarray(cell_block.data, dtype=np.int64)
        physical_tags = _meshio_physical_tags(mesh, block_index, len(data))
        for tag in np.unique(physical_tags):
            tag = int(tag)
            if tag <= 0:
                continue
            tagged_tetrahedra.setdefault(tag, []).append(data[physical_tags == tag, :4])

    points = np.asarray(mesh.points, dtype=float)
    for tag in sorted(tagged_tetrahedra):
        tetrahedra = np.vstack(tagged_tetrahedra[tag])
        faces = _tetrahedral_boundary(tetrahedra)
        surface = _compact_surface(points, faces)
        if surface.faces.size == 0:
            continue
        yield TissueSurface(
            tag=tag,
            name=tissue_display_name(tag),
            slug=tissue_slug(tag),
            surface=surface,
        )


def iter_tissue_surface_arrays(path: Path) -> Iterator[TissueSurface]:
    """Yield each tagged tetrahedral tissue as a complete boundary surface."""
    path = Path(path)
    simnibs_error: Exception | None = None
    try:
        from simnibs import mesh_io

        mesh = mesh_io.read_msh(str(path))
    except Exception as exc:
        simnibs_error = exc
    else:
        found = False
        for tissue in _iter_tissue_surfaces_from_simnibs_mesh(mesh):
            found = True
            yield tissue
        if not found:
            raise RuntimeError(f"Could not extract tissue surfaces from {path}; mesh has no tissue tags")
        return

    try:
        import meshio

        mesh = meshio.read(str(path))
    except Exception as meshio_error:
        raise RuntimeError(
            f"Could not extract tissue surfaces from {path}; "
            f"simnibs={simnibs_error}; meshio={meshio_error}"
        ) from meshio_error
    found = False
    for tissue in _iter_tissue_surfaces_from_meshio_mesh(mesh):
        found = True
        yield tissue
    if not found:
        raise RuntimeError(f"Could not extract tissue surfaces from {path}; mesh has no tissue tags")


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
