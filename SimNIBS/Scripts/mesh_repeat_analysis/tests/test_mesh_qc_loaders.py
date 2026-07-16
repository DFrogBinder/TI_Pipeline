import numpy as np

from mesh_repeat_analysis.post.mesh_qc.loaders import (
    _from_meshio_mesh,
    _iter_tissue_surfaces_from_meshio_mesh,
)


class CellBlock:
    def __init__(self, cell_type, data):
        self.type = cell_type
        self.data = np.asarray(data, dtype=np.int64)


class Mesh:
    def __init__(self, points, cells, cell_data=None):
        self.points = np.asarray(points, dtype=float)
        self.cells = cells
        self.cell_data = cell_data or {}


def test_meshio_loader_prefers_tetrahedral_exterior_boundary():
    mesh = Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        cells=[
            CellBlock("triangle", [[0, 1, 2]]),
            CellBlock("tetra", [[0, 1, 2, 3]]),
        ],
    )

    surface = _from_meshio_mesh(mesh)

    assert surface.faces.shape == (4, 3)
    assert {tuple(sorted(face)) for face in surface.faces} == {
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    }


def test_tissue_loader_reconstructs_complete_boundary_per_tetrahedral_tag():
    mesh = Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        cells=[CellBlock("tetra", [[0, 1, 2, 3], [0, 2, 1, 4]])],
        cell_data={"gmsh:physical": [np.array([5, 5], dtype=np.int64)]},
    )

    tissues = list(_iter_tissue_surfaces_from_meshio_mesh(mesh))

    assert len(tissues) == 1
    assert tissues[0].tag == 5
    assert tissues[0].name == "Scalp"
    assert tissues[0].slug == "tag_05_scalp"
    assert tissues[0].surface.points.shape == (5, 3)
    assert tissues[0].surface.faces.shape == (6, 3)


def test_tissue_loader_keeps_distinct_tissue_tags_separate():
    mesh = Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        cells=[CellBlock("tetra", [[0, 1, 2, 3], [1, 4, 2, 3]])],
        cell_data={"gmsh:physical": [np.array([1, 2], dtype=np.int64)]},
    )

    tissues = list(_iter_tissue_surfaces_from_meshio_mesh(mesh))

    assert [tissue.tag for tissue in tissues] == [1, 2]
    assert [tissue.surface.faces.shape for tissue in tissues] == [(4, 3), (4, 3)]
