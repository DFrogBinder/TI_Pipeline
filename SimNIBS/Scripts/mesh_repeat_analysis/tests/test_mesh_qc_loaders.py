import numpy as np

from mesh_repeat_analysis.post.mesh_qc.loaders import _from_meshio_mesh


class CellBlock:
    def __init__(self, cell_type, data):
        self.type = cell_type
        self.data = np.asarray(data, dtype=np.int64)


class Mesh:
    def __init__(self, points, cells):
        self.points = np.asarray(points, dtype=float)
        self.cells = cells


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

