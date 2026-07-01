import numpy as np

from mesh_repeat_analysis.post.mesh_qc.geometry_qc import SurfaceArrays, compute_qc_metrics


def test_closed_tetrahedron_has_no_boundary_or_nonmanifold_edges():
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array(
            [
                [0, 2, 1],
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
            ]
        ),
    )

    metrics = compute_qc_metrics(surface)

    assert metrics.status == "OK"
    assert metrics.boundary_edges == 0
    assert metrics.nonmanifold_edges == 0
    assert metrics.degenerate_faces == 0
    assert metrics.connected_components == 1


def test_open_square_flags_boundary_edges():
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )

    metrics = compute_qc_metrics(surface)

    assert metrics.status == "FAIL"
    assert "BOUNDARY_EDGES" in metrics.flags
    assert metrics.boundary_edges == 4


def test_degenerate_triangle_is_flagged():
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )

    metrics = compute_qc_metrics(surface)

    assert metrics.status == "FAIL"
    assert "DEGENERATE_FACES" in metrics.flags
    assert metrics.degenerate_faces == 1


def test_nonmanifold_edge_is_flagged():
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
    )

    metrics = compute_qc_metrics(surface)

    assert metrics.status == "FAIL"
    assert "NONMANIFOLD_EDGES" in metrics.flags
    assert metrics.nonmanifold_edges == 1


def test_component_check_can_be_disabled_for_fast_batch_qc():
    surface = SurfaceArrays(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [10.0, 11.0, 10.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
    )

    metrics = compute_qc_metrics(surface, check_components=False)

    assert "DISCONNECTED_COMPONENTS" not in metrics.flags
    assert metrics.connected_components == -1
    assert "BOUNDARY_EDGES" in metrics.flags
