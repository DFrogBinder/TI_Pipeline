from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SurfaceArrays:
    points: np.ndarray
    faces: np.ndarray


@dataclass
class QCMetrics:
    status: str
    flags: tuple[str, ...]
    n_points: int
    n_faces: int
    degenerate_faces: int
    boundary_edges: int
    nonmanifold_edges: int
    connected_components: int
    x_size: float
    y_size: float
    z_size: float

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["flags"] = ";".join(self.flags)
        return row


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.asarray([], dtype=float)
    tri = points[faces[:, :3]]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5


def edge_counts(faces: np.ndarray) -> Counter[tuple[int, int]]:
    unique_edges, counts = edge_count_arrays(faces)
    return Counter({tuple(map(int, edge)): int(count) for edge, count in zip(unique_edges, counts)})


def edge_count_arrays(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
    tri = np.asarray(faces[:, :3], dtype=np.int64)
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def connected_components(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0

    by_edge: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for face_idx, (a, b, c) in enumerate(faces[:, :3]):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(u), int(v)) if u < v else (int(v), int(u))
            by_edge[edge].append(face_idx)

    neighbors: list[set[int]] = [set() for _ in range(len(faces))]
    for face_indices in by_edge.values():
        if len(face_indices) < 2:
            continue
        for face_idx in face_indices:
            neighbors[face_idx].update(i for i in face_indices if i != face_idx)

    seen: set[int] = set()
    components = 0
    for start in range(len(faces)):
        if start in seen:
            continue
        components += 1
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            current = queue.popleft()
            for nxt in neighbors[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return components


def compute_qc_metrics(
    surface: SurfaceArrays,
    *,
    degenerate_area_eps: float = 1e-12,
    check_components: bool = True,
) -> QCMetrics:
    points = np.asarray(surface.points, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    flags: list[str] = []

    n_points = int(points.shape[0])
    n_faces = int(faces.shape[0])
    if n_points == 0 or n_faces == 0:
        flags.append("EMPTY_SURFACE")

    degenerate = 0
    boundary_edges = 0
    nonmanifold_edges = 0
    components = -1 if not check_components else 0
    if n_points and n_faces:
        areas = triangle_areas(points, faces)
        degenerate = int(np.count_nonzero(areas <= degenerate_area_eps))
        _, counts = edge_count_arrays(faces)
        boundary_edges = int(np.count_nonzero(counts == 1))
        nonmanifold_edges = int(np.count_nonzero(counts > 2))
        if check_components:
            components = connected_components(faces)

    if degenerate:
        flags.append("DEGENERATE_FACES")
    if boundary_edges:
        flags.append("BOUNDARY_EDGES")
    if nonmanifold_edges:
        flags.append("NONMANIFOLD_EDGES")
    if components > 1:
        flags.append("DISCONNECTED_COMPONENTS")

    if n_points:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        sizes = maxs - mins
    else:
        sizes = np.asarray([0.0, 0.0, 0.0])

    return QCMetrics(
        status="FAIL" if flags else "OK",
        flags=tuple(flags),
        n_points=n_points,
        n_faces=n_faces,
        degenerate_faces=degenerate,
        boundary_edges=int(boundary_edges),
        nonmanifold_edges=int(nonmanifold_edges),
        connected_components=int(components),
        x_size=float(sizes[0]),
        y_size=float(sizes[1]),
        z_size=float(sizes[2]),
    )
