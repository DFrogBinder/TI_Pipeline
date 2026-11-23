"""Field-independent geometry identities for fixed-mesh validation."""

from __future__ import annotations

import hashlib

import numpy as np


def _update_array(digest: "hashlib._Hash", label: str, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    digest.update(label.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))


def geometry_fingerprint(
    nodes: object, tetrahedra: object, tags: object
) -> str:
    """Hash nodes, connectivity, and tags; result fields are deliberately absent."""

    node_array = np.asarray(nodes, dtype="<f8")
    element_array = np.asarray(tetrahedra, dtype="<i8")
    tag_array = np.asarray(tags, dtype="<i8")
    if node_array.ndim != 2 or node_array.shape[1] != 3:
        raise ValueError("nodes must have shape N x 3")
    if element_array.ndim != 2 or element_array.shape[1] != 4:
        raise ValueError("tetrahedra must have shape M x 4")
    if tag_array.shape != (len(element_array),):
        raise ValueError("tags must have one value per tetrahedron")
    digest = hashlib.sha256()
    _update_array(digest, "nodes", node_array)
    _update_array(digest, "tetrahedra", element_array)
    _update_array(digest, "tags", tag_array)
    return digest.hexdigest()

