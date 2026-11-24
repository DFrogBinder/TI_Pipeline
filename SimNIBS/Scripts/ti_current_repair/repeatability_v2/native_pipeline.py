"""End-to-end native extraction for one completed v2 simulation manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .extraction import extract_three_outcomes, write_extraction_tables
from .mesh_io import load_native_element_field, load_tagged_surface
from .precision import require_float64
from .roi import build_parcel_clipped_roi, build_under_anode_roi


TARGET_LABEL_IDS = {
    "left_hippocampus": (17,),
    "right_m1": (12129,),
}
UNDER_ANODE = {"left_hippocampus": "F8", "right_m1": "Fp2"}
CORTICAL_GM_TAG = 2
HEAD_VOLUME_TAG_MAX_EXCLUSIVE = 100


def under_anode_name(target: str) -> str:
    try:
        return UNDER_ANODE[target]
    except KeyError as error:
        raise ValueError(f"Unsupported v2 target: {target}") from error


def map_ti_target_mask_to_pair1(
    *, ti: object, pair1: object, ti_target_mask: object
) -> np.ndarray:
    """Map one biological-tissue ROI from cropped TI support to augmented pair 1."""

    if ti.tetra_tags is None or pair1.tetra_tags is None:
        raise ValueError("TI and pair-1 meshes require tetrahedral tissue tags")
    ti_tags = np.asarray(ti.tetra_tags, dtype=np.int64)
    pair_tags = np.asarray(pair1.tetra_tags, dtype=np.int64)
    ti_head = (ti_tags > 0) & (ti_tags < HEAD_VOLUME_TAG_MAX_EXCLUSIVE)
    pair_head = (pair_tags > 0) & (pair_tags < HEAD_VOLUME_TAG_MAX_EXCLUSIVE)
    selected = np.asarray(ti_target_mask, dtype=bool)
    if selected.shape != ti_tags.shape or np.any(selected & ~ti_head):
        raise ValueError("Target ROI contains support outside biological head tissues")
    for label, left, right in (
        ("tags", ti_tags[ti_head], pair_tags[pair_head]),
        ("centres", ti.tetra_centres[ti_head], pair1.tetra_centres[pair_head]),
        ("volumes", ti.tetra_volumes[ti_head], pair1.tetra_volumes[pair_head]),
    ):
        if not np.array_equal(left, right):
            raise ValueError(
                f"Cropped TI and augmented pair-1 biological geometry differ: {label}"
            )
    mapped = np.zeros(len(pair_tags), dtype=bool)
    mapped[np.flatnonzero(pair_head)] = selected[ti_head]
    return mapped


def parcel_mask_from_atlas(
    tetra_centres: object,
    atlas_path: Path,
    *,
    label_ids: Iterable[int],
) -> np.ndarray:
    """Sample integer atlas geometry at tetrahedral centres; fields stay native."""

    import nibabel as nib

    centres = require_float64(tetra_centres, "tetrahedron centres")
    image = nib.load(str(atlas_path))
    labels = np.asarray(image.dataobj)
    if labels.ndim != 3 or not np.allclose(labels, np.rint(labels)):
        raise ValueError(f"Atlas must contain three-dimensional integer labels: {atlas_path}")
    voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), centres)
    indices = np.rint(voxel).astype(np.int64)
    valid = np.ones(len(indices), dtype=bool)
    for axis, size in enumerate(labels.shape):
        valid &= (indices[:, axis] >= 0) & (indices[:, axis] < size)
    sampled = np.full(len(indices), -1, dtype=np.int64)
    if np.any(valid):
        valid_indices = indices[valid]
        sampled[valid] = labels[tuple(valid_indices.T)].astype(np.int64, copy=False)
    return np.isin(sampled, tuple(int(value) for value in label_ids))


def _geometry_contract(manifest: dict) -> tuple[dict, dict[str, str]]:
    if manifest["condition"] == "remesh":
        return manifest, {
            str(key): str(value)
            for key, value in manifest["component_geometry_paths"].items()
        }
    cache = json.loads(Path(manifest["cache_manifest"]).read_text(encoding="utf-8"))
    if cache.get("status") != "ready":
        raise ValueError("Fixed simulation points to a non-ready geometry cache")
    return cache, {
        str(key): str(value) for key, value in cache["pair_caches"].items()
    }


def extract_task_manifest(
    *, manifest_path: Path, atlas_path: Path, output_root: Path
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Simulation manifest is not complete: {manifest_path}")
    target = str(manifest["target"])
    try:
        labels = TARGET_LABEL_IDS[target]
    except KeyError as error:
        raise ValueError(f"Unsupported v2 target: {target}") from error
    ti = load_native_element_field(Path(manifest["ti_msh_path"]), "TImax")
    pair1 = load_native_element_field(
        Path(manifest["component_field_paths"]["1"]), "E"
    )
    if pair1.tetra_tags is None:
        raise ValueError("Pair-1 mesh lacks tetrahedral tissue tags")
    parcel = parcel_mask_from_atlas(
        ti.tetra_centres, atlas_path, label_ids=labels
    )
    target_roi = build_parcel_clipped_roi(
        centres=ti.tetra_centres,
        volumes=ti.tetra_volumes,
        parcel_mask=parcel,
        target=target,
    )
    pair1_target_mask = map_ti_target_mask_to_pair1(
        ti=ti, pair1=pair1, ti_target_mask=target_roi.mask
    )
    geometry_contract, geometry_paths = _geometry_contract(manifest)
    anode_name = str(geometry_contract["electrode_names"]["1"][0])
    expected_anode = under_anode_name(target)
    if anode_name.casefold() != expected_anode.casefold():
        raise ValueError(
            f"Pair-1 positive electrode is {anode_name}, expected {expected_anode}"
        )
    positive_surface_tag = int(
        geometry_contract["electrode_surface_tags"]["1"][0]
    )
    surface_nodes, surface_triangles = load_tagged_surface(
        Path(geometry_paths["1"]), positive_surface_tag
    )
    under_roi = build_under_anode_roi(
        electrode_name=expected_anode,
        surface_nodes=surface_nodes,
        surface_triangles=surface_triangles,
        tetra_centres=pair1.tetra_centres,
        tetra_volumes=pair1.tetra_volumes,
        cortical_gm_mask=np.asarray(pair1.tetra_tags) == CORTICAL_GM_TAG,
        target_volume_mm3=100.0,
    )
    common = {
        "subject": manifest["subject"],
        "target": target,
        "condition": manifest["condition"],
        "repeat": int(manifest["repeat"]),
        "repeat_tag": manifest["repeat_tag"],
        "simnibs_version": manifest["simnibs_version"],
    }
    rows = extract_three_outcomes(
        ti_field=ti,
        pair1_field=pair1,
        matched_target_mask=target_roi.mask,
        pair1_matched_target_mask=pair1_target_mask,
        matched_roi_provenance=target_roi.provenance,
        under_anode_mask=under_roi.mask,
        under_anode_provenance=under_roi.provenance,
        common=common,
    )
    geometry_rows = [
        {
            **common,
            "pair": pair,
            "geometry_sha256": manifest["geometry_sha256"][str(pair)],
            "geometry_path": geometry_paths[str(pair)],
            "source_remesh_repeat": manifest.get(
                "source_remesh_repeat", manifest["repeat"]
            ),
            "electrode_placement_invoked": manifest["condition"] == "remesh",
        }
        for pair in (1, 2)
    ]
    written = write_extraction_tables(output_root, rows, geometry_rows)
    return {
        "schema_version": 2,
        "status": "complete",
        "simulation_manifest": str(manifest_path.resolve()),
        "atlas_path": str(atlas_path.resolve()),
        "tables": {key: str(value) for key, value in written.items()},
        "outcome_rows": 3,
        "geometry_rows": 2,
        "source_outputs_modified": False,
    }

