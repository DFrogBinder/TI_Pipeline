"""Median-selected immutable pair-specific fixed-geometry caches."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def select_representative_repeat(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Cannot select a representative repeat from no rows")
    repeats = [int(row["repeat"]) for row in materialized]
    if len(repeats) != len(set(repeats)):
        raise ValueError("Duplicate repeat in representative-selection rows")
    values = np.asarray(
        [row["roi_median_v_per_m"] for row in materialized], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Representative-selection values must be finite")
    median = float(np.median(values))
    selected = min(
        materialized,
        key=lambda row: (
            abs(float(row["roi_median_v_per_m"]) - median),
            int(row["repeat"]),
        ),
    )
    return {**selected, "cell_median_v_per_m": median}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_pair_caches(
    *,
    remesh_manifest: Mapping[str, object],
    cache_root: Path,
    adapter: object,
) -> dict[str, object]:
    """Remove result fields while retaining exact pair geometry and provenance."""

    subject = str(remesh_manifest["subject"])
    target = str(remesh_manifest["target"])
    source_repeat = int(remesh_manifest["repeat"])
    root = cache_root / subject / target
    if root.exists() and any(root.iterdir()):
        manifest_path = root / "cache_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("status") == "ready":
                return {**existing, "manifest_path": str(manifest_path)}
        raise RuntimeError(f"Refusing to overwrite incomplete fixed cache: {root}")
    root.mkdir(parents=True, exist_ok=True)
    pair_caches: dict[str, str] = {}
    geometry_hashes: dict[str, str] = {}
    source_mesh_hashes: dict[str, str] = {}
    for pair in (1, 2):
        key = str(pair)
        source = Path(remesh_manifest["component_geometry_paths"][key])
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Missing remesh pair geometry: {source}")
        source_geometry = adapter.geometry_sha256(source)
        destination = root / f"pair{pair}_geometry_cache.msh"
        adapter.strip_result_fields(source, destination)
        cached_geometry = adapter.geometry_sha256(destination)
        if cached_geometry != source_geometry:
            raise ValueError(f"Geometry changed while preparing pair {pair} cache")
        destination.chmod(0o444)
        pair_caches[key] = str(destination.resolve())
        geometry_hashes[key] = cached_geometry
        source_mesh_hashes[key] = source_geometry
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": "ready",
        "subject": subject,
        "target": target,
        "source_repeat": source_repeat,
        "pair_caches": pair_caches,
        "geometry_sha256": geometry_hashes,
        "source_geometry_sha256": source_mesh_hashes,
        "currents_a": remesh_manifest["currents_a"],
        "conductivity_by_tag": remesh_manifest["conductivity_by_tag"],
        "electrode_surface_tags": remesh_manifest["electrode_surface_tags"],
        "electrode_names": remesh_manifest.get("electrode_names", {}),
        "solver_options": remesh_manifest["solver_options"],
        "n_workers": int(remesh_manifest.get("n_workers", 8)),
        "result_fields_removed": True,
        "pair_specific": True,
    }
    manifest_path = root / "cache_manifest.json"
    _write_json(manifest_path, payload)
    manifest_path.chmod(0o444)
    return {**payload, "manifest_path": str(manifest_path)}

