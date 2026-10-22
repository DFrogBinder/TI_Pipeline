"""Version-pinned adapter for the SimNIBS 3.2.6 headreco meshing APIs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_VERSION = "3.2.6"
EXPECTED_SURFACE_SIGNATURE = (
    "(input_files, out_dir, temp_dir, vertex_density=0.5, catsurf=False)"
)
EXPECTED_VOLUME_SIGNATURE = "(subject_id, input_files, out_dir, keep_air)"
MESH_CLEANUP_METHODS = (
    "fix_surface_labels",
    "fix_thin_tetrahedra",
    "fix_th_node_ordering",
    "fix_tr_node_ordering",
    "elements_volumes_and_areas",
)


class MeshingAPIError(RuntimeError):
    """Raised when the installed v3 meshing contract differs from the pin."""


class MeshValidationError(RuntimeError):
    """Raised when a generated mesh fails the independent scientific checks."""


@dataclass(frozen=True)
class V3MeshingAPI:
    simnibs: Any
    headreco: Any
    hmutils: Any
    mesh_io: Any


@dataclass(frozen=True)
class SurfaceResult:
    surfaces: tuple[Path, ...]
    relabel_surfaces: tuple[Path, ...]
    vertex_density: float


@dataclass(frozen=True)
class MeshValidation:
    mesh_path: str
    tetrahedra: int
    tissue_tags: tuple[int, ...]
    minimum_tetrahedron_volume: float


def load_v3_meshing_api() -> V3MeshingAPI:
    """Import SimNIBS only when running inside the pinned HPC module."""
    simnibs = importlib.import_module("simnibs")
    return V3MeshingAPI(
        simnibs=simnibs,
        headreco=importlib.import_module("simnibs.msh.headreco"),
        hmutils=importlib.import_module("simnibs.msh.hmutils"),
        mesh_io=importlib.import_module("simnibs.msh.mesh_io"),
    )


def probe_v3_meshing_api(api: V3MeshingAPI) -> dict[str, object]:
    """Inspect, but never execute, the version-pinned meshing entry points."""
    version = str(getattr(api.simnibs, "__version__", ""))
    if version != EXPECTED_VERSION:
        raise MeshingAPIError(
            f"Expected SimNIBS version {EXPECTED_VERSION}, found {version or 'unknown'}"
        )
    surface = _required_callable(api.headreco, "make_surface_meshes")
    volume = _required_callable(api.headreco, "make_volume_mesh")
    signatures = {
        "make_surface_meshes": str(inspect.signature(surface)),
        "make_volume_mesh": str(inspect.signature(volume)),
    }
    if signatures["make_surface_meshes"] != EXPECTED_SURFACE_SIGNATURE:
        raise MeshingAPIError(
            "Unexpected make_surface_meshes signature: "
            f"{signatures['make_surface_meshes']}"
        )
    if signatures["make_volume_mesh"] != EXPECTED_VOLUME_SIGNATURE:
        raise MeshingAPIError(
            f"Unexpected make_volume_mesh signature: {signatures['make_volume_mesh']}"
        )
    for name in ("relabel_compartments", "volume_mesh_relabel", "path2bin"):
        _required_callable(api.hmutils, name)
    read_msh = _required_callable(api.mesh_io, "read_msh")
    _required_callable(api.mesh_io, "write_msh")
    if "buffered" not in inspect.signature(read_msh).parameters:
        raise MeshingAPIError("read_msh does not support the required buffered argument")
    mesh_type = getattr(api.mesh_io, "Msh", None)
    if mesh_type is None:
        raise MeshingAPIError("mesh_io.Msh is missing")
    for name in MESH_CLEANUP_METHODS:
        if not callable(getattr(mesh_type, name, None)):
            raise MeshingAPIError(f"mesh_io.Msh.{name} is missing or not callable")
    binaries = {}
    for name in ("gmsh", "meshfix"):
        resolved = api.hmutils.path2bin(name)
        if not resolved:
            raise MeshingAPIError(f"Could not resolve required {name} binary")
        binaries[name] = str(resolved)
    module_files = {
        name: str(getattr(module, "__file__", "unknown"))
        for name, module in (
            ("simnibs", api.simnibs),
            ("headreco", api.headreco),
            ("hmutils", api.hmutils),
            ("mesh_io", api.mesh_io),
        )
    }
    return {
        "status": "ready",
        "simnibs_version": version,
        "module_files": module_files,
        "signatures": signatures,
        "binaries": binaries,
        "mesh_cleanup_methods": list(MESH_CLEANUP_METHODS),
    }


def build_surfaces(
    api: V3MeshingAPI,
    mask_paths: Mapping[str, Path],
    m2m_dir: Path,
    scratch_dir: Path,
    vertex_density: float = 0.5,
) -> SurfaceResult:
    probe_v3_meshing_api(api)
    required = {"WM", "GM", "CSF", "BONE", "SKIN", "EYES", "AIR"}
    if set(mask_paths) != required:
        raise MeshValidationError(
            "Surface-mask set differs from the pinned contract: "
            f"missing={sorted(required - set(mask_paths))}, "
            f"extra={sorted(set(mask_paths) - required)}"
        )
    masks = sorted(
        (Path(path).resolve(strict=True) for path in mask_paths.values()),
        key=lambda path: path.name.lower(),
    )
    if not np.isfinite(vertex_density) or vertex_density <= 0:
        raise MeshValidationError("vertex_density must be finite and positive")
    m2m_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    result = api.headreco.make_surface_meshes(
        [str(path) for path in masks],
        str(m2m_dir),
        str(scratch_dir),
        vertex_density=vertex_density,
        catsurf=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise MeshValidationError(
            "make_surface_meshes did not return (surfaces, relabel_surfaces)"
        )
    surfaces = _validated_surface_paths(result[0], m2m_dir, "surface")
    relabel = _validated_surface_paths(
        result[1], m2m_dir, "relabel surface", allow_empty=True
    )
    if len(surfaces) < 6:
        raise MeshValidationError(
            f"Surface generation returned only {len(surfaces)} primary surfaces"
        )
    return SurfaceResult(surfaces, relabel, float(vertex_density))


def build_volume_mesh(
    api: V3MeshingAPI,
    subject: str,
    surfaces: Sequence[Path],
    relabel_surfaces: Sequence[Path],
    m2m_dir: Path,
    output_mesh: Path,
    keep_air: bool = False,
) -> MeshValidation:
    probe_v3_meshing_api(api)
    surface_paths = _validated_surface_paths(surfaces, m2m_dir, "surface")
    relabel_paths = _validated_surface_paths(
        relabel_surfaces, m2m_dir, "relabel surface", allow_empty=True
    )
    surface_paths = tuple(sorted(surface_paths, key=lambda path: path.name.lower()))
    relabel_paths = tuple(sorted(relabel_paths, key=lambda path: path.name.lower()))
    output_mesh = output_mesh.resolve(strict=False)
    if output_mesh.exists():
        raise MeshValidationError(f"Refusing to overwrite existing mesh: {output_mesh}")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)

    raw_mesh = Path(
        api.headreco.make_volume_mesh(
            subject,
            [str(path) for path in surface_paths],
            str(m2m_dir),
            bool(keep_air),
        )
    ).resolve(strict=True)
    mesh = api.mesh_io.read_msh(str(raw_mesh), buffered=True)
    if any("ventricles" in path.name.lower() for path in surface_paths):
        mesh = api.hmutils.relabel_compartments(mesh, [9, 1009], [3, 1003])
    if relabel_paths:
        mesh = api.hmutils.volume_mesh_relabel(
            mesh,
            [str(path) for path in relabel_paths],
            5,
        )
    mesh.fn = str(output_mesh)
    mesh.fix_surface_labels()
    mesh.fix_thin_tetrahedra()
    mesh.fix_th_node_ordering()
    mesh.fix_tr_node_ordering()
    api.mesh_io.write_msh(mesh, str(output_mesh))
    validation = validate_mesh(api, output_mesh)
    if raw_mesh != output_mesh:
        raw_mesh.unlink()
    return validation


def validate_mesh(api: V3MeshingAPI, mesh_path: Path) -> MeshValidation:
    mesh_path = mesh_path.resolve(strict=True)
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise MeshValidationError(f"Mesh is missing or empty: {mesh_path}")
    mesh = api.mesh_io.read_msh(str(mesh_path), buffered=True)
    try:
        element_types = np.asarray(mesh.elm.elm_type)
        tags = np.asarray(mesh.elm.tag1)
    except AttributeError as exc:
        raise MeshValidationError("Mesh lacks element type/tag arrays") from exc
    tetrahedra = element_types == 4
    tetrahedron_count = int(np.count_nonzero(tetrahedra))
    if tetrahedron_count == 0:
        raise MeshValidationError("Mesh contains no tetrahedra")
    tissue_tags = tuple(sorted(int(value) for value in np.unique(tags[tetrahedra])))
    if tissue_tags != (1, 2, 3, 4, 5, 6):
        raise MeshValidationError(
            "Mesh tetrahedral tissue tags must be exactly (1, 2, 3, 4, 5, 6), "
            f"found {tissue_tags}"
        )
    volume_result = mesh.elements_volumes_and_areas()
    volumes = np.asarray(getattr(volume_result, "value", volume_result), dtype=float)
    if volumes.ndim != 1:
        volumes = volumes.reshape(-1)
    if volumes.size == element_types.size:
        tetrahedron_volumes = volumes[tetrahedra]
    elif volumes.size == tetrahedron_count:
        tetrahedron_volumes = volumes
    else:
        raise MeshValidationError(
            "Mesh volume array length does not match its tetrahedral elements"
        )
    if not np.all(np.isfinite(tetrahedron_volumes)):
        raise MeshValidationError("Mesh tetrahedral volumes must all be finite")
    if not np.all(tetrahedron_volumes > 0):
        raise MeshValidationError("Mesh tetrahedral volumes must all be positive")
    return MeshValidation(
        mesh_path=str(mesh_path),
        tetrahedra=tetrahedron_count,
        tissue_tags=tissue_tags,
        minimum_tetrahedron_volume=float(np.min(tetrahedron_volumes)),
    )


def _required_callable(owner: Any, name: str):
    value = getattr(owner, name, None)
    if not callable(value):
        raise MeshingAPIError(f"Required callable {name} is missing")
    return value


def _validated_surface_paths(
    values: Sequence[str | Path],
    m2m_dir: Path,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[Path, ...]:
    paths = tuple(Path(value).resolve(strict=True) for value in values)
    if not paths and not allow_empty:
        raise MeshValidationError(f"No {label} files were provided")
    root = m2m_dir.resolve(strict=False)
    for path in paths:
        if path.suffix.lower() != ".stl" or not path.is_file():
            raise MeshValidationError(f"Invalid {label} output: {path}")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise MeshValidationError(f"{label.title()} escapes m2m directory: {path}") from exc
    return paths
