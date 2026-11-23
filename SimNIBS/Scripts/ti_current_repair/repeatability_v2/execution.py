"""Task-level remesh and exact-geometry fixed execution primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True)
class SimulationTask:
    subject: str
    target: str
    condition: str
    repeat: int

    def __post_init__(self) -> None:
        if self.condition not in {"remesh", "fixed_mesh"}:
            raise ValueError(f"Unsupported condition: {self.condition}")
        if self.target not in {"left_hippocampus", "right_m1"}:
            raise ValueError(f"Unsupported target: {self.target}")
        if not 1 <= self.repeat <= 40:
            raise ValueError("repeat must be in 1..40")

    @property
    def repeat_tag(self) -> str:
        return f"repeat_{self.repeat:03d}"


@dataclass(frozen=True)
class SourceInputs:
    corrected_segmentation: Path
    atlas: Path
    targets_csv: Path

    @property
    def paths(self) -> tuple[Path, ...]:
        return self.corrected_segmentation, self.atlas, self.targets_csv


class ExecutionAdapter(Protocol):
    version: str

    def rebuild_head(self, segmentation: Path, output: Path) -> Path: ...

    def construct_electrode_geometry(
        self,
        head_mesh: Path,
        *,
        pair: int,
        target: str,
        output: Path,
        seed: int | None,
    ) -> Path: ...

    def solve_component(self, geometry: Path, output: Path) -> Path: ...

    def solve_existing_mesh(
        self, geometry: Path, output: Path, metadata: dict
    ) -> Path: ...

    def combine_ti(self, pair1: Path, pair2: Path, output: Path) -> Path: ...

    def geometry_sha256(self, path: Path) -> str: ...

    def geometry_metadata(self, pair: int) -> dict[str, object]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _task_root(output_root: Path, task: SimulationTask) -> Path:
    return output_root / task.subject / task.target / task.condition / task.repeat_tag


def _validate_inputs(
    inputs: SourceInputs, expected_hashes: Mapping[Path, str]
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for path in inputs.paths:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing or unsafe v2 source input: {path}")
        digest = _sha256(path)
        expected = expected_hashes.get(path)
        if expected != digest:
            raise ValueError(f"Input hash mismatch for {path}: {digest} != {expected}")
        observed[str(path.resolve())] = digest
    return observed


def _validate_existing_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    for record in payload.get("outputs", []):
        output = Path(record["path"])
        if not output.is_file() or _sha256(output) != record["sha256"]:
            return None
    return payload


def run_remesh_task(
    *,
    task: SimulationTask,
    inputs: SourceInputs,
    expected_hashes: Mapping[Path, str],
    output_root: Path,
    adapter: ExecutionAdapter,
) -> dict[str, object]:
    if task.condition != "remesh":
        raise ValueError("run_remesh_task requires condition='remesh'")
    if adapter.version != "4.6.0":
        raise ValueError(f"Expected SimNIBS 4.6.0 adapter, observed {adapter.version}")
    source_hashes = _validate_inputs(inputs, expected_hashes)
    root = _task_root(output_root, task)
    manifest_path = root / "task_manifest.json"
    existing = _validate_existing_manifest(manifest_path)
    if existing is not None:
        return {**existing, "resumed_valid_output": True}
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete v2 task root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    head = adapter.rebuild_head(inputs.corrected_segmentation, root / "head.msh")
    events.append({"operation": "head_mesh_rebuilt", "path": str(head)})
    geometry_paths: dict[str, str] = {}
    geometry_hashes: dict[str, str] = {}
    currents: dict[str, object] = {}
    surface_tags: dict[str, object] = {}
    electrode_names: dict[str, object] = {}
    conductivity_by_tag: dict[str, object] | None = None
    solver_options: object = None
    n_workers: int | None = None
    component_paths: dict[str, Path] = {}
    for pair in (1, 2):
        geometry = adapter.construct_electrode_geometry(
            head,
            pair=pair,
            target=task.target,
            output=root / f"pair{pair}_geometry.msh",
            seed=None,
        )
        geometry_paths[str(pair)] = str(geometry)
        geometry_hashes[str(pair)] = adapter.geometry_sha256(geometry)
        metadata = adapter.geometry_metadata(pair)
        if int(metadata.get("pair", -1)) != pair:
            raise ValueError(f"Adapter returned mismatched metadata for pair {pair}")
        currents[str(pair)] = metadata["currents_a"]
        surface_tags[str(pair)] = metadata["electrode_surface_tags"]
        electrode_names[str(pair)] = metadata["electrode_names"]
        current_conductivity = {
            str(key): value
            for key, value in dict(metadata["conductivity_by_tag"]).items()
        }
        if conductivity_by_tag is None:
            conductivity_by_tag = current_conductivity
        elif conductivity_by_tag != current_conductivity:
            raise ValueError("Pair-specific conductivity provenance differs")
        if solver_options is None:
            solver_options = metadata.get("solver_options")
        elif solver_options != metadata.get("solver_options"):
            raise ValueError("Pair-specific solver options differ")
        current_workers = int(metadata.get("n_workers", 8))
        if n_workers is None:
            n_workers = current_workers
        elif n_workers != current_workers:
            raise ValueError("Pair-specific worker counts differ")
        events.append(
            {
                "operation": "electrode_geometry_constructed",
                "pair": pair,
                "seed": None,
                "path": str(geometry),
                "geometry_sha256": geometry_hashes[str(pair)],
            }
        )
        component = adapter.solve_component(
            geometry, root / f"pair{pair}_field.msh"
        )
        component_paths[str(pair)] = component
        events.append(
            {"operation": "component_solved", "pair": pair, "path": str(component)}
        )
    ti_path = adapter.combine_ti(
        component_paths["1"], component_paths["2"], root / "TI.msh"
    )
    events.append({"operation": "ti_combined", "path": str(ti_path)})
    outputs = [head, *map(Path, geometry_paths.values()), *component_paths.values(), ti_path]
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "simnibs_version": adapter.version,
        "subject": task.subject,
        "target": task.target,
        "condition": task.condition,
        "repeat": task.repeat,
        "repeat_tag": task.repeat_tag,
        "source_hashes": source_hashes,
        "head_mesh_policy": "rebuild_per_repeat",
        "electrode_mesh_policy": "rebuild_unseeded_per_repeat",
        "component_geometry_paths": geometry_paths,
        "geometry_sha256": geometry_hashes,
        "currents_a": currents,
        "conductivity_by_tag": conductivity_by_tag,
        "electrode_surface_tags": surface_tags,
        "electrode_names": electrode_names,
        "solver_options": solver_options,
        "n_workers": n_workers,
        "component_field_paths": {
            key: str(value) for key, value in component_paths.items()
        },
        "ti_msh_path": str(ti_path),
        "events": events,
        "nifti_role": "not_generated_authoritative_stage",
        "outputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in outputs
        ],
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def run_fixed_task(
    *,
    task: SimulationTask,
    cache_manifest: Path,
    output_root: Path,
    adapter: ExecutionAdapter,
) -> dict[str, object]:
    if task.condition != "fixed_mesh":
        raise ValueError("run_fixed_task requires condition='fixed_mesh'")
    if adapter.version != "4.6.0":
        raise ValueError(f"Expected SimNIBS 4.6.0 adapter, observed {adapter.version}")
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cache.get("status") != "ready":
        raise ValueError(f"Fixed cache is not ready: {cache_manifest}")
    if cache.get("subject") != task.subject or cache.get("target") != task.target:
        raise ValueError("Fixed cache does not match the simulation task")
    root = _task_root(output_root, task)
    manifest_path = root / "task_manifest.json"
    existing = _validate_existing_manifest(manifest_path)
    if existing is not None:
        return {**existing, "resumed_valid_output": True}
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete v2 task root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, object]] = []
    component_paths: dict[str, Path] = {}
    observed_geometry: dict[str, str] = {}
    for pair in (1, 2):
        key = str(pair)
        geometry = Path(cache["pair_caches"][key])
        observed = adapter.geometry_sha256(geometry)
        expected = cache["geometry_sha256"][key]
        if observed != expected:
            raise ValueError(
                f"Fixed cache geometry changed for pair {pair}: {observed} != {expected}"
            )
        observed_geometry[key] = observed
        metadata = {
            "pair": pair,
            "currents_a": cache["currents_a"][key],
            "conductivity_by_tag": cache["conductivity_by_tag"],
            "electrode_surface_tags": cache["electrode_surface_tags"][key],
            "solver_options": cache["solver_options"],
            "n_workers": int(cache.get("n_workers", 8)),
        }
        result = adapter.solve_existing_mesh(
            geometry, root / f"pair{pair}_field.msh", metadata
        )
        component_paths[key] = result
        events.append(
            {
                "operation": "existing_mesh_component_solved",
                "pair": pair,
                "geometry_sha256": observed,
                "path": str(result),
            }
        )
    ti_path = adapter.combine_ti(
        component_paths["1"], component_paths["2"], root / "TI.msh"
    )
    events.append({"operation": "ti_combined", "path": str(ti_path)})
    outputs = [*component_paths.values(), ti_path]
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "complete",
        "simnibs_version": adapter.version,
        "subject": task.subject,
        "target": task.target,
        "condition": task.condition,
        "repeat": task.repeat,
        "repeat_tag": task.repeat_tag,
        "cache_manifest": str(cache_manifest.resolve()),
        "source_remesh_repeat": cache["source_repeat"],
        "geometry_sha256": observed_geometry,
        "component_field_paths": {
            key: str(value) for key, value in component_paths.items()
        },
        "ti_msh_path": str(ti_path),
        "electrode_placement_invoked": False,
        "events": events,
        "outputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in outputs
        ],
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}

