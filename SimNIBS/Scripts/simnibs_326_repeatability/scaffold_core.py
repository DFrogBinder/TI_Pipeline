"""Request-driven SimNIBS 3.2.6 ``headreco`` scaffold builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


READY_MARKER = ".simnibs326_scaffold_ready.json"


@dataclass(frozen=True)
class ScaffoldRequest:
    subject: str
    t1: Path
    t2: Path
    corrected_labels: Path
    scaffold_root: Path
    simnibs_module: str
    cat12_compat_script: Path
    cat12_patch_id: str
    head_model_strategy: str
    segmentation_provenance: str
    workflow: str


@dataclass(frozen=True)
class ScaffoldPaths:
    anat: Path
    m2m: Path
    native_mesh: Path
    compatibility_mesh: Path
    ready_marker: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _simnibs_version() -> str:
    import simnibs  # type: ignore

    return str(getattr(simnibs, "__version__", "unknown"))


def scaffold_paths(request: ScaffoldRequest) -> ScaffoldPaths:
    anat = request.scaffold_root / request.subject / "anat"
    m2m = anat / f"m2m_{request.subject}"
    return ScaffoldPaths(
        anat=anat,
        m2m=m2m,
        native_mesh=anat / f"{request.subject}.msh",
        compatibility_mesh=m2m / f"{request.subject}.msh",
        ready_marker=anat / READY_MARKER,
    )


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _nonempty_directory(path: Path) -> bool:
    return path.is_dir() and any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in path.rglob("*")
    )


def _headreco_command(request: ScaffoldRequest) -> list[str]:
    return [
        sys.executable,
        str(request.cat12_compat_script),
        "headreco",
        "all",
        "--noclean",
        request.subject,
        str(request.t1),
        str(request.t2),
    ]


def _marker_binding(
    request: ScaffoldRequest, paths: ScaffoldPaths
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "ready",
        "subject": request.subject,
        "workflow": request.workflow,
        "simnibs_module": request.simnibs_module,
        "cat12_compatibility_patch": request.cat12_patch_id,
        "head_model_strategy": request.head_model_strategy,
        "segmentation_provenance": request.segmentation_provenance,
        "source_t1": str(request.t1),
        "source_t1_sha256": sha256_file(request.t1),
        "source_t2": str(request.t2),
        "source_t2_sha256": sha256_file(request.t2),
        "corrected_v4_labels_provenance_only": str(request.corrected_labels),
        "corrected_v4_labels_sha256": sha256_file(request.corrected_labels),
        "native_mesh": str(paths.native_mesh),
        "native_mesh_sha256": sha256_file(paths.native_mesh),
        "compatibility_mesh": str(paths.compatibility_mesh),
        "compatibility_mesh_sha256": sha256_file(paths.compatibility_mesh),
        "command": _headreco_command(request),
    }


def _marker_matches(
    request: ScaffoldRequest, paths: ScaffoldPaths
) -> tuple[bool, str | None]:
    if not _nonempty_file(paths.ready_marker):
        return False, "ready marker is missing or empty"
    try:
        marker = json.loads(paths.ready_marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"ready marker is unreadable: {exc}"
    if not isinstance(marker, dict):
        return False, "ready marker is not a JSON object"
    try:
        expected = _marker_binding(request, paths)
    except (FileNotFoundError, OSError):
        return False, "marker inputs or meshes are missing"

    # ``workflow`` and the compatibility-mesh hash were added by this shared
    # core. Their absence is accepted for an otherwise fully bound historical
    # balanced-10 marker; newly written markers always contain both fields.
    optional_legacy_fields = {"workflow", "compatibility_mesh_sha256"}
    specially_compared_fields = {"command"}
    differences = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if key not in optional_legacy_fields
        and key not in specially_compared_fields
        and marker.get(key) != value
    }
    for key in optional_legacy_fields:
        if key in marker and marker[key] != expected[key]:
            differences[key] = {
                "expected": expected[key],
                "observed": marker[key],
            }
    expected_command = expected["command"]
    observed_command = marker.get("command")
    command_matches = bool(
        isinstance(expected_command, list)
        and isinstance(observed_command, list)
        and len(observed_command) == len(expected_command)
        and observed_command
        and isinstance(observed_command[0], str)
        and observed_command[0]
        and observed_command[1:] == expected_command[1:]
    )
    if not command_matches:
        differences["command"] = {
            "expected": expected_command,
            "observed": observed_command,
        }
    if differences:
        return False, json.dumps(differences, sort_keys=True)
    return True, None


def validate_scaffold(request: ScaffoldRequest) -> dict[str, object]:
    paths = scaffold_paths(request)
    surfaces = (
        sorted(path for path in paths.m2m.glob("*.stl") if _nonempty_file(path))
        if paths.m2m.is_dir()
        else []
    )
    required = {
        "source_t1": _nonempty_file(request.t1),
        "source_t2": _nonempty_file(request.t2),
        "corrected_v4_labels_provenance_only": _nonempty_file(request.corrected_labels),
        "m2m_directory": paths.m2m.is_dir(),
        "t1fs_conform": _nonempty_file(paths.m2m / "T1fs_conform.nii.gz"),
        "t1fs_nu_conform": _nonempty_file(paths.m2m / "T1fs_nu_conform.nii.gz"),
        "to_mni": _nonempty_directory(paths.m2m / "toMNI"),
        "eeg_positions": _nonempty_directory(paths.m2m / "eeg_positions"),
        "surface_meshes": len(surfaces) >= 5,
        "native_mesh": _nonempty_file(paths.native_mesh),
        "compatibility_mesh": _nonempty_file(paths.compatibility_mesh),
        "ready_marker": _nonempty_file(paths.ready_marker),
    }
    required["mesh_copies_match"] = bool(
        required["native_mesh"]
        and required["compatibility_mesh"]
        and sha256_file(paths.native_mesh) == sha256_file(paths.compatibility_mesh)
    )
    marker_matches, marker_error = _marker_matches(request, paths)
    required["ready_marker_binding"] = marker_matches
    return {
        "subject": request.subject,
        "status": "ready" if all(required.values()) else "incomplete",
        "checks": required,
        "surface_count": len(surfaces),
        "marker_error": marker_error,
        "paths": {
            "anat": str(paths.anat),
            "m2m": str(paths.m2m),
            "native_mesh": str(paths.native_mesh),
            "compatibility_mesh": str(paths.compatibility_mesh),
            "ready_marker": str(paths.ready_marker),
        },
    }


def _validate_inputs(request: ScaffoldRequest) -> None:
    for label, path in (
        ("T1 input", request.t1),
        ("T2 input", request.t2),
        ("corrected-label input", request.corrected_labels),
        ("CAT12 compatibility input", request.cat12_compat_script),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty {label}: {path}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_scaffold(request: ScaffoldRequest) -> dict[str, object]:
    _validate_inputs(request)
    paths = scaffold_paths(request)
    existing = validate_scaffold(request)
    if existing["status"] == "ready":
        existing["action"] = "reused"
        return existing

    if paths.anat.exists():
        shutil.rmtree(paths.anat)
    paths.anat.mkdir(parents=True, exist_ok=True)
    command = _headreco_command(request)
    print(json.dumps({"event": "headreco_scaffold_start", "command": command}))
    subprocess.run(command, cwd=paths.anat, check=True)

    if not _nonempty_file(paths.native_mesh):
        raise FileNotFoundError(
            f"headreco completed without its native mesh: {paths.native_mesh}"
        )
    if not paths.m2m.is_dir():
        raise FileNotFoundError(f"headreco completed without m2m folder: {paths.m2m}")
    shutil.copy2(paths.native_mesh, paths.compatibility_mesh)

    marker = {
        **_marker_binding(request, paths),
        "created_utc": _utc_now(),
        "simnibs_version": _simnibs_version(),
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
    }
    _atomic_json(paths.ready_marker, marker)
    result = validate_scaffold(request)
    if result["status"] != "ready":
        raise RuntimeError(f"Scaffold validation failed: {json.dumps(result)}")
    result["action"] = "built"
    return result
