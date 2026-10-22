"""SimNIBS 3.2.6 backend for promoted corrected-label head models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent
CURRENT_REPAIR_ROOT = SCRIPTS_ROOT / "ti_current_repair"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
_added_current_repair_path = str(CURRENT_REPAIR_ROOT) not in sys.path
if _added_current_repair_path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))
try:
    from simulation_runners import repeatability_experiment as runner  # noqa: E402
finally:
    if _added_current_repair_path:
        sys.path.remove(str(CURRENT_REPAIR_ROOT))

from .headmodels import validate_headmodel_manifest  # noqa: E402
from .settings import (  # noqa: E402
    CampaignPaths,
    SIMNIBS_MODULE,
    SUBJECTS,
    campaign_paths_from_environment,
    subject_paths,
)
from .v3_meshing import build_volume_mesh, load_v3_meshing_api  # noqa: E402


BACKEND_NAME = "simnibs-3.2.6-harmonized-corrected-label-v2"
CONFIG_RUNTIME_KEY = "simnibs_326_harmonized_runtime"
_CONFIG_PATH: Path | None = None
_CAMPAIGN_PATHS: CampaignPaths | None = None
_SIMNIBS_VERSION = "unknown"


class SimulationBoundaryError(RuntimeError):
    """Raised when a task cannot prove the promoted head-model boundary."""


def max_ti(E1_org, E2_org):
    """Calculate the Grossman maximal temporal-interference envelope."""
    import numpy as np

    if E1_org.shape != E2_org.shape or E1_org.ndim != 2 or E1_org.shape[1] != 3:
        raise ValueError("E1 and E2 must have the same N x 3 shape")
    E1 = E1_org.copy()
    E2 = E2_org.copy()
    swap = np.linalg.norm(E2, axis=1) > np.linalg.norm(E1, axis=1)
    E1[swap] = E2[swap]
    E2[swap] = E1_org[swap]
    oppose = np.sum(E1 * E2, axis=1) < 0
    E2[oppose] = -E2[oppose]
    norm1 = np.linalg.norm(E1, axis=1)
    norm2 = np.linalg.norm(E2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_alpha = np.sum(E1 * E2, axis=1) / (norm1 * norm2)
        envelope = (
            2.0
            * np.linalg.norm(np.cross(E2, E1 - E2), axis=1)
            / np.linalg.norm(E1 - E2, axis=1)
        )
    aligned = norm2 <= norm1 * cos_alpha
    envelope[aligned] = 2.0 * norm2[aligned]
    envelope[(norm1 == 0.0) & (norm2 == 0.0)] = 0.0
    return envelope


class _TIAdapter:
    get_maxTI = staticmethod(max_ti)


class _SimNIBS326Adapter:
    def __init__(self, module):
        self._module = module
        self.__version__ = str(getattr(module, "__version__", "unknown"))

    def run_simnibs(self, session):
        session.subpath = str(Path(session.fnamehead).parent)
        session.open_in_gmsh = False
        session.map_to_vol = False
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        return session.run(cpus=cpus)


def _ensure_simnibs_imports_326() -> None:
    global _SIMNIBS_VERSION
    if runner.SIM_MODULE is not None:
        return
    runner._ensure_scientific_imports()
    import simnibs as simnibs_module  # type: ignore

    try:
        from simnibs import mesh_io, sim_struct  # type: ignore
    except ImportError:
        from simnibs.msh import mesh_io  # type: ignore
        from simnibs.simulation import sim_struct  # type: ignore

    _SIMNIBS_VERSION = str(getattr(simnibs_module, "__version__", "unknown"))
    if _SIMNIBS_VERSION != "3.2.6":
        raise SimulationBoundaryError(
            f"Expected SimNIBS 3.2.6 from {SIMNIBS_MODULE}; observed {_SIMNIBS_VERSION}"
        )
    runner.SIM_MODULE = _SimNIBS326Adapter(simnibs_module)
    runner.SIM_MESH_IO = mesh_io
    runner.SIM_STRUCT = sim_struct
    runner.SIM_TI = _TIAdapter()


def _write_label_volume_326(
    mesh_path: Path,
    reference_path: Path,
    output_path: Path,
) -> Path:
    """Export v3 tetrahedral tissue tags with the established assign method."""
    import nibabel as nib

    _ensure_simnibs_imports_326()
    mesh = runner.SIM_MESH_IO.read_msh(str(mesh_path))
    mesh = mesh.crop_mesh(elm_type=4)
    element_data_type = getattr(runner.SIM_MESH_IO, "ElementData", None)
    if not callable(element_data_type) or not callable(
        getattr(element_data_type, "to_nifti", None)
    ):
        raise SimulationBoundaryError(
            "SimNIBS 3.2.6 mesh_io.ElementData.to_nifti is unavailable"
        )
    reference = nib.load(str(reference_path))
    destination = output_path if output_path.suffix else Path(f"{output_path}.nii.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = element_data_type(mesh.elm.tag1)
    labels.mesh = mesh
    labels.to_nifti(
        reference.header["dim"][1:4],
        reference.affine,
        fn=str(destination),
        qform=reference.header.get_qform(),
        method="assign",
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise SimulationBoundaryError(f"Label export did not create {destination}")
    runner.log_event(
        "label_volume_exported",
        implementation="simnibs-3.2.6-mesh-io-element-tags",
        mesh_path=str(mesh_path),
        reference_path=str(reference_path),
        output_path=str(destination),
    )
    return destination


def _validated_promotion(paths: CampaignPaths) -> dict[str, str]:
    manifest = validate_headmodel_manifest(paths)
    promotion_path = paths.headmodel_root / "_pipeline" / "headmodels" / "promoted.json"
    if not promotion_path.is_file():
        raise SimulationBoundaryError("Head models are not promoted")
    try:
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationBoundaryError("Head-model promotion receipt is unreadable") from exc
    manifest_hash = str(manifest["manifest_sha256"])
    if (
        promotion.get("status") != "promoted"
        or promotion.get("confirm_reviewed") is not True
        or promotion.get("manifest_sha256") != manifest_hash
        or promotion.get("subjects") != list(SUBJECTS)
    ):
        raise SimulationBoundaryError(
            "Head-model promotion receipt is stale or incomplete"
        )
    return {
        "manifest_sha256": manifest_hash,
        "promotion_sha256": _sha256(promotion_path),
        "promotion_path": str(promotion_path.resolve()),
    }


def _headmodel_marker(paths: CampaignPaths, subject: str) -> dict[str, Any]:
    resolved = subject_paths(paths, subject)
    marker_path = resolved.headmodel_receipt
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationBoundaryError(
            f"Head-model marker is missing or unreadable for {subject}: {marker_path}"
        ) from exc
    if marker.get("status") != "complete" or marker.get("subject") != subject:
        raise SimulationBoundaryError(f"Head-model marker is incomplete for {subject}")
    for key in ("harmonized_map", "mesh"):
        _validate_hash_record(marker.get(key), key)
    surfaces = marker.get("surfaces")
    if not isinstance(surfaces, list) or len(surfaces) < 5:
        raise SimulationBoundaryError(f"Head-model surfaces are incomplete for {subject}")
    for record in surfaces:
        _validate_hash_record(record, "surface")
        path = Path(str(record["path"])).resolve()
        if not _is_within(path, resolved.staged_m2m.resolve()):
            raise SimulationBoundaryError(f"Promoted surface escapes m2m root: {path}")
    return marker


def _build_remesh_workspace(
    paths: CampaignPaths,
    subject: str,
    workspace,
    *,
    force_mesh: bool,
) -> Path:
    """Stage only promoted references/surfaces and invoke the direct v3 mesher."""
    promotion = _validated_promotion(paths)
    resolved = subject_paths(paths, subject)
    marker = _headmodel_marker(paths, subject)
    workspace.anat_dir.mkdir(parents=True, exist_ok=True)
    if workspace.mesh_dir.exists() or workspace.mesh_dir.is_symlink():
        if workspace.mesh_dir.is_symlink() or workspace.mesh_dir.is_file():
            workspace.mesh_dir.unlink()
        else:
            shutil.rmtree(workspace.mesh_dir)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{workspace.mesh_dir.name}.stage-", dir=workspace.anat_dir)
    )
    promoted = False
    try:
        for name in ("T1fs_conform.nii.gz", "T1fs_nu_conform.nii.gz"):
            _copy_regular(resolved.staged_m2m / name, temporary / name)
        for name in ("eeg_positions", "toMNI"):
            _copy_tree_no_symlinks(resolved.staged_m2m / name, temporary / name)
        staged_surfaces = []
        staged_relabel = []
        for record in marker["surfaces"]:
            source = Path(str(record["path"]))
            destination = temporary / source.name
            _copy_regular(source, destination)
            if _sha256(destination) != record["sha256"]:
                raise SimulationBoundaryError(f"Staged surface hash mismatch: {source}")
            if "air_outer_relabel" in source.name.lower():
                staged_relabel.append(destination)
            else:
                staged_surfaces.append(destination)
        api = load_v3_meshing_api()
        temporary_mesh = temporary / f"{subject}.msh"
        build_volume_mesh(
            api,
            subject,
            staged_surfaces,
            staged_relabel,
            temporary,
            temporary_mesh,
            keep_air=False,
        )
        os.replace(temporary, workspace.mesh_dir)
        promoted = True
        mesh_path = workspace.mesh_path
        payload = {
            "schema_version": 1,
            "status": "mesh_ready",
            "created_at": time.time(),
            "subject": subject,
            "backend": BACKEND_NAME,
            "force_mesh": bool(force_mesh),
            "manifest_sha256": promotion["manifest_sha256"],
            "promotion_path": promotion["promotion_path"],
            "promotion_sha256": promotion["promotion_sha256"],
            "headmodel_marker_path": str(resolved.headmodel_receipt.resolve()),
            "headmodel_marker_sha256": _sha256(resolved.headmodel_receipt),
            "harmonized_map_path": str(resolved.harmonized_label.resolve()),
            "harmonized_map_sha256": _sha256(resolved.harmonized_label),
            "mesh_path": str(mesh_path.resolve()),
            "mesh_sha256": _sha256(mesh_path),
        }
        _atomic_write_json(workspace.anat_dir / ".mesh_ready.json", payload)
        return mesh_path
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)


def _mesh_workspace_harmonized(
    workspace,
    *,
    subject: str,
    force_mesh: bool,
    paths: CampaignPaths | None = None,
) -> Path:
    resolved_paths = paths or _CAMPAIGN_PATHS or campaign_paths_from_environment()
    promotion = _validated_promotion(resolved_paths)
    marker_path = workspace.anat_dir / ".mesh_ready.json"
    lock_path = workspace.anat_dir / ".mesh_build.lock"
    if workspace.mesh_path.is_file() and marker_path.is_file() and not force_mesh:
        marker = _load_json(marker_path)
        if marker.get("seed_mode") == "physical_cache_copy":
            _validate_seeded_fixed_mesh(resolved_paths, workspace, subject, marker)
            return workspace.mesh_path
        _validate_remesh_ready(workspace, subject, marker, promotion)
        return workspace.mesh_path
    with runner._exclusive_lock(lock_path):
        if workspace.mesh_path.is_file() and marker_path.is_file() and not force_mesh:
            marker = _load_json(marker_path)
            if marker.get("seed_mode") == "physical_cache_copy":
                _validate_seeded_fixed_mesh(resolved_paths, workspace, subject, marker)
            else:
                _validate_remesh_ready(workspace, subject, marker, promotion)
            return workspace.mesh_path
        marker_path.unlink(missing_ok=True)
        return _build_remesh_workspace(
            resolved_paths,
            subject,
            workspace,
            force_mesh=force_mesh,
        )


def _validate_seeded_fixed_mesh(
    paths: CampaignPaths,
    workspace,
    subject: str,
    marker: dict[str, Any],
) -> None:
    if marker.get("status") != "mesh_ready" or marker.get("subject") != subject:
        raise SimulationBoundaryError("Selected fixed-mesh cache marker is incomplete")
    checksum = _sha256(workspace.mesh_path)
    if marker.get("mesh_checksum") != checksum:
        raise SimulationBoundaryError("Selected fixed-mesh cache hash mismatch")
    source = Path(str(marker.get("source_mesh_path", ""))).resolve(strict=False)
    allowed = (paths.left_root.resolve(), paths.right_root.resolve())
    if not any(_is_within(source, root) for root in allowed):
        raise SimulationBoundaryError(
            f"Selected fixed mesh does not originate in a new harmonized result root: {source}"
        )
    if not source.is_file() or _sha256(source) != checksum:
        raise SimulationBoundaryError("Selected immutable source mesh changed")


def _validate_remesh_ready(
    workspace,
    subject: str,
    marker: dict[str, Any],
    promotion: dict[str, str],
) -> None:
    expected = {
        "status": "mesh_ready",
        "subject": subject,
        "backend": BACKEND_NAME,
        "manifest_sha256": promotion["manifest_sha256"],
        "promotion_sha256": promotion["promotion_sha256"],
        "mesh_sha256": _sha256(workspace.mesh_path),
    }
    differences = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if differences:
        raise SimulationBoundaryError(
            "Existing remesh workspace provenance differs: "
            + json.dumps(differences, sort_keys=True)
        )


def _validate_campaign_config(
    config_path: Path,
    *,
    paths: CampaignPaths | None = None,
) -> None:
    global _CONFIG_PATH, _CAMPAIGN_PATHS
    resolved_paths = paths or campaign_paths_from_environment()
    payload = _load_json(config_path)
    experiment_root = Path(str(payload.get("experiment_root", ""))).resolve(strict=False)
    allowed_roots = {resolved_paths.left_root.resolve(), resolved_paths.right_root.resolve()}
    if experiment_root not in allowed_roots:
        raise SystemExit(
            "Config experiment_root is not a new harmonized result root: "
            f"{experiment_root}"
        )
    source_root = Path(str(payload.get("source_root", ""))).resolve(strict=False)
    if source_root != resolved_paths.corrected_root.resolve():
        raise SystemExit(
            "Config source_root differs from the pinned corrected read-only source"
        )
    if payload.get("subjects") != list(SUBJECTS):
        raise SystemExit("Config subject order differs from the pinned balanced-10 cohort")
    try:
        promotion = _validated_promotion(resolved_paths)
    except SimulationBoundaryError as exc:
        raise SystemExit(str(exc)) from exc
    runtime = payload.get(CONFIG_RUNTIME_KEY)
    if not isinstance(runtime, dict):
        raise SystemExit(f"Config lacks required {CONFIG_RUNTIME_KEY} provenance")
    expected = {
        "simnibs_module": SIMNIBS_MODULE,
        "backend": BACKEND_NAME,
        "headmodel_root": str(resolved_paths.headmodel_root.resolve()),
        "headmodel_manifest_sha256": promotion["manifest_sha256"],
        "headmodel_promotion_sha256": promotion["promotion_sha256"],
    }
    differences = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if differences:
        raise SystemExit(
            "Config promotion/runtime provenance differs: "
            + json.dumps(differences, sort_keys=True)
        )
    _CONFIG_PATH = config_path.resolve()
    _CAMPAIGN_PATHS = resolved_paths


def _runtime_provenance(subject: str) -> dict[str, str]:
    paths = _CAMPAIGN_PATHS or campaign_paths_from_environment()
    promotion = _validated_promotion(paths)
    resolved = subject_paths(paths, subject)
    return {
        "simnibs_module": SIMNIBS_MODULE,
        "simnibs_version": _SIMNIBS_VERSION,
        "backend": BACKEND_NAME,
        "config": str(_CONFIG_PATH) if _CONFIG_PATH else "",
        "headmodel_root": str(paths.headmodel_root.resolve()),
        "headmodel_manifest_sha256": promotion["manifest_sha256"],
        "headmodel_promotion_sha256": promotion["promotion_sha256"],
        "headmodel_marker_sha256": _sha256(resolved.headmodel_receipt),
        "harmonized_map_sha256": _sha256(resolved.harmonized_label),
    }


_ORIGINAL_WRITE_TASK_MANIFEST = runner._write_task_manifest
_ORIGINAL_WRITE_CONDITION_MANIFEST = runner._write_condition_manifest


def _write_task_manifest_326(*args, **kwargs) -> None:
    _ORIGINAL_WRITE_TASK_MANIFEST(*args, **kwargs)
    workspace = args[0] if args else kwargs["workspace"]
    task = kwargs["task"]
    path = workspace.root / "task_manifest.json"
    payload = _load_json(path)
    payload[CONFIG_RUNTIME_KEY] = _runtime_provenance(task.subject)
    ready = workspace.anat_dir / ".mesh_ready.json"
    payload["mesh_ready_sha256"] = _sha256(ready) if ready.is_file() else None
    _atomic_write_json(path, payload)


def _write_condition_manifest_326(*args, **kwargs) -> None:
    _ORIGINAL_WRITE_CONDITION_MANIFEST(*args, **kwargs)
    config = args[0] if args else kwargs["config"]
    subject = kwargs["subject"]
    condition_name = kwargs["condition_name"]
    path = runner.condition_manifest_path(config, subject, condition_name)
    payload = _load_json(path)
    payload[CONFIG_RUNTIME_KEY] = _runtime_provenance(subject)
    _atomic_write_json(path, payload)


def _validate_existing_task_manifest_326(workspace, *, config) -> None:
    path = workspace.root / "task_manifest.json"
    if not path.is_file():
        raise runner.SimulationInputError(f"Missing task provenance: {path}")
    payload = _load_json(path)
    if payload.get("stimulation") != config.stimulation.to_dict():
        raise runner.SimulationInputError(f"Stimulation provenance differs: {path}")
    runtime = payload.get(CONFIG_RUNTIME_KEY)
    subject = str(payload.get("subject", ""))
    if runtime != _runtime_provenance(subject):
        raise runner.SimulationInputError(
            f"Existing task does not match the current promoted head models: {path}"
        )


def install_backend() -> None:
    runner._ensure_simnibs_imports = _ensure_simnibs_imports_326
    runner._mesh_workspace = _mesh_workspace_harmonized
    runner._write_task_manifest = _write_task_manifest_326
    runner._write_condition_manifest = _write_condition_manifest_326
    runner._validate_existing_task_manifest = _validate_existing_task_manifest_326
    runner.SIM_LABEL_EXPORTER = _write_label_volume_326


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SimulationBoundaryError(f"Promoted reference is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)
    if _sha256(source) != _sha256(destination):
        raise SimulationBoundaryError(f"Promoted reference copy hash mismatch: {source}")


def _copy_tree_no_symlinks(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise SimulationBoundaryError(f"Promoted reference tree is missing: {source}")
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise SimulationBoundaryError(f"Symlink in promoted reference tree: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            _copy_regular(path, target)
        else:
            raise SimulationBoundaryError(f"Non-regular promoted reference: {path}")


def _validate_hash_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise SimulationBoundaryError(f"Missing promoted {label} record")
    path = Path(str(record.get("path", "")))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise SimulationBoundaryError(f"Promoted {label} hash mismatch: {path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return path != root
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SimulationBoundaryError(f"Expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_argument(argv: list[str]) -> Path:
    try:
        index = argv.index("--config")
        return Path(argv[index + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("The harmonized v3 runner requires --config") from exc


def main() -> None:
    config_path = _config_argument(sys.argv[1:])
    _validate_campaign_config(config_path)
    install_backend()
    _ensure_simnibs_imports_326()
    runner.main()


if __name__ == "__main__":
    main()
