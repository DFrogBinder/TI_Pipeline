#!/usr/bin/env python3
"""SimNIBS 3.2.6 backend for the established repeatability task runner.

The task/config layout intentionally matches ``ti_current_repair`` so that its
validators and optimizer-matched spherical ROI tooling remain usable.  Only
the version-specific meshing and SimNIBS execution adapters are replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent
CURRENT_REPAIR_ROOT = SCRIPTS_ROOT / "ti_current_repair"
for candidate in (SCRIPTS_ROOT, CURRENT_REPAIR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from settings import (  # noqa: E402
    HEAD_MODEL_STRATEGY,
    SCAFFOLD_ROOT,
    SEGMENTATION_PROVENANCE,
    SIMNIBS_MODULE,
)
from simulation_runners import repeatability_experiment as runner  # noqa: E402


BACKEND_NAME = "simnibs-3.2.6-headreco-volumemesh"
CONFIG_RUNTIME_KEY = "simnibs_326_runtime"
_CONFIG_PATH: Path | None = None
_SIMNIBS_VERSION = "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_ti(E1_org, E2_org):
    """Calculate the Grossman maximal TI envelope used by SimNIBS TI_utils."""
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
        # v3 headreco stores the canonical head mesh beside m2m_<subject>.
        # We use the validated compatibility copy inside m2m_<subject> and set
        # subpath explicitly so v3 never has to infer the non-v4 layout.
        session.subpath = str(Path(session.fnamehead).parent)
        session.open_in_gmsh = False
        # The shared runner explicitly creates the analysis NIfTIs with
        # msh2nii.  Disable the redundant v3 interpolation step.
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
    if not _SIMNIBS_VERSION.startswith("3.2.6"):
        raise RuntimeError(
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
    """Write the tissue-label NIfTI missing from the v3.2.6 msh2nii CLI.

    This is the same tetrahedra/tag/assign algorithm used by the SimNIBS 4
    ``create_label`` implementation, expressed with the v3 mesh I/O API.
    """
    import nibabel as nib

    _ensure_simnibs_imports_326()
    mesh = runner.SIM_MESH_IO.read_msh(str(mesh_path))
    mesh = mesh.crop_mesh(elm_type=4)
    element_data_type = getattr(runner.SIM_MESH_IO, "ElementData", None)
    if not callable(element_data_type) or not callable(
        getattr(element_data_type, "to_nifti", None)
    ):
        raise RuntimeError(
            "SimNIBS 3.2.6 mesh_io.ElementData.to_nifti is unavailable"
        )

    reference = nib.load(str(reference_path))
    destination = output_path
    if destination.suffix == "":
        destination = Path(f"{destination}.nii.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    label_data = element_data_type(mesh.elm.tag1)
    label_data.mesh = mesh
    label_data.to_nifti(
        reference.header["dim"][1:4],
        reference.affine,
        fn=str(destination),
        qform=reference.header.get_qform(),
        method="assign",
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"SimNIBS 3.2.6 label export did not create a NIfTI: {destination}"
        )
    runner.log_event(
        "label_volume_exported",
        implementation="simnibs-3.2.6-mesh-io-element-tags",
        mesh_path=str(mesh_path),
        reference_path=str(reference_path),
        output_path=str(destination),
    )
    return destination


def _scaffold_m2m(subject: str) -> Path:
    return SCAFFOLD_ROOT / subject / "anat" / f"m2m_{subject}"


def _scaffold_marker(subject: str) -> Path:
    return SCAFFOLD_ROOT / subject / "anat" / ".simnibs326_scaffold_ready.json"


def _copy_scaffold_for_volumemesh(*, subject: str, destination: Path) -> None:
    source = _scaffold_m2m(subject)
    marker = _scaffold_marker(subject)
    if not source.is_dir() or not marker.is_file():
        raise runner.SimulationInputError(
            f"Validated SimNIBS 3.2.6 scaffold is missing for {subject}: {source}"
        )

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True)

    excluded_files = {
        f"{subject}.msh",
        f"{subject}_final_contr.nii.gz",
        "wm_fromMesh.nii.gz",
        "gm_fromMesh.nii.gz",
        "headreco_log.html",
    }
    required_dirs = {"toMNI"}
    for child in source.iterdir():
        if child.name in excluded_files:
            continue
        target = destination / child.name
        if child.is_file() or child.is_symlink():
            shutil.copy2(child, target, follow_symlinks=True)
        elif child.name in required_dirs:
            shutil.copytree(child, target, symlinks=False)

    surfaces = sorted(destination.glob("*.stl"))
    required_files = [
        destination / "T1fs_conform.nii.gz",
        destination / "T1fs_nu_conform.nii.gz",
    ]
    missing = [path for path in required_files if not path.is_file()]
    if len(surfaces) < 5 or missing or not (destination / "toMNI").is_dir():
        raise runner.SimulationInputError(
            "Scaffold clone is insufficient for headreco volumemesh: "
            f"surfaces={len(surfaces)}, missing={missing}, destination={destination}"
        )


def _mesh_workspace_326(
    workspace: runner.WorkspacePaths,
    *,
    subject: str,
    force_mesh: bool,
) -> Path:
    ready_marker = runner._mesh_ready_marker(workspace)
    native_mesh = workspace.anat_dir / f"{subject}.msh"
    lock_path = workspace.anat_dir / ".mesh_build.lock"

    if workspace.mesh_path.is_file() and ready_marker.is_file() and not force_mesh:
        runner.log_event(
            "mesh_reuse",
            subject=subject,
            mesh_path=str(workspace.mesh_path),
            backend=BACKEND_NAME,
        )
        return workspace.mesh_path

    with runner._exclusive_lock(lock_path):
        if workspace.mesh_path.is_file() and ready_marker.is_file() and not force_mesh:
            return workspace.mesh_path
        if ready_marker.exists():
            ready_marker.unlink()
        if native_mesh.exists():
            native_mesh.unlink()

        _copy_scaffold_for_volumemesh(
            subject=subject,
            destination=workspace.mesh_dir,
        )
        command = ["headreco", "volumemesh", "--noclean", subject]
        runner.run_cmd(
            command,
            cwd=str(workspace.anat_dir),
            label="headreco_326_volumemesh",
        )
        if not native_mesh.is_file():
            raise FileNotFoundError(
                f"headreco volumemesh did not create its native mesh: {native_mesh}"
            )
        shutil.copy2(native_mesh, workspace.mesh_path)
        if _sha256(native_mesh) != _sha256(workspace.mesh_path):
            raise RuntimeError(
                f"Native/compatibility mesh checksum mismatch for {subject}"
            )
        _ensure_simnibs_imports_326()
        runner._write_json(
            ready_marker,
            {
                "schema_version": 1,
                "status": "mesh_ready",
                "created_at": time.time(),
                "subject": subject,
                "simnibs_module": SIMNIBS_MODULE,
                "simnibs_version": _SIMNIBS_VERSION,
                "backend": BACKEND_NAME,
                "head_model_strategy": HEAD_MODEL_STRATEGY,
                "segmentation_provenance": SEGMENTATION_PROVENANCE,
                "scaffold_marker": str(_scaffold_marker(subject)),
                "scaffold_marker_sha256": _sha256(_scaffold_marker(subject)),
                "headreco_command": command,
                "native_mesh_path": str(native_mesh),
                "mesh_path": str(workspace.mesh_path),
                "mesh_sha256": _sha256(workspace.mesh_path),
            },
        )
        return workspace.mesh_path


def _runtime_provenance() -> dict[str, str]:
    return {
        "simnibs_module": SIMNIBS_MODULE,
        "simnibs_version": _SIMNIBS_VERSION,
        "backend": BACKEND_NAME,
        "head_model_strategy": HEAD_MODEL_STRATEGY,
        "segmentation_provenance": SEGMENTATION_PROVENANCE,
        "config": str(_CONFIG_PATH) if _CONFIG_PATH else "",
    }


_original_write_task_manifest = runner._write_task_manifest
_original_write_condition_manifest = runner._write_condition_manifest


def _write_task_manifest_326(*args, **kwargs) -> None:
    _original_write_task_manifest(*args, **kwargs)
    workspace = args[0] if args else kwargs["workspace"]
    path = workspace.root / "task_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[CONFIG_RUNTIME_KEY] = _runtime_provenance()
    runner._write_json(path, payload)


def _write_condition_manifest_326(*args, **kwargs) -> None:
    _original_write_condition_manifest(*args, **kwargs)
    config = args[0] if args else kwargs["config"]
    subject = kwargs["subject"]
    condition_name = kwargs["condition_name"]
    path = runner.condition_manifest_path(config, subject, condition_name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[CONFIG_RUNTIME_KEY] = _runtime_provenance()
    runner._write_json(path, payload)


def _validate_existing_task_manifest_326(
    workspace: runner.WorkspacePaths,
    *,
    config,
) -> None:
    path = workspace.root / "task_manifest.json"
    if not path.is_file():
        raise runner.SimulationInputError(f"Missing task provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stimulation") != config.stimulation.to_dict():
        raise runner.SimulationInputError(
            f"Existing output stimulation provenance differs: {path}"
        )
    runtime = payload.get(CONFIG_RUNTIME_KEY, {})
    if runtime.get("simnibs_module") != SIMNIBS_MODULE or runtime.get("backend") != BACKEND_NAME:
        raise runner.SimulationInputError(
            f"Existing output is not a validated SimNIBS 3.2.6 task: {path}"
        )


def _config_argument(argv: list[str]) -> Path:
    try:
        index = argv.index("--config")
        return Path(argv[index + 1]).expanduser().resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("The SimNIBS 3.2.6 runner requires --config") from exc


def _validate_campaign_config(config_path: Path) -> None:
    global _CONFIG_PATH
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = payload.get(CONFIG_RUNTIME_KEY)
    if runtime is not None:
        expected = {
            "simnibs_module": SIMNIBS_MODULE,
            "backend": BACKEND_NAME,
            "head_model_strategy": HEAD_MODEL_STRATEGY,
            "scaffold_root": str(SCAFFOLD_ROOT),
        }
        differences = {
            key: {"expected": value, "observed": runtime.get(key)}
            for key, value in expected.items()
            if runtime.get(key) != value
        }
        if differences:
            raise SystemExit(
                "Config runtime does not match this backend: "
                + json.dumps(differences, sort_keys=True)
            )
    else:
        conditions = payload.get("conditions", [])
        fixed_only = conditions and all(
            item.get("mesh_mode") == "fixed_mesh" for item in conditions
        )
        campaign = Path(str(payload.get("experiment_root", ""))) / "_simnibs326" / "campaign.json"
        if not fixed_only or not campaign.is_file():
            raise SystemExit(
                f"Config lacks {CONFIG_RUNTIME_KEY!r} and is not a seeded fixed-only campaign config"
            )
        campaign_payload = json.loads(campaign.read_text(encoding="utf-8"))
        if campaign_payload.get("simnibs_module") != SIMNIBS_MODULE:
            raise SystemExit(f"Campaign manifest module mismatch: {campaign}")
    _CONFIG_PATH = config_path


def install_backend() -> None:
    runner._ensure_simnibs_imports = _ensure_simnibs_imports_326
    runner._mesh_workspace = _mesh_workspace_326
    runner._write_task_manifest = _write_task_manifest_326
    runner._write_condition_manifest = _write_condition_manifest_326
    runner._validate_existing_task_manifest = _validate_existing_task_manifest_326
    runner.SIM_LABEL_EXPORTER = _write_label_volume_326


def main() -> None:
    config_path = _config_argument(sys.argv[1:])
    _validate_campaign_config(config_path)
    install_backend()
    _ensure_simnibs_imports_326()
    runner.main()


if __name__ == "__main__":
    main()
