#!/usr/bin/env python3
"""Plan, initialize, validate, and advance the SimNIBS 3.2.6 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent
CURRENT_REPAIR_ROOT = SCRIPTS_ROOT / "ti_current_repair"
for candidate in (SCRIPTS_ROOT, CURRENT_REPAIR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiment_config import iter_experiment_tasks, load_experiment_config  # noqa: E402
from pipeline.spherical_fixed_nested_experiment import prepare_fixed  # noqa: E402
from stimulation_config import resolve_confirmed_stimulation  # noqa: E402
from validate_repeatability_task import validate_task  # noqa: E402

from settings import (  # noqa: E402
    ATLAS_DIR,
    CPUS_PER_TASK,
    HEAD_MODEL_STRATEGY,
    MATLAB_MODULE,
    MAX_CONCURRENT,
    MEMORY,
    PARTITION,
    REPEAT_COUNT,
    SCAFFOLD_ROOT,
    SEGMENTATION_PROVENANCE,
    SIMNIBS_MODULE,
    SOURCE_ROOT,
    SUBJECTS,
    TARGETS,
    TARGETS_CSV,
    TIME_LIMIT,
    assert_output_root_isolation,
    source_paths,
    target_settings,
)
from simulation_runner import BACKEND_NAME, CONFIG_RUNTIME_KEY  # noqa: E402
from cat12_compat import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    PATCH_ID,
    probe_matlab_compatibility,
)


SCHEMA_VERSION = 1
REMESH_CONFIG = "remesh_only.json"
PAIRED_CONFIG = "paired_analysis.json"
FIXED_CONFIG = "fixed_mesh_spherical_median.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _write_or_validate(path: Path, payload: dict[str, Any]) -> Path:
    if path.is_file():
        observed = _load_json(path)
        if observed != payload:
            raise RuntimeError(f"Refusing to replace incompatible artifact: {path}")
        return path
    return _write_json_atomic(path, payload)


def _config_dir(root: Path) -> Path:
    return root / "_pipeline" / "configs"


def _config_path(root: Path, name: str) -> Path:
    return _config_dir(root) / name


def _campaign_manifest(root: Path) -> Path:
    return root / "_simnibs326" / "campaign.json"


def _completion_receipt(root: Path) -> Path:
    return root / "_pipeline" / "workflow" / "complete.json"


def _runtime_payload() -> dict[str, object]:
    return {
        "simnibs_module": SIMNIBS_MODULE,
        "backend": BACKEND_NAME,
        "head_model_strategy": HEAD_MODEL_STRATEGY,
        "scaffold_root": str(SCAFFOLD_ROOT),
        "segmentation_provenance": SEGMENTATION_PROVENANCE,
        "ti_envelope_implementation": "internal Grossman maximal-envelope equation",
    }


def remesh_config(target: str) -> dict[str, Any]:
    target_spec = target_settings(target)
    stimulation = resolve_confirmed_stimulation(
        target,
        targets_csv=TARGETS_CSV,
    ).to_dict()
    return {
        "source_root": str(SOURCE_ROOT),
        "experiment_root": str(target_spec.experiment_root),
        "subjects": list(SUBJECTS),
        "conditions": [
            {
                "name": "remesh",
                "mesh_mode": "remesh",
                "repeat_count": REPEAT_COUNT,
                "description": (
                    "Fresh SimNIBS 3.2.6 headreco volumemesh realization from the "
                    "participant's immutable v3 scaffold."
                ),
            }
        ],
        "stimulation": stimulation,
        "analysis": {
            "roi_preset": target,
            "atlas_dir": str(ATLAS_DIR),
            "compare_metric": "median_roi",
        },
        CONFIG_RUNTIME_KEY: _runtime_payload(),
    }


def campaign_scope() -> dict[str, Any]:
    root_isolation = assert_output_root_isolation()
    per_target = len(SUBJECTS) * REPEAT_COUNT
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "SimNIBS 3.2.6 half of the final-132 balanced-10 repeatability experiment",
        "simnibs_module": SIMNIBS_MODULE,
        "source_root": str(SOURCE_ROOT),
        "scaffold_root": str(SCAFFOLD_ROOT),
        "atlas_dir": str(ATLAS_DIR),
        "targets_csv": str(TARGETS_CSV),
        "subjects": list(SUBJECTS),
        "subject_count": len(SUBJECTS),
        "targets": {
            key: {
                "experiment_root": str(value.experiment_root),
                "remesh_tasks": per_target,
                "fixed_mesh_tasks": per_target,
                "simulation_tasks": per_target * 2,
            }
            for key, value in TARGETS.items()
        },
        "conditions": ["remesh", "fixed_mesh"],
        "repeats_per_condition": REPEAT_COUNT,
        "headreco_scaffold_tasks": len(SUBJECTS),
        "remesh_tasks": per_target * len(TARGETS),
        "fixed_mesh_tasks": per_target * len(TARGETS),
        "total_simulation_tasks": per_target * len(TARGETS) * 2,
        "component_fem_solves": per_target * len(TARGETS) * 2 * 2,
        "arrays": {
            "scaffold": f"0-{len(SUBJECTS) - 1}%{len(SUBJECTS)}",
            "left_hippocampus_remesh": f"0-{per_target - 1}%{MAX_CONCURRENT}",
            "left_hippocampus_fixed": f"0-{per_target - 1}%{MAX_CONCURRENT}",
            "right_m1_remesh": f"0-{per_target - 1}%{MAX_CONCURRENT}",
            "right_m1_fixed": f"0-{per_target - 1}%{MAX_CONCURRENT}",
        },
        "resources": {
            "partition": PARTITION,
            "cpus_per_task": CPUS_PER_TASK,
            "memory": MEMORY,
            "time_limit": TIME_LIMIT,
            "max_concurrent": MAX_CONCURRENT,
        },
        "head_model_strategy": HEAD_MODEL_STRATEGY,
        "segmentation_provenance": SEGMENTATION_PROVENANCE,
        "root_isolation": root_isolation,
        "execution_scope": "full requested experiment; no smoke or reduced subset",
    }


def print_scope(scope: dict[str, Any]) -> None:
    print("Scope:")
    print(f"  study: {scope['study']}")
    print(f"  simulation module: {scope['simnibs_module']}")
    print(f"  subjects: {scope['subject_count']}")
    print("  targets: 2 (left hippocampus, right M1)")
    print("  conditions per target: 2 (remesh, fixed_mesh)")
    print(f"  repeats per condition: {scope['repeats_per_condition']}")
    print(f"  remesh tasks: {scope['remesh_tasks']}")
    print(f"  fixed-mesh tasks: {scope['fixed_mesh_tasks']}")
    print(f"  total simulation tasks: {scope['total_simulation_tasks']}")
    print(f"  component FEM solves: {scope['component_fem_solves']}")
    print(f"  one-time v3 headreco scaffold tasks: {scope['headreco_scaffold_tasks']}")
    for label, spec in scope["arrays"].items():
        print(f"  {label} array: {spec}")
    resources = scope["resources"]
    print(
        "  resources: "
        f"{resources['partition']}; {resources['cpus_per_task']} CPU; "
        f"{resources['memory']}; {resources['time_limit']}"
    )
    print(f"  head-model strategy: {scope['head_model_strategy']}")
    print(f"  version boundary: {scope['segmentation_provenance']}")
    print(f"  execution: {scope['execution_scope']}")


def _validate_hpc_inputs() -> dict[str, Any]:
    assert_output_root_isolation()
    missing: list[str] = []
    for subject in SUBJECTS:
        missing.extend(str(path) for path in source_paths(subject) if not path.is_file())
        atlas = ATLAS_DIR / f"{subject}.nii.gz"
        if not atlas.is_file():
            missing.append(str(atlas))
    if not TARGETS_CSV.is_file():
        missing.append(str(TARGETS_CSV))
    if missing:
        raise FileNotFoundError(
            "Campaign inputs are missing (first 20): " + ", ".join(missing[:20])
        )
    stimulation = {
        target: resolve_confirmed_stimulation(target, targets_csv=TARGETS_CSV).to_dict()
        for target in TARGETS
    }
    return {
        "source_files": len(SUBJECTS) * 3,
        "atlas_files": len(SUBJECTS),
        "targets_csv": str(TARGETS_CSV),
        "targets_csv_sha256": _sha256(TARGETS_CSV),
        "stimulation": stimulation,
    }


def initialize(*, dry_run: bool) -> dict[str, Any]:
    inputs = _validate_hpc_inputs()
    scope = campaign_scope()
    planned: dict[str, Any] = {}
    for target, settings in TARGETS.items():
        config = remesh_config(target)
        root = settings.experiment_root
        manifest = {
            **scope,
            "target": target,
            "target_experiment_root": str(root),
            "input_validation": inputs,
            "remesh_config": str(_config_path(root, REMESH_CONFIG)),
            "paired_config_state": "remesh_only_until_fixed_stage_completes",
        }
        planned[target] = {
            "root": str(root),
            "remesh_config": config,
            "campaign_manifest": manifest,
        }
        if dry_run:
            continue
        if root.exists() and any(root.iterdir()) and not _campaign_manifest(root).is_file():
            raise RuntimeError(
                "Refusing to initialize a non-empty unowned output root: " + str(root)
            )
        (root / "_pipeline" / "logs").mkdir(parents=True, exist_ok=True)
        (root / "_simnibs326" / "logs").mkdir(parents=True, exist_ok=True)
        remesh_path = _write_or_validate(_config_path(root, REMESH_CONFIG), config)
        _write_or_validate(_config_path(root, PAIRED_CONFIG), config)
        manifest["remesh_config_sha256"] = _sha256(remesh_path)
        _write_or_validate(_campaign_manifest(root), manifest)
    return {"status": "planned" if dry_run else "initialized", "targets": planned}


def module_preflight(*, receipt: Path | None) -> dict[str, Any]:
    import nibabel as nib  # type: ignore
    import numpy as np
    import simnibs  # type: ignore

    version = str(getattr(simnibs, "__version__", "unknown"))
    if not version.startswith("3.2.6"):
        raise RuntimeError(f"Expected SimNIBS 3.2.6, observed {version}")
    try:
        from simnibs import mesh_io, sim_struct  # type: ignore
    except ImportError:
        from simnibs.msh import mesh_io  # type: ignore
        from simnibs.simulation import sim_struct  # type: ignore
    from simnibs.simulation import cond  # type: ignore

    executables = {
        name: shutil.which(name) for name in ("headreco", "msh2nii", "matlab")
    }
    missing = [name for name, path in executables.items() if path is None]
    if missing:
        raise RuntimeError(f"Module is missing required executables: {missing}")
    required_numpy_aliases = ("bool", "int", "float")
    missing_numpy_aliases = [
        name for name in required_numpy_aliases if name not in np.__dict__
    ]
    if missing_numpy_aliases:
        raise RuntimeError(
            "Legacy NumPy aliases required by SimNIBS 3.2.6 are missing: "
            f"{missing_numpy_aliases}"
        )
    matlab_marker = "SIMNIBS326_MATLAB_OK"
    matlab_command = [
        executables["matlab"],
        "-batch",
        f"disp(['{matlab_marker} ' version])",
    ]
    try:
        matlab_result = subprocess.run(
            matlab_command,
            text=True,
            capture_output=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MATLAB startup probe exceeded 300 seconds") from exc
    matlab_output = matlab_result.stdout + matlab_result.stderr
    if matlab_result.returncode != 0 or matlab_marker not in matlab_output:
        raise RuntimeError(
            "MATLAB startup probe failed: "
            f"returncode={matlab_result.returncode}; output_tail={matlab_output[-4000:]}"
        )
    cat12_compatibility = probe_matlab_compatibility(executables["matlab"])
    help_result = subprocess.run(
        [executables["headreco"], "volumemesh", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    headreco_help = help_result.stdout + help_result.stderr
    if help_result.returncode != 0 or "subject_id" not in headreco_help:
        raise RuntimeError(
            "headreco volumemesh help/API check failed: "
            f"returncode={help_result.returncode}; "
            f"output_tail={headreco_help[-4000:]}"
        )
    msh2nii_help = subprocess.run(
        [executables["msh2nii"], "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    msh2nii_text = msh2nii_help.stdout + msh2nii_help.stderr
    if msh2nii_help.returncode != 0 or "--create_masks" not in msh2nii_text:
        raise RuntimeError(
            "msh2nii mask CLI check failed: "
            f"returncode={msh2nii_help.returncode}; "
            f"output_tail={msh2nii_text[-4000:]}"
        )
    session = sim_struct.SESSION()
    if not callable(getattr(session, "add_tdcslist", None)) or not callable(
        getattr(session, "run", None)
    ):
        raise RuntimeError("SimNIBS 3.2.6 SESSION API is incomplete")
    if not callable(getattr(mesh_io, "read_msh", None)) or not callable(
        getattr(mesh_io, "write_msh", None)
    ):
        raise RuntimeError("SimNIBS 3.2.6 mesh I/O API is incomplete")
    element_data_type = getattr(mesh_io, "ElementData", None)
    if not callable(element_data_type) or not callable(
        getattr(element_data_type, "to_nifti", None)
    ):
        raise RuntimeError(
            "SimNIBS 3.2.6 internal label-volume API is incomplete"
        )
    # headreco 3.2.6 calls nibabel's legacy get_data API.  Check that the
    # module's pinned dependency stack still supports it before launching ten
    # expensive scaffold jobs.
    legacy_image = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), np.eye(4))
    try:
        legacy_image.get_data()
    except Exception as exc:
        raise RuntimeError(
            "The installed nibabel is incompatible with headreco 3.2.6 get_data()"
        ) from exc
    conductivity = {
        entry.name: float(entry.value)
        for entry in cond.standard_cond()
        if getattr(entry, "name", "")
    }
    required_cond = {
        "WM",
        "GM",
        "CSF",
        "Bone",
        "Scalp",
        "Compact_bone",
        "Spongy_bone",
        "Blood",
        "Muscle",
        "Saline",
    }
    if not required_cond.issubset(conductivity):
        raise RuntimeError(
            f"Missing v3 conductivity entries: {sorted(required_cond - set(conductivity))}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "created_utc": _utc_now(),
        "expected_module": SIMNIBS_MODULE,
        "expected_matlab_module": MATLAB_MODULE,
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
        "simnibs_version": version,
        "simnibs_file": str(Path(simnibs.__file__).resolve()),
        "nibabel_version": str(getattr(nib, "__version__", "unknown")),
        "numpy_version": str(np.__version__),
        "numpy_legacy_aliases": {
            name: repr(np.__dict__[name]) for name in required_numpy_aliases
        },
        "mesh_io_module": str(getattr(mesh_io, "__file__", "")),
        "executables": executables,
        "matlab_probe": {
            "command": matlab_command,
            "returncode": matlab_result.returncode,
            "output_tail": matlab_output[-2000:],
        },
        "cat12_compatibility": cat12_compatibility,
        "session_api": ["SESSION", "add_tdcslist", "run"],
        "mesh_api": ["read_msh", "write_msh", "ElementData.to_nifti"],
        "msh2nii_probe": {
            "command": [executables["msh2nii"], "--help"],
            "returncode": msh2nii_help.returncode,
            "supports_create_masks": "--create_masks" in msh2nii_text,
            "supports_create_label": "--create_label" in msh2nii_text,
            "output_tail": msh2nii_text[-2000:],
        },
        "label_volume_implementation": (
            "internal SimNIBS 3.2.6 ElementData.to_nifti tissue-tag assignment"
        ),
        "conductivity_table": conductivity,
        "ti_envelope_implementation": "internal Grossman maximal-envelope equation",
        "head_model_strategy": HEAD_MODEL_STRATEGY,
        "segmentation_provenance": SEGMENTATION_PROVENANCE,
    }
    if receipt is not None:
        _write_json_atomic(receipt.expanduser().resolve(), payload)
    return payload


def validate_module_preflight_receipt(receipt: Path) -> dict[str, Any]:
    path = receipt.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Module preflight receipt is missing: {path}")
    payload = _load_json(path)
    compatibility = payload.get("cat12_compatibility")
    failures: list[str] = []
    if payload.get("status") != "ready":
        failures.append("receipt status is not ready")
    if payload.get("expected_module") != SIMNIBS_MODULE:
        failures.append("SimNIBS module does not match")
    if payload.get("expected_matlab_module") != MATLAB_MODULE:
        failures.append("MATLAB module does not match")
    if not isinstance(compatibility, dict):
        failures.append("CAT12 compatibility evidence is missing")
    else:
        if compatibility.get("status") != "ready":
            failures.append("CAT12 compatibility status is not ready")
        if compatibility.get("patch_id") != PATCH_ID:
            failures.append("CAT12 compatibility patch ID does not match")
        if compatibility.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            failures.append("CAT12 compatibility source hashes do not match")
        if compatibility.get("replacements") != {
            "segment_pre_init_path_insertions": 1,
            "segment_post_init_path_reassertions": 1,
            "xml_error_to_warning_replacements": 2,
        }:
            failures.append("CAT12 compatibility replacement counts do not match")
        if compatibility.get("spm_jobman_initcfg_tested") is not True:
            failures.append("CAT12 spm_jobman initcfg probe did not pass")
        if compatibility.get("post_init_reassertion_tested") is not True:
            failures.append("CAT12 post-init path reassertion probe did not pass")
        if compatibility.get("mat_report_created") is not True:
            failures.append("CAT12 MAT-report probe did not pass")
    if failures:
        raise RuntimeError(
            f"Module preflight receipt is incompatible: {path}: "
            + "; ".join(failures)
        )
    return {
        "status": "ready",
        "receipt": str(path),
        "simnibs_module": SIMNIBS_MODULE,
        "matlab_module": MATLAB_MODULE,
        "cat12_compatibility_patch": PATCH_ID,
    }


def _validate_config_outputs(config_path: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path, validate_paths=False)
    tasks = iter_experiment_tasks(config)
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        checks = validate_task(config, task)
        failed = [check for check in checks if not check.ok]
        if failed:
            failures.append(
                {
                    "task_index": index,
                    "subject": task.subject,
                    "condition": task.condition_name,
                    "repeat_tag": task.repeat_tag,
                    "failed_checks": [check.name for check in failed],
                }
            )
    return {
        "config": str(config_path),
        "expected_tasks": len(tasks),
        "complete_tasks": len(tasks) - len(failures),
        "failures": failures[:50],
        "failure_count": len(failures),
        "status": "complete" if not failures else "incomplete",
    }


def _receipt_scope(config_path: Path, *, target: str, stage: str) -> dict[str, Any]:
    config = load_experiment_config(config_path, validate_paths=False)
    counts = {condition.name: condition.repeat_count for condition in config.conditions}
    total = len(config.subjects) * sum(counts.values())
    if len(set(counts.values())) != 1:
        raise RuntimeError("All final repeatability conditions must have the same repeat count")
    return {
        "target": target,
        "roi": target,
        "stage": stage,
        "subject_count": len(config.subjects),
        "repeats_per_condition": next(iter(counts.values())),
        "conditions": counts,
        "expected_ti_nifti": total,
        "expected_ti_msh": total,
        "simnibs_module": SIMNIBS_MODULE,
        "backend": BACKEND_NAME,
    }


def _write_completion(
    *,
    root: Path,
    config_path: Path,
    target: str,
    stage: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    path = _completion_receipt(root)
    previous = _load_json(path) if path.is_file() else None
    if previous is not None and previous.get("stage") not in (stage, "remesh"):
        raise RuntimeError(f"Refusing unexpected completion-receipt transition: {path}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stage": stage,
        "created_utc": _utc_now(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "scope": _receipt_scope(config_path, target=target, stage=stage),
        "validation": validation,
        "previous_receipt_sha256": _sha256(path) if path.is_file() else None,
    }
    _write_json_atomic(path, payload)
    return payload


def complete_remesh(target: str) -> dict[str, Any]:
    root = target_settings(target).experiment_root
    config_path = _config_path(root, REMESH_CONFIG)
    validation = _validate_config_outputs(config_path)
    if validation["status"] != "complete":
        raise RuntimeError(
            f"Remesh stage is incomplete: {validation['failure_count']} failed tasks"
        )
    return _write_completion(
        root=root,
        config_path=config_path,
        target=target,
        stage="remesh",
        validation=validation,
    )


def prepare_fixed_stage(target: str, *, metrics_csv: Path, dry_run: bool) -> dict[str, Any]:
    root = target_settings(target).experiment_root
    receipt = _load_json(_completion_receipt(root))
    if receipt.get("status") != "complete" or receipt.get("stage") != "remesh":
        raise RuntimeError("A validated remesh completion receipt is required")
    return prepare_fixed(
        source_experiment_root=root,
        metrics_csv=metrics_csv.expanduser().resolve(),
        output_root=root,
        repeat_count=REPEAT_COUNT,
        dry_run=dry_run,
    )


def _combined_config(root: Path) -> dict[str, Any]:
    remesh = _load_json(_config_path(root, REMESH_CONFIG))
    fixed = _load_json(_config_path(root, FIXED_CONFIG))
    if remesh["subjects"] != fixed["subjects"]:
        raise RuntimeError("Remesh and fixed configs have different subjects")
    if remesh["stimulation"] != fixed["stimulation"]:
        raise RuntimeError("Remesh and fixed configs have different stimulation")
    combined = json.loads(json.dumps(remesh))
    combined["conditions"] = remesh["conditions"] + fixed["conditions"]
    return combined


def complete_final(target: str) -> dict[str, Any]:
    root = target_settings(target).experiment_root
    remesh_path = _config_path(root, REMESH_CONFIG)
    fixed_path = _config_path(root, FIXED_CONFIG)
    remesh_validation = _validate_config_outputs(remesh_path)
    fixed_validation = _validate_config_outputs(fixed_path)
    if remesh_validation["status"] != "complete" or fixed_validation["status"] != "complete":
        raise RuntimeError(
            "Final stage is incomplete: "
            f"remesh failures={remesh_validation['failure_count']}, "
            f"fixed failures={fixed_validation['failure_count']}"
        )
    paired_path = _config_path(root, PAIRED_CONFIG)
    current = _load_json(paired_path)
    remesh = _load_json(remesh_path)
    combined = _combined_config(root)
    if current not in (remesh, combined):
        raise RuntimeError(f"Unexpected paired-analysis config state: {paired_path}")
    _write_json_atomic(paired_path, combined)
    validation = {
        "status": "complete",
        "remesh": remesh_validation,
        "fixed_mesh": fixed_validation,
        "total_tasks": remesh_validation["expected_tasks"] + fixed_validation["expected_tasks"],
    }
    return _write_completion(
        root=root,
        config_path=paired_path,
        target=target,
        stage="final",
        validation=validation,
    )


def status(target: str) -> dict[str, Any]:
    root = target_settings(target).experiment_root
    configs = {
        "remesh": _config_path(root, REMESH_CONFIG),
        "fixed_mesh": _config_path(root, FIXED_CONFIG),
        "paired": _config_path(root, PAIRED_CONFIG),
    }
    counts: dict[str, Any] = {}
    for label in ("remesh", "fixed_mesh"):
        path = configs[label]
        if not path.is_file():
            counts[label] = {"configured": False, "complete_outputs": 0}
            continue
        config = load_experiment_config(path, validate_paths=False)
        complete = 0
        for task in iter_experiment_tasks(config):
            repeat = (
                config.experiment_root
                / f"{task.subject}_repeatability"
                / task.condition_name
                / "repeats"
                / task.repeat_tag
                / task.subject
                / "anat"
                / "SimNIBS"
            )
            if (repeat / "ti_brain_only.nii.gz").is_file() and (
                repeat / "Output" / task.subject / "TI.msh"
            ).is_file():
                complete += 1
        counts[label] = {
            "configured": True,
            "expected_outputs": len(iter_experiment_tasks(config)),
            "complete_outputs": complete,
        }
    return {
        "target": target,
        "root": str(root),
        "campaign_manifest": _campaign_manifest(root).is_file(),
        "completion_receipt": (
            _load_json(_completion_receipt(root))
            if _completion_receipt(root).is_file()
            else None
        ),
        "counts": counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--check-paths", action="store_true")

    init = subparsers.add_parser("init")
    init.add_argument("--dry-run", action="store_true")

    module = subparsers.add_parser("module-preflight")
    module.add_argument("--receipt", type=Path, default=None)

    validate_module = subparsers.add_parser("validate-module-receipt")
    validate_module.add_argument("--receipt", type=Path, required=True)

    for name in ("complete-remesh", "complete-final", "status"):
        child = subparsers.add_parser(name)
        child.add_argument("--target", choices=sorted(TARGETS), required=True)

    validate = subparsers.add_parser("validate-stage")
    validate.add_argument("--target", choices=sorted(TARGETS), required=True)
    validate.add_argument("--stage", choices=("remesh", "fixed_mesh"), required=True)

    fixed = subparsers.add_parser("prepare-fixed")
    fixed.add_argument("--target", choices=sorted(TARGETS), required=True)
    fixed.add_argument("--metrics-csv", type=Path, required=True)
    fixed.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        scope = campaign_scope()
        if args.check_paths:
            scope["input_validation"] = _validate_hpc_inputs()
        if args.json:
            print(json.dumps(scope, indent=2, sort_keys=True))
        else:
            print_scope(scope)
        return 0
    if args.command == "init":
        payload = initialize(dry_run=args.dry_run)
    elif args.command == "module-preflight":
        payload = module_preflight(receipt=args.receipt)
    elif args.command == "validate-module-receipt":
        payload = validate_module_preflight_receipt(args.receipt)
    elif args.command == "complete-remesh":
        payload = complete_remesh(args.target)
    elif args.command == "complete-final":
        payload = complete_final(args.target)
    elif args.command == "prepare-fixed":
        payload = prepare_fixed_stage(
            args.target,
            metrics_csv=args.metrics_csv,
            dry_run=args.dry_run,
        )
    elif args.command == "validate-stage":
        root = target_settings(args.target).experiment_root
        name = REMESH_CONFIG if args.stage == "remesh" else FIXED_CONFIG
        payload = _validate_config_outputs(_config_path(root, name))
    else:
        payload = status(args.target)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") not in ("incomplete", "failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
