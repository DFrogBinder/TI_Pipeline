#!/usr/bin/env python3
"""Plan, gate, validate, and advance the harmonized SimNIBS 3.2.6 campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from .headmodels import (
    build_headmodel,
    collect_headmodels,
    promote_headmodels,
    validate_headmodel,
    write_qc_request,
)
from .labels import validate_source_labels
from .reference_stage import collect_reference_stage, stage_subject_reference
from .settings import (
    REPEAT_COUNT,
    SIMNIBS_MODULE,
    SUBJECTS,
    TARGETS_CSV,
    CampaignPaths,
    assert_root_isolation,
    campaign_paths_from_environment,
    campaign_scope as settings_scope,
    subject_paths,
    target_settings,
)
from .simulation_runner import (
    BACKEND_NAME,
    CONFIG_RUNTIME_KEY,
    _validated_promotion,
)
from .v3_meshing import load_v3_meshing_api, probe_v3_meshing_api


PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent
CURRENT_REPAIR_ROOT = SCRIPTS_ROOT / "ti_current_repair"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
_added_repair_path = str(CURRENT_REPAIR_ROOT) not in sys.path
if _added_repair_path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))
try:
    from experiment_config import iter_experiment_tasks, load_experiment_config  # noqa: E402
    from pipeline.spherical_fixed_nested_experiment import prepare_fixed  # noqa: E402
    from stimulation_config import resolve_confirmed_stimulation  # noqa: E402
    from validate_repeatability_task import validate_task  # noqa: E402
finally:
    if _added_repair_path:
        sys.path.remove(str(CURRENT_REPAIR_ROOT))

SCHEMA_VERSION = 1
REMESH_CONFIG = "remesh_only.json"
PAIRED_CONFIG = "paired_analysis.json"
FIXED_CONFIG = "fixed_mesh_spherical_median.json"
REMESH_DESCRIPTION = (
    "Fresh SimNIBS 3.2.6 volume mesh generated directly from the promoted "
    "corrected-label v3 surfaces."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPTS_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def _write_or_validate(path: Path, payload: dict[str, Any]) -> Path:
    if path.is_file():
        if _load_json(path) != payload:
            raise RuntimeError(f"Refusing to replace incompatible artifact: {path}")
        return path
    return _write_json_atomic(path, payload)


def _config_path(root: Path, name: str) -> Path:
    return root / "_pipeline" / "configs" / name


def _campaign_manifest(root: Path) -> Path:
    return root / "_simnibs326" / "campaign.json"


def _completion_receipt(root: Path) -> Path:
    return root / "_pipeline" / "workflow" / "complete.json"


def _resolve_stimulation(target: str, targets_csv: Path = TARGETS_CSV):
    return resolve_confirmed_stimulation(target, targets_csv=targets_csv)


def campaign_scope(paths: CampaignPaths | None = None) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    base = settings_scope(resolved)
    per_target = len(SUBJECTS) * REPEAT_COUNT
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "corrected-label SimNIBS 3.2.6 balanced-10 repeatability",
        **base,
        "headmodel_tasks": len(SUBJECTS),
        "reference_stage_tasks": len(SUBJECTS),
        "remesh_tasks": per_target * 2,
        "fixed_mesh_tasks": per_target * 2,
        "execution_scope": "full requested experiment; no reduced subset",
        "source_policy": "read-only; corrected label maps are hashed, not copied",
        "headmodel_status": "candidate until Stanage module and build evidence exists",
    }


def print_scope(scope: dict[str, Any]) -> None:
    resources = scope["resources"]
    print("Scope:")
    print(f"  study: {scope['study']}")
    print(f"  subjects: {scope['subject_count']}")
    print("  targets: 2 (left hippocampus, right M1)")
    print("  conditions: 2 (remesh, fixed_mesh)")
    print(f"  repeats per condition: {scope['repeats_per_condition']}")
    print(f"  reference-stage tasks: {scope['reference_stage_tasks']}")
    print(f"  head-model tasks: {scope['headmodel_tasks']}")
    print(f"  remesh tasks: {scope['remesh_tasks']}")
    print(f"  fixed-mesh tasks: {scope['fixed_mesh_tasks']}")
    print(f"  total simulation tasks: {scope['total_simulation_tasks']}")
    print(f"  component FEM solves: {scope['component_fem_solves']}")
    for label, value in scope["arrays"].items():
        print(f"  {label} array: {value}")
    print(
        "  resources: "
        f"{resources['partition']}; {resources['cpus_per_task']} CPU; "
        f"{resources['memory']}; {resources['time_limit']}"
    )
    print(f"  module: {resources['simnibs_module']}")
    print("  MATLAB/CAT12 tasks: 0")
    print(f"  execution: {scope['execution_scope']}")


def _inspect_subject_inputs(paths: CampaignPaths, subject: str) -> dict[str, Any]:
    import nibabel as nib

    resolved = subject_paths(paths, subject)
    required_files = {
        "corrected_label": resolved.corrected_label,
        "corrected_t1": resolved.corrected_t1,
        "corrected_t2": resolved.corrected_t2,
        "legacy_ready_marker": resolved.legacy_ready_marker,
        "t1fs_conform": resolved.legacy_m2m / "T1fs_conform.nii.gz",
        "t1fs_nu_conform": resolved.legacy_m2m / "T1fs_nu_conform.nii.gz",
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    required_dirs = (
        resolved.legacy_m2m / "eeg_positions",
        resolved.legacy_m2m / "toMNI",
    )
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise FileNotFoundError(f"Missing inputs for {subject}: {', '.join(missing)}")
    source = nib.load(str(resolved.corrected_label))
    data = validate_source_labels(source.get_fdata(dtype="float64"))
    reference = nib.load(str(resolved.legacy_m2m / "T1fs_conform.nii.gz"))
    files = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in required_files.items()
    }
    source_grid = {
        "shape": list(source.shape),
        "affine": source.affine.astype(float).tolist(),
        "voxel_sizes_mm": [float(v) for v in source.header.get_zooms()[:3]],
    }
    reference_grid = {
        "shape": list(reference.shape),
        "affine": reference.affine.astype(float).tolist(),
        "voxel_sizes_mm": [
            float(v) for v in reference.header.get_zooms()[:3]
        ],
    }
    return {
        "subject": subject,
        "corrected_label": str(resolved.corrected_label.resolve()),
        "corrected_label_sha256": _sha256(resolved.corrected_label),
        "files": files,
        "source_grid": source_grid,
        "reference_grid": reference_grid,
        "source_shape": source_grid["shape"],
        "source_voxel_sizes_mm": source_grid["voxel_sizes_mm"],
        "observed_labels": sorted(int(v) for v in set(data.ravel().tolist())),
        "reference_shape": reference_grid["shape"],
        "reference_voxel_sizes_mm": reference_grid["voxel_sizes_mm"],
        "legacy_ready_marker_sha256": _sha256(resolved.legacy_ready_marker),
    }


def preflight(
    *, paths: CampaignPaths | None = None, targets_csv: Path = TARGETS_CSV
) -> dict[str, Any]:
    """Perform the complete ten-subject input audit without creating outputs."""
    resolved = paths or campaign_paths_from_environment()
    isolation = assert_root_isolation(resolved)
    targets_csv = targets_csv.expanduser().resolve()
    if not targets_csv.is_file():
        raise FileNotFoundError(f"Targets CSV is missing: {targets_csv}")
    missing_atlases = [
        str(resolved.atlas_root / f"{subject}.nii.gz")
        for subject in SUBJECTS
        if not (resolved.atlas_root / f"{subject}.nii.gz").is_file()
    ]
    if missing_atlases:
        raise FileNotFoundError(
            "Missing optimizer atlases (first 10): " + ", ".join(missing_atlases[:10])
        )
    subjects = [_inspect_subject_inputs(resolved, subject) for subject in SUBJECTS]
    stimulation = {
        target: _resolve_stimulation(target, targets_csv).to_dict()
        for target in ("left-hippocampus", "right-m1")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "scope": campaign_scope(resolved),
        "root_isolation": isolation,
        "subjects": subjects,
        "targets_csv": str(targets_csv),
        "targets_csv_sha256": _sha256(targets_csv),
        "stimulation": stimulation,
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
    }


def _runtime_payload(paths: CampaignPaths, promotion: dict[str, str]) -> dict[str, Any]:
    return {
        "simnibs_module": SIMNIBS_MODULE,
        "backend": BACKEND_NAME,
        "headmodel_root": str(paths.headmodel_root.resolve()),
        "headmodel_manifest_sha256": promotion["manifest_sha256"],
        "headmodel_promotion_path": promotion["promotion_path"],
        "headmodel_promotion_sha256": promotion["promotion_sha256"],
        "label_policy": "corrected-v4-to-v3-harmonized-v2",
        "ti_envelope_implementation": "internal Grossman maximal-envelope equation",
    }


def remesh_config(
    target: str,
    *,
    paths: CampaignPaths | None = None,
    targets_csv: Path = TARGETS_CSV,
) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    promotion = _validated_promotion(resolved)
    target_spec = target_settings(target, resolved)
    return {
        "source_root": str(resolved.corrected_root.resolve()),
        "experiment_root": str(target_spec.experiment_root.resolve()),
        "subjects": list(SUBJECTS),
        "conditions": [{
            "name": "remesh",
            "mesh_mode": "remesh",
            "repeat_count": REPEAT_COUNT,
            "description": REMESH_DESCRIPTION,
        }],
        "stimulation": _resolve_stimulation(target, targets_csv).to_dict(),
        "analysis": {
            "roi_preset": target,
            "atlas_dir": str(resolved.atlas_root.resolve()),
            "compare_metric": "median_roi",
        },
        CONFIG_RUNTIME_KEY: _runtime_payload(resolved, promotion),
    }


def initialize(
    *,
    dry_run: bool,
    paths: CampaignPaths | None = None,
    targets_csv: Path = TARGETS_CSV,
) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    assert_root_isolation(resolved)
    # Resolve once before the first write.  A stale promotion therefore cannot
    # leave a partially initialized target root.
    _validated_promotion(resolved)
    configs = {
        target: remesh_config(target, paths=resolved, targets_csv=targets_csv)
        for target in ("left-hippocampus", "right-m1")
    }
    scope = campaign_scope(resolved)
    planned: dict[str, Any] = {}
    for target, config in configs.items():
        root = target_settings(target, resolved).experiment_root
        manifest = {
            **scope,
            "git_commit": _git_commit(),
            "target": target,
            "target_experiment_root": str(root.resolve()),
            "remesh_config": str(_config_path(root, REMESH_CONFIG)),
            "paired_config_state": "remesh-only until fixed preparation completes",
            "runtime": config[CONFIG_RUNTIME_KEY],
        }
        planned[target] = {
            "root": str(root), "remesh_config": config, "campaign_manifest": manifest
        }
        if dry_run:
            continue
        if root.exists() and any(root.iterdir()) and not _campaign_manifest(root).is_file():
            raise RuntimeError(f"Refusing non-empty unowned output root: {root}")
        (root / "_pipeline" / "logs").mkdir(parents=True, exist_ok=True)
        (root / "_simnibs326" / "logs").mkdir(parents=True, exist_ok=True)
        remesh_path = _write_or_validate(_config_path(root, REMESH_CONFIG), config)
        _write_or_validate(_config_path(root, PAIRED_CONFIG), config)
        manifest["remesh_config_sha256"] = _sha256(remesh_path)
        _write_or_validate(_campaign_manifest(root), manifest)
    return {"status": "planned" if dry_run else "initialized", "targets": planned}


def _load_simnibs_fem_api() -> dict[str, Any]:
    import simnibs  # type: ignore

    version = str(getattr(simnibs, "__version__", "unknown"))
    if version != "3.2.6":
        raise RuntimeError(f"Expected SimNIBS 3.2.6, observed {version}")
    try:
        from simnibs import mesh_io, sim_struct  # type: ignore
    except ImportError:
        from simnibs.msh import mesh_io  # type: ignore
        from simnibs.simulation import sim_struct  # type: ignore
    from simnibs.simulation import cond  # type: ignore

    session = sim_struct.SESSION()
    session_ok = callable(getattr(session, "add_tdcslist", None)) and callable(
        getattr(session, "run", None)
    )
    element_type = getattr(mesh_io, "ElementData", None)
    element_ok = callable(element_type) and callable(
        getattr(element_type, "to_nifti", None)
    )
    if not session_ok or not element_ok or not callable(getattr(mesh_io, "read_msh", None)):
        raise RuntimeError("SimNIBS 3.2.6 FEM/mesh export API is incomplete")
    conductivities = sorted(
        entry.name for entry in cond.standard_cond() if getattr(entry, "name", "")
    )
    required = {"WM", "GM", "CSF", "Bone", "Scalp"}
    if not required.issubset(conductivities):
        raise RuntimeError(
            f"Missing required v3 conductivities: {sorted(required - set(conductivities))}"
        )
    return {
        "simnibs_version": version,
        "session_api": session_ok,
        "element_data_api": element_ok,
        "conductivities": conductivities,
    }


def module_preflight(*, receipt: Path | None) -> dict[str, Any]:
    api = load_v3_meshing_api()
    meshing = probe_v3_meshing_api(api)
    fem = _load_simnibs_fem_api()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "expected_module": SIMNIBS_MODULE,
        "python_executable": sys.executable,
        "meshing_api": meshing,
        "fem_api": fem,
        "matlab_required": False,
        "cat12_required": False,
    }
    if receipt is not None:
        _write_json_atomic(receipt.expanduser().resolve(), payload)
    return payload


def validate_module_preflight_receipt(receipt: Path) -> dict[str, Any]:
    receipt = receipt.expanduser().resolve()
    payload = _load_json(receipt)
    if payload.get("status") != "ready":
        raise RuntimeError("Module preflight receipt is not ready")
    if payload.get("expected_module") != SIMNIBS_MODULE:
        raise RuntimeError("Module preflight receipt names a different module")
    if payload.get("matlab_required") is not False or payload.get("cat12_required") is not False:
        raise RuntimeError("Module preflight unexpectedly requires MATLAB/CAT12")
    meshing = payload.get("meshing_api")
    fem = payload.get("fem_api")
    if not isinstance(meshing, dict) or meshing.get("simnibs_version") != "3.2.6":
        raise RuntimeError("Module receipt lacks the pinned v3 meshing probe")
    if not isinstance(fem, dict) or fem.get("simnibs_version") != "3.2.6":
        raise RuntimeError("Module receipt lacks the pinned v3 FEM probe")
    if not fem.get("session_api") or not fem.get("element_data_api"):
        raise RuntimeError("Module receipt records an incomplete FEM API")
    return {
        **payload,
        "receipt": str(receipt),
        "receipt_sha256": _sha256(receipt),
    }


def _validate_config_outputs(config_path: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path, validate_paths=False)
    tasks = iter_experiment_tasks(config)
    failures: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        failed = [check.name for check in validate_task(config, task) if not check.ok]
        if failed:
            failures.append({
                "task_index": index,
                "subject": task.subject,
                "condition": task.condition_name,
                "repeat_tag": task.repeat_tag,
                "failed_checks": failed,
            })
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
    return {
        "target": target,
        "stage": stage,
        "subject_count": len(config.subjects),
        "conditions": counts,
        "expected_ti_nifti": total,
        "expected_ti_msh": total,
        "simnibs_module": SIMNIBS_MODULE,
        "backend": BACKEND_NAME,
    }


def _write_completion(
    *, root: Path, config_path: Path, target: str, stage: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    path = _completion_receipt(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stage": stage,
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "scope": _receipt_scope(config_path, target=target, stage=stage),
        "validation": validation,
        "previous_receipt_sha256": _sha256(path) if path.is_file() else None,
    }
    _write_json_atomic(path, payload)
    return payload


def complete_remesh(
    target: str, *, paths: CampaignPaths | None = None
) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    root = target_settings(target, resolved).experiment_root
    config_path = _config_path(root, REMESH_CONFIG)
    validation = _validate_config_outputs(config_path)
    if validation.get("status") != "complete" or validation.get("complete_tasks") != 400:
        raise RuntimeError(
            f"Remesh stage requires exactly 400 complete tasks; observed {validation}"
        )
    return _write_completion(
        root=root, config_path=config_path, target=target, stage="remesh",
        validation=validation,
    )


def prepare_fixed_stage(
    target: str, *, metrics_csv: Path, dry_run: bool,
    paths: CampaignPaths | None = None,
) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    root = target_settings(target, resolved).experiment_root
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
    for key in ("subjects", "stimulation"):
        if remesh.get(key) != fixed.get(key):
            raise RuntimeError(f"Remesh and fixed configs have different {key}")
    combined = json.loads(json.dumps(remesh))
    combined["conditions"] = remesh["conditions"] + fixed["conditions"]
    return combined


def complete_final(
    target: str, *, paths: CampaignPaths | None = None
) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    root = target_settings(target, resolved).experiment_root
    remesh_path = _config_path(root, REMESH_CONFIG)
    fixed_path = _config_path(root, FIXED_CONFIG)
    remesh = _validate_config_outputs(remesh_path)
    fixed = _validate_config_outputs(fixed_path)
    for label, validation in (("remesh", remesh), ("fixed", fixed)):
        if validation.get("status") != "complete" or validation.get("complete_tasks") != 400:
            raise RuntimeError(f"Final stage {label} condition is not exactly 400/400")
    paired_path = _config_path(root, PAIRED_CONFIG)
    current = _load_json(paired_path)
    remesh_config_payload = _load_json(remesh_path)
    combined = _combined_config(root)
    if current not in (remesh_config_payload, combined):
        raise RuntimeError(f"Unexpected paired config state: {paired_path}")
    _write_json_atomic(paired_path, combined)
    validation = {
        "status": "complete", "remesh": remesh, "fixed_mesh": fixed,
        "total_tasks": 800,
    }
    return _write_completion(
        root=root, config_path=paired_path, target=target, stage="final",
        validation=validation,
    )


def validate_final_receipt(
    target: str, *, paths: CampaignPaths | None = None
) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    root = target_settings(target, resolved).experiment_root
    paired = _config_path(root, PAIRED_CONFIG)
    receipt_path = _completion_receipt(root)
    receipt = _load_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("stage") != "final":
        raise RuntimeError("Final completion receipt is not complete")
    if receipt.get("config_sha256") != _sha256(paired):
        raise RuntimeError("Final receipt does not match the paired config")
    validation = receipt.get("validation", {})
    if not isinstance(validation, dict) or validation.get("total_tasks") != 800:
        raise RuntimeError("Final receipt does not prove 800 tasks")
    for condition in ("remesh", "fixed_mesh"):
        record = validation.get(condition, {})
        if record.get("complete_tasks") != 400 or record.get("failure_count") != 0:
            raise RuntimeError(f"Final receipt does not prove 400 {condition} tasks")
    return {
        "status": "ready", "target": target, "paired_config": str(paired),
        "paired_config_sha256": _sha256(paired),
        "completion_receipt": str(receipt_path),
        "completion_receipt_sha256": _sha256(receipt_path),
        "expected_metric_rows": 800,
    }


def _target_counts(root: Path) -> dict[str, int]:
    counts = {"remesh": 0, "fixed_mesh": 0}
    for condition in counts:
        config_name = REMESH_CONFIG if condition == "remesh" else FIXED_CONFIG
        config_path = _config_path(root, config_name)
        if not config_path.is_file():
            continue
        config = load_experiment_config(config_path, validate_paths=False)
        for task in iter_experiment_tasks(config):
            task_root = (
                root / f"{task.subject}_repeatability" / task.condition_name
                / "repeats" / task.repeat_tag / task.subject / "anat" / "SimNIBS"
            )
            if (task_root / "ti_brain_only.nii.gz").is_file() and (
                task_root / "Output" / task.subject / "TI.msh"
            ).is_file():
                counts[condition] += 1
    return counts


def status(*, paths: CampaignPaths | None = None) -> dict[str, Any]:
    resolved = paths or campaign_paths_from_environment()
    target_counts = {
        target: _target_counts(target_settings(target, resolved).experiment_root)
        for target in ("left-hippocampus", "right-m1")
    }
    initialized = all(
        _campaign_manifest(root).is_file() for root in (resolved.left_root, resolved.right_root)
    )
    references = (resolved.headmodel_root / "_pipeline/reference_stage_collection.json").is_file()
    headmodels = (resolved.headmodel_root / "_pipeline/headmodels/manifest.json").is_file()
    promoted = (resolved.headmodel_root / "_pipeline/headmodels/promoted.json").is_file()
    remesh = all(value["remesh"] == 400 for value in target_counts.values())
    final = all(
        value["remesh"] == 400 and value["fixed_mesh"] == 400
        for value in target_counts.values()
    )
    gates = {
        "initialized": initialized,
        "references_staged": references,
        "headmodels_built": headmodels,
        "promoted": promoted,
        "remesh_complete": remesh,
        "final_complete": final,
    }
    order = (
        ("final-complete", final),
        ("remesh-complete", remesh),
        ("promoted", promoted),
        ("headmodels-built", headmodels),
        ("references-staged", references),
        ("initialized", initialized),
    )
    state = next((name for name, ready in order if ready), "not-initialized")
    return {"status": "ok", "state": state, "gates": gates, "targets": target_counts}


def _render_qc(paths: CampaignPaths, subject: str, render_python: str) -> dict[str, Any]:
    request = write_qc_request(paths, subject)
    renderer = PACKAGE_ROOT / "render_qc.py"
    subprocess.run([render_python, str(renderer), str(request)], check=True)
    return {"status": "rendered", "subject": subject, "request": str(request)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    init = commands.add_parser("init")
    init.add_argument("--dry-run", action="store_true")
    module = commands.add_parser("module-preflight")
    module.add_argument("--receipt", type=Path)
    validate_module = commands.add_parser("validate-module-receipt")
    validate_module.add_argument("--receipt", type=Path, required=True)
    for name in ("stage-reference", "build-headmodel", "validate-headmodel", "render-qc"):
        child = commands.add_parser(name)
        group = child.add_mutually_exclusive_group(required=True)
        group.add_argument("--subject", choices=SUBJECTS)
        group.add_argument("--subject-index", type=int, choices=range(len(SUBJECTS)))
        if name == "render-qc":
            child.add_argument("--render-python", required=True)
    commands.add_parser("collect-references")
    collect = commands.add_parser("collect-headmodels")
    collect.add_argument("--render-python", required=True)
    promote = commands.add_parser("promote-headmodels")
    promote.add_argument("--manifest-sha256", required=True)
    promote.add_argument("--confirm-reviewed", action="store_true")
    for name in ("complete-remesh", "complete-final", "validate-final"):
        child = commands.add_parser(name)
        child.add_argument("--target", choices=("left-hippocampus", "right-m1"), required=True)
    fixed = commands.add_parser("prepare-fixed")
    fixed.add_argument("--target", choices=("left-hippocampus", "right-m1"), required=True)
    fixed.add_argument("--metrics-csv", type=Path, required=True)
    fixed.add_argument("--dry-run", action="store_true")
    commands.add_parser("show-plan")
    commands.add_parser("status")
    return parser


def _selected_subject(args: argparse.Namespace) -> str:
    return args.subject if args.subject is not None else SUBJECTS[args.subject_index]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = campaign_paths_from_environment()
    command = args.command
    if command == "preflight":
        payload = preflight(paths=paths)
    elif command == "init":
        payload = initialize(paths=paths, dry_run=args.dry_run)
    elif command == "module-preflight":
        payload = module_preflight(receipt=args.receipt)
    elif command == "validate-module-receipt":
        payload = validate_module_preflight_receipt(args.receipt)
    elif command == "stage-reference":
        payload = stage_subject_reference(subject_paths(paths, _selected_subject(args))).__dict__
    elif command == "collect-references":
        payload = collect_reference_stage(paths)
    elif command == "build-headmodel":
        payload = build_headmodel(paths, _selected_subject(args), load_v3_meshing_api())
    elif command == "validate-headmodel":
        payload = validate_headmodel(paths, _selected_subject(args), load_v3_meshing_api())
    elif command == "render-qc":
        payload = _render_qc(paths, _selected_subject(args), args.render_python)
    elif command == "collect-headmodels":
        payload = collect_headmodels(paths, args.render_python)
    elif command == "promote-headmodels":
        payload = promote_headmodels(paths, args.manifest_sha256, args.confirm_reviewed)
    elif command == "complete-remesh":
        payload = complete_remesh(args.target, paths=paths)
    elif command == "prepare-fixed":
        payload = prepare_fixed_stage(
            args.target, metrics_csv=args.metrics_csv, dry_run=args.dry_run, paths=paths
        )
    elif command == "complete-final":
        payload = complete_final(args.target, paths=paths)
    elif command == "validate-final":
        payload = validate_final_receipt(args.target, paths=paths)
    elif command == "show-plan":
        payload = {"scope": campaign_scope(paths), "status": status(paths=paths)}
    else:
        payload = status(paths=paths)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") not in ("failed", "incomplete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
