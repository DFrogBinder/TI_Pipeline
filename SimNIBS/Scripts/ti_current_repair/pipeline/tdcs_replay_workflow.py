#!/usr/bin/env python3
"""Preflight, prepare, validate, and submit the full right-M1 tDCS replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
CURRENT_REPAIR_ROOT = HERE.parents[1]
SCRIPTS_ROOT = CURRENT_REPAIR_ROOT.parent
for candidate in (SCRIPTS_ROOT, CURRENT_REPAIR_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from simulation_runners import tdcs_replay as task_runner  # noqa: E402
from tdcs_replay import (  # noqa: E402
    VERSION_MODULES,
    VERSION_PREFIXES,
    VERSION_V3,
    VERSION_V4,
    VERSIONS,
    ReplayConfigurationError,
    ReplayInputError,
    analysis_config,
    build_tasks,
    file_sha256,
    input_root,
    load_json,
    output_root,
    plan_blockers,
    stimulation_from_plan,
    subjects_from_plan,
    validate_audit,
    validate_completed_task,
    validate_plan,
    validate_task_source,
    write_json_atomic,
)


PARTITION = "sheffield"
SIM_CPUS = 8
SIM_MEMORY = "32G"
SIM_TIME = "08:00:00"
SIM_MAX_CONCURRENT = 50
TASK_TIMEOUT_HOURS = 4
MAX_RETRIES = 1
SBATCH_BIN = os.environ.get("SBATCH_BIN", "sbatch")
CONTROL_CPUS = 4
CONTROL_MEMORY = "16G"
CONTROL_TIME = "02:00:00"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _audit_or_none(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.expanduser().is_file():
        return None
    return load_json(path)


def scope_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "existing final-132 balanced-10 right-M1 v4/v3 head models",
        "roi": "right-m1",
        "roi_definition": "optimizer-matched-spherical",
        "subjects": len(subjects_from_plan(plan)),
        "v4_tasks": len(build_tasks(plan, VERSION_V4)),
        "v4_array": "0-799%50",
        "v3_tasks": len(build_tasks(plan, VERSION_V3)),
        "v3_array": "0-399%50",
        "total_simulation_tasks": 1200,
        "expected_tdcs_msh": 1200,
        "expected_tdcs_brain_nifti": 1200,
        "expected_fem_solves": 1200,
        "execution_scope": "full requested experiment; not a smoke or subset",
        "source_policy": "read_only",
        "v4_output_root": str(output_root(plan, VERSION_V4)),
        "v3_output_root": str(output_root(plan, VERSION_V3)),
    }


def print_scope(scope: dict[str, Any]) -> None:
    print("Scope:")
    print(f"  dataset/ROI: {scope['dataset']} / {scope['roi']}")
    print(f"  subjects: {scope['subjects']}")
    print(f"  v4 tasks: {scope['v4_tasks']} ({scope['v4_array']})")
    print(f"  v3 tasks: {scope['v3_tasks']} ({scope['v3_array']})")
    print(f"  total simulation tasks: {scope['total_simulation_tasks']}")
    print(f"  expected TDCS.msh outputs: {scope['expected_tdcs_msh']}")
    print(f"  expected FEM solves: {scope['expected_fem_solves']}")
    print(f"  source policy: {scope['source_policy']}")
    print(f"  execution: {scope['execution_scope']}")


def show_plan(args: argparse.Namespace) -> int:
    plan = load_json(args.plan)
    audit = _audit_or_none(args.audit)
    blockers = plan_blockers(plan, require_submission=False)
    if audit is None:
        blockers.append("passing mesh audit receipt is unavailable")
    else:
        try:
            validate_audit(plan, audit)
        except ReplayConfigurationError as exc:
            blockers.append(str(exc))
    payload = {
        "status": "ready_for_preflight" if not blockers else "blocked",
        "blockers": blockers,
        "scope": scope_payload(plan),
        "modules": VERSION_MODULES,
        "resources": {
            "partition": PARTITION,
            "cpus_per_task": SIM_CPUS,
            "memory": SIM_MEMORY,
            "time_limit": SIM_TIME,
            "max_concurrent": SIM_MAX_CONCURRENT,
            "max_retries": MAX_RETRIES,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def preflight(
    *,
    plan_path: Path,
    audit_path: Path,
    require_submission: bool,
    check_outputs: bool = True,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    plan = load_json(plan_path)
    audit = load_json(audit_path)
    validate_plan(
        plan,
        require_parameters=True,
        require_submission=require_submission,
    )
    validate_audit(plan, audit)
    stimulation = stimulation_from_plan(plan)
    unique_source_meshes: set[Path] = set()
    completed = 0
    incomplete_outputs: list[str] = []
    for version in VERSIONS:
        for task in build_tasks(plan, version):
            if task.source_mesh not in unique_source_meshes:
                validate_task_source(audit, task)
                unique_source_meshes.add(task.source_mesh)
            if task.task_manifest.is_file():
                try:
                    validate_completed_task(
                        plan_path=plan_path,
                        audit_path=audit_path,
                        plan=plan,
                        audit=audit,
                        task=task,
                    )
                except ReplayInputError as exc:
                    incomplete_outputs.append(str(exc))
                else:
                    completed += 1
            elif check_outputs and task.output_root.is_dir() and next(task.output_root.iterdir(), None) is not None:
                incomplete_outputs.append(f"Unproven output directory: {task.output_root}")
    atlas_dir = Path(plan["analysis"]["atlas_dir"]).expanduser().resolve()
    missing_atlases = [
        atlas_dir / f"{subject}.nii.gz"
        for subject in subjects_from_plan(plan)
        if not (atlas_dir / f"{subject}.nii.gz").is_file()
    ]
    if missing_atlases:
        raise ReplayInputError(
            "Missing right-M1 analysis atlas inputs: "
            + ", ".join(str(path) for path in missing_atlases)
        )
    if incomplete_outputs:
        raise ReplayInputError(
            "Existing output state is incomplete or has drifted: "
            + "; ".join(incomplete_outputs[:10])
        )
    return {
        "schema_version": 1,
        "status": "ready",
        "checked_utc": _utc_now(),
        "plan": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "audit": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "scope": scope_payload(plan),
        "stimulation": stimulation.to_dict(),
        "unique_source_meshes_checked": len(unique_source_meshes),
        "existing_completed_tasks": completed,
        "source_roots": {
            version: str(input_root(plan, version)) for version in VERSIONS
        },
        "output_roots": {
            version: str(output_root(plan, version)) for version in VERSIONS
        },
        "source_policy": "read_only",
    }


def command_preflight(args: argparse.Namespace) -> int:
    receipt = preflight(
        plan_path=args.plan,
        audit_path=args.audit,
        require_submission=False,
    )
    print_scope(receipt["scope"])
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def prepare_outputs(plan_path: Path, audit_path: Path) -> dict[str, str]:
    receipt = preflight(
        plan_path=plan_path,
        audit_path=audit_path,
        require_submission=False,
    )
    plan = load_json(plan_path)
    outputs: dict[str, str] = {}
    for version in VERSIONS:
        root = output_root(plan, version)
        config_path = root / "_pipeline" / "configs" / "tdcs_replay.json"
        config = analysis_config(
            plan,
            version,
            plan_sha256=receipt["plan_sha256"],
            audit_sha256=receipt["audit_sha256"],
        )
        if config_path.is_file() and load_json(config_path) != config:
            raise ReplayConfigurationError(
                f"Refusing to overwrite a differing replay config: {config_path}"
            )
        write_json_atomic(config_path, config)
        write_json_atomic(root / "_pipeline" / "workflow" / "preflight.json", receipt)
        outputs[version] = str(config_path)
    return outputs


def command_prepare(args: argparse.Namespace) -> int:
    outputs = prepare_outputs(args.plan, args.audit)
    print(json.dumps({"status": "prepared", "configs": outputs}, indent=2, sort_keys=True))
    return 0


def module_preflight(args: argparse.Namespace) -> int:
    plan_path = args.plan.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    preflight(
        plan_path=plan_path,
        audit_path=audit_path,
        require_submission=False,
        check_outputs=False,
    )
    observed = task_runner.install_backend(args.version)
    receipt = {
        "schema_version": 1,
        "status": "ready",
        "created_utc": _utc_now(),
        "version": args.version,
        "simnibs_module": VERSION_MODULES[args.version],
        "simnibs_version": observed,
        "plan_sha256": file_sha256(plan_path),
        "audit_sha256": file_sha256(audit_path),
    }
    write_json_atomic(args.receipt.expanduser().resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def validate_module_receipt(
    *,
    receipt_path: Path,
    plan_path: Path,
    audit_path: Path,
    version: str,
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    expected = {
        "status": "ready",
        "version": version,
        "simnibs_module": VERSION_MODULES[version],
        "plan_sha256": file_sha256(plan_path),
        "audit_sha256": file_sha256(audit_path),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ReplayConfigurationError(
                f"Module receipt {key} differs: {receipt_path}"
            )
    observed = str(receipt.get("simnibs_version", ""))
    if not observed.startswith(VERSION_PREFIXES[version]):
        raise ReplayConfigurationError(
            f"Module receipt version differs for {version}: {observed}"
        )
    return receipt


def command_validate_module(args: argparse.Namespace) -> int:
    receipt = validate_module_receipt(
        receipt_path=args.receipt,
        plan_path=args.plan,
        audit_path=args.audit,
        version=args.version,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def command_validate_task(args: argparse.Namespace) -> int:
    plan_path = args.plan.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    plan = load_json(plan_path)
    audit = load_json(audit_path)
    validate_plan(plan, require_parameters=True, require_submission=False)
    validate_audit(plan, audit)
    tasks = build_tasks(plan, args.version)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ReplayConfigurationError(
            f"Task index {args.task_index} outside 0..{len(tasks) - 1} for {args.version}."
        )
    task = tasks[args.task_index]
    manifest = validate_completed_task(
        plan_path=plan_path,
        audit_path=audit_path,
        plan=plan,
        audit=audit,
        task=task,
        verify_hashes=True,
    )
    print(json.dumps({"status": "complete", "manifest": str(task.task_manifest), "runtime": manifest["runtime"]}))
    return 0


def finalize_version(args: argparse.Namespace) -> int:
    plan_path = args.plan.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    plan = load_json(plan_path)
    audit = load_json(audit_path)
    validate_plan(plan, require_parameters=True, require_submission=False)
    validate_audit(plan, audit)
    tasks = build_tasks(plan, args.version)
    for task in tasks:
        validate_completed_task(
            plan_path=plan_path,
            audit_path=audit_path,
            plan=plan,
            audit=audit,
            task=task,
        )
    root = output_root(plan, args.version)
    conditions = {"remesh": 40, "fixed_mesh": 40} if args.version == VERSION_V4 else {"fixed_mesh": 40}
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": _utc_now(),
        "version": args.version,
        "plan_sha256": file_sha256(plan_path),
        "audit_sha256": file_sha256(audit_path),
        "scope": {
            "roi": "right-m1",
            "subject_count": 10,
            "conditions": conditions,
            "repeats_per_condition": 40,
            "total_simulation_tasks": len(tasks),
            "expected_field_msh": len(tasks),
            "expected_field_nifti": len(tasks),
            "expected_fem_solves": len(tasks),
            "stimulation_mode": "tdcs",
            "source_policy": "read_only",
        },
        "config": str(root / "_pipeline" / "configs" / "tdcs_replay.json"),
    }
    receipt_path = root / "_pipeline" / "workflow" / "complete.json"
    write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def finalize_all(args: argparse.Namespace) -> int:
    plan_path = args.plan.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    plan = load_json(plan_path)
    receipts: dict[str, dict[str, Any]] = {}
    for version in VERSIONS:
        path = output_root(plan, version) / "_pipeline" / "workflow" / "complete.json"
        payload = load_json(path)
        if payload.get("status") != "complete" or payload.get("version") != version:
            raise ReplayInputError(f"Version completion receipt is invalid: {path}")
        if payload.get("plan_sha256") != file_sha256(plan_path):
            raise ReplayInputError(f"Version completion plan hash differs: {path}")
        if payload.get("audit_sha256") != file_sha256(audit_path):
            raise ReplayInputError(f"Version completion audit hash differs: {path}")
        receipts[version] = {"path": str(path), "sha256": file_sha256(path)}
    combined = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": _utc_now(),
        "scope": scope_payload(plan),
        "plan_sha256": file_sha256(plan_path),
        "audit_sha256": file_sha256(audit_path),
        "version_receipts": receipts,
    }
    path = output_root(plan, VERSION_V4) / "_pipeline" / "workflow" / "all_complete.json"
    write_json_atomic(path, combined)
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0


def _parse_job_id(raw: str) -> str:
    value = raw.strip().split(";", maxsplit=1)[0]
    if not value.isdigit():
        raise RuntimeError(f"Could not parse Slurm job ID from: {raw!r}")
    return value


def _submit(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return _parse_job_id(result.stdout)


def _sbatch_base(*, job_name: str, output: Path) -> list[str]:
    return [
        SBATCH_BIN,
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={PARTITION}",
        f"--output={output}",
        "--open-mode=append",
    ]


def submit_workflow(args: argparse.Namespace) -> int:
    plan_path = args.plan.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    plan = load_json(plan_path)
    scope = scope_payload(plan)
    print_scope(scope)
    if not args.submit:
        blockers = plan_blockers(plan, require_submission=True)
        if not audit_path.is_file():
            blockers.append(f"passing mesh audit receipt is unavailable: {audit_path}")
        else:
            try:
                validate_audit(plan, load_json(audit_path))
            except ReplayConfigurationError as exc:
                blockers.append(str(exc))
        print("Submission: NOT REQUESTED")
        print(json.dumps({"status": "dry_run", "blockers": blockers}, indent=2))
        return 0

    preflight_receipt = preflight(
        plan_path=plan_path,
        audit_path=audit_path,
        require_submission=True,
    )
    prepare_outputs(plan_path, audit_path)
    hpc = CURRENT_REPAIR_ROOT / "hpc_scripts"
    module_script = hpc / "tdcs_replay_module_preflight.slurm"
    array_script = hpc / "tdcs_replay_array.slurm"
    control_script = hpc / "tdcs_replay_control.slurm"
    for path in (module_script, array_script, control_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    log_dirs = {
        version: output_root(plan, version) / "_pipeline" / "logs"
        for version in VERSIONS
    }
    for path in log_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    module_receipts = {
        version: output_root(plan, version) / "_pipeline" / "module_preflight.json"
        for version in VERSIONS
    }

    def exports(version: str, **extra: object) -> str:
        values = {
            "PIPELINE_DIR": CURRENT_REPAIR_ROOT,
            "TDCS_REPLAY_PLAN": plan_path,
            "TDCS_REPLAY_AUDIT": audit_path,
            "TDCS_REPLAY_VERSION": version,
            **extra,
        }
        return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())

    v4_module = _submit(
        _sbatch_base(job_name="tdcs_rmi_v4_check", output=log_dirs[VERSION_V4] / "module-%j.out")
        + [
            "--cpus-per-task=1",
            "--mem=4G",
            "--time=00:20:00",
            f"--export={exports(VERSION_V4, TDCS_REPLAY_MODULE_RECEIPT=module_receipts[VERSION_V4])}",
            str(module_script),
        ]
    )
    v3_module = _submit(
        _sbatch_base(job_name="tdcs_rmi_v3_check", output=log_dirs[VERSION_V3] / "module-%j.out")
        + [
            "--cpus-per-task=1",
            "--mem=4G",
            "--time=00:20:00",
            f"--dependency=afterok:{v4_module}",
            f"--export={exports(VERSION_V3, TDCS_REPLAY_MODULE_RECEIPT=module_receipts[VERSION_V3])}",
            str(module_script),
        ]
    )

    def submit_array(version: str, count: int, dependency: str) -> str:
        return _submit(
            _sbatch_base(job_name=f"tdcs_rmi_{version}", output=log_dirs[version] / "slurm-%A_%a.out")
            + [
                f"--cpus-per-task={SIM_CPUS}",
                f"--mem={SIM_MEMORY}",
                f"--time={SIM_TIME}",
                f"--array=0-{count - 1}%{SIM_MAX_CONCURRENT}",
                f"--dependency=afterok:{dependency}",
                f"--export={exports(version, TDCS_REPLAY_LOG_DIR=log_dirs[version] / 'tasks', TDCS_REPLAY_MODULE_RECEIPT=module_receipts[version], TDCS_REPLAY_MAX_RETRIES=MAX_RETRIES, TDCS_REPLAY_TIMEOUT_HOURS=TASK_TIMEOUT_HOURS)}",
                str(array_script),
            ]
        )

    def submit_control(action: str, version: str, dependency: str) -> str:
        return _submit(
            _sbatch_base(job_name=f"tdcs_rmi_{action}_{version}", output=log_dirs[version] / f"{action}-%j.out")
            + [
                f"--cpus-per-task={CONTROL_CPUS}",
                f"--mem={CONTROL_MEMORY}",
                f"--time={CONTROL_TIME}",
                f"--dependency=afterok:{dependency}",
                f"--export={exports(version, TDCS_REPLAY_ACTION=action)}",
                str(control_script),
            ]
        )

    v4_array = submit_array(VERSION_V4, 800, v3_module)
    v4_finalize = submit_control("finalize-version", VERSION_V4, v4_array)
    v3_array = submit_array(VERSION_V3, 400, v4_finalize)
    v3_finalize = submit_control("finalize-version", VERSION_V3, v3_array)
    all_finalize = submit_control("finalize-all", VERSION_V4, v3_finalize)
    jobs = {
        "v4_module_preflight": v4_module,
        "v3_module_preflight": v3_module,
        "v4_array": v4_array,
        "v4_finalize": v4_finalize,
        "v3_array": v3_array,
        "v3_finalize": v3_finalize,
        "all_finalize": all_finalize,
    }
    submission = {
        "schema_version": 1,
        "status": "submitted",
        "submitted_utc": _utc_now(),
        "scope": scope,
        "preflight": preflight_receipt,
        "jobs": jobs,
    }
    write_json_atomic(
        output_root(plan, VERSION_V4) / "_pipeline" / "workflow" / "submission.json",
        submission,
    )
    print(json.dumps(submission, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_plan_audit(command: argparse.ArgumentParser, *, audit_optional: bool = False) -> None:
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--audit", type=Path, required=not audit_optional)

    show = subparsers.add_parser("show-plan")
    add_plan_audit(show, audit_optional=True)
    preflight_parser = subparsers.add_parser("preflight")
    add_plan_audit(preflight_parser)
    prepare = subparsers.add_parser("prepare")
    add_plan_audit(prepare)
    module = subparsers.add_parser("module-preflight")
    add_plan_audit(module)
    module.add_argument("--version", choices=VERSIONS, required=True)
    module.add_argument("--receipt", type=Path, required=True)
    validate_module = subparsers.add_parser("validate-module-receipt")
    add_plan_audit(validate_module)
    validate_module.add_argument("--version", choices=VERSIONS, required=True)
    validate_module.add_argument("--receipt", type=Path, required=True)
    validate_task_parser = subparsers.add_parser("validate-task")
    add_plan_audit(validate_task_parser)
    validate_task_parser.add_argument("--version", choices=VERSIONS, required=True)
    validate_task_parser.add_argument("--task-index", type=int, required=True)
    finalize = subparsers.add_parser("finalize-version")
    add_plan_audit(finalize)
    finalize.add_argument("--version", choices=VERSIONS, required=True)
    finalize_all_parser = subparsers.add_parser("finalize-all")
    add_plan_audit(finalize_all_parser)
    submit = subparsers.add_parser("submit")
    add_plan_audit(submit)
    submit.add_argument("--submit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "show-plan": show_plan,
        "preflight": command_preflight,
        "prepare": command_prepare,
        "module-preflight": module_preflight,
        "validate-module-receipt": command_validate_module,
        "validate-task": command_validate_task,
        "finalize-version": finalize_version,
        "finalize-all": finalize_all,
        "submit": submit_workflow,
    }
    try:
        return handlers[args.command](args)
    except (ReplayConfigurationError, ReplayInputError, FileNotFoundError, ValueError) as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
