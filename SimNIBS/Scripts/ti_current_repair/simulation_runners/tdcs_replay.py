#!/usr/bin/env python3
"""Run one audited tDCS solve on an existing read-only v3 or v4 head mesh."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
CURRENT_REPAIR_ROOT = HERE.parents[1]
SCRIPTS_ROOT = CURRENT_REPAIR_ROOT.parent
for candidate in (CURRENT_REPAIR_ROOT, SCRIPTS_ROOT):
    try:
        sys.path.remove(str(candidate))
    except ValueError:
        pass
for candidate in (SCRIPTS_ROOT, CURRENT_REPAIR_ROOT):
    sys.path.insert(0, str(candidate))

from simulation_runners import repeatability_experiment as field_runner  # noqa: E402
from tdcs_replay import (  # noqa: E402
    VERSION_MODULES,
    VERSION_PREFIXES,
    VERSION_V3,
    VERSION_V4,
    VERSIONS,
    ReplayConfigurationError,
    ReplayInputError,
    file_sha256,
    load_json,
    required_output_paths,
    stimulation_from_plan,
    task_for_index,
    validate_audit,
    validate_completed_task,
    validate_plan,
    validate_task_source,
    write_json_atomic,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _install_v3_backend() -> str:
    package_root = SCRIPTS_ROOT / "simnibs_326_repeatability"
    backend_path = package_root / "simulation_runner.py"
    if not backend_path.is_file():
        raise ReplayConfigurationError(f"Missing SimNIBS 3.2.6 backend: {backend_path}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    spec = importlib.util.spec_from_file_location(
        "tdcs_replay_simnibs326_backend",
        backend_path,
    )
    if spec is None or spec.loader is None:
        raise ReplayConfigurationError(f"Could not load SimNIBS 3.2.6 backend: {backend_path}")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    backend.install_backend()
    backend._ensure_simnibs_imports_326()
    return str(getattr(field_runner.SIM_MODULE, "__version__", "unknown"))


def install_backend(version: str) -> str:
    if version == VERSION_V3:
        observed = _install_v3_backend()
    elif version == VERSION_V4:
        field_runner._ensure_simnibs_imports()
        observed = str(getattr(field_runner.SIM_MODULE, "__version__", "unknown"))
    else:
        raise ReplayConfigurationError(f"Unsupported replay version: {version}")
    if not observed.startswith(VERSION_PREFIXES[version]):
        raise ReplayConfigurationError(
            f"Expected {VERSION_MODULES[version]}; observed SimNIBS {observed}."
        )
    return observed


def _output_inventory(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ReplayInputError(f"Required tDCS output is missing or empty: {path}")
        inventory[label] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return inventory


def _is_nonempty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def run_task(args: argparse.Namespace) -> int:
    plan_path = args.plan.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    plan = load_json(plan_path)
    audit = load_json(audit_path)
    validate_plan(plan, require_parameters=True, require_submission=False)
    validate_audit(plan, audit)
    task = task_for_index(plan, args.version, args.task_index)
    source_record = validate_task_source(audit, task)
    stimulation = stimulation_from_plan(plan)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "task": task.to_dict(),
                    "source_mesh_sha256": source_record["sha256"],
                    "source_policy": "read_only",
                    "stimulation": stimulation.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if task.task_manifest.is_file():
        try:
            validate_completed_task(
                plan_path=plan_path,
                audit_path=audit_path,
                plan=plan,
                audit=audit,
                task=task,
                verify_hashes=True,
            )
        except ReplayInputError:
            if not args.overwrite:
                raise
        else:
            print(json.dumps({"status": "skipped_complete", "task": task.to_dict()}))
            return 0

    if _is_nonempty_directory(task.output_root) and not args.overwrite:
        raise ReplayInputError(
            "Incomplete output directory already exists; use --overwrite only for an "
            f"intentional task retry: {task.output_root}"
        )
    if args.overwrite and task.task_manifest.exists():
        task.task_manifest.unlink()

    observed_version = install_backend(args.version)
    elapsed = field_runner._run_tdcs_pipeline(
        subject=task.subject,
        subject_dir=task.source_anat_dir,
        output_root=task.output_root,
        fnamehead=task.source_mesh,
        repeat_tag_value=task.repeat_tag,
        condition_name=task.condition,
        stimulation=stimulation,
    )
    outputs = _output_inventory(required_output_paths(task))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": _utc_now(),
        "task": task.to_dict(),
        "source_policy": "read_only",
        "source_mesh": {
            "path": str(task.source_mesh),
            "size_bytes": int(source_record["size_bytes"]),
            "sha256": str(source_record["sha256"]),
        },
        "plan": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "audit": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "stimulation": stimulation.to_dict(),
        "runtime": {
            "simnibs_module": VERSION_MODULES[args.version],
            "simnibs_version": observed_version,
            "elapsed_sec": elapsed,
        },
        "outputs": outputs,
    }
    write_json_atomic(task.task_manifest, manifest)
    validate_completed_task(
        plan_path=plan_path,
        audit_path=audit_path,
        plan=plan,
        audit=audit,
        task=task,
        verify_hashes=True,
    )
    print(json.dumps({"status": "complete", "task": task.to_dict()}, sort_keys=True))
    return 0


def show_task(args: argparse.Namespace) -> int:
    plan = load_json(args.plan)
    validate_plan(plan, require_parameters=False, require_submission=False)
    print(json.dumps(task_for_index(plan, args.version, args.task_index).to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run-task", "show-task"):
        command = subparsers.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--version", choices=VERSIONS, required=True)
        command.add_argument("--task-index", type=int, required=True)
        if name == "run-task":
            command.add_argument("--audit", type=Path, required=True)
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run-task":
            return run_task(args)
        return show_task(args)
    except ReplayInputError as exc:
        print(f"[INPUT ERROR] {exc}", file=sys.stderr)
        return field_runner.SIM_INPUT_EXIT_CODE
    except (ReplayConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
