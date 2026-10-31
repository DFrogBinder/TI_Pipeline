"""Owned dataset preparation and later consolidation lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .planning import build_copy_plan, plan_sha256
from .settings import DatasetConfig
from .validation import validate_scientific_scope


class WorkflowError(RuntimeError):
    """Raised when a dataset root cannot be safely prepared or reused."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def preflight(config: DatasetConfig) -> dict[str, object]:
    """Run the complete source audit without writing the destination."""
    plan = build_copy_plan(config)
    validation = validate_scientific_scope(plan)
    usage = shutil.disk_usage(config.output_root.parent)
    return {
        "status": "ready",
        "workflow": "repeatability_dataset_consolidation",
        "scope": validation.as_dict(),
        "plan_sha256": plan_sha256(plan),
        "planned_items": len(plan.items),
        "terminal_receipts": list(validation.terminal_receipts),
        "free_bytes": usage.free,
        "source_outputs_modified": False,
    }


def _result(config: DatasetConfig, plan_hash: str, scope: dict[str, int], action: str) -> dict[str, object]:
    return {
        "status": "prepared",
        "action": action,
        "workflow": "repeatability_dataset_consolidation",
        "output_root": str(config.output_root.resolve(strict=False)),
        "plan_sha256": plan_hash,
        "scope": scope,
        "source_outputs_modified": False,
    }


def _validate_owned_root(
    config: DatasetConfig,
    plan_payload: dict[str, object],
    plan_hash: str,
    scope: dict[str, int],
) -> dict[str, object]:
    root = config.output_root
    if root.is_symlink():
        raise WorkflowError(f"Refusing symbolic-link dataset root: {root}")
    required = (
        root / "_consolidation/owner.json",
        root / "_consolidation/planned_items.json",
        root / "dataset_manifest.json",
        root / "subjects.txt",
        root / "README.md",
    )
    if not all(path.is_file() and not path.is_symlink() for path in required):
        if (root / "_consolidation/owner.json").exists():
            raise WorkflowError(f"Refusing incomplete owned dataset root: {root}")
        raise WorkflowError(f"Refusing unowned dataset root: {root}")
    try:
        owner = json.loads(required[0].read_text(encoding="utf-8"))
        planned = json.loads(required[1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Unreadable ownership records under {root}") from exc
    expected_root = str(root.resolve())
    if owner.get("workflow") != "repeatability_dataset_consolidation":
        raise WorkflowError(f"Refusing unowned dataset root: {root}")
    if owner.get("output_root") != expected_root:
        raise WorkflowError("Existing owner is bound to a different output root")
    if owner.get("plan_sha256") != plan_hash:
        raise WorkflowError("Existing owner is bound to a different consolidation plan")
    if planned.get("plan_sha256") != plan_hash or planned.get("plan") != plan_payload:
        raise WorkflowError("Existing plan receipt differs from the current consolidation plan")
    return _result(config, plan_hash, scope, "reused")


def _create_skeleton(root: Path) -> None:
    for relative in (
        "data/simnibs326/left_hippocampus",
        "data/simnibs326/right_m1",
        "data/simnibs4/left_hippocampus",
        "data/simnibs4/right_m1",
        "headmodels/simnibs326",
        "headmodels/simnibs4",
        "analyses",
        "supplemental",
        "provenance/configs/historical",
        "provenance/configs/canonical",
        "provenance/receipts",
        "_consolidation/verification",
    ):
        (root / relative).mkdir(parents=True, exist_ok=False)


def _dataset_readme(config: DatasetConfig) -> str:
    return (
        "# Final-132 combined repeatability dataset v1\n\n"
        "Physical, receipt-bound consolidation of SimNIBS 3.2.6 and SimNIBS 4 "
        "left-hippocampus and right-M1 repeatability simulations.\n\n"
        f"Subjects: {len(config.subjects)}\n\n"
        f"Repeats per condition: {config.repeat_count}\n\n"
        f"Expected simulations: {config.expected_simulations}\n\n"
        f"Replacement: {config.replacement_subject} replaces {config.excluded_subject}.\n"
    )


def prepare(config: DatasetConfig) -> dict[str, object]:
    """Create or validate an atomically owned empty dataset skeleton."""
    plan = build_copy_plan(config)
    validation = validate_scientific_scope(plan)
    plan_payload = plan.as_dict()
    plan_hash = plan_sha256(plan)
    scope = validation.as_dict()
    root = config.output_root
    if root.exists() or root.is_symlink():
        return _validate_owned_root(config, plan_payload, plan_hash, scope)
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise WorkflowError(f"Dataset parent is missing or unsafe: {root.parent}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent)
    )
    promoted = False
    try:
        _create_skeleton(temporary)
        created_utc = _utc_now()
        git_commit = _git_commit()
        owner = {
            "schema_version": 1,
            "status": "owned",
            "workflow": "repeatability_dataset_consolidation",
            "created_utc": created_utc,
            "git_commit": git_commit,
            "output_root": str(root.resolve(strict=False)),
            "plan_sha256": plan_hash,
            "subjects": list(config.subjects),
        }
        planned = {
            "schema_version": 1,
            "status": "planned",
            "created_utc": created_utc,
            "plan_sha256": plan_hash,
            "plan": plan_payload,
        }
        manifest = {
            "schema_version": 1,
            "status": "prepared",
            "workflow": "repeatability_dataset_consolidation",
            "created_utc": created_utc,
            "git_commit": git_commit,
            "output_root": str(root.resolve(strict=False)),
            "plan_sha256": plan_hash,
            "subjects": list(config.subjects),
            "excluded_subject": config.excluded_subject,
            "scope": scope,
            "source_outputs_modified": False,
        }
        _write_json(temporary / "_consolidation/owner.json", owner)
        _write_json(temporary / "_consolidation/planned_items.json", planned)
        _write_json(temporary / "dataset_manifest.json", manifest)
        (temporary / "subjects.txt").write_text(
            "\n".join(config.subjects) + "\n", encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            _dataset_readme(config), encoding="utf-8"
        )
        os.replace(temporary, root)
        promoted = True
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)
    return _result(config, plan_hash, scope, "created")
