"""Owned dataset preparation and later consolidation lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .inventory import (
    build_file_inventory,
    build_materialized_inventory,
    sha256_file,
)
from .planning import ConsolidationPlan, CopyItem, build_copy_plan, plan_sha256
from .rsync import compare_item, copy_item
from .settings import DatasetConfig
from .validation import validate_item_identity, validate_scientific_scope


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


def _owned_context(
    config: DatasetConfig,
) -> tuple[ConsolidationPlan, str, dict[str, int]]:
    if not config.output_root.exists() or config.output_root.is_symlink():
        raise WorkflowError("Dataset root must be prepared before copying or verification")
    plan = build_copy_plan(config)
    validation = validate_scientific_scope(plan)
    plan_payload = plan.as_dict()
    plan_hash = plan_sha256(plan)
    scope = validation.as_dict()
    _validate_owned_root(config, plan_payload, plan_hash, scope)
    return plan, plan_hash, scope


def _receipt_name(item: CopyItem) -> str:
    import hashlib

    return f"{hashlib.sha256(item.key.encode('utf-8')).hexdigest()}.json"


def _item_destination(config: DatasetConfig, item: CopyItem) -> Path:
    root = config.output_root.absolute()
    destination = root / item.destination
    component = root
    for part in item.destination.parts:
        component = component / part
        if component.is_symlink():
            raise WorkflowError(
                f"Refusing symlinked destination component: {component}"
            )
    resolved_parent = destination.parent.resolve(strict=False)
    resolved_root = root.resolve()
    if resolved_root != resolved_parent and resolved_root not in resolved_parent.parents:
        raise WorkflowError(f"Planned destination escapes dataset root: {item.destination}")
    return destination


def copy_dataset(config: DatasetConfig) -> dict[str, object]:
    """Resume physical per-item copies while preserving every source."""
    plan, plan_hash, _ = _owned_context(config)
    receipt_root = config.output_root / "_consolidation/copy_items"
    receipt_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    for item in plan.items:
        validate_item_identity(item)
        destination = _item_destination(config, item)
        comparison = compare_item(item.source, destination, checksum=True)
        if comparison.matched:
            action = "reused"
            command = comparison.command
            reused += 1
        else:
            operation = copy_item(item.source, destination)
            action = "copied"
            command = operation.command
            copied += 1
        validate_item_identity(item)
        _write_json(
            receipt_root / _receipt_name(item),
            {
                "schema_version": 1,
                "status": "copied",
                "action": action,
                "created_utc": _utc_now(),
                "plan_sha256": plan_hash,
                "key": item.key,
                "source": str(item.source),
                "destination": str(destination),
                "command": list(command),
                "source_outputs_modified": False,
            },
        )
    return {
        "status": "copied",
        "workflow": "repeatability_dataset_consolidation",
        "plan_sha256": plan_hash,
        "planned_items": len(plan.items),
        "copied_items": copied,
        "reused_items": reused,
        "source_outputs_modified": False,
    }


def _verification_item_payload(
    config: DatasetConfig,
    item: CopyItem,
    *,
    plan_hash: str,
    checksum: bool,
) -> tuple[dict[str, object], tuple[str, ...]]:
    validate_item_identity(item)
    destination = _item_destination(config, item)
    comparison = compare_item(item.source, destination, checksum=checksum)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "matched" if comparison.matched else "mismatch",
        "created_utc": _utc_now(),
        "mode": "checksum" if checksum else "size_mtime",
        "plan_sha256": plan_hash,
        "key": item.key,
        "source": str(item.source),
        "destination": str(destination),
        "command": list(comparison.command),
        "changes": list(comparison.changes),
        "source_outputs_modified": False,
    }
    if checksum:
        source_inventory = build_materialized_inventory(item.source)
        try:
            destination_inventory = build_file_inventory(destination)
        except ValueError as exc:
            payload["status"] = "mismatch"
            payload["inventory_error"] = str(exc)
        else:
            payload["source_inventory_sha256"] = source_inventory.inventory_sha256
            payload["destination_inventory_sha256"] = (
                destination_inventory.inventory_sha256
            )
            if source_inventory.inventory_sha256 != destination_inventory.inventory_sha256:
                payload["status"] = "mismatch"
        validate_item_identity(item)
    return payload, comparison.changes


def verify_dataset(
    config: DatasetConfig,
    *,
    checksum: bool,
) -> dict[str, object]:
    """Compare every planned item and persist tamper-evident item receipts."""
    plan, plan_hash, _ = _owned_context(config)
    mode = "checksum" if checksum else "size_mtime"
    verification_root = config.output_root / "_consolidation/verification" / mode
    items_root = verification_root / "items"
    items_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    changes: list[str] = []
    for item in plan.items:
        payload, item_changes = _verification_item_payload(
            config, item, plan_hash=plan_hash, checksum=checksum
        )
        receipt = items_root / _receipt_name(item)
        _write_json(receipt, payload)
        records.append(
            {
                "key": item.key,
                "status": payload["status"],
                "receipt": str(receipt.relative_to(config.output_root)),
                "receipt_sha256": sha256_file(receipt),
            }
        )
        changes.extend(f"{item.key}: {change}" for change in item_changes)
    mismatches = sum(record["status"] != "matched" for record in records)
    status = "matched" if mismatches == 0 else "mismatch"
    top = {
        "schema_version": 1,
        "status": status,
        "created_utc": _utc_now(),
        "mode": mode,
        "plan_sha256": plan_hash,
        "planned_items": len(plan.items),
        "matched_items": len(plan.items) - mismatches,
        "mismatched_items": mismatches,
        "items": records,
        "source_outputs_modified": False,
    }
    top_path = verification_root / "verification.json"
    _write_json(top_path, top)
    return {
        **{key: value for key, value in top.items() if key != "items"},
        "verification_receipt": str(top_path),
        "verification_receipt_sha256": sha256_file(top_path),
        "changes": changes,
    }


def _load_checksum_verification(
    config: DatasetConfig,
    plan: ConsolidationPlan,
    plan_hash: str,
) -> tuple[dict[str, object], Path]:
    path = (
        config.output_root
        / "_consolidation/verification/checksum/verification.json"
    )
    if not path.is_file() or path.is_symlink():
        raise WorkflowError("A matched checksum verification receipt is required")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("The checksum verification receipt is unreadable") from exc
    if (
        payload.get("status") != "matched"
        or payload.get("mode") != "checksum"
        or payload.get("plan_sha256") != plan_hash
        or payload.get("planned_items") != len(plan.items)
    ):
        raise WorkflowError("A matched checksum verification receipt is required")
    records = payload.get("items")
    if not isinstance(records, list) or len(records) != len(plan.items):
        raise WorkflowError("Checksum verification item scope is incomplete")
    by_key = {item.key: item for item in plan.items}
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("key") not in by_key:
            raise WorkflowError("Unknown verification item receipt")
        key = str(record["key"])
        if key in seen or record.get("status") != "matched":
            raise WorkflowError("Invalid verification item receipt")
        seen.add(key)
        item = by_key[key]
        expected_receipt = (
            config.output_root
            / "_consolidation/verification/checksum/items"
            / _receipt_name(item)
        )
        expected_relative = str(expected_receipt.relative_to(config.output_root))
        if record.get("receipt") != expected_relative:
            raise WorkflowError("A verification item receipt path is invalid")
        receipt = expected_receipt
        if (
            not receipt.is_file()
            or receipt.is_symlink()
            or sha256_file(receipt) != record.get("receipt_sha256")
        ):
            raise WorkflowError("A verification item receipt was changed or removed")
        try:
            item_payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError("A verification item receipt is unreadable") from exc
        if item_payload.get("status") != "matched":
            raise WorkflowError("A verification item receipt is not matched")
        source_inventory = build_materialized_inventory(item.source)
        if source_inventory.inventory_sha256 != item_payload.get(
            "source_inventory_sha256"
        ):
            raise WorkflowError(f"Planned source inventory changed: {item.source}")
        destination = _item_destination(config, item)
        try:
            destination_inventory = build_file_inventory(destination)
        except ValueError as exc:
            raise WorkflowError(f"destination inventory changed: {destination}") from exc
        if destination_inventory.inventory_sha256 != item_payload.get(
            "destination_inventory_sha256"
        ):
            raise WorkflowError(f"destination inventory changed: {destination}")
    if seen != set(by_key):
        raise WorkflowError("Checksum verification item scope is incomplete")
    return payload, path


def _validate_copy_receipts(
    config: DatasetConfig,
    plan: ConsolidationPlan,
    plan_hash: str,
) -> None:
    receipt_root = config.output_root / "_consolidation/copy_items"
    if not receipt_root.is_dir() or receipt_root.is_symlink():
        raise WorkflowError("The copy item receipt directory is missing or unsafe")
    expected = {_receipt_name(item): item for item in plan.items}
    observed = {path.name for path in receipt_root.iterdir() if path.is_file()}
    if observed != set(expected):
        raise WorkflowError("The copy item receipt scope is incomplete or unexpected")
    for name, item in expected.items():
        path = receipt_root / name
        if path.is_symlink():
            raise WorkflowError("A copy item receipt is a symlink")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError("A copy item receipt is unreadable") from exc
        if (
            payload.get("status") != "copied"
            or payload.get("action") not in {"copied", "reused"}
            or payload.get("plan_sha256") != plan_hash
            or payload.get("key") != item.key
            or payload.get("source") != str(item.source)
            or payload.get("destination")
            != str(_item_destination(config, item))
        ):
            raise WorkflowError(f"Invalid copy item receipt: {path}")


def _is_control_file(relative: Path) -> bool:
    if relative.as_posix() in {"README.md", "subjects.txt", "dataset_manifest.json"}:
        return True
    parts = relative.parts
    if parts[:1] != ("_consolidation",):
        return False
    if parts[1:] in {
        ("owner.json",),
        ("planned_items.json",),
        ("complete.json",),
        ("verification", "file_inventory.json"),
    }:
        return True
    if len(parts) == 3 and parts[1] == "copy_items" and parts[2].endswith(".json"):
        return True
    if (
        len(parts) == 4
        and parts[1] == "verification"
        and parts[2] in {"size_mtime", "checksum"}
        and parts[3] == "verification.json"
    ):
        return True
    return (
        len(parts) == 5
        and parts[1] == "verification"
        and parts[2] in {"size_mtime", "checksum"}
        and parts[3] == "items"
        and parts[4].endswith(".json")
    )


def _validate_destination_coverage(
    config: DatasetConfig,
    plan: ConsolidationPlan,
) -> None:
    root = config.output_root
    destinations = tuple(
        (root / item.destination, item.source_type == "directory")
        for item in plan.items
    )
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root)
        if _is_control_file(relative):
            continue
        covered = any(
            path == destination
            or (is_directory and destination in path.parents)
            for destination, is_directory in destinations
        )
        if not covered:
            raise WorkflowError(f"unplanned destination path: {path}")


def _validate_destination_scope(
    config: DatasetConfig,
    plan: ConsolidationPlan,
) -> dict[str, int]:
    root = config.output_root
    if any(
        any(
            part == config.excluded_subject
            or part.startswith(f"{config.excluded_subject}_")
            for part in path.parts
        )
        for path in (root / "data").rglob("*")
    ):
        raise WorkflowError("Canonical data tree contains excluded subject")
    cells = 0
    simulations = 0
    for version in ("simnibs326", "simnibs4"):
        for target in ("left_hippocampus", "right_m1"):
            target_root = root / "data" / version / target
            observed_subjects = {
                path.name.removesuffix("_repeatability")
                for path in target_root.iterdir()
                if path.is_dir() and path.name.endswith("_repeatability")
            }
            if observed_subjects != set(plan.subjects):
                raise WorkflowError(
                    f"Destination subject scope differs under {target_root}"
                )
            for subject in plan.subjects:
                subject_root = target_root / f"{subject}_repeatability"
                observed_conditions = {
                    path.name for path in subject_root.iterdir() if path.is_dir()
                }
                if observed_conditions != {"remesh", "fixed_mesh"}:
                    raise WorkflowError(
                        f"Destination condition scope differs under {subject_root}"
                    )
                for condition in ("remesh", "fixed_mesh"):
                    repeat_root = subject_root / condition / "repeats"
                    tags = {
                        path.name for path in repeat_root.iterdir() if path.is_dir()
                    }
                    expected = {
                        f"repeat_{index:03d}"
                        for index in range(1, plan.repeat_count + 1)
                    }
                    if tags != expected:
                        raise WorkflowError(
                            f"Destination repeat scope differs under {repeat_root}"
                        )
                    for tag in expected:
                        meshes = [
                            path
                            for path in (repeat_root / tag).rglob("TI.msh")
                            if path.is_file() and not path.is_symlink()
                        ]
                        if len(meshes) != 1:
                            raise WorkflowError(
                                f"Destination {tag} has {len(meshes)} TI.msh files"
                            )
                    cells += 1
                    simulations += len(expected)
    if cells != plan.expected_cells or simulations != plan.expected_simulations:
        raise WorkflowError("Destination scientific counts differ from the plan")
    for version in ("simnibs326", "simnibs4"):
        head_root = root / "headmodels" / version
        observed = {path.name for path in head_root.iterdir() if path.is_dir()}
        if observed != set(plan.subjects):
            raise WorkflowError(f"Destination {version} head-model scope differs")
    return {
        "subjects": len(plan.subjects),
        "cells": cells,
        "simulations": simulations,
        "v3_headmodels": len(plan.subjects),
        "v4_scaffolds": len(plan.subjects),
    }


def complete_dataset(config: DatasetConfig) -> dict[str, object]:
    """Write an immutable completion receipt after every live checksum gate passes."""
    plan, plan_hash, _ = _owned_context(config)
    completion_path = config.output_root / "_consolidation/complete.json"
    inventory_path = (
        config.output_root / "_consolidation/verification/file_inventory.json"
    )
    exclusions = (completion_path, inventory_path)
    if completion_path.exists():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        current = build_file_inventory(config.output_root, exclude=exclusions)
        if (
            existing.get("plan_sha256") != plan_hash
            or existing.get("inventory", {}).get("sha256")
            != current.inventory_sha256
        ):
            raise WorkflowError("Refusing to replace a nonmatching existing completion receipt")

    _validate_copy_receipts(config, plan, plan_hash)
    _validate_destination_coverage(config, plan)
    verification, verification_path = _load_checksum_verification(
        config, plan, plan_hash
    )
    scope = _validate_destination_scope(config, plan)
    try:
        inventory = build_file_inventory(config.output_root, exclude=exclusions)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    _write_json(inventory_path, inventory.as_dict())
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "repeatability_dataset_consolidation",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "output_root": str(config.output_root.resolve()),
        "plan_sha256": plan_hash,
        "scope": scope,
        "verification": {
            "mode": "checksum",
            "path": str(verification_path.relative_to(config.output_root)),
            "sha256": sha256_file(verification_path),
            "matched_items": verification["matched_items"],
        },
        "inventory": {
            "path": str(inventory_path.relative_to(config.output_root)),
            "sha256": inventory.inventory_sha256,
            "file_count": len(inventory.entries),
            "total_bytes": inventory.total_bytes,
        },
        "source_outputs_modified": False,
    }
    if completion_path.exists():
        return receipt | {"action": "reused"}
    _write_json(completion_path, receipt)
    return receipt | {"action": "created"}
