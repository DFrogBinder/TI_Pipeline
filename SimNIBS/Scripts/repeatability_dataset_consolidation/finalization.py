"""Resumable, bounded-memory finalization of a copied dataset."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Optional

from .disposition import (
    DispositionError,
    disposition_path,
    retirement_root_metadata,
    validate_retirement_disposition_stream,
    write_retirement_disposition_stream,
)
from .inventory import InventoryEntry, sha256_file
from .planning import ConsolidationPlan, CopyItem
from .settings import DatasetConfig
from .streaming_inventory import (
    InventoryStreamSummary,
    merge_inventory_streams,
    metadata_fingerprint,
    validate_inventory_stream,
    write_inventory_stream,
)
from .workflow import (
    WorkflowError,
    _canonical_config_records,
    _git_commit,
    _is_control_file,
    _item_destination,
    _owned_context,
    _receipt_name,
    _utc_now,
    _validate_copy_receipts,
    _validate_destination_coverage,
    _validate_destination_scope,
    _write_json,
    _write_source_mapping,
)


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"{label} is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError(f"{label} is not a JSON object: {path}")
    return payload


def load_checksum_bindings(
    config: DatasetConfig,
    plan: ConsolidationPlan,
    plan_hash: str,
) -> tuple[dict[str, object], Path, dict[str, dict[str, object]]]:
    """Validate immutable checksum receipts without reading scientific bodies."""
    top_path = (
        config.output_root
        / "_consolidation/verification/checksum/verification.json"
    )
    top = _load_json(top_path, "checksum verification receipt")
    if (
        top.get("status") != "matched"
        or top.get("mode") != "checksum"
        or top.get("plan_sha256") != plan_hash
        or top.get("planned_items") != len(plan.items)
        or top.get("matched_items") != len(plan.items)
        or top.get("mismatched_items") != 0
        or top.get("source_outputs_modified") is not False
    ):
        raise WorkflowError("A matched checksum verification receipt is required")
    records = top.get("items")
    if not isinstance(records, list) or len(records) != len(plan.items):
        raise WorkflowError("Checksum verification item scope is incomplete")

    items_by_key = {item.key: item for item in plan.items}
    bindings: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise WorkflowError("Invalid verification item receipt binding")
        key = record.get("key")
        if not isinstance(key, str) or key not in items_by_key or key in bindings:
            raise WorkflowError("Unknown or duplicate verification item receipt")
        item = items_by_key[key]
        receipt_path = (
            config.output_root
            / "_consolidation/verification/checksum/items"
            / _receipt_name(item)
        )
        expected_relative = str(receipt_path.relative_to(config.output_root))
        recorded_sha = record.get("receipt_sha256")
        if (
            record.get("status") != "matched"
            or record.get("receipt") != expected_relative
            or not isinstance(recorded_sha, str)
            or len(recorded_sha) != 64
            or not receipt_path.is_file()
            or receipt_path.is_symlink()
            or sha256_file(receipt_path) != recorded_sha
        ):
            raise WorkflowError("A verification item receipt was changed or removed")
        payload = _load_json(receipt_path, "Verification item receipt")
        destination = _item_destination(config, item)
        source_inventory = payload.get("source_inventory_sha256")
        destination_inventory = payload.get("destination_inventory_sha256")
        if (
            payload.get("status") != "matched"
            or payload.get("mode") != "checksum"
            or payload.get("plan_sha256") != plan_hash
            or payload.get("key") != item.key
            or payload.get("source") != str(item.source)
            or payload.get("destination") != str(destination)
            or payload.get("source_outputs_modified") is not False
            or not isinstance(source_inventory, str)
            or len(source_inventory) != 64
            or not isinstance(destination_inventory, str)
            or len(destination_inventory) != 64
            or source_inventory != destination_inventory
        ):
            raise WorkflowError("A verification item receipt is not matched")
        bindings[key] = {
            "item": item,
            "receipt_path": receipt_path,
            "receipt_relative": expected_relative,
            "receipt_sha256": recorded_sha,
            "destination_inventory_sha256": destination_inventory,
        }
    if set(bindings) != set(items_by_key):
        raise WorkflowError("Checksum verification item scope is incomplete")
    return top, top_path, bindings


def _checkpoint_paths(config: DatasetConfig, item: CopyItem) -> tuple[Path, Path]:
    name = _receipt_name(item)
    root = config.output_root / "_consolidation/finalization/items"
    return root / name, root / f"{Path(name).stem}.jsonl"


def _summary_from_payload(payload: dict[str, object]) -> InventoryStreamSummary:
    try:
        return InventoryStreamSummary(
            root=str(payload["root"]),
            entry_count=int(payload["entry_count"]),
            total_bytes=int(payload["total_bytes"]),
            inventory_sha256=str(payload["inventory_sha256"]),
            entries_sha256=str(payload["entries_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError("A finalization checkpoint summary is invalid") from exc


def _valid_checkpoint(
    config: DatasetConfig,
    item: CopyItem,
    binding: dict[str, object],
    plan_hash: str,
) -> Optional[tuple[Path, dict[str, object]]]:
    receipt_path, stream_path = _checkpoint_paths(config, item)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    try:
        payload = _load_json(receipt_path, "Finalization checkpoint")
        summary = _summary_from_payload(payload)
        expected_stream = str(stream_path.relative_to(config.output_root))
        if (
            payload.get("status") != "complete"
            or payload.get("plan_sha256") != plan_hash
            or payload.get("key") != item.key
            or payload.get("destination")
            != str(_item_destination(config, item))
            or payload.get("upstream_receipt") != binding["receipt_relative"]
            or payload.get("upstream_receipt_sha256")
            != binding["receipt_sha256"]
            or payload.get("stream") != expected_stream
            or summary.inventory_sha256
            != binding["destination_inventory_sha256"]
        ):
            return None
        validate_inventory_stream(stream_path, summary)
        if metadata_fingerprint(_item_destination(config, item)) != summary.metadata_sha256:
            return None
    except (OSError, ValueError, WorkflowError):
        return None
    return stream_path, payload


def _finalize_item(
    config: DatasetConfig,
    item: CopyItem,
    binding: dict[str, object],
    plan_hash: str,
) -> tuple[Path, dict[str, object], bool]:
    existing = _valid_checkpoint(config, item, binding, plan_hash)
    if existing is not None:
        return existing[0], existing[1], True

    receipt_path, stream_path = _checkpoint_paths(config, item)
    destination = _item_destination(config, item)
    summary = write_inventory_stream(destination, stream_path)
    if summary.inventory_sha256 != binding["destination_inventory_sha256"]:
        raise WorkflowError(f"destination inventory changed: {destination}")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "plan_sha256": plan_hash,
        "key": item.key,
        "destination": str(destination),
        "upstream_receipt": binding["receipt_relative"],
        "upstream_receipt_sha256": binding["receipt_sha256"],
        "stream": str(stream_path.relative_to(config.output_root)),
        **asdict(summary),
    }
    _write_json(receipt_path, payload)
    return stream_path, payload, False


def _load_or_create_disposition(
    config: DatasetConfig,
    plan: ConsolidationPlan,
) -> tuple[dict[str, object], Path]:
    path = disposition_path(config)
    if path.is_file() and not path.is_symlink():
        try:
            return validate_retirement_disposition_stream(config, path), path
        except DispositionError as exc:
            raise WorkflowError(
                "The existing retirement disposition is invalid"
            ) from exc
    payload = write_retirement_disposition_stream(config, plan, path)
    return payload, path


def _control_entries(
    config: DatasetConfig,
    plan: ConsolidationPlan,
    *,
    completed_manifest: dict[str, object],
) -> list[InventoryEntry]:
    root = config.output_root
    manifest_path = root / "dataset_manifest.json"
    entries: list[InventoryEntry] = []
    directory_destinations = frozenset(
        item.destination for item in plan.items if item.source_type == "directory"
    )
    file_destinations = frozenset(
        item.destination for item in plan.items if item.source_type != "directory"
    )
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root)
        if relative in file_destinations or any(
            parent in directory_destinations for parent in relative.parents
        ):
            continue
        if relative.parts[:2] == ("_consolidation", "finalization"):
            continue
        if relative.as_posix() in {
            "_consolidation/complete.json",
            "_consolidation/verification/file_inventory.json",
        }:
            continue
        if not _is_control_file(relative):
            raise WorkflowError(f"unplanned destination path: {path}")
        if path == manifest_path:
            encoded = _json_bytes(completed_manifest)
            entries.append(
                InventoryEntry(
                    path=relative.as_posix(),
                    size_bytes=len(encoded),
                    sha256=hashlib.sha256(encoded).hexdigest(),
                )
            )
        else:
            if not path.is_file() or path.is_symlink():
                raise WorkflowError(f"Unsafe control file: {path}")
            entries.append(
                InventoryEntry(
                    path=relative.as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
    return entries


def _finalize_dataset_impl(config: DatasetConfig) -> dict[str, object]:
    """Resume item checkpoints, build the final inventory, and complete safely."""
    plan, plan_hash, _ = _owned_context(config)
    completion_path = config.output_root / "_consolidation/complete.json"
    inventory_path = (
        config.output_root / "_consolidation/verification/file_inventory.json"
    )
    if completion_path.exists():
        from .workflow import validate_dataset_completion

        try:
            existing = validate_dataset_completion(config)
        except WorkflowError as exc:
            raise WorkflowError(
                "Refusing to replace a nonmatching existing completion receipt: "
                f"{exc}"
            ) from exc
        if existing.get("plan_sha256") != plan_hash:
            raise WorkflowError(
                "Refusing to replace a nonmatching existing completion receipt"
            )
        return {**existing, "action": "reused"}

    _validate_copy_receipts(config, plan, plan_hash)
    verification, verification_path, bindings = load_checksum_bindings(
        config, plan, plan_hash
    )
    manifest_path = config.output_root / "dataset_manifest.json"
    manifest = _load_json(manifest_path, "Dataset manifest")
    if (
        manifest.get("workflow") != "repeatability_dataset_consolidation"
        or manifest.get("output_root") != str(config.output_root.resolve())
        or manifest.get("plan_sha256") != plan_hash
    ):
        raise WorkflowError("The dataset manifest does not match the owned dataset")
    manifest["status"] = "finalizing"
    manifest.pop("completed_utc", None)
    _write_json(manifest_path, manifest)

    streams: list[tuple[Path, Path]] = []
    reused = 0
    for item in plan.items:
        stream_path, _, was_reused = _finalize_item(
            config, item, bindings[item.key], plan_hash
        )
        streams.append((item.destination, stream_path))
        reused += int(was_reused)

    _, canonical_manifest = _canonical_config_records(config, plan, write=False)
    source_mapping = _write_source_mapping(config, plan)
    _validate_destination_coverage(config, plan)
    scope = _validate_destination_scope(config, plan)
    retirement_disposition, retirement_disposition_path = (
        _load_or_create_disposition(config, plan)
    )
    retirement_metadata = retirement_root_metadata(config)
    completed_utc = _utc_now()
    completed_manifest = dict(manifest)
    completed_manifest["status"] = "complete"
    completed_manifest["completed_utc"] = completed_utc
    controls = _control_entries(
        config,
        plan,
        completed_manifest=completed_manifest,
    )
    inventory = merge_inventory_streams(
        streams,
        controls,
        inventory_path,
        root=config.output_root,
    )
    _write_json(manifest_path, completed_manifest)
    if sha256_file(manifest_path) != next(
        entry.sha256 for entry in controls if entry.path == "dataset_manifest.json"
    ):
        raise WorkflowError("The completed dataset manifest differs from its inventory")

    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "repeatability_dataset_consolidation",
        "created_utc": completed_utc,
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
            "file_sha256": sha256_file(inventory_path),
            "file_count": inventory.entry_count,
            "total_bytes": inventory.total_bytes,
        },
        "dataset_manifest": {
            "path": "dataset_manifest.json",
            "sha256": sha256_file(manifest_path),
        },
        "source_mapping": {
            "path": str(source_mapping.relative_to(config.output_root)),
            "sha256": sha256_file(source_mapping),
            "rows": len(plan.items),
        },
        "retirement_disposition": {
            "path": str(retirement_disposition_path.relative_to(config.output_root)),
            "sha256": sha256_file(retirement_disposition_path),
            "entries_sha256": retirement_disposition["entries_sha256"],
            "summary": retirement_disposition["summary"],
            "root_metadata": retirement_metadata,
        },
        "canonical_configs": {
            "path": str(canonical_manifest.relative_to(config.output_root)),
            "sha256": sha256_file(canonical_manifest),
        },
        "finalization": {
            "completed_items": len(plan.items),
            "reused_items": reused,
        },
        "source_outputs_modified": False,
    }
    _write_json(completion_path, receipt)
    return {**receipt, "action": "created"}


def _failure_path(config: DatasetConfig) -> Path:
    return config.output_root / "_consolidation/finalization/failure.json"


def finalize_dataset(config: DatasetConfig) -> dict[str, object]:
    """Run resumable finalization and retain the latest failure for status."""
    try:
        result = _finalize_dataset_impl(config)
    except Exception as exc:
        _write_json(
            _failure_path(config),
            {
                "schema_version": 1,
                "status": "failed",
                "created_utc": _utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
    failure = _failure_path(config)
    if failure.exists() and not failure.is_symlink():
        failure.unlink()
    return result


def finalization_status(config: DatasetConfig) -> dict[str, object]:
    """Return checkpoint progress without mutating the dataset."""
    plan, plan_hash, _ = _owned_context(config)
    root = config.output_root / "_consolidation/finalization/items"
    completed = 0
    for item in plan.items:
        receipt, _ = _checkpoint_paths(config, item)
        if receipt.is_file() and not receipt.is_symlink():
            try:
                payload = _load_json(receipt, "Finalization checkpoint")
            except WorkflowError:
                continue
            if (
                payload.get("status") == "complete"
                and payload.get("plan_sha256") == plan_hash
                and payload.get("key") == item.key
            ):
                completed += 1
    manifest_path = config.output_root / "dataset_manifest.json"
    manifest_state: Optional[str] = None
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest_state = str(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
            )
        except (OSError, json.JSONDecodeError):
            manifest_state = "unreadable"
    last_failure: Optional[dict[str, object]] = None
    failure = _failure_path(config)
    if failure.is_file() and not failure.is_symlink():
        try:
            loaded_failure = json.loads(failure.read_text(encoding="utf-8"))
            if isinstance(loaded_failure, dict):
                last_failure = loaded_failure
        except (OSError, json.JSONDecodeError):
            last_failure = {
                "status": "unreadable",
                "message": "The finalization failure record is unreadable",
            }
    return {
        "status": "complete" if completed == len(plan.items) else "in_progress",
        "plan_sha256": plan_hash,
        "completed_items": completed,
        "expected_items": len(plan.items),
        "manifest_status": manifest_state,
        "inventory_present": (
            config.output_root
            / "_consolidation/verification/file_inventory.json"
        ).is_file(),
        "completion_receipt_present": (
            config.output_root / "_consolidation/complete.json"
        ).is_file(),
        "checkpoint_root": str(root),
        "last_failure": last_failure,
    }
