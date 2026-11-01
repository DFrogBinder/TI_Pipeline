"""Login-node-only archive lifecycle for obsolete v3 reference roots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .inventory import (
    build_file_inventory,
    build_materialized_inventory,
    estimate_materialized_bytes,
)
from .settings import DatasetConfig


class ArchiveError(RuntimeError):
    """Raised when the immutable Shared archive contract is not satisfied."""


@dataclass(frozen=True)
class ArchiveItem:
    key: str
    source: Path
    destination: Path
    source_device: int
    source_inode: int
    source_size: int
    source_mtime_ns: int
    source_ctime_ns: int

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source": str(self.source),
            "destination": str(self.destination),
            "source_device": self.source_device,
            "source_inode": self.source_inode,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "source_ctime_ns": self.source_ctime_ns,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _plan(config: DatasetConfig) -> tuple[tuple[ArchiveItem, ...], str]:
    root = config.shared_archive_root
    if root.is_symlink():
        raise ArchiveError(f"Refusing symbolic-link archive root: {root}")
    items: list[ArchiveItem] = []
    destinations: set[Path] = set()
    for key, source in config.archive_sources.items():
        if not source.is_dir() or source.is_symlink():
            raise ArchiveError(f"Required missing archive source: {source}")
        destination = root / source.name
        if destination in destinations:
            raise ArchiveError(f"Duplicate archive destination: {destination}")
        destinations.add(destination)
        if destination.is_symlink():
            raise ArchiveError(
                f"Refusing symbolic-link archive destination: {destination}"
            )
        resolved_root = root.resolve(strict=False)
        resolved_source = source.resolve()
        if (
            resolved_root == resolved_source
            or resolved_root in resolved_source.parents
            or resolved_source in resolved_root.parents
        ):
            raise ArchiveError("Shared archive root overlaps an archive source")
        info = source.lstat()
        items.append(
            ArchiveItem(
                key=key,
                source=source.resolve(),
                destination=destination.absolute(),
                source_device=int(info.st_dev),
                source_inode=int(info.st_ino),
                source_size=int(info.st_size),
                source_mtime_ns=int(info.st_mtime_ns),
                source_ctime_ns=int(info.st_ctime_ns),
            )
        )
    ordered = tuple(sorted(items, key=lambda item: item.key))
    approved_roots = tuple(item.source for item in ordered)
    for item in ordered:
        for path in item.source.rglob("*"):
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise ArchiveError(f"Broken archive-source symlink: {path}") from exc
            if not any(
                target == root or root in target.parents for root in approved_roots
            ):
                raise ArchiveError(
                    f"Symlink target is outside archive sources: {path} -> {target}"
                )
    encoded = json.dumps(
        [item.as_dict() for item in ordered],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return ordered, hashlib.sha256(encoded).hexdigest()


def _validate_identity(item: ArchiveItem) -> None:
    if not item.source.is_dir() or item.source.is_symlink():
        raise ArchiveError(f"Required missing archive source: {item.source}")
    info = item.source.lstat()
    observed = (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )
    expected = (
        item.source_device,
        item.source_inode,
        item.source_size,
        item.source_mtime_ns,
        item.source_ctime_ns,
    )
    if observed != expected:
        raise ArchiveError(f"Archive source identity changed: {item.source}")


def _archive_capacity(
    config: DatasetConfig,
    items: tuple[ArchiveItem, ...],
) -> dict[str, object]:
    try:
        required = sum(estimate_materialized_bytes(item.source) for item in items)
    except (OSError, ValueError) as exc:
        raise ArchiveError(f"Cannot estimate materialized archive bytes: {exc}") from exc
    usage = shutil.disk_usage(config.shared_archive_root.parent)
    enforced = (
        not config.shared_archive_root.exists()
        and not config.shared_archive_root.is_symlink()
    )
    sufficient = required <= usage.free
    if enforced and not sufficient:
        raise ArchiveError(
            "Insufficient capacity for initial Shared archive creation: "
            f"required {required} bytes, free {usage.free} bytes"
        )
    return {
        "required_bytes": required,
        "free_bytes": usage.free,
        "capacity_sufficient": sufficient,
        "capacity_gate_enforced": enforced,
    }


def archive_preflight(config: DatasetConfig) -> dict[str, object]:
    """Audit exact archive scope without creating either Shared destination."""
    items, plan_hash = _plan(config)
    capacity = _archive_capacity(config, items)
    return {
        "status": "ready",
        "workflow": "repeatability_dataset_consolidation_archive",
        "archive_root": str(config.shared_archive_root),
        "receipt_root": str(config.shared_receipt_root),
        "plan_sha256": plan_hash,
        "source_count": len(items),
        **capacity,
        "mappings": [item.as_dict() for item in items],
        "source_outputs_modified": False,
    }


def _ensure_owner(
    config: DatasetConfig,
    items: tuple[ArchiveItem, ...],
    plan_hash: str,
) -> str:
    root = config.shared_archive_root
    if root.is_symlink():
        raise ArchiveError(f"Refusing symbolic-link archive root: {root}")
    owner_path = root / "_archive/owner.json"
    plan_path = root / "_archive/plan.json"
    if root.exists():
        if not root.is_dir() or not owner_path.is_file() or not plan_path.is_file():
            raise ArchiveError(f"Refusing unowned archive root: {root}")
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArchiveError(f"Unreadable archive owner under {root}") from exc
        if (
            owner.get("workflow")
            != "repeatability_dataset_consolidation_archive"
            or owner.get("archive_root") != str(root.resolve())
            or owner.get("plan_sha256") != plan_hash
            or saved_plan.get("plan_sha256") != plan_hash
            or saved_plan.get("items") != [item.as_dict() for item in items]
        ):
            raise ArchiveError(f"Existing archive root has different ownership: {root}")
        return "reused"

    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ArchiveError(f"Archive parent is missing or unsafe: {root.parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent)
    )
    promoted = False
    try:
        (temporary / "_archive/copy_items").mkdir(parents=True)
        (temporary / "_archive/verification").mkdir()
        created = _utc_now()
        _write_json(
            temporary / "_archive/owner.json",
            {
                "schema_version": 1,
                "status": "owned",
                "workflow": "repeatability_dataset_consolidation_archive",
                "created_utc": created,
                "archive_root": str(root.resolve(strict=False)),
                "plan_sha256": plan_hash,
            },
        )
        _write_json(
            temporary / "_archive/plan.json",
            {
                "schema_version": 1,
                "status": "planned",
                "created_utc": created,
                "plan_sha256": plan_hash,
                "items": [item.as_dict() for item in items],
            },
        )
        os.replace(temporary, root)
        promoted = True
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)
    return "created"


def _run(command: list[str]) -> tuple[str, ...]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArchiveError(
            f"rsync exited {completed.returncode}: {detail or 'no diagnostic'}"
        )
    return tuple(line for line in completed.stdout.splitlines() if line.strip())


def _rsync_command(
    item: ArchiveItem,
    *,
    dry_run: bool,
    checksum: bool,
) -> list[str]:
    command = [
        "rsync",
        "-rltH",
        "--no-perms",
        "--no-owner",
        "--no-group",
        "--copy-links",
        "--protect-args",
        "--itemize-changes",
        "--out-format=%i|%n%L",
    ]
    if dry_run:
        command.extend(("--dry-run", "--delete"))
    else:
        command.append("--partial")
    if checksum:
        command.append("--checksum")
    command.extend((f"{item.source}/", f"{item.destination}/"))
    return command


def _compare(item: ArchiveItem, *, checksum: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    command = _rsync_command(item, dry_run=True, checksum=checksum)
    if not item.destination.exists():
        return tuple(command), (f"missing|{item.destination}",)
    return tuple(command), _run(command)


def _copy(item: ArchiveItem) -> tuple[tuple[str, ...], tuple[str, ...]]:
    item.destination.mkdir(parents=True, exist_ok=True)
    command = _rsync_command(item, dry_run=False, checksum=True)
    return tuple(command), _run(command)


def archive_copy(config: DatasetConfig) -> dict[str, object]:
    """Resume exact whole-tree archive copies without deleting any source."""
    items, plan_hash = _plan(config)
    capacity = _archive_capacity(config, items)
    _ensure_owner(config, items, plan_hash)
    copied = 0
    reused = 0
    for item in items:
        _validate_identity(item)
        compare_command, changes = _compare(item, checksum=True)
        if changes:
            command, _ = _copy(item)
            action = "copied"
            copied += 1
        else:
            command = compare_command
            action = "reused"
            reused += 1
        _validate_identity(item)
        _write_json(
            config.shared_archive_root / "_archive/copy_items" / f"{item.key}.json",
            {
                "schema_version": 1,
                "status": "copied",
                "action": action,
                "created_utc": _utc_now(),
                "plan_sha256": plan_hash,
                "key": item.key,
                "source": str(item.source),
                "destination": str(item.destination),
                "command": list(command),
                "source_outputs_modified": False,
            },
        )
    return {
        "status": "copied",
        "plan_sha256": plan_hash,
        "planned_roots": len(items),
        "copied_roots": copied,
        "reused_roots": reused,
        **capacity,
        "source_outputs_modified": False,
    }


def _completion_path(config: DatasetConfig) -> Path:
    return config.shared_receipt_root / "legacy_v3_archive_v1.complete.json"


def _write_completion(
    config: DatasetConfig,
    plan_hash: str,
    mappings: list[dict[str, object]],
) -> str:
    path = _completion_path(config)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "repeatability_dataset_consolidation_archive",
        "created_utc": _utc_now(),
        "archive_root": str(config.shared_archive_root.resolve()),
        "plan_sha256": plan_hash,
        "mappings": mappings,
        "source_outputs_modified": False,
    }
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ArchiveError("Refusing unsafe existing archive completion receipt")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArchiveError("Unreadable existing archive completion receipt") from exc
        comparable = dict(payload)
        comparable["created_utc"] = existing.get("created_utc")
        if existing != comparable:
            raise ArchiveError("Refusing nonmatching existing archive completion receipt")
        return "reused"
    _write_json(path, payload)
    return "created"


def archive_verify(
    config: DatasetConfig,
    *,
    checksum: bool,
) -> dict[str, object]:
    """Compare all archived roots and complete only after full checksums match."""
    items, plan_hash = _plan(config)
    _ensure_owner(config, items, plan_hash)
    mode = "checksum" if checksum else "size_mtime"
    records: list[dict[str, object]] = []
    changes: list[str] = []
    for item in items:
        _validate_identity(item)
        command, differences = _compare(item, checksum=checksum)
        record: dict[str, object] = {
            "key": item.key,
            "source": str(item.source),
            "destination": str(item.destination),
            "status": "matched" if not differences else "mismatch",
            "command": list(command),
            "changes": list(differences),
        }
        if checksum:
            try:
                source_inventory = build_materialized_inventory(item.source).as_dict()
                destination_inventory = build_file_inventory(item.destination).as_dict()
            except ValueError as exc:
                raise ArchiveError(str(exc)) from exc
            record["source_inventory_sha256"] = source_inventory["inventory_sha256"]
            record["destination_inventory_sha256"] = destination_inventory[
                "inventory_sha256"
            ]
            if source_inventory["inventory_sha256"] != destination_inventory[
                "inventory_sha256"
            ]:
                record["status"] = "mismatch"
        _validate_identity(item)
        records.append(record)
        changes.extend(f"{item.key}: {change}" for change in differences)
    mismatches = sum(record["status"] != "matched" for record in records)
    verification = {
        "schema_version": 1,
        "status": "matched" if mismatches == 0 else "mismatch",
        "created_utc": _utc_now(),
        "mode": mode,
        "plan_sha256": plan_hash,
        "matched_roots": len(items) - mismatches,
        "mismatched_roots": mismatches,
        "mappings": records,
        "source_outputs_modified": False,
    }
    verification_path = (
        config.shared_archive_root
        / "_archive/verification"
        / f"{mode}.json"
    )
    _write_json(verification_path, verification)
    if checksum and mismatches == 0:
        action = _write_completion(config, plan_hash, records)
        return {
            **verification,
            "status": "complete",
            "action": action,
            "completion_receipt": str(_completion_path(config)),
            "completion_receipt_sha256": _sha256_file(_completion_path(config)),
            "changes": changes,
        }
    return {**verification, "changes": changes}


def validate_archive_completion(config: DatasetConfig) -> dict[str, object]:
    """Read-only live validation of the immutable Shared archive receipt."""
    items, plan_hash = _plan(config)
    if not config.shared_archive_root.is_dir():
        raise ArchiveError("The owned Shared archive root is missing")
    _ensure_owner(config, items, plan_hash)
    path = _completion_path(config)
    if not path.is_file() or path.is_symlink():
        raise ArchiveError("The archive completion receipt is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError("The archive completion receipt is unreadable") from exc
    if (
        payload.get("status") != "complete"
        or payload.get("workflow")
        != "repeatability_dataset_consolidation_archive"
        or payload.get("archive_root") != str(config.shared_archive_root.resolve())
        or payload.get("plan_sha256") != plan_hash
    ):
        raise ArchiveError("The archive completion receipt does not match current scope")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != len(items):
        raise ArchiveError("The archive completion receipt has incomplete mappings")
    by_key = {
        str(mapping.get("key")): mapping
        for mapping in mappings
        if isinstance(mapping, dict)
    }
    if set(by_key) != {item.key for item in items}:
        raise ArchiveError("The archive completion receipt has unexpected mappings")
    for item in items:
        _validate_identity(item)
        mapping = by_key[item.key]
        command, changes = _compare(item, checksum=True)
        try:
            source_inventory = build_materialized_inventory(item.source).as_dict()
            destination_inventory = build_file_inventory(item.destination).as_dict()
        except ValueError as exc:
            raise ArchiveError(str(exc)) from exc
        if (
            changes
            or mapping.get("status") != "matched"
            or mapping.get("source") != str(item.source)
            or mapping.get("destination") != str(item.destination)
            or mapping.get("source_inventory_sha256")
            != source_inventory["inventory_sha256"]
            or mapping.get("destination_inventory_sha256")
            != destination_inventory["inventory_sha256"]
            or source_inventory["inventory_sha256"]
            != destination_inventory["inventory_sha256"]
        ):
            raise ArchiveError(
                f"Archived root no longer matches its completion receipt: {item.key}"
            )
        _validate_identity(item)
    return {
        "status": "complete",
        "path": str(path),
        "sha256": _sha256_file(path),
        "plan_sha256": plan_hash,
        "source_count": len(items),
    }
