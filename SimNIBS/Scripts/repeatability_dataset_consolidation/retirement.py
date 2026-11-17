"""Receipt-gated planning, submission, and proof for source retirement."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence

from .archive import ArchiveError, validate_archive_completion
from .disposition import (
    DispositionError,
    disposition_path,
    retirement_root_metadata,
)
from .inventory import sha256_file
from .planning import ConsolidationPlan, build_copy_plan, plan_sha256
from .settings import (
    DatasetConfig,
    covered_retirement_allowlist,
    retirement_allowlist,
)
from .streaming_inventory import metadata_fingerprint
from .validation import validate_scientific_scope
from .workflow import (
    WorkflowError,
    _validate_owned_root,
    validate_dataset_completion,
)


class RetirementError(RuntimeError):
    """Raised when any irreversible-retirement gate is unsatisfied."""


@dataclass(frozen=True)
class RetirementTarget:
    key: str
    path: Path
    requires_archive_receipt: bool
    source_device: int
    source_inode: int

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "path": str(self.path),
            "requires_archive_receipt": self.requires_archive_receipt,
            "source_device": self.source_device,
            "source_inode": self.source_inode,
        }


@dataclass(frozen=True)
class RetirementPlan:
    config: DatasetConfig
    targets: tuple[RetirementTarget, ...]
    dry_run: bool
    plan_sha256: str
    dataset_receipt: Path
    dataset_receipt_sha256: str
    archive_receipt: Path | None
    archive_receipt_sha256: str | None
    retirement_disposition: Path
    retirement_disposition_sha256: str
    retirement_disposition_summary: dict[str, object]
    scheduler_snapshot_sha256: str
    deletion_launcher: Path
    scope: str = "full"
    retirement_root_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class RetirementConfirmations:
    dataset_root: str
    dataset_receipt_sha256: str
    archive_receipt_sha256: str | None
    retirement_disposition_sha256: str
    confirm_retirement: bool


_FULL_SCOPE = "full"
_COVERED_SCOPE = "covered"


def _scope_allowlist(config: DatasetConfig, scope: str) -> dict[str, Path]:
    if scope == _FULL_SCOPE:
        return retirement_allowlist(config)
    if scope == _COVERED_SCOPE:
        return covered_retirement_allowlist(config)
    raise RetirementError(f"Unknown retirement scope: {scope}")


def _scope_workflow(scope: str) -> str:
    if scope == _FULL_SCOPE:
        return "repeatability_dataset_consolidation_retirement"
    if scope == _COVERED_SCOPE:
        return "repeatability_dataset_consolidation_covered_retirement"
    raise RetirementError(f"Unknown retirement scope: {scope}")


def _request_path(config: DatasetConfig, scope: str) -> Path:
    filename = (
        "retirement_request.json"
        if scope == _FULL_SCOPE
        else "covered_retirement_request.json"
    )
    _scope_workflow(scope)
    return config.deletion_log_root / filename


def _completion_path(config: DatasetConfig, scope: str) -> Path:
    filename = (
        "retirement_complete.json"
        if scope == _FULL_SCOPE
        else "covered_retirement_complete.json"
    )
    _scope_workflow(scope)
    return config.deletion_log_root / filename


def _other_scope(scope: str) -> str:
    if scope == _FULL_SCOPE:
        return _COVERED_SCOPE
    if scope == _COVERED_SCOPE:
        return _FULL_SCOPE
    raise RetirementError(f"Unknown retirement scope: {scope}")


@contextmanager
def _submission_lock(config: DatasetConfig):
    config.deletion_log_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.deletion_log_root / "retirement_submission.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RetirementError("Another retirement submission is running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reject_conflicting_request(config: DatasetConfig, scope: str) -> None:
    conflicting = _request_path(config, _other_scope(scope))
    if conflicting.exists() or conflicting.is_symlink():
        raise RetirementError(
            f"A conflicting retirement request already exists: {conflicting}"
        )


def _is_legacy_full_request(request: dict[str, object], scope: str) -> bool:
    return bool(
        scope == _FULL_SCOPE
        and request.get("schema_version") == 1
        and "scope" not in request
        and request.get("workflow") == _scope_workflow(_FULL_SCOPE)
    )


def _validate_request_header(request: dict[str, object], scope: str) -> bool:
    legacy = _is_legacy_full_request(request, scope)
    if legacy:
        return True
    if (
        request.get("schema_version") != 2
        or request.get("workflow") != _scope_workflow(scope)
        or request.get("scope") != scope
    ):
        raise RetirementError("The retirement request receipt binding is invalid")
    return False


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


def _scheduler_snapshot() -> str:
    command = ["squeue", "--noheader", "--format=%i|%j|%Z|%o"]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
    except FileNotFoundError as exc:
        raise RetirementError("Cannot inspect active Slurm jobs: squeue is unavailable") from exc
    if completed.returncode != 0:
        raise RetirementError(
            f"Cannot inspect active Slurm jobs: {completed.stderr.strip()}"
        )
    return completed.stdout


def _run_launcher(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _load_dataset_completion(
    config: DatasetConfig,
) -> tuple[
    ConsolidationPlan,
    Path,
    str,
    Path,
    str,
    dict[str, object],
    dict[str, object],
]:
    try:
        plan = build_copy_plan(config)
        validation = validate_scientific_scope(plan)
        current_plan_hash = plan_sha256(plan)
        _validate_owned_root(
            config, plan.as_dict(), current_plan_hash, validation.as_dict()
        )
    except Exception as exc:
        raise RetirementError(f"Dataset completion receipt preflight failed: {exc}") from exc
    try:
        payload = validate_dataset_completion(config)
    except WorkflowError as exc:
        raise RetirementError(
            f"dataset completion receipt live validation failed: {exc}"
        ) from exc
    completion = config.output_root / "_consolidation/complete.json"
    disposition_record = payload.get("retirement_disposition")
    if not isinstance(disposition_record, dict):
        raise RetirementError("The dataset retirement disposition binding is missing")
    saved_disposition_path = disposition_path(config)
    if (
        disposition_record.get("path")
        != str(saved_disposition_path.relative_to(config.output_root))
        or not saved_disposition_path.is_file()
        or saved_disposition_path.is_symlink()
        or sha256_file(saved_disposition_path) != disposition_record.get("sha256")
    ):
        raise RetirementError("The dataset retirement disposition binding is invalid")
    saved_metadata = disposition_record.get("root_metadata")
    if not isinstance(saved_metadata, dict):
        raise RetirementError("The retirement source metadata binding is missing")
    try:
        live_metadata = retirement_root_metadata(config)
    except DispositionError as exc:
        raise RetirementError("Retirement source metadata is unreadable or unsafe") from exc
    if saved_metadata != live_metadata:
        raise RetirementError("Retirement source metadata changed after dataset completion")
    summary = disposition_record.get("summary")
    if not isinstance(summary, dict):
        raise RetirementError("The retirement disposition summary binding is invalid")
    return (
        plan,
        completion,
        sha256_file(completion),
        saved_disposition_path,
        sha256_file(saved_disposition_path),
        summary,
        saved_metadata,
    )


def _retirement_hash(
    scope: str,
    targets: Sequence[RetirementTarget],
    dataset_receipt_sha256: str,
    archive_receipt_sha256: str | None,
    retirement_disposition_sha256: str,
) -> str:
    encoded = json.dumps(
        {
            "scope": scope,
            "targets": [target.as_dict() for target in targets],
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "archive_receipt_sha256": archive_receipt_sha256,
            "retirement_disposition_sha256": retirement_disposition_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_full_retirement_hash(
    targets: Sequence[RetirementTarget],
    dataset_receipt_sha256: str,
    archive_receipt_sha256: str | None,
    retirement_disposition_sha256: str,
) -> str:
    encoded = json.dumps(
        {
            "targets": [target.as_dict() for target in targets],
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "archive_receipt_sha256": archive_receipt_sha256,
            "retirement_disposition_sha256": retirement_disposition_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_DISPOSITION_COUNTERS = (
    "dataset_copy_files",
    "dataset_copy_bytes",
    "shared_archive_files",
    "shared_archive_bytes",
    "intentional_retirement_files",
    "intentional_retirement_bytes",
)


def _validate_disposition_summary(
    summary: dict[str, object],
    keys: Sequence[str],
    *,
    covered: bool,
) -> None:
    for key in keys:
        record = summary.get(key)
        if not isinstance(record, dict):
            raise RetirementError(
                f"Retirement disposition summary is missing for {key}"
            )
        for counter in _DISPOSITION_COUNTERS:
            value = record.get(counter)
            if type(value) is not int or value < 0:
                raise RetirementError(
                    f"Retirement disposition counter is invalid for {key}: {counter}"
                )
        if covered and (
            record["shared_archive_files"] != 0
            or record["shared_archive_bytes"] != 0
        ):
            raise RetirementError(
                f"Covered retirement target has a Shared-archive disposition: {key}"
            )


def _authoritative_request_disposition(
    config: DatasetConfig,
    request: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        completion = validate_dataset_completion(config)
    except WorkflowError as exc:
        raise RetirementError(
            f"dataset completion receipt live validation failed: {exc}"
        ) from exc
    record = completion.get("retirement_disposition")
    if not isinstance(record, dict):
        raise RetirementError("The authoritative retirement disposition is missing")
    summary = record.get("summary")
    metadata = record.get("root_metadata")
    if not isinstance(summary, dict) or not isinstance(metadata, dict):
        raise RetirementError("The authoritative retirement disposition is invalid")
    if request.get("retirement_disposition_summary") != summary:
        raise RetirementError(
            "The retirement request disposition summary differs from completion"
        )
    return summary, metadata


def _validate_pending_target_metadata(
    target: RetirementTarget,
    root_metadata: dict[str, object],
) -> None:
    expected = root_metadata.get(target.key)
    if not isinstance(expected, dict):
        raise RetirementError(
            f"Retirement source metadata binding is missing for {target.key}"
        )
    path = target.path
    if not path.is_dir() or path.is_symlink():
        raise RetirementError(f"Retirement source metadata changed for {target.key}")
    before = path.lstat()
    try:
        fingerprint = metadata_fingerprint(path, materialize_links=True)
    except (OSError, ValueError) as exc:
        raise RetirementError(
            f"Retirement source metadata changed for {target.key}"
        ) from exc
    after = path.lstat()
    actual = {
        "path": str(path.resolve()),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "metadata_sha256": fingerprint,
    }
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_identity = (
        actual["device"],
        actual["inode"],
        actual["size_bytes"],
        actual["mtime_ns"],
        actual["ctime_ns"],
    )
    if before_identity != after_identity or actual != expected:
        raise RetirementError(f"Retirement source metadata changed for {target.key}")


def _validate_target(path: Path, allowed_root: Path) -> tuple[int, int]:
    if not path.is_dir() or path.is_symlink():
        raise RetirementError(f"Retirement target is missing or unsafe: {path}")
    resolved = path.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise RetirementError(f"Retirement target is outside the allowed root: {resolved}")
    info = path.lstat()
    return int(info.st_dev), int(info.st_ino)


def _plan_retirement(
    config: DatasetConfig,
    *,
    scope: str,
    allowlist: dict[str, Path],
    selected: Sequence[str] | None = None,
) -> RetirementPlan:
    (
        dataset_plan,
        dataset_receipt,
        dataset_sha,
        retirement_disposition,
        retirement_disposition_sha,
        retirement_disposition_summary,
        retirement_root_metadata,
    ) = _load_dataset_completion(config)
    keys = tuple(allowlist) if selected is None else tuple(selected)
    if not keys:
        raise RetirementError("At least one retirement key is required")
    if len(keys) != len(set(keys)):
        raise RetirementError("Duplicate retirement key")
    unknown = [key for key in keys if key not in allowlist]
    if unknown:
        raise RetirementError(f"unknown retirement key: {unknown[0]}")
    if keys != tuple(allowlist):
        raise RetirementError(
            "Retirement must use the complete retirement allowlist in canonical order"
        )
    covered = scope == "covered"
    if scope not in {"full", "covered"}:
        raise RetirementError(f"Unknown retirement scope: {scope}")
    _validate_disposition_summary(
        retirement_disposition_summary,
        keys,
        covered=covered,
    )
    allowed_root = config.output_root.parent.resolve()
    archive_keys = set(config.archive_sources)
    if covered and archive_keys.intersection(keys):
        raise RetirementError("Covered retirement scope contains an archive source")
    targets: list[RetirementTarget] = []
    for key in keys:
        configured_path = allowlist[key]
        device, inode = _validate_target(configured_path, allowed_root)
        path = configured_path.resolve()
        requires_archive = key in archive_keys
        targets.append(
            RetirementTarget(key, path, requires_archive, device, inode)
        )
    for index, left in enumerate(targets):
        for right in targets[index + 1 :]:
            if left.path in right.path.parents or right.path in left.path.parents:
                raise RetirementError("Retirement targets overlap")

    archive_receipt: Path | None = None
    archive_sha: str | None = None
    if any(target.requires_archive_receipt for target in targets):
        try:
            archive_validation = validate_archive_completion(config)
        except ArchiveError as exc:
            raise RetirementError(f"Shared archive completion receipt failed: {exc}") from exc
        archive_receipt = Path(str(archive_validation["path"]))
        archive_sha = str(archive_validation["sha256"])

    scheduler = _scheduler_snapshot()
    for target in targets:
        if str(target.path) in scheduler or target.path.name in scheduler:
            raise RetirementError(
                f"Selected source is referenced by an active Slurm job: {target.path}"
            )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "CamCan_Experiment/HPC_scripts/submit_delete_directory.sh"
    )
    if not launcher.is_file():
        raise RetirementError(f"Hardened deletion launcher is missing: {launcher}")
    target_tuple = tuple(targets)
    return RetirementPlan(
        config=config,
        targets=target_tuple,
        dry_run=True,
        plan_sha256=_retirement_hash(
            scope,
            target_tuple,
            dataset_sha,
            archive_sha,
            retirement_disposition_sha,
        ),
        dataset_receipt=dataset_receipt,
        dataset_receipt_sha256=dataset_sha,
        archive_receipt=archive_receipt,
        archive_receipt_sha256=archive_sha,
        retirement_disposition=retirement_disposition,
        retirement_disposition_sha256=retirement_disposition_sha,
        retirement_disposition_summary=retirement_disposition_summary,
        scheduler_snapshot_sha256=hashlib.sha256(scheduler.encode("utf-8")).hexdigest(),
        deletion_launcher=launcher,
        scope=scope,
        retirement_root_metadata=retirement_root_metadata,
    )


def plan_retirement(
    config: DatasetConfig,
    selected: Sequence[str] | None = None,
) -> RetirementPlan:
    """Build the full read-only exact deletion plan; never submit or delete."""
    return _plan_retirement(
        config,
        scope="full",
        allowlist=retirement_allowlist(config),
        selected=selected,
    )


def plan_covered_retirement(config: DatasetConfig) -> RetirementPlan:
    """Build the exact nonarchive deletion plan; never submit or delete."""
    return _plan_retirement(
        config,
        scope="covered",
        allowlist=covered_retirement_allowlist(config),
    )


def _validate_confirmations(
    plan: RetirementPlan,
    confirmations: RetirementConfirmations,
) -> None:
    if not confirmations.confirm_retirement:
        raise RetirementError("Explicit retirement confirmation is required")
    if confirmations.dataset_root != str(plan.config.output_root.resolve()):
        raise RetirementError("The dataset root confirmation is not exact")
    if confirmations.dataset_receipt_sha256 != plan.dataset_receipt_sha256:
        raise RetirementError("The dataset receipt confirmation is not exact")
    if any(target.requires_archive_receipt for target in plan.targets) and (
        confirmations.archive_receipt_sha256 != plan.archive_receipt_sha256
    ):
        raise RetirementError("The archive receipt confirmation is not exact")
    if not any(target.requires_archive_receipt for target in plan.targets) and (
        confirmations.archive_receipt_sha256 is not None
    ):
        raise RetirementError("The retirement confirmation has an unexpected archive hash")
    if (
        confirmations.retirement_disposition_sha256
        != plan.retirement_disposition_sha256
    ):
        raise RetirementError("The retirement disposition confirmation is not exact")


def _request_payload(
    plan: RetirementPlan,
    jobs: list[dict[str, object]],
    *,
    created_utc: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "submitted" if len(jobs) == len(plan.targets) else "submitting",
        "workflow": _scope_workflow(plan.scope),
        "scope": plan.scope,
        "created_utc": created_utc or _utc_now(),
        "retirement_plan_sha256": plan.plan_sha256,
        "dataset_root": str(plan.config.output_root.resolve()),
        "dataset_receipt": str(plan.dataset_receipt),
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "archive_receipt": str(plan.archive_receipt) if plan.archive_receipt else None,
        "archive_receipt_sha256": plan.archive_receipt_sha256,
        "retirement_disposition": str(plan.retirement_disposition),
        "retirement_disposition_sha256": plan.retirement_disposition_sha256,
        "retirement_disposition_summary": plan.retirement_disposition_summary,
        "allowed_root": str(plan.config.output_root.parent.resolve()),
        "targets": [target.as_dict() for target in plan.targets],
        "jobs": jobs,
    }


def _request_plan(
    config: DatasetConfig,
    scope: str = _FULL_SCOPE,
) -> tuple[RetirementPlan, list[dict[str, object]], Path, str]:
    request_path = _request_path(config, scope)
    if not request_path.is_file() or request_path.is_symlink():
        raise RetirementError("No safe retirement request exists to resume")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetirementError("The retirement request receipt is unreadable") from exc
    if not isinstance(request, dict):
        raise RetirementError("The retirement request receipt is unreadable")
    legacy = _validate_request_header(request, scope)
    if (
        request.get("dataset_root") != str(config.output_root.resolve())
        or request.get("allowed_root") != str(config.output_root.parent.resolve())
    ):
        raise RetirementError("The retirement request receipt binding is invalid")

    dataset_receipt = config.output_root / "_consolidation/complete.json"
    if (
        request.get("dataset_receipt") != str(dataset_receipt)
        or not dataset_receipt.is_file()
        or dataset_receipt.is_symlink()
        or sha256_file(dataset_receipt) != request.get("dataset_receipt_sha256")
    ):
        raise RetirementError("The bound dataset completion receipt changed")
    archive_receipt: Path | None = None
    archive_sha: str | None = None
    if scope == _FULL_SCOPE:
        archive_receipt = (
            config.shared_receipt_root / "legacy_v3_archive_v1.complete.json"
        )
        if (
            request.get("archive_receipt") != str(archive_receipt)
            or not archive_receipt.is_file()
            or archive_receipt.is_symlink()
            or sha256_file(archive_receipt) != request.get("archive_receipt_sha256")
        ):
            raise RetirementError("The bound archive completion receipt changed")
        archive_sha = str(request["archive_receipt_sha256"])
    elif (
        request.get("archive_receipt") is not None
        or request.get("archive_receipt_sha256") is not None
    ):
        raise RetirementError("The covered retirement request has an archive receipt")
    saved_disposition = disposition_path(config)
    if (
        request.get("retirement_disposition") != str(saved_disposition)
        or not saved_disposition.is_file()
        or saved_disposition.is_symlink()
        or sha256_file(saved_disposition)
        != request.get("retirement_disposition_sha256")
    ):
        raise RetirementError("The bound retirement disposition changed")
    disposition_summary = request.get("retirement_disposition_summary")
    if not isinstance(disposition_summary, dict):
        raise RetirementError("The retirement disposition summary is invalid")
    authoritative_summary, root_metadata = _authoritative_request_disposition(
        config,
        request,
    )
    disposition_summary = authoritative_summary

    records = request.get("targets")
    jobs = request.get("jobs")
    allowlist = _scope_allowlist(config, scope)
    archive_keys = set(config.archive_sources)
    if not isinstance(records, list) or not isinstance(jobs, list):
        raise RetirementError("The retirement request scope is invalid")
    if [record.get("key") for record in records if isinstance(record, dict)] != list(
        allowlist
    ) or len(records) != len(allowlist):
        raise RetirementError("The retirement request target scope is invalid")
    _validate_disposition_summary(
        disposition_summary,
        tuple(allowlist),
        covered=scope == _COVERED_SCOPE,
    )
    targets: list[RetirementTarget] = []
    for record in records:
        if not isinstance(record, dict):
            raise RetirementError("The retirement request target binding is invalid")
        key = str(record.get("key", ""))
        expected_path = allowlist[key].resolve(strict=False)
        if (
            Path(str(record.get("path", ""))) != expected_path
            or bool(record.get("requires_archive_receipt")) != (key in archive_keys)
        ):
            raise RetirementError("The retirement request target binding is invalid")
        targets.append(
            RetirementTarget(
                key=key,
                path=expected_path,
                requires_archive_receipt=key in archive_keys,
                source_device=int(record.get("source_device", -1)),
                source_inode=int(record.get("source_inode", -1)),
            )
        )
    target_tuple = tuple(targets)
    hash_arguments = (
        target_tuple,
        str(request.get("dataset_receipt_sha256", "")),
        archive_sha,
        str(request.get("retirement_disposition_sha256", "")),
    )
    expected_hash = (
        _legacy_full_retirement_hash(*hash_arguments)
        if legacy
        else _retirement_hash(scope, *hash_arguments)
    )
    if expected_hash != request.get("retirement_plan_sha256"):
        raise RetirementError("The retirement request plan binding is invalid")
    job_keys = [str(job.get("key", "")) for job in jobs if isinstance(job, dict)]
    if (
        len(job_keys) != len(jobs)
        or len(job_keys) != len(set(job_keys))
        or not set(job_keys).issubset(allowlist)
    ):
        raise RetirementError("The retirement request job scope is invalid")
    launcher = (
        Path(__file__).resolve().parents[1]
        / "CamCan_Experiment/HPC_scripts/submit_delete_directory.sh"
    )
    created = str(request.get("created_utc", ""))
    if not created:
        raise RetirementError("The retirement request creation time is missing")
    return (
        RetirementPlan(
            config=config,
            targets=target_tuple,
            dry_run=True,
            plan_sha256=expected_hash,
            dataset_receipt=dataset_receipt,
            dataset_receipt_sha256=str(request["dataset_receipt_sha256"]),
            archive_receipt=archive_receipt,
            archive_receipt_sha256=archive_sha,
            retirement_disposition=saved_disposition,
            retirement_disposition_sha256=str(
                request["retirement_disposition_sha256"]
            ),
            retirement_disposition_summary={
                str(key): value for key, value in disposition_summary.items()
            },
            scheduler_snapshot_sha256="resume-from-immutable-request",
            deletion_launcher=launcher,
            scope=scope,
            retirement_root_metadata=root_metadata,
        ),
        [dict(job) for job in jobs],
        request_path,
        created,
    )


def _submit_missing_jobs(
    plan: RetirementPlan,
    jobs: list[dict[str, object]],
    request_path: Path,
    *,
    created_utc: str,
    validate_pending: bool,
) -> dict[str, object]:
    target_by_key = {target.key: target for target in plan.targets}
    retryable: dict[str, dict[str, object]] = {}
    for job in jobs:
        key = str(job["key"])
        target = target_by_key[key]
        job_id = str(job.get("job_id", ""))
        expected_terminal = (
            plan.config.deletion_log_root
            / f"delete-directory-{job_id}.receipt.tsv"
        )
        if job.get("terminal_receipt") != str(expected_terminal):
            raise RetirementError("The retirement request terminal receipt path is invalid")
        receipt = _read_terminal_receipt(expected_terminal)
        valid = bool(
            receipt
            and receipt.get("job_id") == job_id
            and receipt.get("target") == str(target.path)
            and receipt.get("target_id")
            == f"{target.source_device}:{target.source_inode}"
            and receipt.get("allowed_root")
            == str(plan.config.output_root.parent.resolve())
        )
        if valid and receipt and receipt.get("status") == "failed":
            retryable[key] = job
    submitted_keys = {
        str(job["key"]) for job in jobs if str(job["key"]) not in retryable
    }
    missing_targets = [
        target for target in plan.targets if target.key not in submitted_keys
    ]
    if validate_pending:
        root_metadata = plan.retirement_root_metadata
        if not isinstance(root_metadata, dict):
            raise RetirementError("Retirement source metadata binding is missing")
        for target in missing_targets:
            _validate_pending_target_metadata(target, root_metadata)
    scheduler = _scheduler_snapshot()
    for target in missing_targets:
        if str(target.path) in scheduler or target.path.name in scheduler:
            raise RetirementError(
                f"Selected source is referenced by an active Slurm job: {target.path}"
            )
    for target in plan.targets:
        if target.key in submitted_keys:
            continue
        current_device, current_inode = _validate_target(
            target.path, plan.config.output_root.parent.resolve()
        )
        if (current_device, current_inode) != (
            target.source_device,
            target.source_inode,
        ):
            raise RetirementError(
                f"Retirement target identity changed before submission: {target.path}"
            )
        command = [
            "bash",
            str(plan.deletion_launcher),
            "--target",
            str(target.path),
            "--confirm",
            str(target.path),
            "--allowed-root",
            str(plan.config.output_root.parent.resolve()),
            "--log-dir",
            str(plan.config.deletion_log_root.resolve()),
        ]
        completed = _run_launcher(command)
        if completed.returncode != 0:
            _write_json(
                request_path,
                _request_payload(plan, jobs, created_utc=created_utc),
            )
            raise RetirementError(
                f"Deletion launcher failed for {target.key}: {completed.stderr.strip()}"
            )
        match = re.search(r"Submitted deletion job: ([0-9]+)", completed.stdout)
        if match is None:
            _write_json(
                request_path,
                _request_payload(plan, jobs, created_utc=created_utc),
            )
            raise RetirementError(f"Could not parse deletion job ID for {target.key}")
        job_id = match.group(1)
        new_job: dict[str, object] = {
            "key": target.key,
            "target": str(target.path),
            "job_id": job_id,
            "request_receipt": str(
                plan.config.deletion_log_root
                / f"delete-directory-{job_id}.request.tsv"
            ),
            "terminal_receipt": str(
                plan.config.deletion_log_root
                / f"delete-directory-{job_id}.receipt.tsv"
            ),
        }
        previous = retryable.get(target.key)
        if previous is None:
            jobs.append(new_job)
        else:
            history = list(previous.get("retry_history", []))
            prior_attempt = {
                key: value
                for key, value in previous.items()
                if key != "retry_history"
            }
            new_job["retry_history"] = history + [prior_attempt]
            index = next(
                index
                for index, job in enumerate(jobs)
                if str(job["key"]) == target.key
            )
            jobs[index] = new_job
        submitted_keys.add(target.key)
        _write_json(
            request_path,
            _request_payload(plan, jobs, created_utc=created_utc),
        )
    return {
        "status": "submitted",
        "complete": False,
        "request_receipt": str(request_path),
        "retirement_plan_sha256": plan.plan_sha256,
        "jobs": jobs,
    }


def _live_plan(config: DatasetConfig, scope: str) -> RetirementPlan:
    if scope == _FULL_SCOPE:
        return plan_retirement(config)
    if scope == _COVERED_SCOPE:
        return plan_covered_retirement(config)
    raise RetirementError(f"Unknown retirement scope: {scope}")


def _submit_retirement_locked(
    plan: RetirementPlan,
    confirmations: RetirementConfirmations,
    *,
    scope: str,
) -> dict[str, object]:
    if plan.scope != scope:
        raise RetirementError("The retirement plan has another scope")
    request_path = _request_path(plan.config, scope)
    if request_path.exists() or request_path.is_symlink():
        stored, jobs, request_path, created = _request_plan(plan.config, scope)
        if stored.plan_sha256 != plan.plan_sha256:
            raise RetirementError("The existing retirement request has another scope")
        _validate_confirmations(stored, confirmations)
        return _submit_missing_jobs(
            stored,
            jobs,
            request_path,
            created_utc=created,
            validate_pending=True,
        )
    _validate_confirmations(plan, confirmations)
    fresh = _live_plan(plan.config, scope)
    if fresh.plan_sha256 != plan.plan_sha256:
        raise RetirementError("The retirement plan changed during confirmation")
    config = plan.config
    config.deletion_log_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []
    created = _utc_now()
    _write_json(
        request_path, _request_payload(plan, jobs, created_utc=created)
    )
    return _submit_missing_jobs(
        plan,
        jobs,
        request_path,
        created_utc=created,
        validate_pending=False,
    )


def _submit_retirement(
    plan: RetirementPlan,
    confirmations: RetirementConfirmations,
    *,
    scope: str,
) -> dict[str, object]:
    with _submission_lock(plan.config):
        _reject_conflicting_request(plan.config, scope)
        return _submit_retirement_locked(
            plan,
            confirmations,
            scope=scope,
        )


def submit_retirement(
    plan: RetirementPlan,
    confirmations: RetirementConfirmations,
) -> dict[str, object]:
    """Submit the full archive-gated retirement plan."""
    return _submit_retirement(
        plan,
        confirmations,
        scope=_FULL_SCOPE,
    )


def submit_covered_retirement(
    plan: RetirementPlan,
    confirmations: RetirementConfirmations,
) -> dict[str, object]:
    """Submit only the exact completed-dataset-covered retirement plan."""
    return _submit_retirement(
        plan,
        confirmations,
        scope=_COVERED_SCOPE,
    )


def _resume_retirement(
    config: DatasetConfig,
    confirmations: RetirementConfirmations,
    *,
    scope: str,
) -> dict[str, object]:
    with _submission_lock(config):
        _reject_conflicting_request(config, scope)
        plan, jobs, request_path, created = _request_plan(config, scope)
        _validate_confirmations(plan, confirmations)
        return _submit_missing_jobs(
            plan,
            jobs,
            request_path,
            created_utc=created,
            validate_pending=True,
        )


def resume_retirement(
    config: DatasetConfig,
    confirmations: RetirementConfirmations,
) -> dict[str, object]:
    """Resume only jobs absent from an immutable partial retirement request."""
    return _resume_retirement(config, confirmations, scope=_FULL_SCOPE)


def resume_covered_retirement(
    config: DatasetConfig,
    confirmations: RetirementConfirmations,
) -> dict[str, object]:
    """Resume only covered jobs absent from the immutable covered request."""
    return _resume_retirement(config, confirmations, scope=_COVERED_SCOPE)


def _read_terminal_receipt(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except (OSError, csv.Error):
        return {"status": "unreadable"}
    if len(rows) != 1:
        return {"status": "unreadable"}
    return {str(key): str(value) for key, value in rows[0].items()}


def _retirement_status(config: DatasetConfig, scope: str) -> dict[str, object]:
    """Prove terminal deletion only from receipts plus independent absence."""
    request_path = _request_path(config, scope)
    if not request_path.is_file() or request_path.is_symlink():
        return {"status": "not_submitted", "complete": False, "targets": []}
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetirementError("The retirement request receipt is unreadable") from exc
    if not isinstance(request, dict):
        raise RetirementError("The retirement request receipt is unreadable")
    legacy = _validate_request_header(request, scope)
    if request.get("dataset_root") != str(config.output_root.resolve()):
        raise RetirementError("The retirement request is bound to another dataset root")
    if request.get("allowed_root") != str(config.output_root.parent.resolve()):
        raise RetirementError("The retirement request has a different allowed root")
    dataset_receipt = Path(str(request.get("dataset_receipt", "")))
    expected_dataset_receipt = config.output_root / "_consolidation/complete.json"
    if dataset_receipt != expected_dataset_receipt:
        raise RetirementError("The retirement request dataset receipt path is invalid")
    if (
        not dataset_receipt.is_file()
        or dataset_receipt.is_symlink()
        or sha256_file(dataset_receipt) != request.get("dataset_receipt_sha256")
    ):
        raise RetirementError("The bound dataset completion receipt changed")
    bound_disposition = Path(str(request.get("retirement_disposition", "")))
    disposition_summary = request.get("retirement_disposition_summary")
    if (
        bound_disposition != disposition_path(config)
        or not bound_disposition.is_file()
        or bound_disposition.is_symlink()
        or sha256_file(bound_disposition)
        != request.get("retirement_disposition_sha256")
        or not isinstance(disposition_summary, dict)
    ):
        raise RetirementError("The bound retirement disposition changed")
    authoritative_summary, _ = _authoritative_request_disposition(config, request)
    disposition_summary = authoritative_summary
    archive_value = request.get("archive_receipt")
    if archive_value:
        archive_receipt = Path(str(archive_value))
        if (
            not archive_receipt.is_file()
            or archive_receipt.is_symlink()
            or sha256_file(archive_receipt)
            != request.get("archive_receipt_sha256")
        ):
            raise RetirementError("The bound archive completion receipt changed")
    target_records = request.get("targets")
    jobs = request.get("jobs")
    if not isinstance(target_records, list) or not isinstance(jobs, list):
        raise RetirementError("The retirement request scope is invalid")
    allowlist = _scope_allowlist(config, scope)
    archive_keys = set(config.archive_sources)
    record_keys: list[str] = []
    for record in target_records:
        if not isinstance(record, dict):
            raise RetirementError("The retirement request target binding is invalid")
        key = str(record.get("key", ""))
        record_keys.append(key)
        if (
            key not in allowlist
            or Path(str(record.get("path", "")))
            != allowlist[key].resolve(strict=False)
            or bool(record.get("requires_archive_receipt")) != (key in archive_keys)
        ):
            raise RetirementError("The retirement request target binding is invalid")
    if record_keys != list(allowlist):
        raise RetirementError("The retirement request target binding is invalid")
    _validate_disposition_summary(
        disposition_summary,
        tuple(allowlist),
        covered=scope == _COVERED_SCOPE,
    )
    expected_archive_receipt = (
        config.shared_receipt_root / "legacy_v3_archive_v1.complete.json"
    )
    if scope == _FULL_SCOPE:
        if not archive_value or Path(str(archive_value)) != expected_archive_receipt:
            raise RetirementError("The retirement request archive receipt path is invalid")
    elif archive_value is not None:
        raise RetirementError("The retirement request has an unexpected archive receipt")
    target_tuple = tuple(
        RetirementTarget(
            key=str(record["key"]),
            path=Path(str(record["path"])),
            requires_archive_receipt=bool(record["requires_archive_receipt"]),
            source_device=int(record["source_device"]),
            source_inode=int(record["source_inode"]),
        )
        for record in target_records
    )
    hash_arguments = (
        target_tuple,
        str(request["dataset_receipt_sha256"]),
        (
            str(request["archive_receipt_sha256"])
            if request.get("archive_receipt_sha256")
            else None
        ),
        str(request.get("retirement_disposition_sha256", "")),
    )
    expected_hash = (
        _legacy_full_retirement_hash(*hash_arguments)
        if legacy
        else _retirement_hash(scope, *hash_arguments)
    )
    if expected_hash != request.get("retirement_plan_sha256"):
        raise RetirementError("The retirement request plan binding is invalid")
    job_by_key = {
        str(job.get("key")): job for job in jobs if isinstance(job, dict)
    }
    expected_keys = {str(target["key"]) for target in target_records}
    if len(job_by_key) != len(jobs) or not set(job_by_key).issubset(expected_keys):
        raise RetirementError("The retirement request job scope is invalid")
    results: list[dict[str, object]] = []
    for target in target_records:
        key = str(target["key"])
        path = Path(str(target["path"]))
        job = job_by_key.get(key)
        expected_terminal = (
            config.deletion_log_root
            / f"delete-directory-{job['job_id']}.receipt.tsv"
            if job is not None
            else None
        )
        if job is not None and job.get("terminal_receipt") != str(expected_terminal):
            raise RetirementError("The retirement request terminal receipt path is invalid")
        receipt_payload = (
            _read_terminal_receipt(expected_terminal)
            if expected_terminal is not None
            else None
        )
        receipt_valid = bool(
            job
            and receipt_payload
            and receipt_payload.get("job_id") == str(job["job_id"])
            and receipt_payload.get("target") == str(path)
            and receipt_payload.get("target_id")
            == f"{target['source_device']}:{target['source_inode']}"
            and receipt_payload.get("allowed_root")
            == str(config.output_root.parent.resolve())
        )
        receipt_status = (
            receipt_payload.get("status") if receipt_valid and receipt_payload else "missing"
        )
        absent = not path.exists() and not path.is_symlink()
        results.append(
            {
                "key": key,
                "target": str(path),
                "job_id": str(job["job_id"]) if job else None,
                "receipt_status": receipt_status,
                "target_absent": absent,
                "complete": receipt_status == "complete" and absent,
            }
        )
    complete = len(results) == len(target_records) and all(
        result["complete"] for result in results
    )
    if complete:
        summary_path = _completion_path(config, scope)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": _utc_now(),
            "retirement_plan_sha256": request["retirement_plan_sha256"],
            "request_receipt": str(request_path),
            "request_receipt_sha256": sha256_file(request_path),
            "targets": results,
        }
        if summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            comparable = dict(summary)
            comparable["completed_utc"] = existing.get("completed_utc")
            if existing != comparable:
                raise RetirementError("Existing retirement completion summary differs")
        else:
            _write_json(summary_path, summary)
        return {"status": "complete", "complete": True, "targets": results}
    has_terminal = any(result["receipt_status"] != "missing" for result in results)
    return {
        "status": "incomplete" if has_terminal else "submitted",
        "complete": False,
        "targets": results,
    }


def retirement_status(config: DatasetConfig) -> dict[str, object]:
    """Report the full archive-gated retirement status."""
    return _retirement_status(config, _FULL_SCOPE)


def covered_retirement_status(config: DatasetConfig) -> dict[str, object]:
    """Report the completed-dataset-covered retirement status."""
    return _retirement_status(config, _COVERED_SCOPE)
