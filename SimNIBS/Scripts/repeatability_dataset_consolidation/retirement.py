"""Receipt-gated planning, submission, and proof for source retirement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence

from .archive import ArchiveError, validate_archive_completion
from .inventory import build_file_inventory, sha256_file
from .planning import ConsolidationPlan, build_copy_plan, plan_sha256
from .settings import DatasetConfig, retirement_allowlist
from .validation import validate_scientific_scope
from .workflow import (
    WorkflowError,
    _load_checksum_verification,
    _validate_copy_receipts,
    _validate_destination_coverage,
    _validate_destination_scope,
    _validate_owned_root,
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
    scheduler_snapshot_sha256: str
    deletion_launcher: Path


@dataclass(frozen=True)
class RetirementConfirmations:
    dataset_root: str
    dataset_receipt_sha256: str
    archive_receipt_sha256: str | None
    confirm_retirement: bool


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
) -> tuple[ConsolidationPlan, Path, str]:
    try:
        plan = build_copy_plan(config)
        validation = validate_scientific_scope(plan)
        current_plan_hash = plan_sha256(plan)
        _validate_owned_root(
            config, plan.as_dict(), current_plan_hash, validation.as_dict()
        )
    except Exception as exc:
        raise RetirementError(f"Dataset completion receipt preflight failed: {exc}") from exc
    completion = config.output_root / "_consolidation/complete.json"
    inventory_path = (
        config.output_root / "_consolidation/verification/file_inventory.json"
    )
    if not completion.is_file() or completion.is_symlink():
        raise RetirementError("The dataset completion receipt is missing or unsafe")
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetirementError("The dataset completion receipt is unreadable") from exc
    if (
        payload.get("status") != "complete"
        or payload.get("workflow") != "repeatability_dataset_consolidation"
        or payload.get("output_root") != str(config.output_root.resolve())
        or payload.get("plan_sha256") != current_plan_hash
    ):
        raise RetirementError("The dataset completion receipt does not match current scope")
    try:
        _validate_copy_receipts(config, plan, current_plan_hash)
        _validate_destination_coverage(config, plan)
        _load_checksum_verification(config, plan, current_plan_hash)
        scope = _validate_destination_scope(config, plan)
        current_inventory = build_file_inventory(
            config.output_root, exclude=(completion, inventory_path)
        )
    except (WorkflowError, ValueError) as exc:
        raise RetirementError(f"Dataset completion receipt live validation failed: {exc}") from exc
    if payload.get("scope") != scope or payload.get("inventory", {}).get(
        "sha256"
    ) != current_inventory.inventory_sha256:
        raise RetirementError("The dataset completion receipt inventory is stale")
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise RetirementError("The dataset file inventory is missing or unsafe")
    try:
        saved_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetirementError("The dataset file inventory is unreadable") from exc
    if saved_inventory != current_inventory.as_dict():
        raise RetirementError("The dataset file inventory differs from live content")
    return plan, completion, sha256_file(completion)


def _retirement_hash(
    targets: Sequence[RetirementTarget],
    dataset_receipt_sha256: str,
    archive_receipt_sha256: str | None,
) -> str:
    encoded = json.dumps(
        {
            "targets": [target.as_dict() for target in targets],
            "dataset_receipt_sha256": dataset_receipt_sha256,
            "archive_receipt_sha256": archive_receipt_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_target(path: Path, allowed_root: Path) -> tuple[int, int]:
    if not path.is_dir() or path.is_symlink():
        raise RetirementError(f"Retirement target is missing or unsafe: {path}")
    resolved = path.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise RetirementError(f"Retirement target is outside the allowed root: {resolved}")
    info = path.lstat()
    return int(info.st_dev), int(info.st_ino)


def _source_is_covered(plan: ConsolidationPlan, root: Path) -> bool:
    resolved = root.resolve()
    return any(
        item.source == resolved or resolved in item.source.parents
        for item in plan.items
    )


def plan_retirement(
    config: DatasetConfig,
    selected: Sequence[str] | None = None,
) -> RetirementPlan:
    """Build a read-only exact deletion plan; never submit or delete."""
    dataset_plan, dataset_receipt, dataset_sha = _load_dataset_completion(config)
    allowlist = retirement_allowlist(config)
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
    allowed_root = config.output_root.parent.resolve()
    archive_keys = set(config.archive_sources)
    targets: list[RetirementTarget] = []
    for key in keys:
        configured_path = allowlist[key]
        device, inode = _validate_target(configured_path, allowed_root)
        path = configured_path.resolve()
        requires_archive = key in archive_keys
        if not requires_archive and not _source_is_covered(dataset_plan, path):
            raise RetirementError(
                f"Dataset completion receipt does not cover retirement source: {key}"
            )
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
        plan_sha256=_retirement_hash(target_tuple, dataset_sha, archive_sha),
        dataset_receipt=dataset_receipt,
        dataset_receipt_sha256=dataset_sha,
        archive_receipt=archive_receipt,
        archive_receipt_sha256=archive_sha,
        scheduler_snapshot_sha256=hashlib.sha256(scheduler.encode("utf-8")).hexdigest(),
        deletion_launcher=launcher,
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


def _request_payload(
    plan: RetirementPlan,
    jobs: list[dict[str, object]],
    *,
    created_utc: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "submitted" if len(jobs) == len(plan.targets) else "submitting",
        "workflow": "repeatability_dataset_consolidation_retirement",
        "created_utc": created_utc or _utc_now(),
        "retirement_plan_sha256": plan.plan_sha256,
        "dataset_root": str(plan.config.output_root.resolve()),
        "dataset_receipt": str(plan.dataset_receipt),
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "archive_receipt": str(plan.archive_receipt) if plan.archive_receipt else None,
        "archive_receipt_sha256": plan.archive_receipt_sha256,
        "allowed_root": str(plan.config.output_root.parent.resolve()),
        "targets": [target.as_dict() for target in plan.targets],
        "jobs": jobs,
    }


def _request_plan(
    config: DatasetConfig,
) -> tuple[RetirementPlan, list[dict[str, object]], Path, str]:
    request_path = config.deletion_log_root / "retirement_request.json"
    if not request_path.is_file() or request_path.is_symlink():
        raise RetirementError("No safe retirement request exists to resume")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetirementError("The retirement request receipt is unreadable") from exc
    if (
        request.get("workflow") != "repeatability_dataset_consolidation_retirement"
        or request.get("dataset_root") != str(config.output_root.resolve())
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

    records = request.get("targets")
    jobs = request.get("jobs")
    allowlist = retirement_allowlist(config)
    archive_keys = set(config.archive_sources)
    if not isinstance(records, list) or not isinstance(jobs, list):
        raise RetirementError("The retirement request scope is invalid")
    if [record.get("key") for record in records if isinstance(record, dict)] != list(
        allowlist
    ) or len(records) != len(allowlist):
        raise RetirementError("The retirement request target scope is invalid")
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
    expected_hash = _retirement_hash(
        target_tuple,
        str(request.get("dataset_receipt_sha256", "")),
        str(request.get("archive_receipt_sha256", "")),
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
            archive_receipt_sha256=str(request["archive_receipt_sha256"]),
            scheduler_snapshot_sha256="resume-from-immutable-request",
            deletion_launcher=launcher,
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


def submit_retirement(
    plan: RetirementPlan,
    confirmations: RetirementConfirmations,
) -> dict[str, object]:
    """Revalidate the plan and delegate one guarded deletion job per target."""
    request_path = plan.config.deletion_log_root / "retirement_request.json"
    if request_path.exists() or request_path.is_symlink():
        stored, jobs, request_path, created = _request_plan(plan.config)
        if stored.plan_sha256 != plan.plan_sha256:
            raise RetirementError("The existing retirement request has another scope")
        _validate_confirmations(stored, confirmations)
        return _submit_missing_jobs(
            stored, jobs, request_path, created_utc=created
        )
    _validate_confirmations(plan, confirmations)
    fresh = plan_retirement(
        plan.config, selected=tuple(target.key for target in plan.targets)
    )
    if fresh.plan_sha256 != plan.plan_sha256:
        raise RetirementError("The retirement plan changed during confirmation")
    config = plan.config
    config.deletion_log_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []
    created = _utc_now()
    _write_json(
        request_path, _request_payload(plan, jobs, created_utc=created)
    )
    return _submit_missing_jobs(plan, jobs, request_path, created_utc=created)


def resume_retirement(
    config: DatasetConfig,
    confirmations: RetirementConfirmations,
) -> dict[str, object]:
    """Resume only jobs absent from an immutable partial retirement request."""
    plan, jobs, request_path, created = _request_plan(config)
    _validate_confirmations(plan, confirmations)
    return _submit_missing_jobs(plan, jobs, request_path, created_utc=created)


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


def retirement_status(config: DatasetConfig) -> dict[str, object]:
    """Prove terminal deletion only from receipts plus independent absence."""
    request_path = config.deletion_log_root / "retirement_request.json"
    if not request_path.is_file() or request_path.is_symlink():
        return {"status": "not_submitted", "complete": False, "targets": []}
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetirementError("The retirement request receipt is unreadable") from exc
    if request.get("workflow") != "repeatability_dataset_consolidation_retirement":
        raise RetirementError("The retirement request receipt is unowned")
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
    allowlist = retirement_allowlist(config)
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
    if len(record_keys) != len(set(record_keys)) or not record_keys:
        raise RetirementError("The retirement request target binding is invalid")
    expected_archive_receipt = (
        config.shared_receipt_root / "legacy_v3_archive_v1.complete.json"
    )
    if any(key in archive_keys for key in record_keys):
        if not archive_value or Path(str(archive_value)) != expected_archive_receipt:
            raise RetirementError("The retirement request archive receipt path is invalid")
    elif archive_value is not None:
        raise RetirementError("The retirement request has an unexpected archive receipt")
    expected_hash = _retirement_hash(
        tuple(
            RetirementTarget(
                key=str(record["key"]),
                path=Path(str(record["path"])),
                requires_archive_receipt=bool(record["requires_archive_receipt"]),
                source_device=int(record["source_device"]),
                source_inode=int(record["source_inode"]),
            )
            for record in target_records
        ),
        str(request["dataset_receipt_sha256"]),
        (
            str(request["archive_receipt_sha256"])
            if request.get("archive_receipt_sha256")
            else None
        ),
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
        summary_path = config.deletion_log_root / "retirement_complete.json"
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
