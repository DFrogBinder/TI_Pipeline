"""Resumable orchestration that stops at the retirement confirmation boundary."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
from typing import Iterator, TextIO

from .archive import (
    ArchiveError,
    archive_copy,
    archive_preflight,
    archive_verify,
    validate_archive_completion,
)
from .retirement import RetirementError, RetirementPlan, plan_retirement
from .settings import DatasetConfig
from .workflow import (
    WorkflowError,
    complete_dataset,
    validate_dataset_completion,
)


class ReadinessError(RuntimeError):
    """Raised when the automatic workflow cannot safely reach deletion readiness."""


class ReadinessLockedError(ReadinessError):
    """Raised when another readiness process owns the exclusive lock."""


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


def _state_path(config: DatasetConfig) -> Path:
    return config.deletion_log_root / "retirement_readiness_state.json"


def _ready_path(config: DatasetConfig) -> Path:
    return config.deletion_log_root / "retirement_ready.json"


def _lock_path(config: DatasetConfig) -> Path:
    return config.deletion_log_root / "retirement_readiness.lock"


def _write_state(
    config: DatasetConfig,
    *,
    status: str,
    stage: str,
    detail: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "stage": stage,
        "updated_utc": _utc_now(),
        "dataset_root": str(config.output_root.resolve(strict=False)),
        "jobs_submitted": 0,
        "deletion_performed": False,
    }
    if detail is not None:
        payload["detail"] = detail
    _write_json(_state_path(config), payload)


@contextmanager
def _exclusive_lock(config: DatasetConfig) -> Iterator[TextIO]:
    config.deletion_log_root.mkdir(parents=True, exist_ok=True)
    path = _lock_path(config)
    if path.is_symlink():
        raise ReadinessError(f"Refusing symbolic-link readiness lock: {path}")
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReadinessLockedError(
                "Another retirement-readiness process already holds the lock"
            ) from exc
        stream.seek(0)
        stream.truncate()
        json.dump(
            {
                "schema_version": 1,
                "status": "locked",
                "acquired_utc": _utc_now(),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "dataset_root": str(config.output_root.resolve(strict=False)),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield stream
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _retire_submit_command(plan: RetirementPlan) -> list[str]:
    command = [
        "bash",
        "repeatability_dataset_consolidation/hpc/submit.sh",
        "retire-submit",
        f"--confirm-output-root={plan.config.output_root}",
        f"--dataset-receipt-sha256={plan.dataset_receipt_sha256}",
    ]
    if plan.archive_receipt_sha256 is not None:
        command.append(
            f"--archive-receipt-sha256={plan.archive_receipt_sha256}"
        )
    command.extend(
        (
            "--retirement-disposition-sha256="
            f"{plan.retirement_disposition_sha256}",
            "--confirm-retirement",
        )
    )
    return command


def _ready_payload(plan: RetirementPlan) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "deletion_ready",
        "workflow": "repeatability_dataset_consolidation_retirement_readiness",
        "created_utc": _utc_now(),
        "dataset_root": str(plan.config.output_root.resolve()),
        "dataset_receipt": str(plan.dataset_receipt),
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "archive_receipt": (
            str(plan.archive_receipt) if plan.archive_receipt is not None else None
        ),
        "archive_receipt_sha256": plan.archive_receipt_sha256,
        "retirement_disposition": str(plan.retirement_disposition),
        "retirement_disposition_sha256": plan.retirement_disposition_sha256,
        "retirement_disposition_summary": plan.retirement_disposition_summary,
        "retirement_plan_sha256": plan.plan_sha256,
        "scheduler_snapshot_sha256": plan.scheduler_snapshot_sha256,
        "targets": [target.as_dict() for target in plan.targets],
        "retire_submit_command": _retire_submit_command(plan),
        "jobs_submitted": 0,
        "deletion_performed": False,
        "source_outputs_modified": False,
    }


def _persist_ready(config: DatasetConfig, payload: dict[str, object]) -> str:
    path = _ready_path(config)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ReadinessError(f"Refusing unsafe readiness receipt: {path}")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReadinessError("The existing readiness receipt is unreadable") from exc
        comparable = dict(payload)
        comparable["created_utc"] = existing.get("created_utc")
        if existing != comparable:
            raise ReadinessError(
                "Refusing to replace a nonmatching retirement-readiness receipt"
            )
        return "reused"
    _write_json(path, payload)
    return "created"


def _prepare_retirement_impl(config: DatasetConfig) -> dict[str, object]:
    _write_state(config, status="running", stage="finalize_dataset")
    complete_dataset(config)
    validate_dataset_completion(config)

    _write_state(config, status="running", stage="archive_preflight")
    archive_preflight(config)
    _write_state(config, status="running", stage="archive_copy")
    archive_copy(config)
    _write_state(config, status="running", stage="archive_quick_verify")
    quick = archive_verify(config, checksum=False)
    if quick.get("status") != "matched":
        raise ReadinessError("Shared archive size/mtime verification did not match")
    _write_state(config, status="running", stage="archive_checksum_verify")
    checksum = archive_verify(config, checksum=True)
    if checksum.get("status") != "complete":
        raise ReadinessError("Shared archive checksum verification did not complete")
    validate_archive_completion(config)

    _write_state(config, status="running", stage="retirement_plan")
    plan = plan_retirement(config)
    payload = _ready_payload(plan)
    action = _persist_ready(config, payload)
    _write_state(config, status="deletion_ready", stage="stop_before_deletion")
    return {
        **payload,
        "action": action,
        "readiness_receipt": str(_ready_path(config)),
    }


def prepare_retirement(config: DatasetConfig) -> dict[str, object]:
    """Reach deletion readiness automatically, but never submit or delete."""
    try:
        with _exclusive_lock(config):
            return _prepare_retirement_impl(config)
    except ReadinessLockedError:
        raise
    except (ArchiveError, RetirementError, WorkflowError, ReadinessError) as exc:
        _write_state(
            config,
            status="failed",
            stage="stopped_before_deletion",
            detail=str(exc),
        )
        raise ReadinessError(str(exc)) from exc
    except Exception as exc:
        _write_state(
            config,
            status="failed",
            stage="stopped_before_deletion",
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise


def _lock_is_active(config: DatasetConfig) -> bool:
    path = _lock_path(config)
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        raise ReadinessError(f"The readiness lock is unsafe: {path}")
    with path.open("r", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return False


def _read_json(path: Path, label: str) -> dict[str, object] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ReadinessError(f"The {label} is unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"The {label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ReadinessError(f"The {label} is not a JSON object")
    return payload


def readiness_status(config: DatasetConfig) -> dict[str, object]:
    """Report external orchestration state without creating or changing files."""
    if _lock_is_active(config):
        state = _read_json(_state_path(config), "readiness state") or {}
        return {
            **state,
            "status": "running",
            "lock": str(_lock_path(config)),
            "jobs_submitted": 0,
            "deletion_performed": False,
        }

    state = _read_json(_state_path(config), "readiness state")
    if state is not None and state.get("status") == "failed":
        return {
            **state,
            "status": "failed",
            "jobs_submitted": 0,
            "deletion_performed": False,
        }

    ready = _read_json(_ready_path(config), "retirement readiness receipt")
    if ready is not None:
        if (
            ready.get("status") != "deletion_ready"
            or ready.get("workflow")
            != "repeatability_dataset_consolidation_retirement_readiness"
            or ready.get("dataset_root")
            != str(config.output_root.resolve(strict=False))
            or ready.get("jobs_submitted") != 0
            or ready.get("deletion_performed") is not False
            or not isinstance(ready.get("targets"), list)
            or not isinstance(ready.get("retire_submit_command"), list)
        ):
            raise ReadinessError("The retirement readiness receipt binding is invalid")
        return {
            **ready,
            "readiness_receipt": str(_ready_path(config)),
        }

    if state is None:
        return {
            "status": "not_started",
            "jobs_submitted": 0,
            "deletion_performed": False,
        }
    return {
        **state,
        "status": "interrupted",
        "jobs_submitted": 0,
        "deletion_performed": False,
    }
