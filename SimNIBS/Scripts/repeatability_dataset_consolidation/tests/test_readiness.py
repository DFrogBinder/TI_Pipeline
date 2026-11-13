from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.inventory import sha256_file
from repeatability_dataset_consolidation.planning import build_copy_plan
from repeatability_dataset_consolidation.readiness import (
    ReadinessError,
    prepare_retirement,
    readiness_status,
)
from repeatability_dataset_consolidation.settings import retirement_allowlist
from repeatability_dataset_consolidation.workflow import (
    copy_dataset,
    prepare,
    verify_dataset,
)


def _seed_archive_sources(config) -> None:
    for key, root in config.archive_sources.items():
        (root / f"{key}.txt").write_text(f"{key}\n", encoding="utf-8")


def _prepare_checksum_matched(config) -> None:
    _seed_archive_sources(config)
    prepare(config)
    copy_dataset(config)
    assert verify_dataset(config, checksum=True)["status"] == "matched"


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int, bytes | None], ...]:
    if not root.exists():
        return ()
    records: list[tuple[str, int, int, bytes | None]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda value: str(value)):
        info = path.lstat()
        records.append(
            (
                str(path),
                info.st_mtime_ns,
                info.st_size,
                path.read_bytes() if path.is_file() and not path.is_symlink() else None,
            )
        )
    return tuple(records)


def test_prepare_retirement_stops_before_deletion(
    miniature_config, monkeypatch
) -> None:
    config = miniature_config
    _prepare_checksum_matched(config)
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )

    result = prepare_retirement(config)

    assert result["status"] == "deletion_ready"
    assert result["jobs_submitted"] == 0
    assert result["deletion_performed"] is False
    ready_path = config.deletion_log_root / "retirement_ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    dataset_receipt = config.output_root / "_consolidation/complete.json"
    archive_receipt = (
        config.shared_receipt_root / "legacy_v3_archive_v1.complete.json"
    )
    disposition = (
        config.output_root / "provenance/retirement/source_disposition.json"
    )
    assert ready["dataset_receipt_sha256"] == sha256_file(dataset_receipt)
    assert ready["archive_receipt_sha256"] == sha256_file(archive_receipt)
    assert ready["retirement_disposition_sha256"] == sha256_file(disposition)
    assert ready["retirement_plan_sha256"] == result["retirement_plan_sha256"]
    assert ready["scheduler_snapshot_sha256"] == result[
        "scheduler_snapshot_sha256"
    ]
    assert [target["key"] for target in ready["targets"]] == list(
        retirement_allowlist(config)
    )
    command = ready["retire_submit_command"]
    assert command[:3] == [
        "bash",
        "repeatability_dataset_consolidation/hpc/submit.sh",
        "retire-submit",
    ]
    assert f"--confirm-output-root={config.output_root}" in command
    assert f"--dataset-receipt-sha256={ready['dataset_receipt_sha256']}" in command
    assert f"--archive-receipt-sha256={ready['archive_receipt_sha256']}" in command
    assert (
        f"--retirement-disposition-sha256="
        f"{ready['retirement_disposition_sha256']}"
    ) in command
    assert "--confirm-retirement" in command
    assert ready["jobs_submitted"] == 0
    assert ready["deletion_performed"] is False
    assert all(
        path.is_dir() and not path.is_symlink()
        for path in retirement_allowlist(config).values()
    )
    assert not (config.deletion_log_root / "retirement_request.json").exists()


def test_prepare_retirement_never_crosses_the_deletion_boundary(
    miniature_config, monkeypatch
) -> None:
    config = miniature_config
    _prepare_checksum_matched(config)
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    launcher_calls: list[object] = []
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher",
        lambda *args, **kwargs: launcher_calls.append((args, kwargs))
        or pytest.fail("the deletion launcher must not be called"),
    )

    result = prepare_retirement(config)

    assert result["status"] == "deletion_ready"
    assert launcher_calls == []
    assert not (config.deletion_log_root / "retirement_request.json").exists()
    assert all(path.exists() for path in retirement_allowlist(config).values())


def test_prepare_retirement_fails_closed_on_dataset_archive_and_scheduler_changes(
    miniature_config, monkeypatch
) -> None:
    config = miniature_config
    _prepare_checksum_matched(config)
    first = build_copy_plan(config).items[0]
    destination = config.output_root / first.destination
    changed = next(path for path in destination.rglob("*") if path.is_file())
    changed.write_bytes(changed.read_bytes() + b"changed")
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    with pytest.raises(ReadinessError, match="destination|inventory|metadata"):
        prepare_retirement(config)
    assert not (config.deletion_log_root / "retirement_ready.json").exists()


def test_prepare_retirement_rejects_archive_mismatch_active_job_and_lock(
    miniature_config, monkeypatch
) -> None:
    config = miniature_config
    _prepare_checksum_matched(config)
    original_verify = __import__(
        "repeatability_dataset_consolidation.readiness", fromlist=["archive_verify"]
    ).archive_verify

    def mismatch(config, *, checksum):
        if checksum:
            return {"status": "mismatch"}
        return original_verify(config, checksum=checksum)

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.readiness.archive_verify", mismatch
    )
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    with pytest.raises(ReadinessError, match="checksum verification"):
        prepare_retirement(config)

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.readiness.archive_verify",
        original_verify,
    )
    active = next(iter(retirement_allowlist(config).values())).resolve()
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: f"123|active|/work|python worker.py {active}\n",
    )
    with pytest.raises(ReadinessError, match="active Slurm job"):
        prepare_retirement(config)

    lock = config.deletion_log_root / "retirement_readiness.lock"
    with lock.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = _tree_snapshot(config.deletion_log_root)
        with pytest.raises(ReadinessError, match="already holds the lock"):
            prepare_retirement(config)
        assert _tree_snapshot(config.deletion_log_root) == before


def test_readiness_status_is_read_only_for_absent_running_failed_and_ready(
    miniature_config, monkeypatch
) -> None:
    config = miniature_config
    before = _tree_snapshot(config.deletion_log_root)
    assert readiness_status(config)["status"] == "not_started"
    assert _tree_snapshot(config.deletion_log_root) == before

    config.deletion_log_root.mkdir(parents=True)
    lock = config.deletion_log_root / "retirement_readiness.lock"
    with lock.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = _tree_snapshot(config.deletion_log_root)
        assert readiness_status(config)["status"] == "running"
        assert _tree_snapshot(config.deletion_log_root) == before

    state = config.deletion_log_root / "retirement_readiness_state.json"
    state.write_text(
        json.dumps({"status": "failed", "stage": "archive_copy"}) + "\n",
        encoding="utf-8",
    )
    before = _tree_snapshot(config.deletion_log_root)
    assert readiness_status(config)["status"] == "failed"
    assert _tree_snapshot(config.deletion_log_root) == before

    _prepare_checksum_matched(config)
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    prepare_retirement(config)
    before = _tree_snapshot(config.deletion_log_root)
    status = readiness_status(config)
    assert status["status"] == "deletion_ready"
    assert status["jobs_submitted"] == 0
    assert status["deletion_performed"] is False
    assert _tree_snapshot(config.deletion_log_root) == before

    source = config.sources.paper_analysis / "analysis.json"
    source.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ReadinessError, match="metadata changed"):
        prepare_retirement(config)
    assert readiness_status(config)["status"] == "failed"
