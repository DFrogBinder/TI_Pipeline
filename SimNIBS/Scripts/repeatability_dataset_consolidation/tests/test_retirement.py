from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import repeatability_dataset_consolidation.retirement as retirement_module
from repeatability_dataset_consolidation.archive import archive_copy, archive_verify
from repeatability_dataset_consolidation.retirement import (
    RetirementConfirmations,
    RetirementError,
    covered_retirement_status,
    plan_covered_retirement,
    plan_retirement,
    resume_covered_retirement,
    resume_retirement,
    retirement_status,
    submit_covered_retirement,
    submit_retirement,
)
from repeatability_dataset_consolidation.settings import (
    covered_retirement_allowlist,
    retirement_allowlist,
)
from repeatability_dataset_consolidation.workflow import (
    complete_dataset,
    copy_dataset,
    prepare,
    verify_dataset,
)


@pytest.fixture
def dataset_retirement_ready(miniature_config, monkeypatch):
    for key, root in miniature_config.archive_sources.items():
        (root / f"{key}.txt").write_text(f"{key}\n", encoding="utf-8")
    prepare(miniature_config)
    copy_dataset(miniature_config)
    assert verify_dataset(miniature_config, checksum=True)["status"] == "matched"
    complete_dataset(miniature_config)
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    return miniature_config


@pytest.fixture
def retirement_ready(dataset_retirement_ready):
    config = dataset_retirement_ready
    archive_copy(config)
    assert archive_verify(config, checksum=True)["status"] == "complete"
    return config


def test_covered_plan_uses_only_dataset_covered_roots_without_archive_receipt(
    dataset_retirement_ready,
) -> None:
    config = dataset_retirement_ready

    plan = plan_covered_retirement(config)

    assert plan.scope == "covered"
    assert tuple(target.key for target in plan.targets) == tuple(
        covered_retirement_allowlist(config)
    )
    assert len(plan.targets) == 10
    assert all(not target.requires_archive_receipt for target in plan.targets)
    assert plan.archive_receipt is None
    assert plan.archive_receipt_sha256 is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda summary, key: summary.pop(key), "missing"),
        (
            lambda summary, key: summary[key].__setitem__(
                "dataset_copy_files", True
            ),
            "counter",
        ),
        (
            lambda summary, key: summary[key].__setitem__(
                "intentional_retirement_bytes", -1
            ),
            "counter",
        ),
        (
            lambda summary, key: summary[key].__setitem__(
                "shared_archive_files", 1
            ),
            "Shared-archive",
        ),
    ),
)
def test_covered_plan_rejects_unsafe_disposition_summary(
    dataset_retirement_ready, monkeypatch, mutation, message
) -> None:
    config = dataset_retirement_ready
    loaded = list(retirement_module._load_dataset_completion(config))
    loaded[5] = {
        key: dict(value) for key, value in loaded[5].items()
    }
    first_key = next(iter(covered_retirement_allowlist(config)))
    mutation(loaded[5], first_key)
    monkeypatch.setattr(
        retirement_module,
        "_load_dataset_completion",
        lambda unused_config: tuple(loaded),
    )

    with pytest.raises(RetirementError, match=message):
        plan_covered_retirement(config)


def test_retirement_plan_is_dry_run_exact_and_receipt_bound(retirement_ready) -> None:
    config = retirement_ready
    plan = plan_retirement(config)

    assert plan.dry_run is True
    assert {target.key for target in plan.targets} == set(retirement_allowlist(config))
    assert {target.path for target in plan.targets} == {
        path.resolve() for path in retirement_allowlist(config).values()
    }
    assert all(target.path != config.sources.v4_scaffolds for target in plan.targets)
    assert len(plan.dataset_receipt_sha256) == 64
    assert len(plan.archive_receipt_sha256 or "") == 64
    assert {
        target.key for target in plan.targets if target.requires_archive_receipt
    } == set(config.archive_sources)

    with pytest.raises(RetirementError, match="complete retirement allowlist"):
        plan_retirement(config, selected=("paper_analysis", "legacy_v3_left"))
    with pytest.raises(RetirementError, match="unknown retirement key"):
        plan_retirement(config, selected=("arbitrary_path",))
    with pytest.raises(RetirementError, match="At least one retirement key"):
        plan_retirement(config, selected=())


def test_retirement_plan_uses_bound_checkpoints_not_full_body_rehashes(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    completion = json.loads(
        (config.output_root / "_consolidation/complete.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = completion["retirement_disposition"]["root_metadata"]
    assert set(metadata) == set(retirement_allowlist(config))
    assert all(len(record["metadata_sha256"]) == 64 for record in metadata.values())

    monkeypatch.setattr(
        retirement_module,
        "_load_checksum_verification",
        lambda *args, **kwargs: pytest.fail("must not re-hash dataset bodies"),
        raising=False,
    )
    monkeypatch.setattr(
        retirement_module,
        "build_retirement_disposition",
        lambda *args, **kwargs: pytest.fail("must not rebuild the disposition"),
        raising=False,
    )

    assert plan_retirement(config).dry_run is True


def test_retirement_plan_fails_when_source_metadata_changes(
    retirement_ready,
) -> None:
    config = retirement_ready
    source = config.sources.paper_analysis / "analysis.json"
    original = source.read_bytes()
    source.write_bytes(b"[]\n" if original != b"[]\n" else b"{}\n")

    with pytest.raises(RetirementError, match="metadata changed"):
        plan_retirement(config)


def test_retirement_plan_rejects_tampering_and_active_jobs(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    completion = config.output_root / "_consolidation/complete.json"
    original = completion.read_bytes()
    completion.write_text('{"status":"forged"}\n', encoding="utf-8")
    with pytest.raises(RetirementError, match="dataset completion receipt"):
        plan_retirement(config)
    completion.write_bytes(original)

    target = config.sources.paper_analysis.resolve()
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: f"123|job|{target}|python worker.py {target}\n",
    )
    with pytest.raises(RetirementError, match="active Slurm job"):
        plan_retirement(config)


def _confirmations(plan, *, archive: bool = True) -> RetirementConfirmations:
    return RetirementConfirmations(
        dataset_root=str(plan.config.output_root.resolve()),
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        archive_receipt_sha256=(plan.archive_receipt_sha256 if archive else None),
        retirement_disposition_sha256=plan.retirement_disposition_sha256,
        confirm_retirement=True,
    )


def test_covered_submission_uses_distinct_receipts_and_exact_ten_jobs(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {6000 + len(commands)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )

    unexpected_archive = RetirementConfirmations(
        dataset_root=str(config.output_root.resolve()),
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        archive_receipt_sha256="a" * 64,
        retirement_disposition_sha256=plan.retirement_disposition_sha256,
        confirm_retirement=True,
    )
    with pytest.raises(RetirementError, match="unexpected archive"):
        submit_covered_retirement(plan, unexpected_archive)

    submitted = submit_covered_retirement(plan, _confirmations(plan))

    assert submitted["status"] == "submitted"
    assert len(commands) == 10
    assert [command[command.index("--target") + 1] for command in commands] == [
        str(target.path) for target in plan.targets
    ]
    covered_request = config.deletion_log_root / "covered_retirement_request.json"
    assert covered_request.is_file()
    assert not (config.deletion_log_root / "retirement_request.json").exists()
    request = json.loads(covered_request.read_text(encoding="utf-8"))
    assert request["schema_version"] == 2
    assert request["scope"] == "covered"
    assert request["archive_receipt"] is None
    assert request["archive_receipt_sha256"] is None
    assert covered_retirement_status(config)["status"] == "submitted"
    with pytest.raises(RetirementError, match="conflicting retirement request"):
        resume_retirement(config, _confirmations(plan))

    repeated = resume_covered_retirement(config, _confirmations(plan))
    assert repeated["jobs"] == submitted["jobs"]
    assert len(commands) == 10

    for index, target in enumerate(plan.targets, start=1):
        _write_terminal_receipt(
            config.deletion_log_root
            / f"delete-directory-{6000 + index}.receipt.tsv",
            job_id=str(6000 + index),
            target=target,
            status="complete",
        )
        shutil.rmtree(target.path)

    completed = covered_retirement_status(config)
    assert completed["status"] == "complete"
    assert (config.deletion_log_root / "covered_retirement_complete.json").is_file()
    assert not (config.deletion_log_root / "retirement_complete.json").exists()


def test_submission_lock_rejects_concurrent_covered_submission(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {6200 + len(commands)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )

    with retirement_module._submission_lock(config):
        with pytest.raises(RetirementError, match="Another retirement submission"):
            submit_covered_retirement(plan, _confirmations(plan))

    assert commands == []
    assert not (
        config.deletion_log_root / "covered_retirement_request.json"
    ).exists()


def test_covered_resume_rejects_archive_key_forgery(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    submitted = 0

    def fake_run(command, **kwargs):
        nonlocal submitted
        submitted += 1
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {6500 + submitted}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    submit_covered_retirement(plan, _confirmations(plan))
    request_path = config.deletion_log_root / "covered_retirement_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    archive_key, archive_path = next(iter(config.archive_sources.items()))
    request["targets"][-1] = {
        "key": archive_key,
        "path": str(archive_path.resolve()),
        "requires_archive_receipt": True,
        "source_device": archive_path.lstat().st_dev,
        "source_inode": archive_path.lstat().st_ino,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RetirementError, match="target scope"):
        resume_covered_retirement(config, _confirmations(plan))
    with pytest.raises(RetirementError, match="target binding"):
        covered_retirement_status(config)


def test_covered_partial_submission_resumes_only_missing_targets(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    initial_calls: list[str] = []

    def fail_second(command, **kwargs):
        target = command[command.index("--target") + 1]
        initial_calls.append(target)
        if len(initial_calls) == 2:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {6700 + len(initial_calls)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fail_second
    )
    with pytest.raises(RetirementError, match="launcher failed"):
        submit_covered_retirement(plan, _confirmations(plan))

    first_target = plan.targets[0]
    shutil.rmtree(first_target.path)
    resumed_calls: list[str] = []

    def succeed(command, **kwargs):
        target = command[command.index("--target") + 1]
        resumed_calls.append(target)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {6800 + len(resumed_calls)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", succeed
    )
    resumed = resume_covered_retirement(config, _confirmations(plan))

    assert str(first_target.path) not in resumed_calls
    assert len(resumed["jobs"]) == len(plan.targets)


def test_covered_resume_revalidates_pending_source_metadata(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    calls = 0

    def fail_second(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {6900 + calls}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fail_second
    )
    with pytest.raises(RetirementError, match="launcher failed"):
        submit_covered_retirement(plan, _confirmations(plan))

    pending = plan.targets[1]
    (pending.path / "added-after-request.txt").write_text("unsafe\n", encoding="utf-8")
    resumed_calls: list[list[str]] = []
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher",
        lambda command, **kwargs: resumed_calls.append(list(command))
        or subprocess.CompletedProcess(
            command, 0, stdout="[INFO] Submitted deletion job: 6999\n", stderr=""
        ),
    )

    with pytest.raises(RetirementError, match="metadata changed"):
        resume_covered_retirement(config, _confirmations(plan))
    assert resumed_calls == []


def test_covered_resume_revalidates_live_completed_dataset(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    calls = 0

    def fail_second(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {7100 + calls}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fail_second
    )
    with pytest.raises(RetirementError, match="launcher failed"):
        submit_covered_retirement(plan, _confirmations(plan))
    manifest = config.output_root / "dataset_manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RetirementError, match="live validation failed"):
        resume_covered_retirement(config, _confirmations(plan))


@pytest.mark.parametrize("first_scope", ("full", "covered"))
def test_full_and_covered_requests_cannot_submit_overlapping_jobs(
    retirement_ready, monkeypatch, first_scope
) -> None:
    config = retirement_ready
    full_plan = plan_retirement(config)
    covered_plan = plan_covered_retirement(config)
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {7300 + len(commands)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    if first_scope == "full":
        submit_retirement(full_plan, _confirmations(full_plan))
        expected_calls = len(full_plan.targets)
        second = lambda: submit_covered_retirement(
            covered_plan, _confirmations(covered_plan)
        )
    else:
        submit_covered_retirement(covered_plan, _confirmations(covered_plan))
        expected_calls = len(covered_plan.targets)
        second = lambda: submit_retirement(full_plan, _confirmations(full_plan))

    with pytest.raises(RetirementError, match="conflicting retirement request"):
        second()
    assert len(commands) == expected_calls


def test_covered_request_summary_must_equal_authoritative_completion_summary(
    dataset_retirement_ready, monkeypatch
) -> None:
    config = dataset_retirement_ready
    plan = plan_covered_retirement(config)
    submitted = 0

    def fake_run(command, **kwargs):
        nonlocal submitted
        submitted += 1
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {7600 + submitted}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    submit_covered_retirement(plan, _confirmations(plan))
    request_path = config.deletion_log_root / "covered_retirement_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    first_key = next(iter(covered_retirement_allowlist(config)))
    request["retirement_disposition_summary"][first_key][
        "dataset_copy_files"
    ] += 1
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(RetirementError, match="summary differs"):
        resume_covered_retirement(config, _confirmations(plan))
    with pytest.raises(RetirementError, match="summary differs"):
        covered_retirement_status(config)


def test_pre_scope_full_request_remains_resumable_and_reportable(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config)
    legacy_hash_payload = {
        "targets": [target.as_dict() for target in plan.targets],
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "archive_receipt_sha256": plan.archive_receipt_sha256,
        "retirement_disposition_sha256": plan.retirement_disposition_sha256,
    }
    legacy_hash = hashlib.sha256(
        json.dumps(
            legacy_hash_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    request = retirement_module._request_payload(plan, [])
    request["schema_version"] = 1
    request.pop("scope")
    request["retirement_plan_sha256"] = legacy_hash
    request_path = config.deletion_log_root / "retirement_request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert retirement_status(config)["status"] == "submitted"
    submitted = 0

    def fake_run(command, **kwargs):
        nonlocal submitted
        submitted += 1
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {7800 + submitted}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    resumed = resume_retirement(config, _confirmations(plan))
    assert len(resumed["jobs"]) == len(plan.targets)


def test_submission_requires_exact_confirmations_and_delegates_one_job_each(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config)
    wrong = RetirementConfirmations(
        dataset_root=str(config.output_root) + "-wrong",
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        archive_receipt_sha256=plan.archive_receipt_sha256,
        retirement_disposition_sha256=plan.retirement_disposition_sha256,
        confirm_retirement=True,
    )
    with pytest.raises(RetirementError, match="dataset root confirmation"):
        submit_retirement(plan, wrong)
    with pytest.raises(RetirementError, match="archive receipt confirmation"):
        submit_retirement(plan, _confirmations(plan, archive=False))
    wrong_disposition = RetirementConfirmations(
        dataset_root=str(config.output_root.resolve()),
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        archive_receipt_sha256=plan.archive_receipt_sha256,
        retirement_disposition_sha256="0" * 64,
        confirm_retirement=True,
    )
    with pytest.raises(RetirementError, match="disposition confirmation"):
        submit_retirement(plan, wrong_disposition)

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        job_id = 7000 + len(commands)
        return subprocess.CompletedProcess(
            command, 0, stdout=f"[INFO] Submitted deletion job: {job_id}\n", stderr=""
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    result = submit_retirement(plan, _confirmations(plan))

    assert result["status"] == "submitted"
    assert result["complete"] is False
    assert len(commands) == len(plan.targets)
    for command, target in zip(commands, plan.targets):
        assert command[:2] == ["bash", str(plan.deletion_launcher)]
        assert command[command.index("--target") + 1] == str(target.path)
        assert command[command.index("--confirm") + 1] == str(target.path)
        assert command[command.index("--allowed-root") + 1] == str(
            config.output_root.parent.resolve()
        )
        assert command[command.index("--log-dir") + 1] == str(
            config.deletion_log_root.resolve()
        )
        assert target.path.is_dir()

    request = json.loads(
        (config.deletion_log_root / "retirement_request.json").read_text()
    )
    assert request["status"] == "submitted"
    assert [job["job_id"] for job in request["jobs"]] == [
        str(7000 + index) for index in range(1, len(plan.targets) + 1)
    ]
    repeated = submit_retirement(plan, _confirmations(plan))
    assert repeated["jobs"] == result["jobs"]


def test_partial_submission_resumes_without_replanning_deleted_sources(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config)
    calls: list[str] = []

    def fail_second(command, **kwargs):
        target = command[command.index("--target") + 1]
        calls.append(target)
        if len(calls) == 2:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {8100 + len(calls)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fail_second
    )
    with pytest.raises(RetirementError, match="launcher failed"):
        submit_retirement(plan, _confirmations(plan))

    request_path = config.deletion_log_root / "retirement_request.json"
    request = json.loads(request_path.read_text())
    assert len(request["jobs"]) == 1
    first_target = plan.targets[0]
    shutil.rmtree(first_target.path)

    resumed_calls: list[str] = []

    def succeed(command, **kwargs):
        target = command[command.index("--target") + 1]
        resumed_calls.append(target)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {8200 + len(resumed_calls)}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", succeed
    )
    active_target = plan.targets[1]
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: f"123|active|/work|python worker.py {active_target.path}\n",
    )
    with pytest.raises(RetirementError, match="active Slurm job"):
        resume_retirement(config, _confirmations(plan))
    assert resumed_calls == []
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    resumed = resume_retirement(config, _confirmations(plan))

    assert str(first_target.path) not in resumed_calls
    assert len(resumed["jobs"]) == len(plan.targets)


def _write_terminal_receipt(path: Path, *, job_id: str, target, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "job_id",
                "status",
                "target",
                "target_id",
                "allowed_root",
                "started_at",
                "completed_at",
                "message",
            ]
        )
        writer.writerow(
            [
                job_id,
                status,
                str(target.path),
                f"{target.source_device}:{target.source_inode}",
                str(target.path.parent),
                "start",
                "end",
                "done",
            ]
        )


def test_status_requires_terminal_receipts_and_independent_absence(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config)

    submitted = 0

    def fake_run(command, **kwargs):
        nonlocal submitted
        submitted += 1
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {8000 + submitted}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    submit_retirement(plan, _confirmations(plan))
    assert retirement_status(config)["status"] == "submitted"

    receipt = config.deletion_log_root / "delete-directory-8001.receipt.tsv"
    _write_terminal_receipt(
        receipt, job_id="8001", target=plan.targets[0], status="complete"
    )
    present = retirement_status(config)
    assert present["status"] == "incomplete"
    assert present["targets"][0]["target_absent"] is False

    for index, target in enumerate(plan.targets, start=1):
        target_receipt = (
            config.deletion_log_root / f"delete-directory-{8000 + index}.receipt.tsv"
        )
        _write_terminal_receipt(
            target_receipt,
            job_id=str(8000 + index),
            target=target,
            status="complete",
        )
        shutil.rmtree(target.path)
    completed = retirement_status(config)
    assert completed["status"] == "complete"
    assert completed["complete"] is True
    assert (config.deletion_log_root / "retirement_complete.json").is_file()


def test_status_keeps_failed_or_missing_receipts_incomplete(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config)

    submitted = 0

    def fake_run(command, **kwargs):
        nonlocal submitted
        submitted += 1
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {9000 + submitted}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    submit_retirement(plan, _confirmations(plan))
    request_path = config.deletion_log_root / "retirement_request.json"
    request = json.loads(request_path.read_text())
    nested_index = next(
        index
        for index, target in enumerate(request["targets"])
        if target["key"] == "nested_analysis"
    )
    request["targets"][nested_index]["path"] = str(
        config.output_root.parent / "forged"
    )
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RetirementError, match="target binding"):
        retirement_status(config)
    nested_target = next(
        target for target in plan.targets if target.key == "nested_analysis"
    )
    request["targets"][nested_index]["path"] = str(nested_target.path)
    request_path.write_text(json.dumps(request), encoding="utf-8")

    nested_job = next(job for job in request["jobs"] if job["key"] == "nested_analysis")
    receipt = config.deletion_log_root / f"delete-directory-{nested_job['job_id']}.receipt.tsv"
    _write_terminal_receipt(
        receipt,
        job_id=str(nested_job["job_id"]),
        target=nested_target,
        status="failed",
    )
    shutil.rmtree(nested_target.path)

    result = retirement_status(config)
    assert result["status"] == "incomplete"
    nested_result = next(
        target for target in result["targets"] if target["key"] == "nested_analysis"
    )
    assert nested_result["receipt_status"] == "failed"


def test_resume_retries_only_a_valid_terminal_failed_job(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config)
    submitted = 0

    def initial_submit(command, **kwargs):
        nonlocal submitted
        submitted += 1
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"[INFO] Submitted deletion job: {9500 + submitted}\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher",
        initial_submit,
    )
    submit_retirement(plan, _confirmations(plan))
    failed_target = plan.targets[0]
    failed_receipt = config.deletion_log_root / "delete-directory-9501.receipt.tsv"
    _write_terminal_receipt(
        failed_receipt,
        job_id="9501",
        target=failed_target,
        status="failed",
    )

    retry_commands: list[list[str]] = []

    def retry_submit(command, **kwargs):
        retry_commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="[INFO] Submitted deletion job: 9901\n",
            stderr="",
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher",
        retry_submit,
    )
    result = resume_retirement(config, _confirmations(plan))

    assert len(retry_commands) == 1
    assert retry_commands[0][retry_commands[0].index("--target") + 1] == str(
        failed_target.path
    )
    retried = next(job for job in result["jobs"] if job["key"] == failed_target.key)
    assert retried["job_id"] == "9901"
    assert [attempt["job_id"] for attempt in retried["retry_history"]] == ["9501"]
