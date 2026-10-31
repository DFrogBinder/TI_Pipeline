from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from repeatability_dataset_consolidation.archive import archive_copy, archive_verify
from repeatability_dataset_consolidation.retirement import (
    RetirementConfirmations,
    RetirementError,
    plan_retirement,
    retirement_status,
    submit_retirement,
)
from repeatability_dataset_consolidation.settings import retirement_allowlist
from repeatability_dataset_consolidation.workflow import (
    complete_dataset,
    copy_dataset,
    prepare,
    verify_dataset,
)


@pytest.fixture
def retirement_ready(miniature_config, monkeypatch):
    for key, root in miniature_config.archive_sources.items():
        (root / f"{key}.txt").write_text(f"{key}\n", encoding="utf-8")
    prepare(miniature_config)
    copy_dataset(miniature_config)
    assert verify_dataset(miniature_config, checksum=True)["status"] == "matched"
    complete_dataset(miniature_config)
    archive_copy(miniature_config)
    assert archive_verify(miniature_config, checksum=True)["status"] == "complete"
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: "",
    )
    return miniature_config


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

    subset = plan_retirement(config, selected=("paper_analysis", "legacy_v3_left"))
    assert [target.key for target in subset.targets] == [
        "paper_analysis",
        "legacy_v3_left",
    ]
    with pytest.raises(RetirementError, match="unknown retirement key"):
        plan_retirement(config, selected=("arbitrary_path",))
    with pytest.raises(RetirementError, match="At least one retirement key"):
        plan_retirement(config, selected=())


def test_retirement_plan_rejects_tampering_and_active_jobs(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    completion = config.output_root / "_consolidation/complete.json"
    original = completion.read_bytes()
    completion.write_text('{"status":"forged"}\n', encoding="utf-8")
    with pytest.raises(RetirementError, match="dataset completion receipt"):
        plan_retirement(config, selected=("paper_analysis",))
    completion.write_bytes(original)

    target = config.sources.paper_analysis.resolve()
    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._scheduler_snapshot",
        lambda: f"123|job|{target}|python worker.py {target}\n",
    )
    with pytest.raises(RetirementError, match="active Slurm job"):
        plan_retirement(config, selected=("paper_analysis",))


def _confirmations(plan, *, archive: bool = True) -> RetirementConfirmations:
    return RetirementConfirmations(
        dataset_root=str(plan.config.output_root.resolve()),
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        archive_receipt_sha256=(plan.archive_receipt_sha256 if archive else None),
        confirm_retirement=True,
    )


def test_submission_requires_exact_confirmations_and_delegates_one_job_each(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config, selected=("paper_analysis", "legacy_v3_left"))
    wrong = RetirementConfirmations(
        dataset_root=str(config.output_root) + "-wrong",
        dataset_receipt_sha256=plan.dataset_receipt_sha256,
        archive_receipt_sha256=plan.archive_receipt_sha256,
        confirm_retirement=True,
    )
    with pytest.raises(RetirementError, match="dataset root confirmation"):
        submit_retirement(plan, wrong)
    with pytest.raises(RetirementError, match="archive receipt confirmation"):
        submit_retirement(plan, _confirmations(plan, archive=False))

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
    assert len(commands) == 2
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
    assert [job["job_id"] for job in request["jobs"]] == ["7001", "7002"]
    with pytest.raises(RetirementError, match="request already exists"):
        submit_retirement(plan, _confirmations(plan))


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
    plan = plan_retirement(config, selected=("paper_analysis",))

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="[INFO] Submitted deletion job: 8001\n", stderr=""
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

    shutil.rmtree(plan.targets[0].path)
    completed = retirement_status(config)
    assert completed["status"] == "complete"
    assert completed["complete"] is True
    assert (config.deletion_log_root / "retirement_complete.json").is_file()


def test_status_keeps_failed_or_missing_receipts_incomplete(
    retirement_ready, monkeypatch
) -> None:
    config = retirement_ready
    plan = plan_retirement(config, selected=("nested_analysis",))

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="[INFO] Submitted deletion job: 8002\n", stderr=""
        )

    monkeypatch.setattr(
        "repeatability_dataset_consolidation.retirement._run_launcher", fake_run
    )
    submit_retirement(plan, _confirmations(plan))
    request_path = config.deletion_log_root / "retirement_request.json"
    request = json.loads(request_path.read_text())
    request["targets"][0]["path"] = str(config.output_root.parent / "forged")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RetirementError, match="target binding"):
        retirement_status(config)
    request["targets"][0]["path"] = str(plan.targets[0].path)
    request_path.write_text(json.dumps(request), encoding="utf-8")

    receipt = config.deletion_log_root / "delete-directory-8002.receipt.tsv"
    _write_terminal_receipt(
        receipt, job_id="8002", target=plan.targets[0], status="failed"
    )
    shutil.rmtree(plan.targets[0].path)

    result = retirement_status(config)
    assert result["status"] == "incomplete"
    assert result["targets"][0]["receipt_status"] == "failed"
