from __future__ import annotations

from dataclasses import replace
import json

import pytest

from repeatability_dataset_consolidation import cli
from repeatability_dataset_consolidation.retirement import RetirementPlan


def test_cli_preflight_prints_scope_and_is_read_only(miniature_config, capsys) -> None:
    result = cli.main(["preflight"], config=miniature_config)

    assert result == 0
    output = capsys.readouterr().out
    assert "subjects: 2" in output
    assert "subject-condition cells: 16" in output
    assert "simulation records: 32" in output
    assert '"status": "ready"' in output
    assert not miniature_config.output_root.exists()


def test_cli_dispatches_all_nonretirement_modes(
    miniature_config, monkeypatch, capsys
) -> None:
    calls: list[tuple[str, object]] = []

    def operation(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs))
            status = "matched" if name == "verify_dataset" else "ok"
            return {"status": status, "name": name}

        return run

    for name in (
        "prepare",
        "copy_dataset",
        "verify_dataset",
        "complete_dataset",
        "dataset_status",
        "archive_preflight",
        "archive_copy",
        "archive_verify",
        "retirement_status",
    ):
        monkeypatch.setattr(cli, name, operation(name))

    root = str(miniature_config.output_root)
    archive = str(miniature_config.shared_archive_root)
    modes = (
        (["prepare"], "prepare"),
        (["copy", f"--confirm-output-root={root}"], "copy_dataset"),
        (["verify"], "verify_dataset"),
        (["checksum"], "complete_dataset"),
        (["status"], "dataset_status"),
        (["archive-preflight"], "archive_preflight"),
        (["archive-copy", f"--confirm-archive-root={archive}"], "archive_copy"),
        (["archive-verify"], "archive_verify"),
        (["archive-checksum"], "archive_verify"),
        (["retire-status"], "retirement_status"),
    )
    for arguments, expected in modes:
        calls.clear()
        assert cli.main(arguments, config=miniature_config) == 0
        assert calls[-1][0] == expected
    output = capsys.readouterr().out
    assert "jobs submitted: 0" in output


def test_cli_requires_exact_copy_confirmations(miniature_config, monkeypatch) -> None:
    monkeypatch.setattr(cli, "copy_dataset", lambda config: {"status": "unexpected"})
    monkeypatch.setattr(cli, "archive_copy", lambda config: {"status": "unexpected"})

    assert cli.main(["copy"], config=miniature_config) == 2
    assert (
        cli.main(
            ["copy", "--confirm-output-root=/wrong"], config=miniature_config
        )
        == 2
    )
    assert cli.main(["archive-copy"], config=miniature_config) == 2


def test_cli_retirement_defaults_to_plan_and_requires_all_submit_gates(
    miniature_config, monkeypatch, capsys
) -> None:
    fake_plan = RetirementPlan(
        config=miniature_config,
        targets=(),
        dry_run=True,
        plan_sha256="a" * 64,
        dataset_receipt=miniature_config.output_root / "complete.json",
        dataset_receipt_sha256="b" * 64,
        archive_receipt=None,
        archive_receipt_sha256=None,
        scheduler_snapshot_sha256="c" * 64,
        deletion_launcher=miniature_config.output_root / "delete.sh",
    )
    submissions: list[object] = []
    monkeypatch.setattr(cli, "plan_retirement", lambda config, selected=None: fake_plan)
    monkeypatch.setattr(
        cli,
        "submit_retirement",
        lambda plan, confirmations: submissions.append(confirmations)
        or {"status": "submitted"},
    )

    assert cli.main(["retire-plan"], config=miniature_config) == 0
    assert submissions == []
    assert (
        cli.main(
            [
                "retire-submit",
                f"--confirm-output-root={miniature_config.output_root}",
                f"--dataset-receipt-sha256={'b' * 64}",
            ],
            config=miniature_config,
        )
        == 2
    )
    assert submissions == []
    assert (
        cli.main(
            [
                "retire-submit",
                f"--confirm-output-root={miniature_config.output_root}",
                f"--dataset-receipt-sha256={'b' * 64}",
                "--confirm-retirement",
            ],
            config=miniature_config,
        )
        == 0
    )
    assert len(submissions) == 1
    output = capsys.readouterr().out
    assert "retirement default: plan only" in output


def test_cli_resumes_an_existing_retirement_request_without_replanning(
    miniature_config, monkeypatch
) -> None:
    request = miniature_config.deletion_log_root / "retirement_request.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "plan_retirement",
        lambda *args, **kwargs: pytest.fail("resume must not rebuild the live plan"),
    )
    monkeypatch.setattr(
        cli,
        "resume_retirement",
        lambda config, confirmations: calls.append(confirmations)
        or {"status": "submitted"},
    )

    assert (
        cli.main(
            [
                "retire-submit",
                f"--confirm-output-root={miniature_config.output_root}",
                f"--dataset-receipt-sha256={'a' * 64}",
                f"--archive-receipt-sha256={'b' * 64}",
                "--confirm-retirement",
            ],
            config=miniature_config,
        )
        == 0
    )
    assert len(calls) == 1


def test_production_scope_block_reports_full_dataset(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "preflight", lambda config: {"status": "ready"})
    config = cli.production_config()

    assert cli.main(["preflight"], config=config) == 0
    output = capsys.readouterr().out
    assert "subjects: 10" in output
    assert "subject-condition cells: 80" in output
    assert "simulation records: 3200" in output
    assert "jobs submitted: 0" in output

