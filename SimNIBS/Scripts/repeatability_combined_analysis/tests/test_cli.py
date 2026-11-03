from __future__ import annotations

import json
from pathlib import Path

from repeatability_combined_analysis import cli


def test_scope_prints_exact_numeric_scope_without_writes(
    analysis_config, capsys
) -> None:
    assert cli.main(["scope"], config=analysis_config) == 0
    output = capsys.readouterr().out
    assert "subjects: 10" in output
    assert "versions: 2" in output
    assert "targets: 2" in output
    assert "conditions: 2" in output
    assert "repeats per cell: 40" in output
    assert "participant cells: 80" in output
    assert "run-level rows: 3200" in output
    assert "meshing tasks: 0" in output
    assert "FEM simulation tasks: 0" in output
    assert not analysis_config.output_root.exists()


def test_preflight_cli_is_read_only(analysis_config, capsys) -> None:
    assert cli.main(["preflight"], config=analysis_config) == 0
    payload = json.loads(capsys.readouterr().out.split("\n{", 1)[1].join(("{", "")))
    assert payload["status"] == "ready"
    assert not analysis_config.output_root.exists()


def test_prepare_cli_requires_exact_confirmation(analysis_config, capsys) -> None:
    assert cli.main(["prepare"], config=analysis_config) == 2
    error = json.loads(capsys.readouterr().err)
    assert "confirm-output-root" in error["error"]
    assert not analysis_config.output_root.exists()

    assert (
        cli.main(
            [
                "prepare",
                f"--confirm-output-root={analysis_config.output_root}",
            ],
            config=analysis_config,
        )
        == 0
    )
    assert analysis_config.output_root.is_dir()


def test_execute_cli_cannot_bypass_ownership(analysis_config, capsys) -> None:
    analysis_config.output_root.mkdir()
    assert cli.main(["execute"], config=analysis_config) == 2
    error = json.loads(capsys.readouterr().err)
    assert "unowned" in error["error"].lower()


def test_status_and_rebind_cli_are_read_only(
    analysis_config, tmp_path: Path, capsys
) -> None:
    assert cli.main(["status"], config=analysis_config) == 0
    assert '"state": "absent"' in capsys.readouterr().out

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    before = sorted(dataset.rglob("*"))
    assert (
        cli.main(
            ["rebind-preflight", f"--dataset-root={dataset}"],
            config=analysis_config,
        )
        == 2
    )
    capsys.readouterr()
    assert sorted(dataset.rglob("*")) == before

