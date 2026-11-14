from __future__ import annotations

import json

import simnibs_segmentation_determinism.cli as cli


def _payload(capsys):
    return json.loads(capsys.readouterr().out)


def test_scope_cli_prints_stable_json_without_creating_output(
    configured_experiment, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "load_config", lambda args: configured_experiment)

    assert cli.main(["scope"]) == 0
    payload = _payload(capsys)

    assert payload["workflow"] == "simnibs_segmentation_determinism"
    assert payload["scientific_segmentation_tasks"] == 20
    assert payload["scheduler_tasks"] == 23
    assert not configured_experiment.output_root.exists()


def test_preflight_init_and_status_cli(configured_experiment, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda args: configured_experiment)

    assert cli.main(["preflight"]) == 0
    assert _payload(capsys)["status"] == "ready"
    assert cli.main(["init"]) == 0
    assert _payload(capsys)["status"] == "prepared"
    assert cli.main(["status"]) == 0
    assert _payload(capsys)["state"] == "prepared"


def test_cli_errors_are_machine_readable(configured_experiment, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda args: configured_experiment)

    assert cli.main(["run-repeat", "--arm", "bad", "--array-index", "0"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"


def test_install_cli_requires_confirmation_and_exposes_read_only_preflight(
    configured_experiment,
    dataset_receipt_writer,
    staging_completer,
    monkeypatch,
    capsys,
):
    config = configured_experiment
    staging_completer(config)
    dataset_receipt_writer(config)
    monkeypatch.setattr(cli, "load_config", lambda args: config)

    assert cli.main(["install-preflight"]) == 0
    assert _payload(capsys)["status"] == "ready"
    assert (
        cli.main(
            [
                "install-extension",
                "--confirm-dataset-root",
                str(config.dataset_root.parent),
            ]
        )
        == 2
    )
    assert "confirmation" in capsys.readouterr().err


def test_cli_staging_root_override_is_independent(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    staging = tmp_path / "staging"
    config = cli.load_config(
        cli.build_parser().parse_args(
            [
                "--dataset-root",
                str(dataset),
                "--staging-root",
                str(staging),
                "scope",
            ]
        )
    )
    assert config.dataset_root == dataset
    assert config.output_root == staging
