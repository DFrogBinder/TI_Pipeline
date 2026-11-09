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
