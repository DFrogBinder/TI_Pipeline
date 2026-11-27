import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def literal_config() -> dict:
    return {
        "schema_version": 1,
        "campaign_id": "raw-charm-401",
        "input": {
            "manifest": "inputs/collection.tsv",
            "delimiter": "tab",
            "ready_status": "complete",
            "columns": {
                "subject_id": "subject",
                "realization_id": None,
                "label_path": "collected_map",
                "label_sha256": "sha256",
                "label_bytes": "bytes",
                "status": "status",
            },
            "segmentation_version": "4.0.1",
            "excluded_task_keys": [],
        },
        "output": {
            "root": "outputs",
            "task_manifest": "campaign/task_manifest.tsv",
            "preflight_summary": "campaign/preflight.json",
            "mesh_template": (
                "subjects/{subject_id}/{realization_id}/anat/"
                "m2m_{subject_id}/{subject_id}.msh"
            ),
            "result_template": "results/{task_key}.json",
        },
        "software": {
            "backend": "direct_charm_create_mesh",
            "expected_version": "4.0.1",
            "module": "SimNIBS/4.0.1-foss-2023a",
            "python": "python",
            "settings_path": None,
            "settings_policy": "installation_charm_ini",
        },
        "quality": {
            "require_tetrahedra": True,
            "required_tissue_tags": [],
        },
        "execution": {
            "expected_tasks": 2,
            "partition": "sheffield",
            "cpus_per_task": 8,
            "memory": "32G",
            "time_limit": "08:00:00",
            "max_concurrent": 50,
            "workers_per_array_task": 1,
            "local_staging": False,
            "max_retries": 2,
        },
        "resume": {"reuse_validated": True, "overwrite": False},
    }


def write_config(tmp_path: Path, payload: dict, name: str = "campaign.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_loads_complete_config_and_resolves_paths_relative_to_config(tmp_path):
    from charm_mesh_campaign.config import load_campaign_config

    config_path = write_config(tmp_path, literal_config())
    config = load_campaign_config(config_path)

    assert config.campaign_id == "raw-charm-401"
    assert config.input.manifest == (tmp_path / "inputs/collection.tsv").resolve()
    assert config.output.root == (tmp_path / "outputs").resolve()
    assert config.output.task_manifest == (
        tmp_path / "outputs/campaign/task_manifest.tsv"
    ).resolve()
    assert config.software.expected_version == "4.0.1"
    assert config.software.settings_path is None
    assert config.execution.expected_tasks == 2
    assert len(config.sha256) == 64


def test_explicit_settings_path_is_resolved_relative_to_config(tmp_path):
    from charm_mesh_campaign.config import load_campaign_config

    payload = literal_config()
    payload["software"]["settings_path"] = "settings/charm.ini"
    payload["software"]["settings_policy"] = "explicit_file"
    config = load_campaign_config(write_config(tmp_path, payload))

    assert config.software.settings_path == (tmp_path / "settings/charm.ini").resolve()


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update({"unexpected": 1}), "unknown field"),
        (lambda value: value.pop("software"), "missing field"),
        (lambda value: value.update({"schema_version": 2}), "schema_version"),
        (
            lambda value: value["software"].update({"backend": "meshmesh"}),
            "backend",
        ),
        (
            lambda value: value["software"].update(
                {"settings_policy": "installation_charm_ini", "settings_path": "x.ini"}
            ),
            "settings_path",
        ),
        (
            lambda value: value["execution"].update({"workers_per_array_task": 3}),
            "divide",
        ),
    ],
)
def test_rejects_invalid_or_ambiguous_configuration(tmp_path, mutator, match):
    from charm_mesh_campaign.config import load_campaign_config

    payload = literal_config()
    mutator(payload)
    with pytest.raises(ValueError, match=match):
        load_campaign_config(write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "template",
    [
        "/tmp/escaped/{task_key}.msh",
        "../escaped/{task_key}.msh",
        "subjects/{unknown}.msh",
        "subjects/static.msh",
    ],
)
def test_rejects_unsafe_or_non_unique_output_templates(tmp_path, template):
    from charm_mesh_campaign.config import load_campaign_config

    payload = literal_config()
    payload["output"]["mesh_template"] = template
    with pytest.raises(ValueError, match="mesh_template"):
        load_campaign_config(write_config(tmp_path, payload))


def test_configuration_hash_is_deterministic_and_changes_with_meaning(tmp_path):
    from charm_mesh_campaign.config import load_campaign_config

    payload = literal_config()
    first = load_campaign_config(write_config(tmp_path, payload, "first.json"))
    reordered = {key: payload[key] for key in reversed(payload)}
    second = load_campaign_config(write_config(tmp_path, reordered, "second.json"))
    changed_payload = copy.deepcopy(payload)
    changed_payload["execution"]["memory"] = "64G"
    changed = load_campaign_config(write_config(tmp_path, changed_payload, "changed.json"))

    assert first.sha256 == second.sha256
    assert changed.sha256 != first.sha256


def test_does_not_expand_environment_variables_in_paths(tmp_path, monkeypatch):
    from charm_mesh_campaign.config import load_campaign_config

    monkeypatch.setenv("CAMPAIGN_INPUT", "/outside/input.tsv")
    payload = literal_config()
    payload["input"]["manifest"] = "$CAMPAIGN_INPUT"
    config = load_campaign_config(write_config(tmp_path, payload))

    assert config.input.manifest == (tmp_path / "$CAMPAIGN_INPUT").resolve()
