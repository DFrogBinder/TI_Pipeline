import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def campaign_payload(tmp_path: Path, expected_tasks: int = 2) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "input": {
            "manifest": "inputs/collection.tsv",
            "delimiter": "tab",
            "ready_status": "complete",
            "columns": {
                "subject_id": "subject",
                "realization_id": "repeat",
                "label_path": "label",
                "label_sha256": "sha256",
                "label_bytes": "bytes",
                "status": "status",
            },
            "segmentation_version": "4.0.1",
            "excluded_task_keys": [],
        },
        "output": {
            "root": "outputs",
            "task_manifest": "campaign/tasks.tsv",
            "preflight_summary": "campaign/preflight.json",
            "mesh_template": "subjects/{subject_id}/{realization_id}/{task_key}.msh",
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
        "quality": {"require_tetrahedra": True, "required_tissue_tags": []},
        "execution": {
            "expected_tasks": expected_tasks,
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


def make_config(tmp_path: Path, rows: list[dict], *, expected_tasks: int | None = None):
    from charm_mesh_campaign.config import load_campaign_config

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest = inputs / "collection.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    payload = campaign_payload(tmp_path, expected_tasks or len(rows))
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_campaign_config(config_path)


def map_row(tmp_path: Path, subject: str, repeat: str, content: bytes) -> dict:
    path = tmp_path / f"{subject}_{repeat}.nii.gz"
    path.write_bytes(content)
    return {
        "subject": subject,
        "repeat": repeat,
        "label": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": str(len(content)),
        "status": "complete",
    }


def test_builds_one_collision_safe_task_per_segmentation_realization(tmp_path):
    from charm_mesh_campaign.tasks import build_task_manifest, read_task_manifest

    rows = [
        map_row(tmp_path, "sub-CC000001", "seg_001", b"first"),
        map_row(tmp_path, "sub-CC000001", "seg_002", b"second"),
    ]
    config = make_config(tmp_path, rows)

    summary = build_task_manifest(config)
    tasks = read_task_manifest(config.output.task_manifest)

    assert summary["status"] == "ready"
    assert summary["tasks"] == 2
    assert [task.task_id for task in tasks] == [0, 1]
    assert [task.task_key for task in tasks] == [
        "sub-CC000001__seg_001",
        "sub-CC000001__seg_002",
    ]
    assert len({task.mesh_path for task in tasks}) == 2
    assert all(task.status == "ready" for task in tasks)
    assert all(task.config_sha256 == config.sha256 for task in tasks)
    assert all(task.mesh_path.is_relative_to(config.output.root) for task in tasks)


def test_without_realization_column_uses_explicit_single_realization_id(tmp_path):
    from charm_mesh_campaign.config import load_campaign_config
    from charm_mesh_campaign.tasks import build_task_manifest, read_task_manifest

    row = map_row(tmp_path, "sub-CC000001", "unused", b"one")
    row.pop("repeat")
    config = make_config(tmp_path, [row], expected_tasks=1)
    payload = json.loads(config.source_path.read_text())
    payload["input"]["columns"]["realization_id"] = None
    config.source_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_campaign_config(config.source_path)

    build_task_manifest(config)
    task = read_task_manifest(config.output.task_manifest)[0]
    assert task.realization_id == "seg_001"


@pytest.mark.parametrize("defect", ["hash", "bytes", "status", "missing", "symlink"])
def test_blocks_invalid_input_evidence(tmp_path, defect):
    from charm_mesh_campaign.tasks import build_task_manifest, read_task_manifest

    row = map_row(tmp_path, "sub-CC000001", "seg_001", b"content")
    if defect == "hash":
        row["sha256"] = "0" * 64
    elif defect == "bytes":
        row["bytes"] = "999"
    elif defect == "status":
        row["status"] = "failed"
    elif defect == "missing":
        Path(row["label"]).unlink()
    else:
        target = Path(row["label"])
        link = tmp_path / "linked-label.nii.gz"
        link.symlink_to(target)
        row["label"] = str(link)
    config = make_config(tmp_path, [row], expected_tasks=1)

    summary = build_task_manifest(config)
    task = read_task_manifest(config.output.task_manifest)[0]

    assert summary["status"] == "blocked"
    assert task.status == "blocked"
    assert task.message != "ready"


def test_exact_expected_count_is_a_submission_gate(tmp_path):
    from charm_mesh_campaign.tasks import build_task_manifest, read_task_manifest

    row = map_row(tmp_path, "sub-CC000001", "seg_001", b"one")
    config = make_config(tmp_path, [row], expected_tasks=2)

    summary = build_task_manifest(config)

    assert summary["status"] == "blocked"
    assert summary["tasks_expected"] == 2
    assert read_task_manifest(config.output.task_manifest)[0].status == "blocked"


def test_explicit_exclusion_is_recorded_and_applied_before_expected_count(tmp_path):
    from charm_mesh_campaign.config import load_campaign_config
    from charm_mesh_campaign.tasks import build_task_manifest, read_task_manifest

    rows = [
        map_row(tmp_path, "sub-CC000001", "seg_001", b"one"),
        map_row(tmp_path, "sub-CC000002", "seg_001", b"two"),
    ]
    config = make_config(tmp_path, rows, expected_tasks=1)
    payload = json.loads(config.source_path.read_text())
    payload["input"]["excluded_task_keys"] = ["sub-CC000002__seg_001"]
    config.source_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_campaign_config(config.source_path)

    summary = build_task_manifest(config)

    assert summary["status"] == "ready"
    assert summary["excluded_task_keys"] == ["sub-CC000002__seg_001"]
    assert [task.task_key for task in read_task_manifest(config.output.task_manifest)] == [
        "sub-CC000001__seg_001"
    ]


@pytest.mark.parametrize("collision", ["identity", "output"])
def test_rejects_duplicate_identity_or_rendered_output(tmp_path, collision):
    from charm_mesh_campaign.tasks import build_task_manifest

    if collision == "identity":
        rows = [
            map_row(tmp_path, "sub-CC000001", "seg_001", b"one"),
            map_row(tmp_path, "sub-CC000001", "seg_001", b"two"),
        ]
    else:
        rows = [
            map_row(tmp_path, "sub-CC000001", "seg_001", b"one"),
            map_row(tmp_path, "sub-CC000002", "seg_001", b"two"),
        ]
    config = make_config(tmp_path, rows)
    if collision == "output":
        unsafe_output = replace(
            config.output,
            mesh_template="subjects/shared.msh",
            result_template="results/{task_key}.json",
        )
        config = replace(config, output=unsafe_output)
    with pytest.raises(ValueError, match="duplicate (task_key|mesh output)"):
        build_task_manifest(config)


def test_rendered_paths_cannot_escape_output_root(tmp_path, monkeypatch):
    from charm_mesh_campaign.tasks import _render_output_path

    output_root = tmp_path / "outputs"
    with pytest.raises(ValueError, match="outside output root"):
        _render_output_path(
            output_root,
            "{task_key}.msh",
            {"task_key": "../escape", "task_id": 0, "subject_id": "s", "realization_id": "r"},
        )
