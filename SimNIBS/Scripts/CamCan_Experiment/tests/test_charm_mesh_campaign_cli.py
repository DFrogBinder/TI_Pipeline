import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "charm_mesh_campaign"
sys.path.insert(0, str(ROOT))


def fake_config(tmp_path: Path, *, workers: int = 2):
    config_path = tmp_path / "campaign.json"
    config_path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        source_path=config_path,
        campaign_id="test-campaign",
        sha256="a" * 64,
        output=SimpleNamespace(
            root=tmp_path / "outputs",
            task_manifest=tmp_path / "outputs/campaign/tasks.tsv",
            preflight_summary=tmp_path / "outputs/campaign/preflight.json",
        ),
        software=SimpleNamespace(
            module="SimNIBS/4.0.1-foss-2023a", python="python"
        ),
        execution=SimpleNamespace(
            expected_tasks=3,
            partition="sheffield",
            cpus_per_task=8,
            memory="32G",
            time_limit="08:00:00",
            max_concurrent=50,
            workers_per_array_task=workers,
            local_staging=False,
            max_retries=2,
        ),
    )


def test_submission_command_uses_explicit_config_resources_and_packing(tmp_path):
    from charm_mesh_campaign.submit_mesh_campaign import build_sbatch_command

    config = fake_config(tmp_path)
    command = build_sbatch_command(config, ready_tasks=3)
    rendered = " ".join(command)

    assert "--array=0-1%50" in command
    assert "--partition=sheffield" in command
    assert "--cpus-per-task=8" in command
    assert "--mem=32G" in command
    assert "--time=08:00:00" in command
    assert "TI_CHARM_MESH_SIMNIBS_MODULE=SimNIBS/4.0.1-foss-2023a" in rendered
    assert "TI_CHARM_MESH_PYTHON=python" in rendered
    assert f"TI_CHARM_MESH_LOG_ROOT={config.output.root / 'logs'}" in rendered
    assert "TI_CHARM_MESH_WORKERS_PER_ARRAY_TASK=2" in rendered
    assert "TI_CHARM_MESH_THREADS_PER_WORKER=4" in rendered


def test_submit_is_dry_run_by_default(tmp_path, monkeypatch):
    from charm_mesh_campaign import submit_mesh_campaign as submission

    config = fake_config(tmp_path, workers=1)
    monkeypatch.setattr(submission, "_ready_scope", lambda value: 3)
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("submitted")),
    )

    payload = submission.submit_campaign(config, submit=False)

    assert payload["status"] == "dry_run"
    assert payload["submitted"] is False
    assert payload["tasks"] == 3
    assert payload["array_elements"] == 3


def test_explicit_submit_parses_job_id_and_writes_receipt(tmp_path, monkeypatch):
    from charm_mesh_campaign import submit_mesh_campaign as submission

    config = fake_config(tmp_path, workers=1)
    monkeypatch.setattr(submission, "_ready_scope", lambda value: 3)
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12345;stanage\n"),
    )

    payload = submission.submit_campaign(config, submit=True)

    assert payload["status"] == "submitted"
    assert payload["job_id"] == "12345"
    receipt = json.loads(
        (config.output.root / "campaign/submission_receipt.json").read_text()
    )
    assert receipt["config_sha256"] == "a" * 64


def test_legacy_worker_module_is_parameterized_with_identical_default():
    text = (
        ROOT
        / "charm_segmentation_batch/mesh_collected_segmentations_array.slurm"
    ).read_text(encoding="utf-8")
    assert (
        'SIMNIBS_MODULE="${TI_CHARM_MESH_SIMNIBS_MODULE:-SimNIBS/4.0.1-foss-2023a}"'
        in text
    )
    assert 'module load "${SIMNIBS_MODULE}"' in text


def test_new_worker_and_legacy_launchers_pass_bash_syntax():
    paths = [
        PACKAGE / "mesh_campaign_array.slurm",
        ROOT / "charm_segmentation_batch/mesh_collected_segmentations_array.slurm",
        ROOT / "charm_segmentation_batch/submit_collected_meshes.sh",
    ]
    for path in paths:
        subprocess.run(["bash", "-n", str(path)], check=True)
    generic_worker = (PACKAGE / "mesh_campaign_array.slurm").read_text(
        encoding="utf-8"
    )
    assert 'if wait "${pid}"; then' in generic_worker
    assert "TASK_EXIT=$?" in generic_worker
