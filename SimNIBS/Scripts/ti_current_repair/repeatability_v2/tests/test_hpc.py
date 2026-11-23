from __future__ import annotations

import json
from pathlib import Path
import subprocess

from ti_current_repair.repeatability_v2.hpc import (
    build_sbatch_command,
    task_for_stage,
)
from ti_current_repair.repeatability_v2.orchestration import (
    build_production_stage_plan,
)
from ti_current_repair.repeatability_v2.worker import simulation_manifest_path


def test_task_mapping_covers_exact_full_scope() -> None:
    left_first = task_for_stage("left_hippocampus_remesh", 0)
    left_last = task_for_stage("left_hippocampus_remesh", 399)
    right_fixed = task_for_stage("right_m1_fixed", 40)
    cache_last = task_for_stage("fixed_cache_preparation", 19)

    assert left_first == {
        "subject_index": 0,
        "target": "left_hippocampus",
        "condition": "remesh",
        "repeat": 1,
    }
    assert left_last == {
        "subject_index": 9,
        "target": "left_hippocampus",
        "condition": "remesh",
        "repeat": 40,
    }
    assert right_fixed == {
        "subject_index": 1,
        "target": "right_m1",
        "condition": "fixed_mesh",
        "repeat": 1,
    }
    assert cache_last == {"subject_index": 9, "target": "right_m1"}


def test_sbatch_command_encodes_array_resources_and_afterok_dependencies(
    tmp_path: Path,
) -> None:
    stage = build_production_stage_plan()[4]
    command = build_sbatch_command(
        stage=stage,
        worker_script=tmp_path / "worker.slurm",
        dependency_job_ids=("123", "456"),
        profile=tmp_path / "profile.json",
    )

    assert command[:2] == ["sbatch", "--parsable"]
    assert "--array=0-399%50" in command
    assert "--cpus-per-task=8" in command
    assert "--mem=32G" in command
    assert "--time=08:00:00" in command
    assert "--dependency=afterok:123:456" in command
    assert any(value.startswith("--export=ALL,V2_STAGE=left_hippocampus_fixed") for value in command)


def test_hpc_shell_scripts_parse_and_do_not_name_simnibs401() -> None:
    package = Path(__file__).resolve().parents[1]
    submit = package / "hpc" / "submit.sh"
    worker = package / "hpc" / "worker.slurm"

    assert subprocess.run(["bash", "-n", str(submit)], check=False).returncode == 0
    assert subprocess.run(["bash", "-n", str(worker)], check=False).returncode == 0
    assert "4.0.1" not in submit.read_text(encoding="utf-8")
    assert "4.0.1" not in worker.read_text(encoding="utf-8")
    assert "SIMNIBS_MODULE" in submit.read_text(encoding="utf-8")
    assert "EXPECTED_SIMNIBS_VERSION" in submit.read_text(encoding="utf-8")
    assert "mkdir -p \"${V2_LOG_ROOT}\"" in submit.read_text(encoding="utf-8")
    assert '"--output=${V2_LOG_ROOT}/%x-%A_%a.out"' in submit.read_text(
        encoding="utf-8"
    )


def test_worker_manifest_path_matches_condition_and_repeat_layout(tmp_path: Path) -> None:
    path = simulation_manifest_path(
        output_root=tmp_path,
        subject="sub-CC110174",
        target="right_m1",
        condition="fixed_mesh",
        repeat=40,
    )
    assert path == (
        tmp_path
        / "sub-CC110174"
        / "right_m1"
        / "fixed_mesh"
        / "repeat_040"
        / "task_manifest.json"
    )
