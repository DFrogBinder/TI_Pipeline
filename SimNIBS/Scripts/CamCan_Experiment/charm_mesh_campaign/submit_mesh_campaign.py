"""Build and optionally submit a full configuration-driven Slurm campaign."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from .config import CampaignConfig
from .tasks import read_task_manifest


WORKER = Path(__file__).with_name("mesh_campaign_array.slurm").resolve()


def _ready_scope(config: CampaignConfig) -> int:
    if not config.output.preflight_summary.is_file():
        raise ValueError("preflight summary is missing; run preflight first")
    summary = json.loads(config.output.preflight_summary.read_text(encoding="utf-8"))
    if summary.get("status") != "ready":
        raise ValueError("campaign preflight is not ready")
    if summary.get("config_sha256") != config.sha256:
        raise ValueError("preflight summary belongs to a different configuration")
    tasks = read_task_manifest(config.output.task_manifest)
    if len(tasks) != config.execution.expected_tasks:
        raise ValueError("task manifest count differs from configured full scope")
    if any(
        task.status != "ready" or task.config_sha256 != config.sha256 for task in tasks
    ):
        raise ValueError("task manifest contains blocked or stale rows")
    return len(tasks)


def build_sbatch_command(
    config: CampaignConfig, *, ready_tasks: int, sbatch_bin: str = "sbatch"
) -> list[str]:
    if ready_tasks != config.execution.expected_tasks:
        raise ValueError(
            f"refusing partial scope: {ready_tasks} ready tasks, "
            f"expected {config.execution.expected_tasks}"
        )
    execution = config.execution
    array_elements = math.ceil(ready_tasks / execution.workers_per_array_task)
    threads = execution.cpus_per_task // execution.workers_per_array_task
    log_dir = config.output.root / "logs"
    exports = {
        "TI_CHARM_MESH_CONFIG": str(config.source_path),
        "TI_CHARM_MESH_PACKAGE_ROOT": str(Path(__file__).resolve().parents[1]),
        "TI_CHARM_MESH_LOG_ROOT": str(config.output.root / "logs"),
        "TI_CHARM_MESH_SIMNIBS_MODULE": config.software.module,
        "TI_CHARM_MESH_PYTHON": config.software.python,
        "TI_CHARM_MESH_TASKS": str(ready_tasks),
        "TI_CHARM_MESH_MAX_RETRIES": str(execution.max_retries),
        "TI_CHARM_MESH_WORKERS_PER_ARRAY_TASK": str(
            execution.workers_per_array_task
        ),
        "TI_CHARM_MESH_THREADS_PER_WORKER": str(threads),
        "TI_CHARM_MESH_LOCAL_STAGING": "1" if execution.local_staging else "0",
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    return [
        sbatch_bin,
        "--parsable",
        f"--job-name=charm_mesh_{config.campaign_id}",
        f"--partition={execution.partition}",
        f"--cpus-per-task={execution.cpus_per_task}",
        f"--mem={execution.memory}",
        f"--time={execution.time_limit}",
        f"--array=0-{array_elements - 1}%{execution.max_concurrent}",
        f"--output={log_dir}/charm-mesh-%A_%a.out",
        f"--export={export_arg}",
        str(WORKER),
    ]


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def submit_campaign(
    config: CampaignConfig, *, submit: bool = False, sbatch_bin: str = "sbatch"
) -> dict[str, object]:
    """Return the full command, submitting only when explicitly requested."""

    ready_tasks = _ready_scope(config)
    command = build_sbatch_command(
        config, ready_tasks=ready_tasks, sbatch_bin=sbatch_bin
    )
    array_elements = math.ceil(
        ready_tasks / config.execution.workers_per_array_task
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "dry_run",
        "submitted": False,
        "campaign_id": config.campaign_id,
        "config_sha256": config.sha256,
        "tasks": ready_tasks,
        "array_elements": array_elements,
        "workers_per_array_task": config.execution.workers_per_array_task,
        "command": command,
        "segmentation_rerun": False,
        "simulation_requested": False,
    }
    if not submit:
        return payload
    (config.output.root / "logs").mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    raw_job_id = completed.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", raw_job_id):
        raise RuntimeError(f"could not parse sbatch job ID: {completed.stdout!r}")
    payload.update(
        {
            "status": "submitted",
            "submitted": True,
            "job_id": raw_job_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    _write_json_atomic(
        config.output.root / "campaign" / "submission_receipt.json", payload
    )
    return payload
