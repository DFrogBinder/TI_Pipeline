#!/usr/bin/env python3
"""Staged defacing repeatability experiment pipeline.

This module intentionally keeps the orchestration layer dependency-light.  HPC
stages exchange JSON manifests and use Slurm arrays, while local stages create
inputs, provenance records, summaries, and smoke-checkable status reports.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, Iterable, Mapping, Sequence


PIPELINE_STATE_DIR = "_pipeline"
ANALYSIS_DIR = "_analysis"
DEFAULT_SUBJECT = "sub-001"
METRIC_COLUMNS = [
    "median_roi_ti",
    "mean_roi_ti",
    "peak_roi_ti",
    "median_head_ti",
    "mean_head_ti",
    "peak_head_ti",
]
MESH_COLUMNS = ["mesh_nodes", "mesh_cells"]
HOTSPOT_COLUMNS = ["hotspot_x", "hotspot_y", "hotspot_z"]
BASE_PER_REPEAT_FIELDS = [
    "condition",
    "repeat",
    *METRIC_COLUMNS,
    *HOTSPOT_COLUMNS,
    "high_field_voxels",
    "mesh_nodes",
    "mesh_cells",
    "mesh_checksum",
]


@dataclass(frozen=True)
class CommandResult:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pipeline_dir(experiment_root: Path) -> Path:
    return Path(experiment_root) / PIPELINE_STATE_DIR


def config_path(experiment_root: Path) -> Path:
    return pipeline_dir(experiment_root) / "config.json"


def stage_status_path(experiment_root: Path) -> Path:
    return pipeline_dir(experiment_root) / "stage_status.json"


def events_path(experiment_root: Path) -> Path:
    return pipeline_dir(experiment_root) / "events.jsonl"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    return path


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def log_event(experiment_root: Path, event: str, **fields: Any) -> None:
    append_jsonl(
        events_path(experiment_root),
        {
            "timestamp": utc_now(),
            "event": event,
            **fields,
        },
    )


def load_stage_status(experiment_root: Path) -> dict[str, Any]:
    path = stage_status_path(experiment_root)
    if path.exists():
        return read_json(path)
    return {}


def update_stage_status(experiment_root: Path, stage: str, status: str, **fields: Any) -> None:
    stage_status = load_stage_status(experiment_root)
    stage_status[stage] = {
        "status": status,
        "updated_at": utc_now(),
        **fields,
    }
    write_json(stage_status_path(experiment_root), stage_status)


def default_simulation_config() -> dict[str, Any]:
    return {
        "pair1": {"anode": "F10", "cathode": "P8", "current_a": 0.002},
        "pair2": {"anode": "T7", "cathode": "P7", "current_a": 0.001588656},
        "electrode_shape": "ellipse",
        "electrode_radius_mm": 10,
        "electrode_thickness_mm": 1,
        "electrode_conductivity": 0.85,
        "element_size": 0.1,
        "map_to_fsavg": False,
        "map_to_mni": True,
        "write_volume_outputs": True,
        "runner_command": None,
    }


def build_config(
    *,
    source_root: Path,
    experiment_root: Path,
    subject: str,
    repeat_count_full_face: int,
    repeat_count_defaced: int,
    roi_preset: str,
    simulation_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    simulation = default_simulation_config()
    if simulation_overrides:
        simulation.update(simulation_overrides)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_root": str(Path(source_root).resolve()),
        "experiment_root": str(Path(experiment_root).resolve()),
        "subject": subject,
        "conditions": [
            {
                "name": "full_face",
                "input_mode": "original",
                "repeat_count": int(repeat_count_full_face),
            },
            {
                "name": "defaced",
                "input_mode": "fsl_deface",
                "repeat_count": int(repeat_count_defaced),
            },
        ],
        "roi": {
            "preset": roi_preset,
            "description": "Configured ROI used by downstream report tasks.",
        },
        "simulation": simulation,
    }


def init_experiment(
    *,
    source_root: Path | str,
    experiment_root: Path | str,
    subject: str,
    repeat_count_full_face: int = 40,
    repeat_count_defaced: int = 40,
    roi_preset: str = "left-hippocampus",
    simulation_overrides: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    config = build_config(
        source_root=Path(source_root),
        experiment_root=experiment_root,
        subject=subject,
        repeat_count_full_face=repeat_count_full_face,
        repeat_count_defaced=repeat_count_defaced,
        roi_preset=roi_preset,
        simulation_overrides=simulation_overrides,
    )
    if dry_run:
        return config

    for dirname in [
        PIPELINE_STATE_DIR,
        "inputs/original",
        "inputs/full_face",
        "inputs/defaced",
        "runs",
        "logs",
        ANALYSIS_DIR,
        "figures",
    ]:
        (experiment_root / dirname).mkdir(parents=True, exist_ok=True)
    write_json(config_path(experiment_root), config)
    write_json(stage_status_path(experiment_root), {})
    update_stage_status(experiment_root, "init", "complete", config=str(config_path(experiment_root)))
    log_event(experiment_root, "init.complete", config=str(config_path(experiment_root)))
    return config


def load_config(experiment_root: Path | str) -> dict[str, Any]:
    path = config_path(Path(experiment_root))
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config: {path}")
    return read_json(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, source_path: Path | None = None) -> dict[str, Any]:
    path = Path(path)
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if source_path is not None:
        record["source_path"] = str(source_path)
    return record


def modality_score(path: Path, subject: str, modality: str) -> tuple[int, str]:
    text = str(path).lower()
    score = 0
    if subject.lower() in text:
        score -= 20
    if "/anat/" in text:
        score -= 10
    if f"_{modality.lower()}" in path.name.lower():
        score -= 5
    return (score, str(path))


def find_modality_file(source_root: Path, subject: str, modality: str) -> Path:
    source_root = Path(source_root)
    candidates: list[Path] = []
    for pattern in (f"**/{subject}*{modality}*.nii", f"**/{subject}*{modality}*.nii.gz", f"**/*{modality}*.nii", f"**/*{modality}*.nii.gz"):
        candidates.extend(path for path in source_root.glob(pattern) if path.is_file())
    candidates = sorted(set(candidates), key=lambda path: modality_score(path, subject, modality))
    if not candidates:
        raise FileNotFoundError(f"Could not find {modality} NIfTI under {source_root} for {subject}")
    return candidates[0]


def replace_path(target: Path) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()


def link_or_copy(src: Path, dst: Path, *, copy_inputs: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    replace_path(dst)
    if copy_inputs:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(str(Path(src).resolve()), dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_inputs(experiment_root: Path | str, *, copy_inputs: bool = False) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    config = load_config(experiment_root)
    source_root = Path(config["source_root"])
    subject = config["subject"]
    source_t1 = find_modality_file(source_root, subject, "T1w")
    source_t2 = find_modality_file(source_root, subject, "T2w")

    original_t1 = experiment_root / "inputs" / "original" / "T1w.nii.gz"
    original_t2 = experiment_root / "inputs" / "original" / "T2w.nii.gz"
    full_face_t1 = experiment_root / "inputs" / "full_face" / "T1w.nii.gz"
    full_face_t2 = experiment_root / "inputs" / "full_face" / "T2w.nii.gz"

    link_or_copy(source_t1, original_t1, copy_inputs=copy_inputs)
    link_or_copy(source_t2, original_t2, copy_inputs=copy_inputs)
    link_or_copy(original_t1, full_face_t1, copy_inputs=copy_inputs)
    link_or_copy(original_t2, full_face_t2, copy_inputs=copy_inputs)

    manifest = {
        "created_at": utc_now(),
        "subject": subject,
        "source_root": str(source_root),
        "copy_inputs": copy_inputs,
        "original": {
            "t1": file_record(original_t1, source_t1),
            "t2": file_record(original_t2, source_t2),
        },
        "conditions": {
            "full_face": {
                "t1": str(full_face_t1),
                "t2": str(full_face_t2),
                "input_mode": "original",
            },
            "defaced": {
                "t1": str(experiment_root / "inputs" / "defaced" / "T1w.nii.gz"),
                "t2": str(experiment_root / "inputs" / "defaced" / "T2w.nii.gz"),
                "input_mode": "fsl_deface",
            },
        },
    }
    write_json(pipeline_dir(experiment_root) / "input_manifest.json", manifest)
    update_stage_status(experiment_root, "prepare-inputs", "complete", manifest=str(pipeline_dir(experiment_root) / "input_manifest.json"))
    log_event(experiment_root, "prepare-inputs.complete", manifest=str(pipeline_dir(experiment_root) / "input_manifest.json"))
    return manifest


def run_command(cmd: Sequence[str], **kwargs: Any) -> CommandResult:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
    return CommandResult(
        args=list(cmd),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def deface_inputs(
    experiment_root: Path | str,
    *,
    fsl_deface_bin: str = "fsl_deface",
    command_runner: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    input_manifest = read_json(pipeline_dir(experiment_root) / "input_manifest.json")
    original_t1 = Path(input_manifest["original"]["t1"]["path"])
    original_t2 = Path(input_manifest["original"]["t2"]["path"])
    defaced_t1 = experiment_root / "inputs" / "defaced" / "T1w.nii.gz"
    defaced_t2 = experiment_root / "inputs" / "defaced" / "T2w.nii.gz"
    defaced_t1.parent.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}
    commands: list[dict[str, Any]] = []
    for key, src, dst in (("t1", original_t1, defaced_t1), ("t2", original_t2, defaced_t2)):
        result = command_runner([fsl_deface_bin, str(src), str(dst)])
        commands.append(
            {
                "modality": key,
                "args": list(result.args),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            update_stage_status(experiment_root, "deface-inputs", "failed", modality=key, stderr=result.stderr)
            log_event(experiment_root, "deface-inputs.failed", modality=key, returncode=result.returncode)
            raise RuntimeError(f"{fsl_deface_bin} failed for {src}: {result.stderr}")
        if not dst.exists():
            raise FileNotFoundError(f"{fsl_deface_bin} did not create expected output: {dst}")
        outputs[key] = file_record(dst, src)

    manifest = {
        "created_at": utc_now(),
        "subject": input_manifest["subject"],
        "tool": fsl_deface_bin,
        "commands": commands,
        "defaced": outputs,
    }
    write_json(pipeline_dir(experiment_root) / "defacing_manifest.json", manifest)
    update_stage_status(experiment_root, "deface-inputs", "complete", manifest=str(pipeline_dir(experiment_root) / "defacing_manifest.json"))
    log_event(experiment_root, "deface-inputs.complete", manifest=str(pipeline_dir(experiment_root) / "defacing_manifest.json"))
    return manifest


def condition_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {condition["name"]: dict(condition) for condition in config["conditions"]}


def expected_condition_input(experiment_root: Path, condition: str, modality: str) -> Path:
    return experiment_root / "inputs" / condition / f"{modality}.nii.gz"


def repeat_tag(repeat: int) -> str:
    return f"repeat-{repeat:03d}"


def ti_mesh_path(experiment_root: Path, condition: str, repeat: int, subject: str) -> Path:
    return (
        experiment_root
        / "runs"
        / condition
        / repeat_tag(repeat)
        / subject
        / "anat"
        / "SimNIBS"
        / "Output"
        / subject
        / "TI.msh"
    )


def plan_simulation_tasks(experiment_root: Path | str) -> list[dict[str, Any]]:
    experiment_root = Path(experiment_root)
    config = load_config(experiment_root)
    subject = config["subject"]
    tasks: list[dict[str, Any]] = []
    task_index = 0
    for condition in config["conditions"]:
        condition_name = condition["name"]
        for repeat in range(1, int(condition["repeat_count"]) + 1):
            run_root = experiment_root / "runs" / condition_name / repeat_tag(repeat)
            anat_dir = run_root / subject / "anat"
            output_dir = anat_dir / "SimNIBS" / "Output" / subject
            task = {
                "task_index": task_index,
                "condition": condition_name,
                "input_mode": condition["input_mode"],
                "subject": subject,
                "repeat": repeat,
                "repeat_tag": repeat_tag(repeat),
                "t1_path": str(expected_condition_input(experiment_root, condition_name, "T1w")),
                "t2_path": str(expected_condition_input(experiment_root, condition_name, "T2w")),
                "run_root": str(run_root),
                "anat_dir": str(anat_dir),
                "output_dir": str(output_dir),
                "ti_mesh_path": str(output_dir / "TI.msh"),
                "simulation_spec_path": str(run_root / "simulation_spec.json"),
                "command_script_path": str(run_root / "run_simulation_commands.sh"),
                "summary_path": str(experiment_root / ANALYSIS_DIR / "repeat_summaries" / f"{condition_name}_{repeat_tag(repeat)}.json"),
            }
            tasks.append(task)
            task_index += 1

    manifest_path = pipeline_dir(experiment_root) / "simulation_tasks.json"
    write_json(manifest_path, tasks)
    update_stage_status(experiment_root, "plan-simulations", "complete", task_count=len(tasks), manifest=str(manifest_path))
    log_event(experiment_root, "plan-simulations.complete", task_count=len(tasks), manifest=str(manifest_path))
    return tasks


def load_task(task_manifest: Path, task_index: int) -> dict[str, Any]:
    tasks = read_json(task_manifest)
    for task in tasks:
        if int(task["task_index"]) == int(task_index):
            return task
    raise IndexError(f"Task index {task_index} not found in {task_manifest}")


def simulation_spec(task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "task": dict(task),
        "simulation": config["simulation"],
        "roi": config["roi"],
        "regenerate_charm": True,
    }


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_simulation_command_lines(task: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    runner_template = config["simulation"].get("runner_command") or os.environ.get("DEFACING_SIMULATION_RUNNER")
    charm_line = " ".join(
        [
            "charm",
            shell_quote(str(task["subject"])),
            shell_quote(str(task["t1_path"])),
            shell_quote(str(task["t2_path"])),
            "--forcerun",
            "--outdir",
            shell_quote(str(Path(str(task["anat_dir"])))),
        ]
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"mkdir -p {shell_quote(str(Path(str(task['run_root']))))}",
        f"mkdir -p {shell_quote(str(Path(str(task['output_dir']))))}",
        charm_line,
    ]
    if runner_template:
        values = {
            "config": str(config_path(Path(config["experiment_root"]))),
            "task_manifest": str(pipeline_dir(Path(config["experiment_root"])) / "simulation_tasks.json"),
            "task_index": str(task["task_index"]),
            "simulation_spec": str(task["simulation_spec_path"]),
            "run_root": str(task["run_root"]),
            "output_dir": str(task["output_dir"]),
            "ti_mesh": str(task["ti_mesh_path"]),
        }
        lines.append(Template(runner_template).safe_substitute(values))
    else:
        lines.append(
            "echo 'No DEFACING_SIMULATION_RUNNER or simulation.runner_command configured; "
            "simulation spec written for downstream SimNIBS execution.' >&2"
        )
        lines.append("exit 64")
    return lines


def run_simulation_task(task_manifest: Path | str, task_index: int, *, dry_run: bool = False) -> dict[str, Any]:
    task_manifest = Path(task_manifest)
    task = load_task(task_manifest, task_index)
    experiment_root = Path(task_manifest).parents[0].parent
    config = load_config(experiment_root)
    Path(task["run_root"]).mkdir(parents=True, exist_ok=True)
    spec = simulation_spec(task, config)
    write_json(Path(task["simulation_spec_path"]), spec)
    command_script = Path(task["command_script_path"])
    command_script.write_text("\n".join(build_simulation_command_lines(task, config)) + "\n", encoding="utf-8")
    command_script.chmod(command_script.stat().st_mode | 0o100)
    if dry_run:
        return {"task": task, "simulation_spec": str(task["simulation_spec_path"]), "command_script": str(command_script)}
    result = run_command(["bash", str(command_script)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or f"Simulation task {task_index} failed")
    return {"task": task, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def parse_sbatch_job_id(output: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return match.group(1) if match else None


def slurm_script(name: str) -> Path:
    return module_root() / "slurm" / name


def run_sbatch(args: Sequence[str]) -> CommandResult:
    sbatch = os.environ.get("SBATCH_BIN", "sbatch")
    return run_command([sbatch, *args])


def record_submission(experiment_root: Path, payload: Mapping[str, Any]) -> None:
    append_jsonl(pipeline_dir(experiment_root) / "submissions.jsonl", payload)


def submit_simulations(experiment_root: Path | str, *, max_concurrent: int = 50) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    tasks = plan_simulation_tasks(experiment_root)
    if not tasks:
        raise RuntimeError("No simulation tasks planned")
    manifest = pipeline_dir(experiment_root) / "simulation_tasks.json"
    script = slurm_script("run_defacing_simulation_array.slurm")
    array_spec = f"0-{len(tasks) - 1}%{int(max_concurrent)}"
    export_spec = ",".join(
        [
            "ALL",
            f"EXPERIMENT_ROOT={experiment_root}",
            f"TASK_MANIFEST={manifest}",
            f"CONFIG={config_path(experiment_root)}",
        ]
    )
    result = run_sbatch([f"--array={array_spec}", f"--export={export_spec}", str(script)])
    job_id = parse_sbatch_job_id(result.stdout)
    submission = {
        "timestamp": utc_now(),
        "stage": "submit-simulations",
        "job_id": job_id,
        "array": array_spec,
        "task_count": len(tasks),
        "task_manifest": str(manifest),
        "script": str(script),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    record_submission(experiment_root, submission)
    status = "submitted" if result.returncode == 0 else "failed"
    update_stage_status(experiment_root, "submit-simulations", status, job_id=job_id, task_count=len(tasks), array=array_spec)
    log_event(experiment_root, f"submit-simulations.{status}", job_id=job_id, task_count=len(tasks), array=array_spec)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "sbatch failed for simulation array")
    return submission


def plan_report_tasks(experiment_root: Path | str) -> list[dict[str, Any]]:
    experiment_root = Path(experiment_root)
    config = load_config(experiment_root)
    tasks = [
        {
            "task_index": index,
            "condition": condition["name"],
            "subject": config["subject"],
            "experiment_root": str(experiment_root),
            "repeat_summaries_dir": str(experiment_root / ANALYSIS_DIR / "repeat_summaries"),
            "per_repeat_metrics": str(experiment_root / ANALYSIS_DIR / "per_repeat_metrics.csv"),
        }
        for index, condition in enumerate(config["conditions"])
    ]
    manifest = pipeline_dir(experiment_root) / "report_tasks.json"
    write_json(manifest, tasks)
    update_stage_status(experiment_root, "plan-reports", "complete", task_count=len(tasks), manifest=str(manifest))
    log_event(experiment_root, "plan-reports.complete", task_count=len(tasks), manifest=str(manifest))
    return tasks


def flatten_repeat_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    roi = summary.get("roi", {}) if isinstance(summary.get("roi"), Mapping) else {}
    whole_head = summary.get("whole_head", {}) if isinstance(summary.get("whole_head"), Mapping) else {}
    mesh = summary.get("mesh", {}) if isinstance(summary.get("mesh"), Mapping) else {}
    hotspot = whole_head.get("hotspot", [None, None, None])
    if not isinstance(hotspot, Sequence) or isinstance(hotspot, (str, bytes)):
        hotspot = [None, None, None]
    hotspot_values = list(hotspot)[:3] + [None, None, None]
    row: dict[str, Any] = {
        "condition": summary.get("condition"),
        "repeat": summary.get("repeat"),
        "median_roi_ti": summary.get("median_roi_ti", roi.get("median_ti")),
        "mean_roi_ti": summary.get("mean_roi_ti", roi.get("mean_ti")),
        "peak_roi_ti": summary.get("peak_roi_ti", roi.get("peak_ti")),
        "median_head_ti": summary.get("median_head_ti", whole_head.get("median_ti")),
        "mean_head_ti": summary.get("mean_head_ti", whole_head.get("mean_ti")),
        "peak_head_ti": summary.get("peak_head_ti", whole_head.get("peak_ti")),
        "hotspot_x": summary.get("hotspot_x", hotspot_values[0]),
        "hotspot_y": summary.get("hotspot_y", hotspot_values[1]),
        "hotspot_z": summary.get("hotspot_z", hotspot_values[2]),
        "high_field_voxels": summary.get("high_field_voxels", whole_head.get("high_field_voxels")),
        "mesh_nodes": summary.get("mesh_nodes", mesh.get("nodes")),
        "mesh_cells": summary.get("mesh_cells", mesh.get("cells")),
        "mesh_checksum": summary.get("mesh_checksum", mesh.get("checksum")),
    }
    tissue_counts = mesh.get("tissue_label_counts", {})
    if isinstance(tissue_counts, Mapping):
        for label, count in tissue_counts.items():
            label_text = str(label)
            if not label_text.startswith("tissue_"):
                label_text = f"tissue_{label_text}_count"
            row[label_text] = count
    for key, value in summary.items():
        if key.startswith("tissue_") and key.endswith("_count"):
            row[key] = value
    return row


def rows_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    extra = sorted({key for row in rows for key in row if key not in BASE_PER_REPEAT_FIELDS})
    return [*BASE_PER_REPEAT_FIELDS, *extra]


def run_report_task(task_manifest: Path | str, task_index: int) -> dict[str, Any]:
    task_manifest = Path(task_manifest)
    report_task = load_task(task_manifest, task_index)
    experiment_root = Path(report_task["experiment_root"])
    condition = report_task["condition"]
    simulation_manifest = pipeline_dir(experiment_root) / "simulation_tasks.json"
    simulation_tasks = read_json(simulation_manifest)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in simulation_tasks:
        if task.get("condition") != condition:
            continue
        summary_path = Path(task["summary_path"])
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        row = flatten_repeat_summary(read_json(summary_path))
        row["condition"] = row.get("condition") or condition
        row["repeat"] = row.get("repeat") or task.get("repeat")
        rows.append(row)
    rows = sorted(rows, key=lambda row: int(parse_float(row.get("repeat")) or 0))
    out = experiment_root / ANALYSIS_DIR / "report_tasks" / f"{condition}_per_repeat_metrics.csv"
    write_csv_rows(out, rows_fieldnames(rows), rows)
    payload = {
        "condition": condition,
        "condition_metrics_csv": out,
        "row_count": len(rows),
        "missing_summary_count": len(missing),
        "missing_summaries": missing,
    }
    write_json(experiment_root / ANALYSIS_DIR / "report_tasks" / f"{condition}_report_task.json", payload)
    log_event(experiment_root, "report-task.complete", condition=condition, row_count=len(rows), missing_summary_count=len(missing))
    return payload


def aggregate_report_outputs(experiment_root: Path | str) -> dict[str, Path]:
    experiment_root = Path(experiment_root)
    report_dir = experiment_root / ANALYSIS_DIR / "report_tasks"
    csv_files = sorted(report_dir.glob("*_per_repeat_metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No report task CSV files found under {report_dir}")
    rows: list[dict[str, Any]] = []
    for csv_file in csv_files:
        rows.extend(read_csv_rows(csv_file))
    rows = sorted(rows, key=lambda row: (str(row.get("condition")), int(parse_float(row.get("repeat")) or 0)))
    per_repeat = experiment_root / ANALYSIS_DIR / "per_repeat_metrics.csv"
    write_csv_rows(per_repeat, rows_fieldnames(rows), rows)
    outputs = aggregate_analysis(experiment_root)
    outputs["per_repeat_metrics_csv"] = per_repeat
    update_stage_status(experiment_root, "aggregate-reports", "complete", per_repeat_metrics=str(per_repeat), row_count=len(rows))
    log_event(experiment_root, "aggregate-reports.complete", per_repeat_metrics=str(per_repeat), row_count=len(rows))
    return outputs


def submit_reports(experiment_root: Path | str, *, max_concurrent: int = 10) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    tasks = plan_report_tasks(experiment_root)
    if not tasks:
        raise RuntimeError("No report tasks planned")
    manifest = pipeline_dir(experiment_root) / "report_tasks.json"
    script = slurm_script("run_defacing_report_array.slurm")
    array_spec = f"0-{len(tasks) - 1}%{int(max_concurrent)}"
    export_spec = ",".join(
        [
            "ALL",
            f"EXPERIMENT_ROOT={experiment_root}",
            f"REPORT_TASK_MANIFEST={manifest}",
            f"CONFIG={config_path(experiment_root)}",
        ]
    )
    result = run_sbatch([f"--array={array_spec}", f"--export={export_spec}", str(script)])
    job_id = parse_sbatch_job_id(result.stdout)
    submission = {
        "timestamp": utc_now(),
        "stage": "analyze",
        "job_id": job_id,
        "array": array_spec,
        "task_count": len(tasks),
        "task_manifest": str(manifest),
        "script": str(script),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    record_submission(experiment_root, submission)
    status = "submitted" if result.returncode == 0 else "failed"
    update_stage_status(experiment_root, "analyze", status, job_id=job_id, task_count=len(tasks), array=array_spec)
    log_event(experiment_root, f"analyze.{status}", job_id=job_id, task_count=len(tasks), array=array_spec)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "sbatch failed for report array")
    return submission


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def numeric_values(rows: Iterable[Mapping[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = parse_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def group_by_condition(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("condition")), []).append(row)
    return grouped


def tissue_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(key for key in row if key.startswith("tissue_") and key.endswith("_count"))
    return sorted(columns)


def rows_by_repeat(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[int, Mapping[str, Any]]:
    selected: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("condition") != condition:
            continue
        repeat = parse_float(row.get("repeat"))
        if repeat is not None:
            selected[int(repeat)] = row
    return selected


def euclidean(row_a: Mapping[str, Any], row_b: Mapping[str, Any]) -> float | None:
    a = [parse_float(row_a.get(column)) for column in HOTSPOT_COLUMNS]
    b = [parse_float(row_b.get(column)) for column in HOTSPOT_COLUMNS]
    if any(value is None for value in a + b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def dice_from_counts(a: Any, b: Any) -> float | None:
    value_a = parse_float(a)
    value_b = parse_float(b)
    if value_a is None or value_b is None:
        return None
    denom = value_a + value_b
    if denom == 0:
        return 1.0
    return 2 * min(value_a, value_b) / denom


def aggregate_analysis(experiment_root: Path | str) -> dict[str, str]:
    experiment_root = Path(experiment_root)
    analysis_dir = experiment_root / ANALYSIS_DIR
    per_repeat = analysis_dir / "per_repeat_metrics.csv"
    if not per_repeat.exists():
        raise FileNotFoundError(f"Missing per-repeat metrics: {per_repeat}")

    rows = read_csv_rows(per_repeat)
    grouped = group_by_condition(rows)
    tissues = tissue_columns(rows)
    metrics = METRIC_COLUMNS + MESH_COLUMNS + tissues

    condition_summary: dict[str, dict[str, Any]] = {}
    condition_csv_rows: list[dict[str, Any]] = []
    mesh_summary: dict[str, dict[str, Any]] = {}
    for condition, condition_rows in sorted(grouped.items()):
        condition_summary[condition] = {}
        mesh_summary[condition] = {"tissue_label_counts": {}}
        for metric in metrics:
            metric_stats = stats(numeric_values(condition_rows, metric))
            condition_summary[condition][metric] = metric_stats
            condition_csv_rows.append({"condition": condition, "metric": metric, **metric_stats})
            if metric in MESH_COLUMNS:
                mesh_summary[condition][metric] = metric_stats
            elif metric in tissues:
                mesh_summary[condition]["tissue_label_counts"][metric] = metric_stats
        checksums = sorted({row.get("mesh_checksum") for row in condition_rows if row.get("mesh_checksum")})
        mesh_summary[condition]["mesh_checksums"] = checksums

    comparison: dict[str, Any] = {"defaced_minus_full_face": {}, "defaced_vs_full_face": {}}
    if "full_face" in condition_summary and "defaced" in condition_summary:
        comparison_rows: list[dict[str, Any]] = []
        for metric in metrics:
            full_mean = condition_summary["full_face"].get(metric, {}).get("mean")
            defaced_mean = condition_summary["defaced"].get(metric, {}).get("mean")
            if full_mean is None or defaced_mean is None:
                continue
            delta = defaced_mean - full_mean
            comparison["defaced_minus_full_face"][metric] = {"mean_delta": delta}
            comparison_rows.append(
                {
                    "comparison": "defaced_minus_full_face",
                    "metric": metric,
                    "mean_delta": delta,
                }
            )

        full_by_repeat = rows_by_repeat(rows, "full_face")
        defaced_by_repeat = rows_by_repeat(rows, "defaced")
        paired_repeats = sorted(set(full_by_repeat).intersection(defaced_by_repeat))
        hotspot_distances = [
            distance
            for distance in (euclidean(full_by_repeat[repeat], defaced_by_repeat[repeat]) for repeat in paired_repeats)
            if distance is not None
        ]
        high_field_dice = [
            dice
            for dice in (
                dice_from_counts(full_by_repeat[repeat].get("high_field_voxels"), defaced_by_repeat[repeat].get("high_field_voxels"))
                for repeat in paired_repeats
            )
            if dice is not None
        ]
        comparison["defaced_vs_full_face"]["hotspot_distance_mean_mm"] = (
            statistics.fmean(hotspot_distances) if hotspot_distances else None
        )
        comparison["defaced_vs_full_face"]["high_field_dice_mean"] = statistics.fmean(high_field_dice) if high_field_dice else None
        comparison_rows.extend(
            [
                {
                    "comparison": "defaced_vs_full_face",
                    "metric": "hotspot_distance_mean_mm",
                    "mean_delta": comparison["defaced_vs_full_face"]["hotspot_distance_mean_mm"],
                },
                {
                    "comparison": "defaced_vs_full_face",
                    "metric": "high_field_dice_mean",
                    "mean_delta": comparison["defaced_vs_full_face"]["high_field_dice_mean"],
                },
            ]
        )
    else:
        comparison_rows = []

    condition_summary_json = write_json(analysis_dir / "condition_summary.json", condition_summary)
    condition_summary_csv = write_csv_rows(
        analysis_dir / "condition_summary.csv",
        ["condition", "metric", "count", "mean", "median", "min", "max", "std"],
        condition_csv_rows,
    )
    comparison_summary_json = write_json(analysis_dir / "comparison_summary.json", comparison)
    comparison_summary_csv = write_csv_rows(
        analysis_dir / "comparison_summary.csv",
        ["comparison", "metric", "mean_delta"],
        comparison_rows,
    )
    mesh_metrics_json = write_json(analysis_dir / "mesh_metrics.json", mesh_summary)
    mesh_rows = []
    for condition, condition_mesh in mesh_summary.items():
        for metric in MESH_COLUMNS:
            if metric in condition_mesh:
                mesh_rows.append({"condition": condition, "metric": metric, **condition_mesh[metric]})
        for metric, metric_stats in condition_mesh.get("tissue_label_counts", {}).items():
            mesh_rows.append({"condition": condition, "metric": metric, **metric_stats})
    mesh_metrics_csv = write_csv_rows(
        analysis_dir / "mesh_metrics.csv",
        ["condition", "metric", "count", "mean", "median", "min", "max", "std"],
        mesh_rows,
    )
    update_stage_status(experiment_root, "aggregate-analysis", "complete", per_repeat_metrics=str(per_repeat))
    log_event(experiment_root, "aggregate-analysis.complete", per_repeat_metrics=str(per_repeat))
    return {
        "condition_summary_json": condition_summary_json,
        "condition_summary_csv": condition_summary_csv,
        "comparison_summary_json": comparison_summary_json,
        "comparison_summary_csv": comparison_summary_csv,
        "mesh_metrics_json": mesh_metrics_json,
        "mesh_metrics_csv": mesh_metrics_csv,
    }


def analyze_experiment(experiment_root: Path | str, *, max_concurrent: int = 10) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    submission = submit_reports(experiment_root, max_concurrent=max_concurrent)
    outputs = None
    per_repeat = experiment_root / ANALYSIS_DIR / "per_repeat_metrics.csv"
    if per_repeat.exists():
        outputs = aggregate_analysis(experiment_root)
    return {"submission": submission, "aggregation": outputs}


def analyze_local(experiment_root: Path | str) -> dict[str, Path]:
    experiment_root = Path(experiment_root)
    if (experiment_root / ANALYSIS_DIR / "per_repeat_metrics.csv").exists():
        return aggregate_analysis(experiment_root)
    return aggregate_report_outputs(experiment_root)


def svg_bar_chart(title: str, rows: Sequence[Mapping[str, Any]], label_key: str, value_key: str) -> str:
    width = 900
    height = max(220, 90 + 38 * len(rows))
    max_value = max([abs(parse_float(row.get(value_key)) or 0.0) for row in rows] + [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" fill="#111111">{escape_xml(title)}</text>',
    ]
    zero_x = 260
    bar_max = width - zero_x - 60
    for index, row in enumerate(rows):
        y = 68 + index * 38
        label = str(row.get(label_key, ""))
        value = parse_float(row.get(value_key)) or 0.0
        bar_width = abs(value) / max_value * bar_max
        color = "#356a9a" if value >= 0 else "#b65c42"
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial, sans-serif" font-size="14" fill="#222222">{escape_xml(label)}</text>')
        parts.append(f'<rect x="{zero_x}" y="{y}" width="{bar_width:.2f}" height="24" fill="{color}"/>')
        parts.append(
            f'<text x="{zero_x + bar_width + 8:.2f}" y="{y + 17}" font-family="Arial, sans-serif" font-size="13" fill="#222222">{value:.6g}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_figures(experiment_root: Path | str) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    analysis_dir = experiment_root / ANALYSIS_DIR
    figures_dir = experiment_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    condition_csv = analysis_dir / "condition_summary.csv"
    comparison_csv = analysis_dir / "comparison_summary.csv"
    if not condition_csv.exists():
        raise FileNotFoundError(f"Missing condition summary: {condition_csv}")
    if not comparison_csv.exists():
        raise FileNotFoundError(f"Missing comparison summary: {comparison_csv}")

    condition_rows = [
        {
            **row,
            "label": f"{row.get('condition')} {row.get('metric')}",
        }
        for row in read_csv_rows(condition_csv)
        if row.get("metric") in {"median_roi_ti", "mean_roi_ti", "peak_roi_ti", "median_head_ti"}
    ]
    comparison_rows = [
        {
            **row,
            "label": str(row.get("metric")),
        }
        for row in read_csv_rows(comparison_csv)
        if row.get("comparison") == "defaced_minus_full_face"
        and row.get("metric") in {"median_roi_ti", "mean_roi_ti", "peak_roi_ti", "median_head_ti"}
    ]
    condition_svg = figures_dir / "condition_metric_means.svg"
    comparison_svg = figures_dir / "comparison_metric_deltas.svg"
    condition_svg.write_text(svg_bar_chart("Condition metric means", condition_rows, "label", "mean"), encoding="utf-8")
    comparison_svg.write_text(svg_bar_chart("Defaced minus full-face deltas", comparison_rows, "label", "mean_delta"), encoding="utf-8")

    manifest = {
        "created_at": utc_now(),
        "figures": {
            "condition_metric_means": str(condition_svg),
            "comparison_metric_deltas": str(comparison_svg),
        },
        "sources": {
            "condition_summary": str(condition_csv),
            "comparison_summary": str(comparison_csv),
        },
    }
    write_json(figures_dir / "figure_manifest.json", manifest)
    update_stage_status(experiment_root, "make-figures", "complete", manifest=str(figures_dir / "figure_manifest.json"))
    log_event(experiment_root, "make-figures.complete", manifest=str(figures_dir / "figure_manifest.json"))
    return manifest


def status_experiment(experiment_root: Path | str) -> dict[str, Any]:
    experiment_root = Path(experiment_root)
    missing: list[str] = []
    warnings: list[str] = []
    config: dict[str, Any] | None = None
    if not config_path(experiment_root).exists():
        missing.append(str(config_path(experiment_root)))
    else:
        config = load_config(experiment_root)
    for path in [
        pipeline_dir(experiment_root) / "input_manifest.json",
        pipeline_dir(experiment_root) / "events.jsonl",
        pipeline_dir(experiment_root) / "stage_status.json",
    ]:
        if not path.exists():
            missing.append(str(path))
    for path in [
        experiment_root / "inputs" / "full_face" / "T1w.nii.gz",
        experiment_root / "inputs" / "full_face" / "T2w.nii.gz",
        experiment_root / "inputs" / "defaced" / "T1w.nii.gz",
        experiment_root / "inputs" / "defaced" / "T2w.nii.gz",
    ]:
        if not path.exists():
            missing.append(str(path))
    if config is not None:
        for condition in config["conditions"]:
            for repeat in range(1, int(condition["repeat_count"]) + 1):
                mesh = ti_mesh_path(experiment_root, condition["name"], repeat, config["subject"])
                if not mesh.exists():
                    missing.append(str(mesh))
    for path in [
        experiment_root / ANALYSIS_DIR / "per_repeat_metrics.csv",
        experiment_root / ANALYSIS_DIR / "condition_summary.csv",
        experiment_root / ANALYSIS_DIR / "comparison_summary.csv",
        experiment_root / ANALYSIS_DIR / "mesh_metrics.csv",
        experiment_root / "figures" / "figure_manifest.json",
    ]:
        if not path.exists():
            missing.append(str(path))
    stage_status = load_stage_status(experiment_root)
    if stage_status.get("submit-simulations", {}).get("status") == "failed":
        warnings.append("Simulation submission failed")
    if stage_status.get("analyze", {}).get("status") == "failed":
        warnings.append("Report submission failed")
    return {
        "experiment_root": str(experiment_root),
        "ok": not missing and not warnings,
        "missing": missing,
        "warnings": warnings,
        "stage_status": stage_status,
    }


def print_status(status: Mapping[str, Any]) -> None:
    print(f"Experiment root: {status['experiment_root']}")
    print(f"OK: {status['ok']}")
    if status["missing"]:
        print("Missing:")
        for item in status["missing"]:
            print(f"  - {item}")
    if status["warnings"]:
        print("Warnings:")
        for item in status["warnings"]:
            print(f"  - {item}")


def load_simulation_overrides(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return read_json(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create experiment config and state directory")
    init_parser.add_argument("--source-root", required=True, type=Path)
    init_parser.add_argument("--experiment-root", required=True, type=Path)
    init_parser.add_argument("--subject", required=True)
    init_parser.add_argument("--repeat-count-full-face", type=int, default=40)
    init_parser.add_argument("--repeat-count-defaced", type=int, default=40)
    init_parser.add_argument("--roi-preset", default="left-hippocampus")
    init_parser.add_argument("--simulation-overrides", type=Path)
    init_parser.add_argument("--dry-run", action="store_true")

    prepare_parser = subparsers.add_parser("prepare-inputs", help="Stage original and full-face inputs")
    prepare_parser.add_argument("--experiment-root", required=True, type=Path)
    prepare_parser.add_argument("--copy-inputs", action="store_true")

    deface_parser = subparsers.add_parser("deface-inputs", help="Create defaced T1/T2 inputs with fsl_deface")
    deface_parser.add_argument("--experiment-root", required=True, type=Path)
    deface_parser.add_argument("--fsl-deface-bin", default="fsl_deface")

    submit_parser = subparsers.add_parser("submit-simulations", help="Submit the simulation Slurm array")
    submit_parser.add_argument("--experiment-root", required=True, type=Path)
    submit_parser.add_argument("--max-concurrent", type=int, default=50)

    analyze_parser = subparsers.add_parser("analyze", help="Submit report tasks and aggregate metrics when available")
    analyze_parser.add_argument("--experiment-root", required=True, type=Path)
    analyze_parser.add_argument("--max-concurrent", type=int, default=10)
    analyze_parser.add_argument("--local-only", action="store_true", help="Only aggregate existing per-repeat metrics")

    figures_parser = subparsers.add_parser("make-figures", help="Generate SVG summary figures from analysis CSVs")
    figures_parser.add_argument("--experiment-root", required=True, type=Path)

    status_parser = subparsers.add_parser("status", help="Inspect pipeline completeness")
    status_parser.add_argument("--experiment-root", required=True, type=Path)
    status_parser.add_argument("--json", action="store_true")

    run_sim_parser = subparsers.add_parser("run-simulation-task", help=argparse.SUPPRESS)
    run_sim_parser.add_argument("--task-manifest", required=True, type=Path)
    run_sim_parser.add_argument("--task-index", required=True, type=int)
    run_sim_parser.add_argument("--dry-run", action="store_true")

    run_report_parser = subparsers.add_parser("run-report-task", help=argparse.SUPPRESS)
    run_report_parser.add_argument("--task-manifest", required=True, type=Path)
    run_report_parser.add_argument("--task-index", required=True, type=int)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        config = init_experiment(
            source_root=args.source_root,
            experiment_root=args.experiment_root,
            subject=args.subject,
            repeat_count_full_face=args.repeat_count_full_face,
            repeat_count_defaced=args.repeat_count_defaced,
            roi_preset=args.roi_preset,
            simulation_overrides=load_simulation_overrides(args.simulation_overrides),
            dry_run=args.dry_run,
        )
        print(json.dumps(config, indent=2, sort_keys=True, default=json_default))
    elif args.command == "prepare-inputs":
        print(json.dumps(prepare_inputs(args.experiment_root, copy_inputs=args.copy_inputs), indent=2, sort_keys=True, default=json_default))
    elif args.command == "deface-inputs":
        print(json.dumps(deface_inputs(args.experiment_root, fsl_deface_bin=args.fsl_deface_bin), indent=2, sort_keys=True, default=json_default))
    elif args.command == "submit-simulations":
        print(json.dumps(submit_simulations(args.experiment_root, max_concurrent=args.max_concurrent), indent=2, sort_keys=True, default=json_default))
    elif args.command == "analyze":
        if args.local_only:
            print(json.dumps(analyze_local(args.experiment_root), indent=2, sort_keys=True, default=json_default))
        else:
            print(json.dumps(analyze_experiment(args.experiment_root, max_concurrent=args.max_concurrent), indent=2, sort_keys=True, default=json_default))
    elif args.command == "make-figures":
        print(json.dumps(make_figures(args.experiment_root), indent=2, sort_keys=True, default=json_default))
    elif args.command == "status":
        status = status_experiment(args.experiment_root)
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True, default=json_default))
        else:
            print_status(status)
    elif args.command == "run-simulation-task":
        print(json.dumps(run_simulation_task(args.task_manifest, args.task_index, dry_run=args.dry_run), indent=2, sort_keys=True, default=json_default))
    elif args.command == "run-report-task":
        print(json.dumps(run_report_task(args.task_manifest, args.task_index), indent=2, sort_keys=True, default=json_default))
    else:
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
