#!/usr/bin/env python3
"""Validate one paired mesh-repeatability Slurm task."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PIPELINE_ROOT = HERE.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiment_config import (  # noqa: E402
    ExperimentConfig,
    ExperimentTask,
    iter_experiment_tasks,
    load_experiment_config,
    subject_condition_mesh_cache_root,
    subject_condition_repeats_root,
)


@dataclass(frozen=True)
class OutputCheck:
    name: str
    path: str
    ok: bool
    reason: str
    size_bytes: int | None = None


def _first_match(parent: Path, pattern: str) -> Path | None:
    if not parent.is_dir():
        return None
    matches = sorted(path for path in parent.glob(pattern) if path.is_file())
    return matches[0] if matches else None


def _check_file(path: Path, *, name: str, min_bytes: int = 1) -> OutputCheck:
    if not path.is_file():
        return OutputCheck(name=name, path=str(path), ok=False, reason="missing")
    size = path.stat().st_size
    if size < min_bytes:
        return OutputCheck(
            name=name,
            path=str(path),
            ok=False,
            reason=f"too small: {size} bytes < {min_bytes} bytes",
            size_bytes=size,
        )
    return OutputCheck(name=name, path=str(path), ok=True, reason="present", size_bytes=size)


def _check_first_match(parent: Path, pattern: str, *, name: str) -> OutputCheck:
    path = _first_match(parent, pattern)
    if path is None:
        return OutputCheck(
            name=name,
            path=str(parent / pattern),
            ok=False,
            reason="missing",
        )
    return _check_file(path, name=name)


def _task_from_index(config: ExperimentConfig, task_index: int) -> ExperimentTask:
    tasks = iter_experiment_tasks(config)
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(
            f"task_index {task_index} is out of range. Valid range: 0..{len(tasks) - 1}"
        )
    return tasks[task_index]


def validate_task(config: ExperimentConfig, task: ExperimentTask) -> list[OutputCheck]:
    repeat_root = subject_condition_repeats_root(
        config,
        task.subject,
        task.condition_name,
    ) / task.repeat_tag
    subject_root = repeat_root / task.subject
    anat_dir = subject_root / "anat"
    sim_root = anat_dir / "SimNIBS"
    output_root = sim_root / "Output" / task.subject
    mesh_dir = anat_dir / f"m2m_{task.subject}"

    checks = [
        OutputCheck(
            name="repeat_root",
            path=str(repeat_root),
            ok=repeat_root.is_dir(),
            reason="present" if repeat_root.is_dir() else "missing",
        ),
        OutputCheck(
            name="subject_root",
            path=str(subject_root),
            ok=subject_root.is_dir(),
            reason="present" if subject_root.is_dir() else "missing",
        ),
        OutputCheck(
            name="anat_dir",
            path=str(anat_dir),
            ok=anat_dir.is_dir(),
            reason="present" if anat_dir.is_dir() else "missing",
        ),
        _check_file(subject_root / "task_manifest.json", name="task_manifest"),
        _check_file(mesh_dir / f"{task.subject}.msh", name="head_mesh"),
        _check_file(sim_root / "ti_brain_only.nii.gz", name="ti_brain_only"),
        _check_file(output_root / "TI.msh", name="ti_msh"),
        _check_first_match(output_root / "Volume_Labels", "TI_Volumetric_*", name="volume_labels"),
        _check_first_match(output_root / "Volume_Base", "TI_Volumetric_*", name="volume_base"),
    ]

    if task.mesh_mode == "fixed_mesh":
        cache_mesh = (
            subject_condition_mesh_cache_root(config, task.subject, task.condition_name)
            / f"m2m_{task.subject}"
            / f"{task.subject}.msh"
        )
        checks.append(_check_file(cache_mesh, name="mesh_cache_mesh"))

    return checks


def _print_result(config: ExperimentConfig, task: ExperimentTask, checks: list[OutputCheck]) -> None:
    print(
        "[INFO] Validating task: "
        f"subject={task.subject} condition={task.condition_name} "
        f"repeat={task.repeat_tag} mesh_mode={task.mesh_mode}"
    )
    print(f"[INFO] Config: {config.config_path}")
    print(f"[INFO] Experiment root: {config.experiment_root}")

    for check in checks:
        status = "OK" if check.ok else "FAIL"
        size = "" if check.size_bytes is None else f" ({check.size_bytes} bytes)"
        print(f"[{status}] {check.name}: {check.path}{size} - {check.reason}")

    if all(check.ok for check in checks):
        print("[INFO] Repeatability task validation passed.")
    else:
        print("[ERROR] Repeatability task validation failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether one paired repeatability task produced final outputs."
    )
    parser.add_argument("--config", required=True, help="Experiment JSON config.")
    parser.add_argument("--task-index", type=int, required=True, help="0-based task index.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config, validate_paths=False)
    task = _task_from_index(config, args.task_index)
    checks = validate_task(config, task)
    _print_result(config, task, checks)
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
