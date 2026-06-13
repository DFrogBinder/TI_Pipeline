#!/usr/bin/env python3
"""Repair and compare Repeatability runs affected by the pair-2 current bug."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve()
PIPELINE_ROOT = HERE.parents[1]
SCRIPTS_ROOT = HERE.parents[2]
for path in (PIPELINE_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_config import (  # noqa: E402
    ExperimentConfig,
    iter_experiment_tasks,
    load_experiment_config,
    subject_condition_repeats_root,
    subject_condition_root,
)
from ti_current_repair.core import (  # noqa: E402
    CurrentSpec,
    ElectrodePair,
    RepairKey,
    RepairTask,
    SimulationSpec,
    compare_repaired_run_outputs,
    iter_with_rich_progress,
    repair_pair2_rerun_run,
    repair_scaled_run,
    write_comparison_summary,
    write_json,
)


REPEATABILITY_CURRENT_SPEC = CurrentSpec(
    pair1_label="F10-P8",
    pair1_current_ma=2.0,
    pair2_label="T7-P7",
    pair2_intended_current_ma=1.588656,
    pair2_original_current_ma=2.0,
)

CAMCAN_CONDUCTIVITIES = {
    "WM": 0.126,
    "GM": 0.276,
    "CSF": 1.65,
    "Skull": 0.01,
    "Scalp": 0.465,
    "Eye": 0.5,
    "Muscle": 0.16,
    "Saline": 1.4,
}


def _resolve_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_config_for_root(config_path: str | Path, root: str | Path) -> ExperimentConfig:
    config = load_experiment_config(config_path, validate_paths=False)
    return replace(config, experiment_root=_resolve_root(root))


def _filter_repeats(tasks, repeats: Sequence[int] | None):
    if repeats is None:
        return tasks
    wanted = set(int(value) for value in repeats)
    return [task for task in tasks if task.repeat_index in wanted]


def build_repeatability_repair_tasks(
    *,
    config_path: str | Path,
    original_root: str | Path,
    output_root: str | Path,
    subjects: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
) -> list[RepairTask]:
    original_config = _load_config_for_root(config_path, original_root)
    output_config = _load_config_for_root(config_path, output_root)
    experiment_tasks = iter_experiment_tasks(
        original_config,
        subjects=list(subjects) if subjects else None,
        condition_names=list(conditions) if conditions else None,
    )
    experiment_tasks = _filter_repeats(experiment_tasks, repeats)

    repair_tasks: list[RepairTask] = []
    for task in experiment_tasks:
        original_repeat_root = (
            subject_condition_repeats_root(original_config, task.subject, task.condition_name)
            / task.repeat_tag
        )
        output_repeat_root = (
            subject_condition_repeats_root(output_config, task.subject, task.condition_name)
            / task.repeat_tag
        )
        original_subject_root = original_repeat_root / task.subject
        output_subject_root = output_repeat_root / task.subject
        repair_tasks.append(
            RepairTask(
                key=RepairKey(
                    experiment="repeatability",
                    subject=task.subject,
                    condition=task.condition_name,
                    repeat_tag=task.repeat_tag,
                ),
                original_subject_root=original_subject_root,
                output_subject_root=output_subject_root,
                original_anat_dir=original_subject_root / "anat",
                output_anat_dir=output_subject_root / "anat",
                current_spec=REPEATABILITY_CURRENT_SPEC,
                metadata={
                    "config_path": str(Path(config_path).expanduser().resolve()),
                    "mesh_mode": task.mesh_mode,
                    "repeat_index": task.repeat_index,
                },
            )
        )
    return repair_tasks


def _copy_condition_manifests(
    *,
    config_path: str | Path,
    original_root: str | Path,
    output_root: str | Path,
    tasks: Sequence[RepairTask],
) -> None:
    if not tasks:
        return
    original_config = _load_config_for_root(config_path, original_root)
    output_config = _load_config_for_root(config_path, output_root)
    seen: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.key.subject, task.key.condition or "")
        if key in seen:
            continue
        seen.add(key)
        src = subject_condition_root(original_config, task.key.subject, task.key.condition or "") / "condition_manifest.json"
        dst = subject_condition_root(output_config, task.key.subject, task.key.condition or "") / "condition_manifest.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            dst.write_bytes(src.read_bytes())
        else:
            write_json(
                dst,
                {
                    "subject": task.key.subject,
                    "condition": task.key.condition,
                    "repair_note": "Original condition_manifest.json was not present; repair script wrote this placeholder.",
                },
            )


def repeatability_simulation_spec(task: RepairTask) -> SimulationSpec:
    subject = task.key.subject
    return SimulationSpec(
        subject=subject,
        head_mesh=task.output_anat_dir / f"m2m_{subject}" / f"{subject}.msh",
        pair2=ElectrodePair("T7", "P7", REPEATABILITY_CURRENT_SPEC.pair2_intended_current_a),
        electrode_radius_mm=10.0,
        electrode_thickness_mm=2.0,
        electrode_shape="ellipse",
        electrode_conductivity=1.4,
        conductivity_by_name=CAMCAN_CONDUCTIVITIES,
    )


def _select_task_index(tasks: list[RepairTask], task_index: int | None) -> list[RepairTask]:
    if task_index is None:
        return tasks
    if task_index < 0 or task_index >= len(tasks):
        raise SystemExit(f"Task index {task_index} is out of range for {len(tasks)} task(s).")
    return [tasks[task_index]]


def _progress_enabled(setting: str) -> bool:
    if setting == "always":
        return True
    if setting == "never":
        return False
    return sys.stderr.isatty()


def _print_plan(tasks: Sequence[RepairTask]) -> None:
    for index, task in enumerate(tasks):
        print(
            json.dumps(
                {
                    "task_index": index,
                    "subject": task.key.subject,
                    "condition": task.key.condition,
                    "repeat_tag": task.key.repeat_tag,
                    "original_subject_root": str(task.original_subject_root),
                    "output_subject_root": str(task.output_subject_root),
                    "scale_factor": task.current_spec.scale_factor,
                },
                sort_keys=True,
            )
        )


def _run_repair_mode(args: argparse.Namespace) -> int:
    tasks = build_repeatability_repair_tasks(
        config_path=args.config,
        original_root=args.original_root,
        output_root=args.output_root,
        subjects=args.subjects,
        conditions=args.conditions,
        repeats=args.repeats,
    )
    if args.count_only:
        print(len(tasks))
        return 0
    if args.dry_run:
        _print_plan(_select_task_index(tasks, args.task_index))
        return 0

    selected = _select_task_index(tasks, args.task_index)
    _copy_condition_manifests(
        config_path=args.config,
        original_root=args.original_root,
        output_root=args.output_root,
        tasks=selected,
    )
    for task in selected:
        if args.mode == "scaled":
            manifest = repair_scaled_run(task, overwrite=args.overwrite)
        elif args.mode == "pair2-rerun":
            manifest = repair_pair2_rerun_run(
                task,
                repeatability_simulation_spec(task),
                overwrite=args.overwrite,
            )
        else:
            raise AssertionError(args.mode)
        print(f"[INFO] wrote manifest: {manifest}")
    return 0


def _run_compare_mode(args: argparse.Namespace) -> int:
    if not args.scaled_root or not args.pair2_rerun_root:
        raise SystemExit("--mode compare requires --scaled-root and --pair2-rerun-root.")
    scaled_tasks = build_repeatability_repair_tasks(
        config_path=args.config,
        original_root=args.scaled_root,
        output_root=args.scaled_root,
        subjects=args.subjects,
        conditions=args.conditions,
        repeats=args.repeats,
    )
    rerun_tasks = build_repeatability_repair_tasks(
        config_path=args.config,
        original_root=args.pair2_rerun_root,
        output_root=args.pair2_rerun_root,
        subjects=args.subjects,
        conditions=args.conditions,
        repeats=args.repeats,
    )
    if len(scaled_tasks) != len(rerun_tasks):
        raise SystemExit(
            f"Scaled and pair2-rerun roots resolved different task counts: "
            f"{len(scaled_tasks)} vs {len(rerun_tasks)}."
        )
    if args.count_only:
        print(len(scaled_tasks))
        return 0

    pairs = list(zip(scaled_tasks, rerun_tasks, strict=True))
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(pairs):
            raise SystemExit(f"Task index {args.task_index} is out of range for {len(pairs)} comparison task(s).")
        pairs = [pairs[args.task_index]]
    if args.dry_run:
        for index, (scaled, rerun) in enumerate(pairs):
            print(
                json.dumps(
                    {
                        "comparison_index": index,
                        "key": scaled.key.identity_tuple(),
                        "scaled_root": str(scaled.output_subject_root),
                        "pair2_rerun_root": str(rerun.output_subject_root),
                    },
                    sort_keys=True,
                )
            )
        return 0

    output_root = _resolve_root(args.output_root)
    rows = []
    for scaled, rerun in iter_with_rich_progress(
        pairs,
        total=len(pairs),
        description="Comparing Repeatability repairs",
        enabled=_progress_enabled(args.progress),
    ):
        rows.append(
            compare_repaired_run_outputs(
                left_task=scaled,
                right_task=rerun,
                output_root=output_root,
                experiment="repeatability",
            )
        )
    summary = write_comparison_summary(rows, output_root)
    print(f"[INFO] wrote comparison summary: {summary['summary_json']}")
    print(f"[INFO] wrote comparison table: {summary['summary_csv']}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Repeatability runs by scaling or rerunning pair 2, or compare repaired roots."
    )
    parser.add_argument("--config", required=True, help="Repeatability experiment JSON config.")
    parser.add_argument("--mode", required=True, choices=("scaled", "pair2-rerun", "compare"))
    parser.add_argument("--original-root", help="Original Repeatability experiment root.")
    parser.add_argument("--output-root", required=True, help="Repair output root or comparison output root.")
    parser.add_argument("--scaled-root", help="Scaled repair root for compare mode.")
    parser.add_argument("--pair2-rerun-root", help="Pair2-rerun repair root for compare mode.")
    parser.add_argument("--subjects", nargs="+", help="Optional subject IDs to include.")
    parser.add_argument("--conditions", nargs="+", help="Optional condition names to include.")
    parser.add_argument("--repeats", nargs="+", type=int, help="Optional repeat indexes to include.")
    parser.add_argument("--task-index", type=int, help="Run only one zero-based repair/comparison task.")
    parser.add_argument("--count-only", action="store_true", help="Print task count and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned tasks without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing repaired subject tree.")
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Show Rich progress during compare mode: auto for interactive stderr, always, or never.",
    )
    args = parser.parse_args(argv)
    if args.mode in {"scaled", "pair2-rerun"} and not args.original_root:
        parser.error("--mode scaled and --mode pair2-rerun require --original-root.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "compare":
        return _run_compare_mode(args)
    return _run_repair_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
