#!/usr/bin/env python3
"""Repair and compare CamCan runs affected by the pair-2 current bug."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve()
CAMCAN_ROOT = HERE.parents[1]
SCRIPTS_ROOT = HERE.parents[2]
for path in (CAMCAN_ROOT, SCRIPTS_ROOT, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from target_montages import MontageSpec, PairSpec, resolve_montage_preset  # noqa: E402
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


DEFAULT_DATASET_GLOB = "*_Data_*"
DATASET_PATTERN = re.compile(r"^.+_Data_\d+$")


CAMCAN_CONDUCTIVITIES = {
    "WM": 0.126,
    "GM": 0.276,
    "CSF": 1.65,
    "Skull": 0.01,
    "Scalp": 0.465,
    "Eye": 0.5,
    "Muscle": 0.16,
}


def _resolve_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def should_skip_unaffected_preset(
    pair1: PairSpec,
    pair2: PairSpec,
    *,
    include_unaffected: bool = False,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> bool:
    if include_unaffected:
        return False
    return abs(pair1.current_a - pair2.current_a) <= max(abs_tol, rel_tol * max(abs(pair1.current_a), abs(pair2.current_a)))


def camcan_scale_factor(montage: MontageSpec) -> float:
    if montage.pair1.current_a == 0:
        raise ZeroDivisionError(f"Montage {montage.name!r} has zero pair-1 current.")
    return montage.pair2.current_a / montage.pair1.current_a


def current_spec_for_montage(montage: MontageSpec) -> CurrentSpec:
    return CurrentSpec(
        pair1_label=f"{montage.pair1.anode}-{montage.pair1.cathode}",
        pair1_current_ma=montage.pair1.current_a * 1e3,
        pair2_label=f"{montage.pair2.anode}-{montage.pair2.cathode}",
        pair2_intended_current_ma=montage.pair2.current_a * 1e3,
        pair2_original_current_ma=montage.pair1.current_a * 1e3,
    )


def discover_camcan_datasets(
    original_root: str | Path,
    *,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
    datasets: Sequence[str] | None = None,
) -> list[Path]:
    root = _resolve_root(original_root)
    if datasets:
        return [root / dataset for dataset in datasets]
    discovered = [
        path
        for path in root.glob(dataset_glob)
        if path.is_dir() and DATASET_PATTERN.match(path.name)
    ]
    return sorted(discovered, key=lambda item: item.name.casefold())


def discover_subjects(dataset_root: Path, subjects: Sequence[str] | None = None) -> list[str]:
    if subjects:
        return list(subjects)
    return sorted(
        path.name
        for path in dataset_root.glob("sub-*")
        if path.is_dir()
    )


def build_camcan_repair_tasks(
    *,
    original_root: str | Path,
    output_root: str | Path,
    montage_preset: str,
    datasets: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    include_unaffected: bool = False,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
) -> list[RepairTask]:
    montage = resolve_montage_preset(montage_preset)
    if should_skip_unaffected_preset(
        montage.pair1,
        montage.pair2,
        include_unaffected=include_unaffected,
    ):
        return []

    output_root_path = _resolve_root(output_root)
    current_spec = current_spec_for_montage(montage)
    tasks: list[RepairTask] = []
    for dataset_root in discover_camcan_datasets(
        original_root,
        dataset_glob=dataset_glob,
        datasets=datasets,
    ):
        for subject in discover_subjects(dataset_root, subjects):
            original_subject_root = dataset_root / subject
            output_subject_root = output_root_path / dataset_root.name / subject
            tasks.append(
                RepairTask(
                    key=RepairKey(
                        experiment="camcan",
                        subject=subject,
                        dataset=dataset_root.name,
                    ),
                    original_subject_root=original_subject_root,
                    output_subject_root=output_subject_root,
                    original_anat_dir=original_subject_root / "anat",
                    output_anat_dir=output_subject_root / "anat",
                    current_spec=current_spec,
                    montage_name=montage.name,
                    metadata={
                        "montage_roi": montage.roi,
                        "montage_description": montage.description,
                        "scale_factor": camcan_scale_factor(montage),
                    },
                )
            )
    return tasks


def camcan_simulation_spec(task: RepairTask, montage: MontageSpec) -> SimulationSpec:
    subject = task.key.subject
    conductivities = {
        **CAMCAN_CONDUCTIVITIES,
        "Saline": montage.electrode_conductivity,
    }
    return SimulationSpec(
        subject=subject,
        head_mesh=task.output_anat_dir / f"m2m_{subject}" / f"{subject}.msh",
        pair2=ElectrodePair(
            montage.pair2.anode,
            montage.pair2.cathode,
            montage.pair2.current_a,
        ),
        electrode_radius_mm=montage.electrode_radius_mm,
        electrode_thickness_mm=montage.electrode_thickness_mm,
        electrode_shape=montage.electrode_shape,
        electrode_conductivity=montage.electrode_conductivity,
        conductivity_by_name=conductivities,
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
                    "dataset": task.key.dataset,
                    "subject": task.key.subject,
                    "original_subject_root": str(task.original_subject_root),
                    "output_subject_root": str(task.output_subject_root),
                    "montage": task.montage_name,
                    "scale_factor": task.current_spec.scale_factor,
                },
                sort_keys=True,
            )
        )


def _write_unaffected_manifest(args: argparse.Namespace, montage: MontageSpec) -> Path:
    return write_json(
        _resolve_root(args.output_root) / "current_repair_unaffected_manifest.json",
        {
            "experiment": "camcan",
            "mode": args.mode,
            "montage": montage.name,
            "pair1_current_a": montage.pair1.current_a,
            "pair2_current_a": montage.pair2.current_a,
            "status": "unaffected_skipped",
            "reason": "pair1.current_a equals pair2.current_a",
        },
    )


def _run_repair_mode(args: argparse.Namespace) -> int:
    montage = resolve_montage_preset(args.montage_preset)
    tasks = build_camcan_repair_tasks(
        original_root=args.original_root,
        output_root=args.output_root,
        montage_preset=args.montage_preset,
        datasets=args.datasets,
        subjects=args.subjects,
        include_unaffected=args.include_unaffected,
        dataset_glob=args.dataset_glob,
    )
    if args.count_only:
        print(len(tasks))
        return 0
    if not tasks and should_skip_unaffected_preset(
        montage.pair1,
        montage.pair2,
        include_unaffected=args.include_unaffected,
    ):
        manifest = _write_unaffected_manifest(args, montage)
        print(f"[INFO] montage unaffected; wrote manifest: {manifest}")
        return 0
    if args.dry_run:
        _print_plan(_select_task_index(tasks, args.task_index))
        return 0

    for task in _select_task_index(tasks, args.task_index):
        if args.mode == "scaled":
            manifest = repair_scaled_run(task, overwrite=args.overwrite)
        elif args.mode == "pair2-rerun":
            manifest = repair_pair2_rerun_run(
                task,
                camcan_simulation_spec(task, montage),
                overwrite=args.overwrite,
            )
        else:
            raise AssertionError(args.mode)
        print(f"[INFO] wrote manifest: {manifest}")
    return 0


def _run_compare_mode(args: argparse.Namespace) -> int:
    if not args.scaled_root or not args.pair2_rerun_root:
        raise SystemExit("--mode compare requires --scaled-root and --pair2-rerun-root.")
    scaled_tasks = build_camcan_repair_tasks(
        original_root=args.scaled_root,
        output_root=args.scaled_root,
        montage_preset=args.montage_preset,
        datasets=args.datasets,
        subjects=args.subjects,
        include_unaffected=True,
        dataset_glob=args.dataset_glob,
    )
    rerun_tasks = build_camcan_repair_tasks(
        original_root=args.pair2_rerun_root,
        output_root=args.pair2_rerun_root,
        montage_preset=args.montage_preset,
        datasets=args.datasets,
        subjects=args.subjects,
        include_unaffected=True,
        dataset_glob=args.dataset_glob,
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
        description="Comparing CamCan repairs",
        enabled=_progress_enabled(args.progress),
    ):
        rows.append(
            compare_repaired_run_outputs(
                left_task=scaled,
                right_task=rerun,
                output_root=output_root,
                experiment="camcan",
            )
        )
    summary = write_comparison_summary(rows, output_root)
    print(f"[INFO] wrote comparison summary: {summary['summary_json']}")
    print(f"[INFO] wrote comparison table: {summary['summary_csv']}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair CamCan batch roots by scaling/rerunning pair 2, or compare repaired roots."
    )
    parser.add_argument("--mode", required=True, choices=("scaled", "pair2-rerun", "compare"))
    parser.add_argument("--original-root", help="Original CamCan batch root containing *_Data_## directories.")
    parser.add_argument("--output-root", required=True, help="Repair output root or comparison output root.")
    parser.add_argument("--scaled-root", help="Scaled repair root for compare mode.")
    parser.add_argument("--pair2-rerun-root", help="Pair2-rerun repair root for compare mode.")
    parser.add_argument("--montage-preset", required=True, help="Montage preset key from target_montages.py.")
    parser.add_argument("--datasets", nargs="+", help="Optional dataset directory names to include.")
    parser.add_argument("--subjects", nargs="+", help="Optional subject IDs to include.")
    parser.add_argument("--dataset-glob", default=DEFAULT_DATASET_GLOB, help="Dataset glob for discovery.")
    parser.add_argument("--include-unaffected", action="store_true", help="Do not skip equal-current presets.")
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
