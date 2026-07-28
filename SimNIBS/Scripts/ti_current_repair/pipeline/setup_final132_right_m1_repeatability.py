#!/usr/bin/env python3
"""Prepare and submit the full final-132 balanced-10 right-M1 repeatability study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
CURRENT_REPAIR_ROOT = HERE.parents[1]
SCRIPTS_ROOT = CURRENT_REPAIR_ROOT.parent
for import_root in (CURRENT_REPAIR_ROOT, SCRIPTS_ROOT):
    try:
        sys.path.remove(str(import_root))
    except ValueError:
        pass
for import_root in (SCRIPTS_ROOT, CURRENT_REPAIR_ROOT):
    sys.path.insert(0, str(import_root))

from pipeline import staged_median_fixed_experiment as staged  # noqa: E402
from stimulation_config import resolve_confirmed_stimulation  # noqa: E402


SUBJECTS = [
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC520083",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
]
REPEAT_COUNT = 40
ROI_PRESET = "right-m1"
COMPARE_METRIC = "median_roi"
MAX_CONCURRENT = 50
ANALYSIS_MAX_CONCURRENT = 10

DEFAULT_SOURCE_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected"
)
DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1"
)
DEFAULT_ATLAS_DIR = Path("/mnt/parscratch/users/cop23bi/ZIPs/atlases")
DEFAULT_TARGETS_CSV = SCRIPTS_ROOT / "utils" / "targets.csv"


def _resolved(value: Path) -> Path:
    return value.expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _assert_empty_or_initialized(experiment_root: Path) -> None:
    manifest = experiment_root / "_pipeline" / "experiment_manifest.json"
    if manifest.is_file() or not experiment_root.exists():
        return
    if any(experiment_root.iterdir()):
        raise RuntimeError(
            "Refusing to initialize the right-M1 study in a non-empty directory "
            f"without a pipeline manifest: {experiment_root}"
        )


def _expected_manifest(
    *,
    source_root: Path,
    experiment_root: Path,
    atlas_dir: Path,
    targets_csv: Path,
) -> dict[str, Any]:
    return {
        "source_root": str(source_root),
        "experiment_root": str(experiment_root),
        "subjects": SUBJECTS,
        "repeat_count": REPEAT_COUNT,
        "roi_preset": ROI_PRESET,
        "compare_metric": COMPARE_METRIC,
        "atlas_dir": str(atlas_dir),
        "stimulation": resolve_confirmed_stimulation(
            ROI_PRESET,
            targets_csv=targets_csv,
        ).to_dict(),
    }


def _validate_existing_manifest(
    manifest_path: Path,
    *,
    expected: dict[str, Any],
) -> None:
    observed = _load_json(manifest_path)
    mismatches = [
        key
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "Existing experiment manifest does not match the requested full "
            "right-M1 repeatability study. Refusing to overwrite or mix studies. "
            f"Differing fields: {', '.join(mismatches)} ({manifest_path})"
        )


def _initialize_or_validate(
    *,
    source_root: Path,
    experiment_root: Path,
    atlas_dir: Path,
    targets_csv: Path,
) -> Path:
    manifest_path = experiment_root / "_pipeline" / "experiment_manifest.json"
    expected = _expected_manifest(
        source_root=source_root,
        experiment_root=experiment_root,
        atlas_dir=atlas_dir,
        targets_csv=targets_csv,
    )
    _assert_empty_or_initialized(experiment_root)

    if manifest_path.is_file():
        _validate_existing_manifest(manifest_path, expected=expected)
        print(f"[INFO] Reusing validated right-M1 experiment: {experiment_root}")
        return manifest_path

    staged.main(
        [
            "init",
            "--source-root",
            str(source_root),
            "--experiment-root",
            str(experiment_root),
            "--subjects",
            ",".join(SUBJECTS),
            "--repeat-count",
            str(REPEAT_COUNT),
            "--roi-preset",
            ROI_PRESET,
            "--montage-preset",
            ROI_PRESET,
            "--targets-csv",
            str(targets_csv),
            "--atlas-dir",
            str(atlas_dir),
            "--compare-metric",
            COMPARE_METRIC,
        ]
    )
    _validate_existing_manifest(manifest_path, expected=expected)
    return manifest_path


def _assert_full_scope(experiment_root: Path) -> dict[str, Any]:
    scope = staged._workflow_scope(
        experiment_root,
        max_concurrent=MAX_CONCURRENT,
        analysis_max_concurrent=ANALYSIS_MAX_CONCURRENT,
    )
    expected = {
        "subject_count": 10,
        "repeats_per_condition": 40,
        "remesh_tasks": 400,
        "fixed_mesh_tasks": 400,
        "total_simulation_tasks": 800,
        "remesh_array": "0-399%50",
        "fixed_mesh_array": "0-399%50",
        "analysis_array": "0-9%10",
        "expected_ti_msh": 800,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": scope.get(key)}
        for key, expected_value in expected.items()
        if scope.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Resolved workflow does not match the full requested right-M1 scope: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return scope


def _submit_all_args(experiment_root: Path, *, dry_run: bool) -> list[str]:
    args = [
        "submit-all",
        "--experiment-root",
        str(experiment_root),
        "--max-concurrent",
        str(MAX_CONCURRENT),
        "--analysis-max-concurrent",
        str(ANALYSIS_MAX_CONCURRENT),
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def run(args: argparse.Namespace) -> int:
    source_root = _resolved(args.source_root)
    experiment_root = _resolved(args.experiment_root)
    atlas_dir = _resolved(args.atlas_dir)
    targets_csv = _resolved(args.targets_csv)

    _initialize_or_validate(
        source_root=source_root,
        experiment_root=experiment_root,
        atlas_dir=atlas_dir,
        targets_csv=targets_csv,
    )
    scope = _assert_full_scope(experiment_root)

    print("Scope:")
    print("  dataset: final-132 balanced-10 corrected repeatability cohort")
    print("  ROI and montage: right M1")
    print(f"  subjects: {scope['subject_count']}")
    print("  conditions: 2 (remesh, fixed_mesh)")
    print(f"  repeats per condition: {scope['repeats_per_condition']}")
    print(f"  simulation tasks: {scope['total_simulation_tasks']}")
    print(f"  remesh array: {scope['remesh_array']}")
    print(f"  fixed-mesh array: {scope['fixed_mesh_array']}")
    print(f"  analysis array: {scope['analysis_array']}")
    print(f"  expected TI.msh outputs: {scope['expected_ti_msh']}")
    print("  execution: full requested experiment; not a smoke or subset")

    staged.main(_submit_all_args(experiment_root, dry_run=True))
    if args.preflight:
        print("[INFO] Right-M1 preflight passed without submitting jobs.")
        print(
            "[INFO] Submit with: bash "
            "ti_current_repair/hpc_scripts/"
            "submit_final132_right_m1_repeatability.sh"
        )
        return 0

    return staged.main(_submit_all_args(experiment_root, dry_run=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Initialize/validate the isolated experiment and print the full plan without sbatch.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
