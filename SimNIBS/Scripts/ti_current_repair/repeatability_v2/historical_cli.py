"""Command line interface for historical native component-field extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repeatability_combined_analysis_v2.workflow import run_preliminary_analysis

from .historical import (
    VERSION_SPECS,
    collect_version,
    extract_task,
    initialize_output,
    preflight_historical,
    preliminary_metric_sources,
    task_matrix,
)


DEFAULT_DATASET_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_historical_component_analysis_v1"
)
DEFAULT_V4_ATLAS_DIR = Path("/mnt/parscratch/users/cop23bi/ZIPs/atlases")


def _shared(parser: argparse.ArgumentParser, *, atlas: bool = False) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    if atlas:
        parser.add_argument("--v4-atlas-dir", type=Path, default=DEFAULT_V4_ATLAS_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only native extraction of historical TI and pair-1 component fields"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _shared(preflight, atlas=True)
    preflight.add_argument(
        "--skip-native-probe",
        action="store_true",
        help="Path-only developer check; HPC preflight must retain native probes.",
    )
    initialize = commands.add_parser("initialize-output")
    _shared(initialize)
    extract = commands.add_parser("extract-task")
    _shared(extract, atlas=True)
    extract.add_argument("--version", choices=tuple(VERSION_SPECS), required=True)
    extract.add_argument("--task-index", type=int, required=True)
    collect = commands.add_parser("collect-version")
    _shared(collect)
    collect.add_argument("--version", choices=tuple(VERSION_SPECS), required=True)
    analyse = commands.add_parser("analyse-preliminary")
    _shared(analyse)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight_historical(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            v4_atlas_dir=args.v4_atlas_dir,
            probe_native=not args.skip_native_probe,
        )
    elif args.command == "initialize-output":
        result = initialize_output(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
    elif args.command == "extract-task":
        tasks = task_matrix(args.version)
        if args.task_index < 0 or args.task_index >= len(tasks):
            raise SystemExit(
                f"task index {args.task_index} outside 0..{len(tasks) - 1}"
            )
        result = extract_task(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            v4_atlas_dir=args.v4_atlas_dir,
            task=tasks[args.task_index],
        )
    elif args.command == "collect-version":
        result = collect_version(
            output_root=args.output_root,
            version=args.version,
        )
    else:
        sources = preliminary_metric_sources(args.output_root)
        result = run_preliminary_analysis(
            sources=sources,
            output_root=args.output_root / "preliminary_analysis",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
