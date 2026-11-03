"""Command-line interface for the guarded combined repeatability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .settings import AnalysisConfig, production_config
from .workflow import (
    WorkflowError,
    execute,
    preflight,
    prepare,
    rebind_preflight,
    status,
)


MODES = ("scope", "preflight", "prepare", "execute", "status", "rebind-preflight")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--confirm-output-root")
    parser.add_argument("--dataset-root", type=Path)
    return parser


def _scope(config: AnalysisConfig) -> dict[str, int]:
    return {
        "subjects": len(config.subjects),
        "versions": 2,
        "targets": 2,
        "conditions": 2,
        "repeats_per_cell": config.repeat_count,
        "participant_cells": config.expected_cells,
        "run_level_rows": config.expected_rows,
        "meshing_tasks": 0,
        "fem_simulation_tasks": 0,
    }


def _print_scope(config: AnalysisConfig) -> None:
    scope = _scope(config)
    print("Scope:")
    print("  workflow: final-132 combined repeatability analysis")
    print(f"  output root: {config.output_root}")
    print(f"  subjects: {scope['subjects']}")
    print(f"  versions: {scope['versions']}")
    print(f"  targets: {scope['targets']}")
    print(f"  conditions: {scope['conditions']}")
    print(f"  repeats per cell: {scope['repeats_per_cell']}")
    print(f"  participant cells: {scope['participant_cells']}")
    print(f"  run-level rows: {scope['run_level_rows']}")
    print(f"  meshing tasks: {scope['meshing_tasks']}")
    print(f"  FEM simulation tasks: {scope['fem_simulation_tasks']}")
    print("  source campaigns: read-only")
    print("  active combined-dataset copy: not read or modified")
    print("  submission default: none")


def main(
    argv: Sequence[str] | None = None,
    *,
    config: AnalysisConfig | None = None,
) -> int:
    args = _parser().parse_args(argv)
    current = config if config is not None else production_config()
    _print_scope(current)
    try:
        if args.mode == "scope":
            result = {"status": "scope", "scope": _scope(current)}
        elif args.mode == "preflight":
            result = preflight(current)
        elif args.mode == "prepare":
            result = prepare(current, args.confirm_output_root)
        elif args.mode == "execute":
            result = execute(current)
        elif args.mode == "status":
            result = status(current)
        else:
            if args.dataset_root is None:
                raise WorkflowError("rebind-preflight requires --dataset-root")
            result = rebind_preflight(current, args.dataset_root)
    except (WorkflowError, ValueError) as error:
        print(
            json.dumps({"status": "error", "error": str(error)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
