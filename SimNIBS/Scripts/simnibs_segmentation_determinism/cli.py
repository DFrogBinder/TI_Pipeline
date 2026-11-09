from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from .settings import ExperimentConfig, scope_payload
from .workflow import collect, initialize, run_repeat, status, workflow_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("scope", "preflight", "init", "collect", "analyze", "status"):
        subparsers.add_parser(command)
    repeat = subparsers.add_parser("run-repeat")
    repeat.add_argument("--arm", required=True)
    repeat.add_argument("--array-index", required=True, type=int)
    repeat.add_argument("--attempt", type=int, default=1)
    return parser


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_env()
    changes = {}
    if args.source_root is not None:
        changes["source_root"] = args.source_root
    if args.output_root is not None:
        changes["output_root"] = args.output_root
    return replace(config, **changes)


def _print(payload, *, stream=None) -> None:
    print(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        file=sys.stdout if stream is None else stream,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args)
        if args.command == "scope":
            payload = scope_payload(config)
        elif args.command == "preflight":
            payload = workflow_preflight(config)
        elif args.command == "init":
            payload = initialize(config)
        elif args.command == "run-repeat":
            payload = run_repeat(
                config,
                arm=args.arm,
                array_index=args.array_index,
                attempt=args.attempt,
            )
        elif args.command in {"collect", "analyze"}:
            payload = collect(config)
        else:
            payload = status(config)
    except Exception as exc:  # CLI boundary emits one stable machine-readable error.
        _print({"status": "error", "error": str(exc)}, stream=sys.stderr)
        return 2
    _print(payload)
    return 0
