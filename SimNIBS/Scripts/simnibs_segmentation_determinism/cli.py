from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .extension_import import install_extension, install_preflight
from .settings import ExperimentConfig, scope_payload
from .workflow import (
    collect,
    initialize,
    run_repeat,
    runtime_preflight,
    status,
    workflow_preflight,
)


def build_visualizations(config: ExperimentConfig):
    """Load the optional plotting stack only for the explicit figures command."""
    from .visualizations import build_visualizations as build

    return build(config)


def build_journal_spatial_figures(config: ExperimentConfig):
    """Load the optional journal plotting stack only for its explicit command."""
    from .journal_spatial_figures import build_journal_spatial_figures as build

    return build(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--staging-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "scope",
        "preflight",
        "init",
        "collect",
        "analyze",
        "figures",
        "journal-spatial-figures",
        "status",
        "runtime-preflight",
        "install-preflight",
    ):
        subparsers.add_parser(command)
    install = subparsers.add_parser("install-extension")
    install.add_argument("--confirm-dataset-root", type=Path, required=True)
    repeat = subparsers.add_parser("run-repeat")
    repeat.add_argument("--arm", required=True)
    repeat.add_argument("--array-index", required=True, type=int)
    repeat.add_argument("--attempt", type=int, default=1)
    return parser


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_env()
    if args.dataset_root is not None:
        config = config.with_dataset_root(args.dataset_root)
    if args.staging_root is not None:
        config = config.with_staging_root(args.staging_root)
    return config


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
        elif args.command == "runtime-preflight":
            payload = runtime_preflight(config)
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
        elif args.command == "figures":
            payload = build_visualizations(config)
        elif args.command == "journal-spatial-figures":
            payload = build_journal_spatial_figures(config)
        elif args.command == "install-preflight":
            payload = install_preflight(config)
        elif args.command == "install-extension":
            payload = install_extension(
                config,
                confirm_dataset_root=args.confirm_dataset_root,
            )
        else:
            payload = status(config)
    except Exception as exc:  # CLI boundary emits one stable machine-readable error.
        _print({"status": "error", "error": str(exc)}, stream=sys.stderr)
        return 2
    _print(payload)
    return 0
