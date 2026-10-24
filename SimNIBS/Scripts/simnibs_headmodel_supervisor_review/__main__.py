"""CLI for the receipt-bound SimNIBS supervisor review package."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from .package import (
    build_package,
    preflight,
    select_subject,
    status,
    validate_package,
)
from .settings import ReviewPaths, paths_from_environment


def _add_paths(parser: argparse.ArgumentParser) -> None:
    defaults = paths_from_environment()
    for field in fields(ReviewPaths):
        option = "--" + field.name.replace("_", "-")
        parser.add_argument(option, type=Path, default=getattr(defaults, field.name))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "select", "build", "status", "validate"):
        child = subparsers.add_parser(command)
        _add_paths(child)
        if command == "build":
            child.add_argument("--confirm-output-root", type=Path, required=True)
    return parser


def _paths(args: argparse.Namespace) -> ReviewPaths:
    return ReviewPaths(
        **{field.name: getattr(args, field.name) for field in fields(ReviewPaths)}
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.command == "build":
        expected = paths.output_root
        observed = args.confirm_output_root.expanduser().resolve(strict=False)
        if observed != expected:
            parser.error(f"--confirm-output-root must exactly match {expected}")
    actions = {
        "preflight": preflight,
        "select": select_subject,
        "build": build_package,
        "status": status,
        "validate": validate_package,
    }
    try:
        payload = actions[args.command](paths)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.command == "status" and payload.get("status") == "absent":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
