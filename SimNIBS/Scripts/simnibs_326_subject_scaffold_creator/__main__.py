"""CLI for the generic SimNIBS 3.2.6 subject scaffold creator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .creator import build, preflight, status, validate
from .settings import (
    DEFAULT_CORRECTED_SOURCE_ROOT,
    DEFAULT_FINAL132_SUBJECTS,
    DEFAULT_OUTPUT_ROOT,
    CreatorConfig,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "build", "validate", "status"):
        child = subparsers.add_parser(name)
        child.add_argument("--subject", required=True)
        child.add_argument(
            "--corrected-source-root",
            type=Path,
            default=DEFAULT_CORRECTED_SOURCE_ROOT,
        )
        child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        child.add_argument(
            "--final132-subjects", type=Path, default=DEFAULT_FINAL132_SUBJECTS
        )
        if name == "build":
            child.add_argument("--confirm-output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = CreatorConfig(
        subject=args.subject,
        corrected_source_root=args.corrected_source_root,
        output_root=args.output_root,
        final132_subjects_file=args.final132_subjects,
    )
    if args.command == "build":
        expected = config.output_root.expanduser().resolve(strict=False)
        observed = args.confirm_output_root.expanduser().resolve(strict=False)
        if observed != expected:
            _parser().error(
                f"--confirm-output-root must exactly match {expected}"
            )
    actions = {
        "preflight": preflight,
        "build": build,
        "validate": validate,
        "status": status,
    }
    try:
        payload = actions[args.command](config)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
