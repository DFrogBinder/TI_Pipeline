"""Command-line interface for the finite-support audit."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .config import cell_from_index, default_output_root
from .io_utils import utc_now, write_json_atomic
from .preflight import preflight, scope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("scope", help="Print the fixed full-study audit scope.")
    check = subparsers.add_parser("preflight", help="Verify all retained inputs and logs.")
    check.add_argument("--output-root", type=Path, default=default_output_root())
    check.add_argument("--quick", action="store_true", help="Skip the full run-level inventory.")
    cell = subparsers.add_parser("audit-cell", help="Audit one target-subject array cell.")
    cell.add_argument("--cell-index", type=int, required=True)
    cell.add_argument("--output-root", type=Path, default=default_output_root())
    collector = subparsers.add_parser("collect", help="Validate and collect all 20 cells.")
    collector.add_argument("--output-root", type=Path, default=default_output_root())
    collector.add_argument("--archive", type=Path)
    renderer = subparsers.add_parser(
        "render-request", help="Render one cell request with a plotting-capable Python."
    )
    renderer.add_argument("--request", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "scope":
        result = scope()
    elif args.command == "preflight":
        result = preflight(output_root=args.output_root, full_scan=not args.quick)
    elif args.command == "audit-cell":
        try:
            from .workflow import audit_cell

            result = audit_cell(cell_index=args.cell_index, output_root=args.output_root)
        except Exception as exc:
            target, _, subject = cell_from_index(args.cell_index)
            failure = {
                "status": "failed",
                "created_utc": utc_now(),
                "cell_index": args.cell_index,
                "target": target.key,
                "subject": subject,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "source_outputs_modified": False,
            }
            write_json_atomic(
                args.output_root / "cells" / target.key / subject / "failed.json",
                failure,
            )
            raise
    elif args.command == "render-request":
        from .cell_render import render_request

        result = render_request(request_path=args.request)
    else:
        from .collect import collect

        result = collect(output_root=args.output_root, archive_path=args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
