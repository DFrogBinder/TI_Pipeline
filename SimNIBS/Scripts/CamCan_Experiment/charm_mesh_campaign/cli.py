"""Command-line interface for reusable CHARM meshing campaigns."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_campaign_config, normalized_config
from .runner import run_task, validate_campaign
from .submit_mesh_campaign import submit_campaign
from .tasks import build_task_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "describe", "validate", "submit"):
        child = commands.add_parser(name)
        child.add_argument("--config", required=True)
        if name == "validate":
            child.add_argument("--skip-hashes", action="store_true")
        if name == "submit":
            child.add_argument("--submit", action="store_true")
            child.add_argument("--sbatch-bin", default="sbatch")
    task = commands.add_parser("run-task")
    task.add_argument("--config", required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--staging-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = load_campaign_config(args.config)
        if args.command == "preflight":
            payload = build_task_manifest(config)
        elif args.command == "describe":
            payload = {
                "campaign_id": config.campaign_id,
                "config_path": str(config.source_path),
                "config_sha256": config.sha256,
                "configuration": normalized_config(config),
                "submission_gated": True,
            }
        elif args.command == "run-task":
            payload = run_task(
                config, args.task_index, staging_root=args.staging_root
            )
        elif args.command == "validate":
            payload = validate_campaign(config, verify_hashes=not args.skip_hashes)
        else:
            payload = submit_campaign(
                config, submit=args.submit, sbatch_bin=args.sbatch_bin
            )
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        if payload.get("status") in {"blocked", "incomplete"}:
            return 2
        return 0
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
