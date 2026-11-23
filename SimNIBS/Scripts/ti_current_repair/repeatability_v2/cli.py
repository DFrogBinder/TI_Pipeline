"""Command-line gates for v2 preflight, release, stage planning, and workers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .orchestration import (
    build_production_stage_plan,
    preflight,
    production_release_gate,
)
from .profile import load_profile, verify_profile_source_inputs


def _json(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repeatability-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--profile", required=True)
    preflight_parser.add_argument("--evidence", required=True)
    preflight_parser.add_argument("--output-root", required=True)
    release = subparsers.add_parser("release-gate")
    release.add_argument("--local-validation", required=True)
    release.add_argument("--hpc-pilot", required=True)
    subparsers.add_parser("stage-plan")
    worker = subparsers.add_parser("run-stage")
    worker.add_argument("--profile", required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--task-index", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        profile = load_profile(Path(args.profile))
        requested = Path(args.output_root).expanduser().resolve(strict=False)
        if requested != profile.output_root:
            raise ValueError("--output-root differs from the operational profile")
        evidence = _json(args.evidence)
        source_verification = verify_profile_source_inputs(profile)
        evidence["input_hashes_valid"] = source_verification["status"] == "verified"
        result = preflight(
            config=profile.study_config,
            evidence=evidence,
            output_root=requested,
        )
        result["profile_source_verification"] = source_verification
    elif args.command == "release-gate":
        result = production_release_gate(
            local_validation=_json(args.local_validation),
            hpc_pilot=_json(args.hpc_pilot),
        )
    elif args.command == "stage-plan":
        result = {
            "schema_version": 2,
            "stages": [asdict(stage) for stage in build_production_stage_plan()],
        }
    else:
        from .worker import run_stage

        result = run_stage(
            profile=load_profile(Path(args.profile)),
            stage=args.stage,
            task_index=args.task_index,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0

