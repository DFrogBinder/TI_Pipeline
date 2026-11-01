"""Login-node command line interface for consolidation and retirement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Sequence

from .archive import ArchiveError, archive_copy, archive_preflight, archive_verify
from .retirement import (
    RetirementConfirmations,
    RetirementError,
    RetirementPlan,
    plan_retirement,
    resume_retirement,
    retirement_status,
    submit_retirement,
)
from .settings import DatasetConfig, production_config
from .workflow import (
    WorkflowError,
    complete_dataset,
    copy_dataset,
    dataset_status,
    preflight,
    prepare,
    verify_dataset,
)


MODES = (
    "preflight",
    "prepare",
    "copy",
    "verify",
    "checksum",
    "status",
    "archive-preflight",
    "archive-copy",
    "archive-verify",
    "archive-checksum",
    "retire-plan",
    "retire-submit",
    "retire-status",
)


class CliError(RuntimeError):
    """Raised for an incomplete or non-exact operator confirmation."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, verify, archive, and separately retire repeatability data."
    )
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--confirm-output-root")
    parser.add_argument("--confirm-archive-root")
    parser.add_argument("--dataset-receipt-sha256")
    parser.add_argument("--archive-receipt-sha256")
    parser.add_argument("--retirement-disposition-sha256")
    parser.add_argument("--confirm-retirement", action="store_true")
    return parser


def _print_scope(config: DatasetConfig) -> None:
    print("Scope:")
    print("  workflow: final-132 combined repeatability dataset")
    print(f"  dataset root: {config.output_root}")
    print(f"  Shared archive root: {config.shared_archive_root}")
    print(f"  subjects: {len(config.subjects)}")
    print("  versions: 2 (SimNIBS 3.2.6, SimNIBS 4)")
    print("  targets: 2 (left hippocampus, right M1)")
    print("  conditions: 2 (remesh, fixed_mesh)")
    print(f"  repeats per condition: {config.repeat_count}")
    print(f"  subject-condition cells: {config.expected_cells}")
    print(f"  simulation records: {config.expected_simulations}")
    print("  execution: login-node filesystem operations only")
    print("  jobs submitted: 0")
    print("  retirement default: plan only")


def _plan_dict(plan: RetirementPlan) -> dict[str, object]:
    return {
        "status": "ready",
        "dry_run": plan.dry_run,
        "plan_sha256": plan.plan_sha256,
        "dataset_receipt": str(plan.dataset_receipt),
        "dataset_receipt_sha256": plan.dataset_receipt_sha256,
        "archive_receipt": str(plan.archive_receipt) if plan.archive_receipt else None,
        "archive_receipt_sha256": plan.archive_receipt_sha256,
        "retirement_disposition": str(plan.retirement_disposition),
        "retirement_disposition_sha256": plan.retirement_disposition_sha256,
        "retirement_disposition_summary": plan.retirement_disposition_summary,
        "targets": [target.as_dict() for target in plan.targets],
        "jobs_submitted": 0,
    }


def _exact(value: str | None, expected: Path, label: str) -> None:
    if value != str(expected):
        raise CliError(f"{label} must exactly equal: {expected}")


def _sha(value: str | None, label: str) -> str:
    if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CliError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def main(
    argv: Sequence[str] | None = None,
    *,
    config: DatasetConfig | None = None,
) -> int:
    args = _parser().parse_args(argv)
    current = config if config is not None else production_config()
    _print_scope(current)
    try:
        if args.mode == "preflight":
            result = preflight(current)
        elif args.mode == "prepare":
            result = prepare(current)
        elif args.mode == "copy":
            _exact(
                args.confirm_output_root,
                current.output_root,
                "--confirm-output-root",
            )
            result = copy_dataset(current)
        elif args.mode == "verify":
            result = verify_dataset(current, checksum=False)
        elif args.mode == "checksum":
            verification = verify_dataset(current, checksum=True)
            result = (
                complete_dataset(current)
                if verification.get("status") == "matched"
                else verification
            )
        elif args.mode == "status":
            result = dataset_status(current)
        elif args.mode == "archive-preflight":
            result = archive_preflight(current)
        elif args.mode == "archive-copy":
            _exact(
                args.confirm_archive_root,
                current.shared_archive_root,
                "--confirm-archive-root",
            )
            result = archive_copy(current)
        elif args.mode == "archive-verify":
            result = archive_verify(current, checksum=False)
        elif args.mode == "archive-checksum":
            result = archive_verify(current, checksum=True)
        elif args.mode == "retire-plan":
            result = _plan_dict(plan_retirement(current))
        elif args.mode == "retire-submit":
            _exact(
                args.confirm_output_root,
                current.output_root,
                "--confirm-output-root",
            )
            request_exists = (
                current.deletion_log_root / "retirement_request.json"
            ).is_file()
            plan = None if request_exists else plan_retirement(current)
            confirmations = RetirementConfirmations(
                dataset_root=args.confirm_output_root,
                dataset_receipt_sha256=_sha(
                    args.dataset_receipt_sha256,
                    "--dataset-receipt-sha256",
                ),
                archive_receipt_sha256=(
                    _sha(
                        args.archive_receipt_sha256,
                        "--archive-receipt-sha256",
                    )
                    if request_exists
                    or (plan is not None and plan.archive_receipt_sha256 is not None)
                    else None
                ),
                retirement_disposition_sha256=_sha(
                    args.retirement_disposition_sha256,
                    "--retirement-disposition-sha256",
                ),
                confirm_retirement=args.confirm_retirement,
            )
            if not args.confirm_retirement:
                raise CliError("retire-submit requires --confirm-retirement")
            result = (
                resume_retirement(current, confirmations)
                if request_exists
                else submit_retirement(plan, confirmations)
            )
        else:
            result = retirement_status(current)
    except (ArchiveError, CliError, RetirementError, WorkflowError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))
    return 0
