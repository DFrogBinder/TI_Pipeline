#!/usr/bin/env python3
"""Build the reusable balanced-10 SimNIBS 3.2.6 headreco scaffolds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .cat12_compat import PATCH_ID
    from .scaffold_core import (
        ScaffoldRequest,
        build_scaffold as _build_scaffold,
        scaffold_paths as _scaffold_paths,
        validate_scaffold as _validate_scaffold,
    )
    from .settings import (
        HEAD_MODEL_STRATEGY,
        SEGMENTATION_PROVENANCE,
        SIMNIBS_MODULE,
        SCAFFOLD_ROOT,
        SUBJECTS,
        assert_output_root_isolation,
        source_paths,
    )
except ImportError:  # Direct execution from the historical Slurm launcher.
    from cat12_compat import PATCH_ID
    from scaffold_core import (  # type: ignore
        ScaffoldRequest,
        build_scaffold as _build_scaffold,
        scaffold_paths as _scaffold_paths,
        validate_scaffold as _validate_scaffold,
    )
    from settings import (  # type: ignore
        HEAD_MODEL_STRATEGY,
        SEGMENTATION_PROVENANCE,
        SIMNIBS_MODULE,
        SCAFFOLD_ROOT,
        SUBJECTS,
        assert_output_root_isolation,
        source_paths,
    )


SUBJECT_PATTERN = re.compile(r"^sub-CC[0-9]{6}$")


def _subject_from_index(index: int) -> str:
    if index < 0 or index >= len(SUBJECTS):
        raise IndexError(f"subject index {index} outside 0..{len(SUBJECTS) - 1}")
    return SUBJECTS[index]


def _canonical_subject(value: str) -> str:
    if not SUBJECT_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "subject must use canonical form sub-CC######"
        )
    return value


def _request(subject: str) -> ScaffoldRequest:
    t1, t2, corrected_labels = source_paths(subject)
    return ScaffoldRequest(
        subject=subject,
        t1=t1,
        t2=t2,
        corrected_labels=corrected_labels,
        scaffold_root=SCAFFOLD_ROOT,
        simnibs_module=SIMNIBS_MODULE,
        cat12_compat_script=Path(__file__).resolve().parent / "cat12_compat.py",
        cat12_patch_id=PATCH_ID,
        head_model_strategy=HEAD_MODEL_STRATEGY,
        segmentation_provenance=SEGMENTATION_PROVENANCE,
        workflow="simnibs_326_repeatability",
    )


def scaffold_paths(subject: str) -> dict[str, Path]:
    paths = _scaffold_paths(_request(subject))
    return {
        "anat": paths.anat,
        "m2m": paths.m2m,
        "native_mesh": paths.native_mesh,
        "compatibility_mesh": paths.compatibility_mesh,
        "ready_marker": paths.ready_marker,
    }


def validate_scaffold(subject: str) -> dict[str, object]:
    return _validate_scaffold(_request(subject))


def build_scaffold(subject: str) -> dict[str, object]:
    assert_output_root_isolation()
    return _build_scaffold(_request(subject))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--json", action="store_true")

    for name in ("build", "validate"):
        child = subparsers.add_parser(name)
        group = child.add_mutually_exclusive_group(required=True)
        group.add_argument("--subject-index", type=int)
        group.add_argument("--subject", type=_canonical_subject)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        payload = {
            "simnibs_module": SIMNIBS_MODULE,
            "task_count": len(SUBJECTS),
            "subjects": list(SUBJECTS),
            "head_model_strategy": HEAD_MODEL_STRATEGY,
            "segmentation_provenance": SEGMENTATION_PROVENANCE,
        }
        print(json.dumps(payload, indent=2) if args.json else len(SUBJECTS))
        return 0

    subject = args.subject or _subject_from_index(args.subject_index)
    payload = (
        build_scaffold(subject)
        if args.command == "build"
        else validate_scaffold(subject)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
