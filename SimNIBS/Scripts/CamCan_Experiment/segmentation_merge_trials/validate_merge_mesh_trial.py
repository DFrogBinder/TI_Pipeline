#!/usr/bin/env python3
"""Validate completed mesh-only merge trial outputs for one subject."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


INCOMPLETE_EXIT_CODE = 125
INPUT_EXIT_CODE = 126

STRATEGY_DIRS = {
    "candidate_a": "candidate_A_full_charm_fallback",
    "candidate_b": "candidate_B_solid_charm_head_skin_base",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_strategy(out_root: Path, subject: str, strategy: str) -> list[str]:
    errors: list[str] = []
    strategy_root = out_root / subject / STRATEGY_DIRS[strategy]
    marker = strategy_root / "COMPLETE.json"
    qc = strategy_root / "merge_qc.json"

    if not marker.is_file():
        return [f"{strategy}: missing marker {marker}"]
    if not qc.is_file():
        errors.append(f"{strategy}: missing QC JSON {qc}")

    try:
        payload = load_json(marker)
    except Exception as exc:
        return [f"{strategy}: could not read marker {marker}: {exc}"]

    for key in ("merged_segmentation_path", "installed_charm_label_path", "mesh_path"):
        value = payload.get(key)
        if not value:
            errors.append(f"{strategy}: marker missing {key}")
            continue
        path = Path(value)
        if not path.is_file():
            errors.append(f"{strategy}: {key} missing on disk: {path}")
        elif path.stat().st_size <= 0:
            errors.append(f"{strategy}: {key} is empty: {path}")

    preview = payload.get("preview_path")
    if preview:
        preview_path = Path(preview)
        if not preview_path.is_file():
            errors.append(f"{strategy}: preview missing: {preview_path}")

    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--strategies",
        choices=["both", "candidate_a", "candidate_b"],
        default="both",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_root = Path(args.out_root).expanduser().resolve()
    subject_root = out_root / args.subject
    if not subject_root.is_dir():
        print(f"[ERROR] Subject trial output directory missing: {subject_root}", file=sys.stderr)
        return INPUT_EXIT_CODE

    strategies = ["candidate_a", "candidate_b"] if args.strategies == "both" else [args.strategies]
    errors: list[str] = []
    for strategy in strategies:
        errors.extend(validate_strategy(out_root, args.subject, strategy))

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return INCOMPLETE_EXIT_CODE

    print(f"[INFO] Merge-mesh trial validation passed for {args.subject}: {', '.join(strategies)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
