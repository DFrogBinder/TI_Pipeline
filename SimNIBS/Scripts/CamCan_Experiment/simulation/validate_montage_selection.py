#!/usr/bin/env python3
"""Fail closed when a CamCan dataset is paired with the wrong montage preset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.camcan_dataset import (  # noqa: E402
    electrode_names_for_config,
    sha256_file,
    validate_dataset_montage,
)


def read_manifest_dataset_names(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    names = tuple(dict.fromkeys(row.get("dataset_name", "").strip() for row in rows))
    return tuple(name for name in names if name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--dataset-name", action="append")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--expected-targets-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_names = (
        read_manifest_dataset_names(args.manifest.expanduser())
        if args.manifest
        else tuple(args.dataset_name)
    )
    targets_csv = args.targets_csv.expanduser().resolve()
    actual_hash = sha256_file(targets_csv)
    if args.expected_targets_sha256 and actual_hash != args.expected_targets_sha256:
        raise SystemExit(
            "targets.csv hash changed after submission: "
            f"{actual_hash} != {args.expected_targets_sha256}"
        )

    config = validate_dataset_montage(dataset_names, args.preset)
    names = electrode_names_for_config(config, targets_csv)
    print(
        json.dumps(
            {
                "status": "ready",
                "datasets": len(dataset_names),
                "dataset_prefix": config.dataset_prefix,
                "targets_roi": config.targets_roi,
                "montage_preset": config.montage_preset,
                "electrode_names": list(names),
                "targets_csv": str(targets_csv),
                "targets_csv_sha256": actual_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
