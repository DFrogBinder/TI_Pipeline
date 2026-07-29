#!/usr/bin/env python3
"""Validate and reuse a completed CamCan analysis as a source-data receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ROIS = [
    "Left_Hippocampus",
    "Left_M1",
    "Right_DLPC",
    "Right_Thalamus",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(payload: dict, key: str, expected: object) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Source analysis receipt has {key}={actual!r}; "
            f"expected {expected!r}."
        )


def validate_receipt(
    receipt_path: Path,
    *,
    expected_subjects: int,
    expected_records: int,
    expected_post_campaign_root: Path,
) -> dict:
    if not receipt_path.is_file():
        raise RuntimeError(
            f"Source analysis receipt is missing: {receipt_path}"
        )

    resolved_receipt = receipt_path.resolve()
    resolved_root = expected_post_campaign_root.resolve()
    if not resolved_receipt.is_relative_to(resolved_root):
        raise RuntimeError(
            "Source analysis receipt is outside the expected post-processing "
            f"campaign root: {resolved_receipt} (expected below {resolved_root})."
        )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    require_equal(payload, "analysis_schema_version", 4)
    require_equal(payload, "status", "complete")
    require_equal(payload, "subjects", expected_subjects)
    require_equal(payload, "rois", EXPECTED_ROIS)
    require_equal(payload, "repeats_per_subject_roi", 10)
    require_equal(payload, "repeat_level_records", expected_records)
    require_equal(
        payload,
        "subject_level_records",
        expected_subjects * len(EXPECTED_ROIS),
    )
    require_equal(payload, "mni_baselines", len(EXPECTED_ROIS))
    require_equal(
        payload,
        "execution_mode",
        "full_image_metric_extraction_and_aggregation",
    )

    return {
        "complete": expected_records,
        "hashes_verified": False,
        "incomplete": 0,
        "stage": "simulations",
        "status": "complete",
        "tasks": expected_records,
        "validation_mode": "prior_complete_analysis_receipt",
        "source_analysis_receipt": str(resolved_receipt),
        "source_analysis_receipt_sha256": sha256_file(receipt_path),
        "source_analysis_schema_version": payload["analysis_schema_version"],
        "source_analysis_execution_mode": payload["execution_mode"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument(
        "--expected-post-campaign-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    result = validate_receipt(
        args.receipt,
        expected_subjects=args.expected_subjects,
        expected_records=args.expected_records,
        expected_post_campaign_root=args.expected_post_campaign_root,
    )
    result["summary"] = str(args.output_summary)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_summary.write_text(
        "validation_mode\tstatus\ttasks\tsource_analysis_receipt\t"
        "source_analysis_receipt_sha256\n"
        f"{result['validation_mode']}\t{result['status']}\t"
        f"{result['tasks']}\t{result['source_analysis_receipt']}\t"
        f"{result['source_analysis_receipt_sha256']}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
