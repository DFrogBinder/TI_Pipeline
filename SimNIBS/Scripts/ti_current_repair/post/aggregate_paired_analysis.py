#!/usr/bin/env python3
"""Aggregate per-subject paired analysis outputs without rerunning NIfTI analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "subject",
    "status",
    "analysis_root",
    "baseline_condition",
    "comparison_condition",
    "compare_metric",
    "baseline_std",
    "comparison_std",
    "std_ratio_comparison_over_baseline",
    "std_reduction_percent",
    "baseline_cv_percent",
    "comparison_cv_percent",
    "cv_reduction_percent",
    "error_type",
    "error_message",
]


def _discover_subjects(experiment_root: Path) -> list[str]:
    analysis_root = experiment_root / "_analysis"
    return sorted(path.name for path in analysis_root.glob("sub-*") if path.is_dir())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _failure_row(subject: str, analysis_root: Path, error_type: str, error_message: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "status": "failed",
        "analysis_root": str(analysis_root),
        "baseline_condition": "",
        "comparison_condition": "",
        "compare_metric": "",
        "baseline_std": "",
        "comparison_std": "",
        "std_ratio_comparison_over_baseline": "",
        "std_reduction_percent": "",
        "baseline_cv_percent": "",
        "comparison_cv_percent": "",
        "cv_reduction_percent": "",
        "error_type": error_type,
        "error_message": error_message,
    }


def _subject_row(experiment_root: Path, subject: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    subject_root = experiment_root / "_analysis" / subject
    comparison_json = subject_root / "condition_comparison.json"
    if not comparison_json.is_file():
        message = f"Missing per-subject comparison JSON: {comparison_json}"
        return _failure_row(subject, subject_root, "FileNotFoundError", message), {
            "subject": subject,
            "analysis_root": str(subject_root),
            "error_type": "FileNotFoundError",
            "error_message": message,
        }
    try:
        payload = _load_json(comparison_json)
        compare_metric = str(payload.get("compare_metric", ""))
        metric_rows = payload.get("metric_rows", [])
        if not isinstance(metric_rows, list):
            raise ValueError("condition_comparison.json metric_rows must be a list")
        primary = next(
            (row for row in metric_rows if isinstance(row, dict) and row.get("metric") == compare_metric),
            None,
        )
        if primary is None:
            raise ValueError(f"Metric row not found for compare_metric={compare_metric!r}")
        return {
            "subject": subject,
            "status": "complete",
            "analysis_root": str(subject_root),
            "baseline_condition": payload.get("baseline_condition", ""),
            "comparison_condition": payload.get("comparison_condition", ""),
            "compare_metric": compare_metric,
            "baseline_std": primary.get("baseline_std", ""),
            "comparison_std": primary.get("comparison_std", ""),
            "std_ratio_comparison_over_baseline": primary.get("std_ratio_comparison_over_baseline", ""),
            "std_reduction_percent": primary.get("std_reduction_percent", ""),
            "baseline_cv_percent": primary.get("baseline_cv_percent", ""),
            "comparison_cv_percent": primary.get("comparison_cv_percent", ""),
            "cv_reduction_percent": primary.get("cv_reduction_percent", ""),
            "error_type": "",
            "error_message": "",
        }, None
    except Exception as exc:
        return _failure_row(subject, subject_root, type(exc).__name__, str(exc)), {
            "subject": subject,
            "analysis_root": str(subject_root),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows)


def aggregate_paired_summary(
    *,
    experiment_root: Path,
    subjects: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    experiment_root = experiment_root.expanduser().resolve()
    output_dir = output_dir or experiment_root / "_analysis"
    subjects = subjects or _discover_subjects(experiment_root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for subject in subjects:
        row, failure = _subject_row(experiment_root, subject)
        rows.append(row)
        if failure is not None:
            failures.append(failure)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_root": str(experiment_root),
        "subjects_total": len(rows),
        "subjects_succeeded": sum(1 for row in rows if row.get("status") == "complete"),
        "subjects_failed": sum(1 for row in rows if row.get("status") != "complete"),
        "subjects": rows,
        "failures": failures,
    }
    (output_dir / "paired_condition_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "paired_condition_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "paired_condition_summary.csv", rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--subjects", default=None, help="Optional comma-separated subject IDs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subjects = [item.strip() for item in args.subjects.split(",") if item.strip()] if args.subjects else None
    summary = aggregate_paired_summary(
        experiment_root=args.experiment_root,
        subjects=subjects,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["subjects_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
