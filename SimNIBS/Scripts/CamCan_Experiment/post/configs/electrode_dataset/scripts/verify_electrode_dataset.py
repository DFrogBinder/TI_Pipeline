#!/usr/bin/env python3
"""Verify electrode dataset files used by electrode-distance post-processing."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROI_ELECTRODE_SETS = (
    Path(__file__).resolve().parents[1].parent
    / "electrode_examples"
    / "roi_electrode_sets.csv"
)
DEFAULT_TARGETS_CSV = Path(__file__).resolve().parents[5] / "utils" / "targets.csv"
TARGET_ROI_ALIASES = {
    "ctx-lh-g-precentral": "left-m1",
    "ctx-lh-g-front-middle": "left-dlpc",
    "left-hippocampus": "left-hippocampus",
    "left-thalamus": "left-thalamus",
    "left-pallidum": "left-pallidum",
    "ctx-rh-g-precentral": "right-m1",
    "ctx-rh-g-front-middle": "right-dlpc",
    "right-hippocampus": "right-hippocampus",
    "right-thalamus": "right-thalamus",
    "right-pallidum": "right-pallidum",
}
REQUIRED_COLUMNS = {"subject", "electrode", "x", "y", "z"}


def slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def read_roi_electrode_sets(path: Path) -> dict[str, list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            slug(row["roi_alias"]): row["electrode_names"].split()
            for row in csv.DictReader(handle)
        }


def read_target_electrode_sets(path: Path) -> dict[str, list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        TARGET_ROI_ALIASES.get(slug(row["roi"]), slug(row["roi"])): row["pair1"].split("-")
        + row["pair2"].split("-")
        for row in rows
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def finite_coordinate(row: dict[str, str]) -> bool:
    try:
        return all(math.isfinite(float(row[axis])) for axis in ("x", "y", "z"))
    except (TypeError, ValueError):
        return False


def verify_dataset(
    dataset_dir: Path,
    roi_electrode_sets_csv: Path,
    targets_csv: Path | None,
) -> dict[str, object]:
    expected_by_roi: dict[str, list[str]] = {}
    if roi_electrode_sets_csv.is_file():
        expected_by_roi.update(read_roi_electrode_sets(roi_electrode_sets_csv))
    if targets_csv and targets_csv.is_file():
        expected_by_roi.update(read_target_electrode_sets(targets_csv))

    if not expected_by_roi:
        raise SystemExit(
            f"No expected electrode sets were parsed from {roi_electrode_sets_csv} or {targets_csv}"
        )
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    report_rows: list[dict[str, str | int]] = []
    errors: list[str] = []
    total_subject_files = 0
    total_rows = 0

    for manifest in manifest_rows:
        roi_alias = slug(manifest["roi_alias"])
        expected_electrodes = expected_by_roi.get(roi_alias)
        if not expected_electrodes:
            errors.append(f"{roi_alias}: no expected electrode set found")
            continue

        roi_dir = dataset_dir / roi_alias
        consolidated = roi_dir / "electrode_centers.csv"
        if not consolidated.is_file():
            errors.append(f"{roi_alias}: missing consolidated file {consolidated}")
            continue

        consolidated_rows = read_csv(consolidated)
        subjects = sorted({row["subject"] for row in consolidated_rows})
        if len(subjects) != int(manifest["subjects"]):
            errors.append(
                f"{roi_alias}: manifest subject count {manifest['subjects']} "
                f"!= consolidated count {len(subjects)}"
            )

        for subject in subjects:
            subject_file = roi_dir / subject / "electrodes.csv"
            if not subject_file.is_file():
                errors.append(f"{roi_alias}/{subject}: missing {subject_file}")
                continue
            rows = read_csv(subject_file)
            electrodes = [row["electrode"] for row in rows]
            bad_coords = [row["electrode"] for row in rows if not finite_coordinate(row)]
            if [row["subject"] for row in rows] != [subject] * len(rows):
                errors.append(f"{roi_alias}/{subject}: subject column does not match file path")
            if electrodes != expected_electrodes:
                errors.append(
                    f"{roi_alias}/{subject}: electrodes {electrodes} != expected {expected_electrodes}"
                )
            if bad_coords:
                errors.append(
                    f"{roi_alias}/{subject}: non-finite coordinate(s) for {', '.join(bad_coords)}"
                )
            total_subject_files += 1
            total_rows += len(rows)

        report_rows.append(
            {
                "roi_alias": roi_alias,
                "subjects": len(subjects),
                "electrodes_per_subject": len(expected_electrodes),
                "rows": len(consolidated_rows),
                "expected_electrode_names": " ".join(expected_electrodes),
            }
        )

    report_csv = dataset_dir / "verification_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "roi_alias",
                "subjects",
                "electrodes_per_subject",
                "rows",
                "expected_electrode_names",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "dataset_dir": str(dataset_dir),
        "roi_electrode_sets_csv": str(roi_electrode_sets_csv),
        "targets_csv": str(targets_csv) if targets_csv else None,
        "roi_count": len(report_rows),
        "subject_files": total_subject_files,
        "rows": total_rows,
        "errors": errors,
        "status": "ok" if not errors else "error",
        "report_csv": str(report_csv),
    }
    (dataset_dir / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--roi-electrode-sets-csv", type=Path, default=DEFAULT_ROI_ELECTRODE_SETS)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summary = verify_dataset(
        dataset_dir=args.dataset_dir.expanduser(),
        roi_electrode_sets_csv=args.roi_electrode_sets_csv.expanduser(),
        targets_csv=args.targets_csv.expanduser() if args.targets_csv else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict and summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
