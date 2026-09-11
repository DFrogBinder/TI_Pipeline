#!/usr/bin/env python3
"""Prepare corrected remesh/fixed-mesh tables for the repeatability paper.

The historical optimizer-ROI tables contain the authoritative remesh rows and
fixed-mesh rows generated from anatomically selected representative meshes.
The correction tables contain only fixed-mesh rows generated from the
spherical-ROI median selections.  This adapter keeps the former, replaces the
latter, and writes one balanced 800-row table per target without modifying any
source file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC = "roi_median_v_per_m"
EXPECTED_SUBJECTS = 10
EXPECTED_REPEATS = 40
TARGETS = {
    "left_hippocampus": {
        "roi": "Left_Hippocampus",
        "filename": "left_hippocampus_optimizer_roi_metrics.csv",
    },
    "right_m1": {
        "roi": "Right_M1",
        "filename": "right_m1_optimizer_roi_metrics.csv",
    },
}
REQUIRED_COLUMNS = {
    "schema_version",
    "subject",
    "condition",
    "repeat_tag",
    "roi",
    METRIC,
    "roi_voxels",
    "finite_roi_voxels",
    "nonfinite_roi_voxels",
    "finite_roi_fraction",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv_atomic(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fieldnames} for row in rows
        )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: object, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected finite {label}, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite {label}, got {value!r}")
    return parsed


def _validate_condition_rows(
    rows: list[dict[str, str]],
    *,
    path: Path,
    target: str,
    condition: str,
) -> dict[str, Any]:
    spec = TARGETS[target]
    if not rows:
        raise ValueError(f"No {condition} rows found in {path}")
    missing = sorted(REQUIRED_COLUMNS.difference(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if {row["condition"] for row in rows} != {condition}:
        raise ValueError(f"{path} contains unexpected conditions")
    if {row["roi"] for row in rows} != {spec['roi']}:
        raise ValueError(f"{path} contains the wrong ROI for {target}")
    if {int(row["schema_version"]) for row in rows} != {2}:
        raise ValueError(f"{path} does not contain schema-version 2 rows")

    subjects = sorted({row["subject"] for row in rows})
    if len(subjects) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"{path} contains {len(subjects)} {condition} subjects, "
            f"expected {EXPECTED_SUBJECTS}"
        )
    expected_tags = {
        f"repeat_{index:03d}" for index in range(1, EXPECTED_REPEATS + 1)
    }
    keys: set[tuple[str, str]] = set()
    runs_with_nonfinite_support = 0
    minimum_finite_fraction = 1.0
    for subject in subjects:
        selected = [row for row in rows if row["subject"] == subject]
        tags = {row["repeat_tag"] for row in selected}
        if len(selected) != EXPECTED_REPEATS or tags != expected_tags:
            raise ValueError(f"{path}: incomplete repeats for {subject}/{condition}")
        for row in selected:
            key = (subject, row["repeat_tag"])
            if key in keys:
                raise ValueError(f"{path}: duplicate key {subject}/{row['repeat_tag']}")
            keys.add(key)
            _finite(row[METRIC], label=f"{subject}/{condition}/{row['repeat_tag']}")
            finite_fraction = _finite(
                row["finite_roi_fraction"],
                label=f"{subject}/{condition}/{row['repeat_tag']}/finite fraction",
            )
            if not 0.0 < finite_fraction <= 1.0:
                raise ValueError(f"{path}: invalid finite ROI fraction {finite_fraction}")
            roi_voxels = int(row["roi_voxels"])
            finite_voxels = int(row["finite_roi_voxels"])
            nonfinite_voxels = int(row["nonfinite_roi_voxels"])
            if roi_voxels <= 0 or finite_voxels <= 0:
                raise ValueError(f"{path}: invalid ROI support for {subject}")
            if finite_voxels + nonfinite_voxels != roi_voxels:
                raise ValueError(f"{path}: inconsistent ROI support for {subject}")
            if not math.isclose(
                finite_fraction,
                finite_voxels / roi_voxels,
                rel_tol=0.0,
                abs_tol=5e-4,
            ):
                raise ValueError(f"{path}: inconsistent finite ROI fraction")
            minimum_finite_fraction = min(minimum_finite_fraction, finite_fraction)
            if nonfinite_voxels > 0:
                runs_with_nonfinite_support += 1

    return {
        "rows": len(rows),
        "subjects": subjects,
        "condition": condition,
        "runs_with_nonfinite_roi_values": runs_with_nonfinite_support,
        "minimum_finite_roi_fraction": minimum_finite_fraction,
    }


def prepare_target(
    *,
    target: str,
    historical_csv: Path,
    corrected_fixed_csv: Path,
    output_csv: Path,
) -> dict[str, Any]:
    historical_fields, historical_rows = _read_csv(historical_csv)
    corrected_fields, corrected_rows = _read_csv(corrected_fixed_csv)
    expected_condition_rows = EXPECTED_SUBJECTS * EXPECTED_REPEATS
    if len(historical_rows) != 2 * expected_condition_rows or {
        row.get("condition") for row in historical_rows
    } != {"remesh", "fixed_mesh"}:
        raise ValueError(
            f"{historical_csv} must contain the historical 800-row, "
            "two-condition table"
        )
    if len(corrected_rows) != expected_condition_rows or {
        row.get("condition") for row in corrected_rows
    } != {"fixed_mesh"}:
        raise ValueError(
            f"{corrected_fixed_csv} must contain only the 400 corrected "
            "fixed-mesh rows"
        )
    historical_remesh = [
        row for row in historical_rows if row.get("condition") == "remesh"
    ]
    corrected_fixed = [
        row for row in corrected_rows if row.get("condition") == "fixed_mesh"
    ]
    historical_summary = _validate_condition_rows(
        historical_remesh,
        path=historical_csv,
        target=target,
        condition="remesh",
    )
    corrected_summary = _validate_condition_rows(
        corrected_fixed,
        path=corrected_fixed_csv,
        target=target,
        condition="fixed_mesh",
    )
    if historical_summary["subjects"] != corrected_summary["subjects"]:
        raise ValueError(f"{target}: remesh and corrected fixed subjects differ")

    fieldnames = list(historical_fields)
    fieldnames.extend(field for field in corrected_fields if field not in fieldnames)
    combined = sorted(
        historical_remesh + corrected_fixed,
        key=lambda row: (row["subject"], row["condition"], row["repeat_tag"]),
    )
    expected_rows = EXPECTED_SUBJECTS * 2 * EXPECTED_REPEATS
    if len(combined) != expected_rows:
        raise ValueError(
            f"{target}: expected {expected_rows} combined rows, got {len(combined)}"
        )
    combined_keys = {
        (row["subject"], row["condition"], row["repeat_tag"]) for row in combined
    }
    if len(combined_keys) != expected_rows:
        raise ValueError(f"{target}: combined rows contain duplicate keys")

    _write_csv_atomic(output_csv, fieldnames=fieldnames, rows=combined)
    return {
        "status": "complete",
        "target": target,
        "roi": TARGETS[target]["roi"],
        "rows": len(combined),
        "subjects": EXPECTED_SUBJECTS,
        "conditions": ["remesh", "fixed_mesh"],
        "repeats_per_subject_condition": EXPECTED_REPEATS,
        "historical_remesh": historical_summary,
        "corrected_fixed": corrected_summary,
        "inputs": {
            "historical_csv": {
                "path": str(historical_csv),
                "sha256": _sha256(historical_csv),
            },
            "corrected_fixed_csv": {
                "path": str(corrected_fixed_csv),
                "sha256": _sha256(corrected_fixed_csv),
            },
        },
        "output": {
            "path": str(output_csv),
            "sha256": _sha256(output_csv),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "left_hippocampus": (
            args.left_historical_csv.expanduser().resolve(),
            args.left_corrected_fixed_csv.expanduser().resolve(),
        ),
        "right_m1": (
            args.right_historical_csv.expanduser().resolve(),
            args.right_corrected_fixed_csv.expanduser().resolve(),
        ),
    }
    targets: dict[str, Any] = {}
    for target, (historical_csv, corrected_csv) in jobs.items():
        output_csv = output_dir / str(TARGETS[target]["filename"])
        targets[target] = prepare_target(
            target=target,
            historical_csv=historical_csv,
            corrected_fixed_csv=corrected_csv,
            output_csv=output_csv,
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "analysis": "corrected spherical-median fixed-mesh paper input assembly",
        "source_outputs_modified": False,
        "targets": targets,
        "expected_rows": 1600,
        "observed_rows": sum(int(item["rows"]) for item in targets.values()),
        "interpretation_boundary": (
            "The 40 runs within each participant and condition are technical "
            "repeats. Participants, not runs, are the population-level units."
        ),
    }
    _write_json_atomic(output_dir / "assembly_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-historical-csv", type=Path, required=True)
    parser.add_argument("--left-corrected-fixed-csv", type=Path, required=True)
    parser.add_argument("--right-historical-csv", type=Path, required=True)
    parser.add_argument("--right-corrected-fixed-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
