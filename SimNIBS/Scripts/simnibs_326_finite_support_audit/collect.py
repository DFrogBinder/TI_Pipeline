"""Validate and collect all 20 finite-support audit cells."""

from __future__ import annotations

import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .config import (
    CONDITIONS,
    REPEAT_COUNT,
    SCHEMA_VERSION,
    SUBJECTS,
    atlas_root,
    target_specs,
)
from .io_utils import (
    file_manifest,
    read_csv,
    sha256_file,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
)
from .report import build_report


PACKAGE_ROOT = Path(__file__).resolve().parent


def _workflow_code_files() -> list[Path]:
    files = [PACKAGE_ROOT / "README.md"]
    files.extend(sorted(PACKAGE_ROOT.glob("*.py")))
    files.extend(sorted(PACKAGE_ROOT.glob("*.json")))
    files.extend(sorted((PACKAGE_ROOT / "hpc").glob("*")))
    return [path for path in files if path.is_file()]


def _load_cells(output_root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    rows: list[dict[str, str]] = []
    source_files: list[Path] = []
    for target in target_specs().values():
        for subject in SUBJECTS:
            cell_root = output_root / "cells" / target.key / subject
            receipt_path = cell_root / "complete.json"
            failure_path = cell_root / "failed.json"
            runs_path = cell_root / "runs.csv"
            if not receipt_path.is_file() or not runs_path.is_file():
                raise FileNotFoundError(f"Incomplete audit cell: {cell_root}")
            if failure_path.is_file() and failure_path.stat().st_mtime >= receipt_path.stat().st_mtime:
                raise RuntimeError(f"Audit cell has a current failure receipt: {failure_path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "complete":
                raise RuntimeError(f"Non-complete receipt: {receipt_path}")
            if receipt.get("target") != target.key or receipt.get("subject") != subject:
                raise RuntimeError(f"Cell receipt identity mismatch: {receipt_path}")
            if receipt.get("runs_csv_sha256") != sha256_file(runs_path):
                raise RuntimeError(f"Cell CSV checksum mismatch: {runs_path}")
            cell_rows = read_csv(runs_path)
            if len(cell_rows) != len(CONDITIONS) * REPEAT_COUNT:
                raise RuntimeError(f"Expected 80 rows in {runs_path}; got {len(cell_rows)}")
            rows.extend(cell_rows)
            summary_path = cell_root / "cell_summary.json"
            figures = sorted((cell_root / "figures").glob("*.png"))
            if not summary_path.is_file() or len(figures) != len(CONDITIONS):
                raise FileNotFoundError(f"Cell summary or paired diagnostic panels missing: {cell_root}")
            source_files.extend([runs_path, receipt_path, summary_path])
            source_files.extend(figures)
    return rows, source_files


def _validate_rows(rows: list[dict[str, str]]) -> None:
    expected = len(target_specs()) * len(SUBJECTS) * len(CONDITIONS) * REPEAT_COUNT
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} paired rows; collected {len(rows)}")
    keys = {
        (row["target"], row["subject"], row["condition"], row["repeat_tag"])
        for row in rows
    }
    if len(keys) != expected:
        raise RuntimeError("Duplicate or missing target-subject-condition-repeat keys")
    if any(int(row["schema_version"]) != SCHEMA_VERSION for row in rows):
        raise RuntimeError("Schema version mismatch")
    versions = {row.get("v3_mesh_reader_simnibs_version", "") for row in rows}
    if not versions or any(not version.startswith("3.2.6") for version in versions):
        raise RuntimeError(f"Unexpected direct-mesh reader versions: {sorted(versions)}")


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["target"], row["subject"], row["condition"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (target, subject, condition), selected in sorted(groups.items()):
        if len(selected) != REPEAT_COUNT:
            raise RuntimeError(f"{target}/{subject}/{condition}: expected {REPEAT_COUNT} rows")
        v3_fraction = [float(row["v3_finite_roi_fraction"]) for row in selected]
        v4_fraction = [float(row["v4_finite_roi_fraction"]) for row in selected]
        summary_rows.append(
            {
                "target": target,
                "subject": subject,
                "condition": condition,
                "runs": len(selected),
                "v3_finite_roi_fraction_mean": mean(v3_fraction),
                "v3_finite_roi_fraction_min": min(v3_fraction),
                "v3_finite_roi_fraction_max": max(v3_fraction),
                "v4_finite_roi_fraction_mean": mean(v4_fraction),
                "v4_finite_roi_fraction_min": min(v4_fraction),
                "v4_finite_roi_fraction_max": max(v4_fraction),
                "v4_minus_v3_fraction_mean": mean(v4_fraction) - mean(v3_fraction),
                "v3_raw_nonfinite_voxels_max": max(int(row["v3_raw_nonfinite_voxels"]) for row in selected),
                "v3_mask_excluded_voxels_max": max(int(row["v3_mask_excluded_voxels"]) for row in selected),
                "v3_unexplained_nonfinite_voxels_max": max(
                    int(row["v3_unexplained_nonfinite_voxels"]) for row in selected
                ),
                "v3_finite_outside_brain_mask_voxels_max": max(
                    int(row["v3_finite_outside_brain_mask_voxels"]) for row in selected
                ),
                "v3_final_finite_raw_nonfinite_voxels_max": max(
                    int(row["v3_final_finite_raw_nonfinite_voxels"]) for row in selected
                ),
                "v3_finite_value_mismatch_voxels_max": max(
                    int(row["v3_finite_value_mismatch_voxels"]) for row in selected
                ),
                "v4_raw_nonfinite_voxels_max": max(
                    int(row["v4_raw_nonfinite_voxels"]) for row in selected
                ),
                "v4_mask_excluded_voxels_max": max(
                    int(row["v4_mask_excluded_voxels"]) for row in selected
                ),
                "v4_unexplained_nonfinite_voxels_max": max(
                    int(row["v4_unexplained_nonfinite_voxels"]) for row in selected
                ),
                "v4_finite_outside_brain_mask_voxels_max": max(
                    int(row["v4_finite_outside_brain_mask_voxels"]) for row in selected
                ),
                "v4_final_finite_raw_nonfinite_voxels_max": max(
                    int(row["v4_final_finite_raw_nonfinite_voxels"]) for row in selected
                ),
                "v4_finite_value_mismatch_voxels_max": max(
                    int(row["v4_finite_value_mismatch_voxels"]) for row in selected
                ),
                "v3_mesh_roi_nonfinite_elements_max": max(
                    int(row["v3_mesh_roi_nonfinite_elements"]) for row in selected
                ),
                "v3_cause_counts_json": json.dumps(
                    Counter(row["v3_cause"] for row in selected), sort_keys=True
                ),
                "v4_cause_counts_json": json.dumps(
                    Counter(row["v4_cause"] for row in selected), sort_keys=True
                ),
                "production_log_status_counts_json": json.dumps(
                    Counter(row["production_log_status"] for row in selected), sort_keys=True
                ),
                "failed_command_result_events": sum(
                    int(row["log_failed_command_result_events"]) for row in selected
                ),
            }
        )
    return summary_rows


def collect(*, output_root: Path, archive_path: Path | None = None) -> dict[str, Any]:
    rows, cell_files = _load_cells(output_root)
    for optional_name in ("production_sacct.tsv", "submission.tsv"):
        optional_path = output_root / optional_name
        if optional_path.is_file():
            cell_files.append(optional_path)
    _validate_rows(rows)
    rows.sort(
        key=lambda row: (
            tuple(target_specs()).index(row["target"]),
            SUBJECTS.index(row["subject"]),
            CONDITIONS.index(row["condition"]),
            int(row["repeat_number"]),
        )
    )
    summary_rows = _aggregate(rows)
    all_runs_path = output_root / "all_runs.csv"
    summary_csv_path = output_root / "audit_summary.csv"
    write_csv_atomic(all_runs_path, rows, list(rows[0]))
    write_csv_atomic(summary_csv_path, summary_rows, list(summary_rows[0]))
    sources = {
        target.key: {
            "roi": target.roi,
            "atlas_labels": list(target.atlas_labels),
            "v3_root": str(target.v3_root),
            "v4_root": str(target.v4_root),
            "v3_remesh_job_id": target.remesh_job_id,
            "v3_fixed_job_id": target.fixed_job_id,
        }
        for target in target_specs().values()
    }
    summary = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "mode": "read_only_finite_support_forensics",
        "study_versions": {"v3": "SimNIBS 3.2.6", "v4": "SimNIBS 4.0.1"},
        "output_root": str(output_root.resolve()),
        "atlas_root": str(atlas_root()),
        "sources": sources,
        "subjects": len(SUBJECTS),
        "targets": len(target_specs()),
        "conditions": len(CONDITIONS),
        "repeats_per_condition": REPEAT_COUNT,
        "cells": len(SUBJECTS) * len(target_specs()),
        "paired_rows": len(rows),
        "v3_runs_audited": len(rows),
        "matched_v4_runs_audited": len(rows),
        "direct_v3_mesh_fields_audited": len(rows),
        "v3_mesh_reader_simnibs_versions": sorted(
            {row["v3_mesh_reader_simnibs_version"] for row in rows}
        ),
        "diagnostic_panels": len(SUBJECTS) * len(target_specs()) * len(CONDITIONS),
        "v3_cause_counts": dict(Counter(row["v3_cause"] for row in rows)),
        "v4_cause_counts": dict(Counter(row["v4_cause"] for row in rows)),
        "production_log_status_counts": dict(
            Counter(row["production_log_status"] for row in rows)
        ),
        "minimum_v3_finite_roi_fraction": min(float(row["v3_finite_roi_fraction"]) for row in rows),
        "minimum_v4_finite_roi_fraction": min(float(row["v4_finite_roi_fraction"]) for row in rows),
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
        "groups": summary_rows,
    }
    summary_json_path = output_root / "audit_summary.json"
    write_json_atomic(summary_json_path, summary)
    report_path = output_root / "audit_report.md"
    report_path.write_text(build_report(summary, summary_rows), encoding="utf-8")
    code_files = _workflow_code_files()
    manifest_payload = {
        **summary,
        "files": file_manifest(
            [all_runs_path, summary_csv_path, summary_json_path, report_path, *cell_files],
            root=output_root,
        ),
        "workflow_code": [
            {
                "path": str(path.relative_to(PACKAGE_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in code_files
        ],
    }
    manifest_path = output_root / "manifest.json"
    write_json_atomic(manifest_path, manifest_payload)
    checksum_files = [
        all_runs_path,
        summary_csv_path,
        summary_json_path,
        report_path,
        manifest_path,
        *cell_files,
    ]
    checksums_path = output_root / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_root)}\n"
            for path in sorted(checksum_files)
        ),
        encoding="utf-8",
    )

    archive = archive_path or (output_root / "finite_support_forensics_v1.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as bundle:
        for path in sorted([*checksum_files, checksums_path]):
            bundle.add(path, arcname=str(path.relative_to(output_root)))
        for path in code_files:
            bundle.add(
                path,
                arcname=str(Path("workflow_code") / path.relative_to(PACKAGE_ROOT)),
            )
    temporary.replace(archive)
    archive_checksum = archive.with_suffix(archive.suffix + ".sha256")
    archive_checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return {**summary, "manifest": str(manifest_path), "archive": str(archive)}
