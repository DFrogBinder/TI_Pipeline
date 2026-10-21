"""Execute one read-only subject-target forensic audit cell."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib

from .audit import classify_run_cause
from .config import (
    CONDITIONS,
    REPEAT_COUNT,
    SCHEMA_VERSION,
    atlas_root,
    cell_from_index,
    production_task_id,
    repeat_root,
)
from .imaging import audit_nifti_stages
from .io_utils import sha256_file, utc_now, write_csv_atomic, write_json_atomic
from .logs import select_task_logs, summarize_task_logs
from .mesh import audit_mesh_field
from .paths import resolve_run_artifacts


SUPPORT_FIELDS = (
    "roi_voxels",
    "final_finite_voxels",
    "raw_nonfinite_voxels",
    "mask_excluded_voxels",
    "unexplained_nonfinite_voxels",
    "finite_outside_brain_mask_voxels",
    "final_finite_raw_nonfinite_voxels",
    "finite_value_mismatch_voxels",
    "finite_roi_fraction",
)
MESH_FIELDS = (
    "mesh_field_elements",
    "mesh_field_nonfinite_elements",
    "mesh_tetrahedra",
    "mesh_roi_elements",
    "mesh_roi_finite_elements",
    "mesh_roi_nonfinite_elements",
    "mesh_roi_min_v_per_m",
    "mesh_roi_max_v_per_m",
    "mesh_size_bytes",
)
LOG_FIELDS = (
    "attempt_count",
    "maximum_retry_index",
    "command_result_events",
    "failed_command_result_events",
    "error_line_count",
    "grid_check_events",
    "all_grid_checks_passed",
    "simulation_done_observed",
    "validation_passed_observed",
    "final_attempt_success",
)
RUN_FIELDS = [
    "schema_version",
    "target",
    "roi",
    "subject",
    "subject_index",
    "condition",
    "repeat_number",
    "repeat_tag",
    "production_job_id",
    "production_task_id",
    "atlas_path",
    "v3_t1_path",
    "v3_raw_field_path",
    "v3_tissue_labels_path",
    "v3_masked_field_path",
    "v3_mesh_path",
    "v4_t1_path",
    "v4_raw_field_path",
    "v4_tissue_labels_path",
    "v4_masked_field_path",
    "v4_mesh_path",
    *[f"v3_{field}" for field in SUPPORT_FIELDS],
    "v3_tissue_label_counts_json",
    "v3_roi_metadata_json",
    "v3_grid_json",
    "v3_cause",
    *[f"v4_{field}" for field in SUPPORT_FIELDS],
    "v4_tissue_label_counts_json",
    "v4_roi_metadata_json",
    "v4_grid_json",
    "v4_cause",
    *[f"v3_{field}" for field in MESH_FIELDS],
    "v3_mesh_field_name",
    "v3_mesh_reader_simnibs_version",
    "v3_mesh_roi_mapping_available",
    *[f"log_{field}" for field in LOG_FIELDS],
    "log_paths_json",
    "production_log_status",
    "finite_roi_fraction_v4_minus_v3",
]


def _atlas_path(subject: str) -> Path:
    root = atlas_root()
    for suffix in (".nii.gz", ".nii"):
        candidate = root / f"{subject}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing subject atlas: {root}/{subject}.nii[.gz]")


def _serializable_audit(audit: dict[str, Any], prefix: str) -> dict[str, Any]:
    support = audit["support"]
    row = {f"{prefix}_{field}": support[field] for field in SUPPORT_FIELDS}
    row[f"{prefix}_tissue_label_counts_json"] = json.dumps(
        support["tissue_label_counts"], sort_keys=True, separators=(",", ":")
    )
    row[f"{prefix}_roi_metadata_json"] = json.dumps(
        audit["roi_metadata"], sort_keys=True, separators=(",", ":")
    )
    row[f"{prefix}_grid_json"] = json.dumps(
        audit["grid"], sort_keys=True, separators=(",", ":")
    )
    row[f"{prefix}_cause"] = audit["cause"]
    return row


def _production_log_status(summary: dict[str, Any]) -> str:
    if int(summary["attempt_count"]) == 0:
        return "missing"
    if bool(summary["final_attempt_success"]):
        return "successful_final_attempt"
    return "final_attempt_not_proven_successful"


def _run_external_renderer(request_path: Path) -> dict[str, Any]:
    render_python_raw = os.environ.get("FINITE_SUPPORT_RENDER_PYTHON", "")
    if not render_python_raw:
        raise RuntimeError(
            "FINITE_SUPPORT_RENDER_PYTHON is required for diagnostic rendering"
        )
    render_python = Path(render_python_raw).expanduser()
    if not render_python.is_file() or not os.access(render_python, os.X_OK):
        raise RuntimeError(
            f"FINITE_SUPPORT_RENDER_PYTHON is not executable: {render_python}"
        )
    command = [
        str(render_python),
        "-u",
        "-m",
        "simnibs_326_finite_support_audit",
        "render-request",
        "--request",
        str(request_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no renderer output").strip()
        raise RuntimeError(
            f"External diagnostic renderer failed with exit code "
            f"{completed.returncode}: {detail[-4000:]}"
        )
    receipt_path = request_path.parent / "render_complete.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Renderer did not write its completion receipt: {receipt_path}")
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _validate_render_receipt(
    *,
    receipt: dict[str, Any],
    request_path: Path,
    target: str,
    subject: str,
    expected_figures: dict[str, Path],
) -> None:
    if receipt.get("status") != "complete":
        raise RuntimeError("Renderer receipt is not complete")
    if receipt.get("target") != target or receipt.get("subject") != subject:
        raise RuntimeError("Renderer receipt identity mismatch")
    if receipt.get("request_sha256") != sha256_file(request_path):
        raise RuntimeError("Renderer request checksum mismatch")
    records = receipt.get("figures")
    if not isinstance(records, list) or len(records) != len(expected_figures):
        raise RuntimeError("Renderer receipt has the wrong figure count")
    by_condition = {str(record.get("condition")): record for record in records}
    if set(by_condition) != set(expected_figures):
        raise RuntimeError("Renderer receipt has the wrong conditions")
    for condition, expected_path in expected_figures.items():
        record = by_condition[condition]
        recorded_path = Path(str(record.get("path", "")))
        if recorded_path.resolve() != expected_path.resolve():
            raise RuntimeError(f"Renderer figure path mismatch for {condition}")
        if not expected_path.is_file() or expected_path.stat().st_size == 0:
            raise RuntimeError(f"Renderer figure is missing or empty: {expected_path}")
        if int(record.get("size_bytes", -1)) != expected_path.stat().st_size:
            raise RuntimeError(f"Renderer figure size mismatch for {condition}")
        if record.get("sha256") != sha256_file(expected_path):
            raise RuntimeError(f"Renderer figure checksum mismatch for {condition}")


def audit_cell(*, cell_index: int, output_root: Path) -> dict[str, Any]:
    """Audit 80 v3 runs and 80 matched v4 runs for one subject-target cell."""

    target, subject_index, subject = cell_from_index(cell_index)
    atlas_path = _atlas_path(subject)
    atlas_image = nib.load(str(atlas_path))
    roi_cache: dict[tuple[Any, ...], tuple[Any, dict[str, Any], bool]] = {}
    cell_root = output_root / "cells" / target.key / subject
    rows: list[dict[str, Any]] = []
    render_conditions: dict[str, Any] = {}

    for condition in CONDITIONS:
        job_id = target.remesh_job_id if condition == "remesh" else target.fixed_job_id
        log_dir = target.v3_root / "_simnibs326" / "logs" / (
            "remesh" if condition == "remesh" else "fixed"
        )
        render_versions: dict[str, Any] = {}
        for repeat_number in range(1, REPEAT_COUNT + 1):
            v3_paths = resolve_run_artifacts(
                repeat_root(target.v3_root, subject, condition, repeat_number), subject
            )
            v4_paths = resolve_run_artifacts(
                repeat_root(target.v4_root, subject, condition, repeat_number), subject
            )
            v3 = audit_nifti_stages(
                raw_field_path=v3_paths["raw_field"],
                masked_field_path=v3_paths["masked_field"],
                tissue_labels_path=v3_paths["tissue_labels"],
                atlas_path=atlas_path,
                roi=target.roi,
                atlas_labels=target.atlas_labels,
                atlas_image=atlas_image,
                roi_cache=roi_cache,
            )
            v4 = audit_nifti_stages(
                raw_field_path=v4_paths["raw_field"],
                masked_field_path=v4_paths["masked_field"],
                tissue_labels_path=v4_paths["tissue_labels"],
                atlas_path=atlas_path,
                roi=target.roi,
                atlas_labels=target.atlas_labels,
                atlas_image=atlas_image,
                roi_cache=roi_cache,
            )
            mesh = audit_mesh_field(
                v3_paths["mesh"],
                roi_mask=v3["diagnostic"]["roi_mask"],
                roi_affine=v3["diagnostic"]["reference_img"].affine,
            )
            v3["cause"] = classify_run_cause(
                mesh_nonfinite_voxels=int(mesh["mesh_roi_nonfinite_elements"]),
                support=v3["support"],
            )
            v4["cause"] = classify_run_cause(
                mesh_nonfinite_voxels=0,
                support=v4["support"],
            )
            task_id = production_task_id(subject_index, repeat_number)
            task_logs = select_task_logs(log_dir, job_id=job_id, task_id=task_id)
            log_summary = summarize_task_logs(task_logs)

            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "target": target.key,
                "roi": target.roi,
                "subject": subject,
                "subject_index": subject_index,
                "condition": condition,
                "repeat_number": repeat_number,
                "repeat_tag": f"repeat_{repeat_number:03d}",
                "production_job_id": job_id,
                "production_task_id": task_id,
                "atlas_path": str(atlas_path),
            }
            for version, paths in (("v3", v3_paths), ("v4", v4_paths)):
                for key in ("t1", "raw_field", "tissue_labels", "masked_field", "mesh"):
                    row[f"{version}_{key}_path"] = str(paths[key])
            row.update(_serializable_audit(v3, "v3"))
            row.update(_serializable_audit(v4, "v4"))
            row.update({f"v3_{field}": mesh[field] for field in MESH_FIELDS})
            row["v3_mesh_field_name"] = mesh["mesh_field_name"]
            row["v3_mesh_reader_simnibs_version"] = mesh["simnibs_version"]
            row["v3_mesh_roi_mapping_available"] = int(mesh["mesh_roi_elements"]) > 0
            row.update({f"log_{field}": log_summary[field] for field in LOG_FIELDS})
            row["log_paths_json"] = json.dumps(log_summary["log_paths"], separators=(",", ":"))
            row["production_log_status"] = _production_log_status(log_summary)
            row["finite_roi_fraction_v4_minus_v3"] = (
                float(v4["support"]["finite_roi_fraction"])
                - float(v3["support"]["finite_roi_fraction"])
            )
            rows.append(row)

            if repeat_number == 1:
                render_versions = {
                    "v3": {
                        "t1": str(v3_paths["t1"]),
                        "raw_field": str(v3_paths["raw_field"]),
                        "tissue_labels": str(v3_paths["tissue_labels"]),
                        "masked_field": str(v3_paths["masked_field"]),
                        "cause": str(v3["cause"]),
                    },
                    "v4": {
                        "t1": str(v4_paths["t1"]),
                        "raw_field": str(v4_paths["raw_field"]),
                        "tissue_labels": str(v4_paths["tissue_labels"]),
                        "masked_field": str(v4_paths["masked_field"]),
                        "cause": str(v4["cause"]),
                    },
                }
        if set(render_versions) != {"v3", "v4"}:
            raise RuntimeError(f"Missing repeat_001 render inputs for {condition}")
        render_conditions[condition] = {
            "output_path": str(
                cell_root / "figures" / f"{condition}_repeat_001_v3_v4.png"
            ),
            "versions": render_versions,
        }

    if len(rows) != len(CONDITIONS) * REPEAT_COUNT:
        raise RuntimeError(f"Expected 80 rows, produced {len(rows)}")
    runs_path = cell_root / "runs.csv"
    write_csv_atomic(runs_path, rows, RUN_FIELDS)
    render_request = {
        "schema_version": SCHEMA_VERSION,
        "target": target.key,
        "target_label": target.roi.replace("_", " "),
        "roi": target.roi,
        "atlas_labels": list(target.atlas_labels),
        "subject": subject,
        "atlas_path": str(atlas_path),
        "conditions": render_conditions,
    }
    request_path = cell_root / "render_request.json"
    write_json_atomic(request_path, render_request)
    render_receipt = _run_external_renderer(request_path)
    expected_figures = {
        condition: Path(str(payload["output_path"]))
        for condition, payload in render_conditions.items()
    }
    _validate_render_receipt(
        receipt=render_receipt,
        request_path=request_path,
        target=target.key,
        subject=subject,
        expected_figures=expected_figures,
    )
    figure_paths = [expected_figures[condition] for condition in CONDITIONS]
    summary = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "mode": "read_only_finite_support_forensics",
        "cell_index": cell_index,
        "target": target.key,
        "roi": target.roi,
        "subject_index": subject_index,
        "subject": subject,
        "conditions": list(CONDITIONS),
        "repeats_per_condition": REPEAT_COUNT,
        "paired_rows": len(rows),
        "v3_runs_audited": len(rows),
        "matched_v4_runs_audited": len(rows),
        "direct_v3_mesh_fields_audited": len(rows),
        "v3_mesh_reader_simnibs_versions": sorted(
            {str(row["v3_mesh_reader_simnibs_version"]) for row in rows}
        ),
        "production_log_status_counts": dict(Counter(row["production_log_status"] for row in rows)),
        "v3_cause_counts": dict(Counter(row["v3_cause"] for row in rows)),
        "v4_cause_counts": dict(Counter(row["v4_cause"] for row in rows)),
        "minimum_v3_finite_roi_fraction": min(float(row["v3_finite_roi_fraction"]) for row in rows),
        "minimum_v4_finite_roi_fraction": min(float(row["v4_finite_roi_fraction"]) for row in rows),
        "runs_csv": str(runs_path),
        "runs_csv_sha256": sha256_file(runs_path),
        "figures": [str(path) for path in figure_paths],
        "render_request": str(request_path),
        "render_request_sha256": sha256_file(request_path),
        "render_receipt": str(cell_root / "render_complete.json"),
        "render_python": render_receipt.get("renderer_python"),
        "render_matplotlib_version": render_receipt.get("matplotlib_version"),
        "source_outputs_modified": False,
        "meshing_tasks": 0,
        "fem_tasks": 0,
    }
    write_json_atomic(cell_root / "cell_summary.json", summary)
    write_json_atomic(cell_root / "complete.json", summary)
    return summary
