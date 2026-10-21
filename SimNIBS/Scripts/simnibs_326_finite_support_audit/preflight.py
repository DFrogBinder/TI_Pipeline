"""Dependency-light, read-only source inventory before submitting the audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import (
    CONDITIONS,
    REPEAT_COUNT,
    SUBJECTS,
    atlas_root,
    production_task_id,
    repeat_root,
    target_specs,
)
from .logs import select_task_logs
from .paths import resolve_run_artifacts


def _overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def assert_output_isolated(output_root: Path) -> None:
    sources = [atlas_root()]
    for target in target_specs().values():
        sources.extend([target.v3_root, target.v4_root])
    overlaps = [source for source in sources if _overlap(output_root, source)]
    if overlaps:
        raise RuntimeError(
            f"Audit output root overlaps read-only sources: {output_root}; {overlaps}"
        )


def _atlas_path(subject: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = atlas_root() / f"{subject}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"Missing atlas for {subject} under {atlas_root()}")


def _completion_receipt(root: Path) -> Path:
    path = root / "_pipeline" / "workflow" / "complete.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing production completion receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise RuntimeError(f"Production receipt is not complete: {path}")
    return path


def scope() -> dict[str, Any]:
    paired_runs = len(target_specs()) * len(SUBJECTS) * len(CONDITIONS) * REPEAT_COUNT
    return {
        "mode": "read_only_finite_support_forensics",
        "subjects": len(SUBJECTS),
        "targets": len(target_specs()),
        "conditions": len(CONDITIONS),
        "repeats_per_condition": REPEAT_COUNT,
        "audit_array_cells": len(target_specs()) * len(SUBJECTS),
        "v3_runs": paired_runs,
        "matched_v4_runs": paired_runs,
        "direct_v3_mesh_field_reads": paired_runs,
        "production_log_audits": paired_runs,
        "paired_diagnostic_panels": len(target_specs()) * len(SUBJECTS) * len(CONDITIONS),
        "meshing_tasks": 0,
        "fem_tasks": 0,
        "source_outputs_modified": False,
    }


def preflight(*, output_root: Path, full_scan: bool = True) -> dict[str, Any]:
    assert_output_isolated(output_root)
    for subject in SUBJECTS:
        _atlas_path(subject)
    receipts = []
    for target in target_specs().values():
        receipts.append(str(_completion_receipt(target.v3_root)))
        if not target.v4_root.is_dir():
            raise FileNotFoundError(target.v4_root)

    artifact_sets = 0
    v3_logs_found = 0
    if full_scan:
        for target in target_specs().values():
            for subject_index, subject in enumerate(SUBJECTS):
                for condition in CONDITIONS:
                    job_id = (
                        target.remesh_job_id
                        if condition == "remesh"
                        else target.fixed_job_id
                    )
                    log_dir = target.v3_root / "_simnibs326" / "logs" / (
                        "remesh" if condition == "remesh" else "fixed"
                    )
                    for repeat_number in range(1, REPEAT_COUNT + 1):
                        resolve_run_artifacts(
                            repeat_root(target.v3_root, subject, condition, repeat_number),
                            subject,
                        )
                        resolve_run_artifacts(
                            repeat_root(target.v4_root, subject, condition, repeat_number),
                            subject,
                        )
                        artifact_sets += 2
                        task_id = production_task_id(subject_index, repeat_number)
                        if select_task_logs(log_dir, job_id=job_id, task_id=task_id):
                            v3_logs_found += 1
        expected = scope()["v3_runs"]
        if v3_logs_found != expected:
            raise RuntimeError(
                f"Expected immutable logs for {expected} v3 runs; found {v3_logs_found}"
            )
    return {
        "status": "ready",
        **scope(),
        "output_root": str(output_root.resolve()),
        "output_root_isolated": True,
        "atlases_found": len(SUBJECTS),
        "v3_completion_receipts": receipts,
        "artifact_sets_resolved": artifact_sets,
        "v3_task_logs_found": v3_logs_found,
        "full_scan": full_scan,
    }
