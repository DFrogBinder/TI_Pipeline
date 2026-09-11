#!/usr/bin/env python3
"""Read-only final audit for the spherical-fixed and nested experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiment_config import load_experiment_config  # noqa: E402
from pipeline.spherical_fixed_nested_experiment import (  # noqa: E402
    SPHERICAL_SELECTION_METRIC,
    collect_simulation_status,
)


DEFAULT_LEFT_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_balanced_10_spherical_fixed_v1"
)
DEFAULT_RIGHT_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/"
    "final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1"
)
DEFAULT_NESTED_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/final_132_repeatability_nested_40x40_v1"
)
EXPECTED_NESTED_SUBJECT = "sub-CC320616"
EXPECTED_NESTED_TARGET = "left-hippocampus"
EXPECTED_NESTED_POOL_SHA256 = (
    "433d76e75b9debf419d3aa6cbc9d657cccd164cf3a2919eb344fb32572324b1b"
)
EXPECTED_FIXED_SELECTIONS = {
    "left-hippocampus": {
        "sub-CC110174": "repeat_011",
        "sub-CC121144": "repeat_013",
        "sub-CC310407": "repeat_002",
        "sub-CC320616": "repeat_018",
        "sub-CC410432": "repeat_004",
        "sub-CC420071": "repeat_019",
        "sub-CC520083": "repeat_001",
        "sub-CC520127": "repeat_012",
        "sub-CC610631": "repeat_027",
        "sub-CC720941": "repeat_008",
    },
    "right-m1": {
        "sub-CC110174": "repeat_016",
        "sub-CC121144": "repeat_006",
        "sub-CC310407": "repeat_022",
        "sub-CC320616": "repeat_001",
        "sub-CC410432": "repeat_024",
        "sub-CC420071": "repeat_035",
        "sub-CC520083": "repeat_023",
        "sub-CC520127": "repeat_007",
        "sub-CC610631": "repeat_004",
        "sub-CC720941": "repeat_005",
    },
}


class AuditError(RuntimeError):
    """Raised when a final-result invariant is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_manifest_hash(manifest: dict[str, Any], key: str, path: Path) -> None:
    expected = str(manifest.get(key, ""))
    _require(bool(expected), f"Manifest lacks {key}: {path}")
    _require(_sha256(path) == expected, f"SHA-256 mismatch for {path}")


def _verify_checksum_file(output_root: Path) -> int:
    checksum_path = output_root / "checksums.sha256"
    _require(checksum_path.is_file(), f"Missing checksum file: {checksum_path}")
    checked = 0
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        _require(
            len(parts) == 2,
            f"Malformed checksum line {line_number}: {checksum_path}",
        )
        expected, relative = parts
        candidate = output_root / relative
        _require(candidate.is_file(), f"Checksummed file is missing: {candidate}")
        _require(_sha256(candidate) == expected, f"Checksum mismatch: {candidate}")
        checked += 1
    _require(checked > 0, f"Checksum file is empty: {checksum_path}")
    return checked


def _expected_repeat_tags(repeats: int) -> set[str]:
    return {f"repeat_{index:03d}" for index in range(1, repeats + 1)}


def _audit_optimizer_metrics(
    *,
    output_root: Path,
    expected_subjects: set[str],
    expected_conditions: set[str],
    expected_repeats: int,
    expected_roi: str,
) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    metrics_path = output_root / "optimizer_roi_metrics.csv"
    manifest = _load_json(manifest_path)
    rows = _read_csv(metrics_path)
    expected_rows = (
        len(expected_subjects) * len(expected_conditions) * expected_repeats
    )
    _require(manifest.get("status") == "complete", f"Incomplete {manifest_path}")
    _require(int(manifest.get("schema_version", -1)) == 2, "Metric schema mismatch")
    _require(int(manifest.get("subject_count", -1)) == len(expected_subjects), "Metric subject count mismatch")
    _require(int(manifest.get("expected_fields", -1)) == expected_rows, "Metric expected-field count mismatch")
    _require(len(rows) == expected_rows, f"Expected {expected_rows} metric rows, got {len(rows)}")
    _require({row["subject"] for row in rows} == expected_subjects, "Metric subjects mismatch")
    _require({row["condition"] for row in rows} == expected_conditions, "Metric conditions mismatch")
    _require({row["roi"] for row in rows} == {expected_roi}, "Metric ROI mismatch")
    keys = {(row["subject"], row["condition"], row["repeat_tag"]) for row in rows}
    _require(len(keys) == expected_rows, "Metric rows contain duplicate keys")
    expected_tags = _expected_repeat_tags(expected_repeats)
    for subject in expected_subjects:
        for condition in expected_conditions:
            selected = [
                row
                for row in rows
                if row["subject"] == subject and row["condition"] == condition
            ]
            _require(
                {row["repeat_tag"] for row in selected} == expected_tags,
                f"Incomplete metric repeats for {subject}/{condition}",
            )
            for row in selected:
                value = float(row[SPHERICAL_SELECTION_METRIC])
                _require(
                    math.isfinite(value),
                    f"Non-finite spherical median for {subject}/{condition}/{row['repeat_tag']}",
                )
    validation = manifest.get("validation")
    _require(isinstance(validation, dict), "Metric manifest lacks validation")
    _require(int(validation.get("rows", -1)) == expected_rows, "Metric validation row count mismatch")
    _require(
        bool(validation.get("all_runs_have_finite_roi_values")),
        "At least one run has no finite spherical-ROI values",
    )
    _require(str(manifest.get("combined_csv_sha256", "")) == _sha256(metrics_path), "Combined metric CSV hash mismatch")
    checked_files = _verify_checksum_file(output_root)
    return {
        "status": "complete",
        "rows": len(rows),
        "subjects": len(expected_subjects),
        "conditions": len(expected_conditions),
        "repeats_per_subject_condition": expected_repeats,
        "roi": expected_roi,
        "metric": SPHERICAL_SELECTION_METRIC,
        "checksummed_files": checked_files,
        "runs_with_nonfinite_roi_values": int(
            validation.get("runs_with_nonfinite_roi_values", 0)
        ),
        "minimum_finite_roi_fraction": float(
            validation.get("minimum_finite_roi_fraction", 0.0)
        ),
    }


def _audit_simulations(root: Path, config_path: Path, expected_tasks: int) -> dict[str, Any]:
    status = collect_simulation_status(config_path)
    _require(status.get("status") == "complete", f"Incomplete simulations: {root}")
    _require(int(status.get("expected_ti_nifti", -1)) == expected_tasks, "Expected simulation count mismatch")
    _require(int(status.get("observed_complete_tasks", -1)) == expected_tasks, "Observed simulation count mismatch")
    _require(int(status.get("incomplete_task_count", -1)) == 0, "Incomplete simulation tasks remain")
    _require(int(status.get("cache_issue_count", -1)) == 0, "Mesh-cache validation failed")
    completion = _load_json(root / "_pipeline/workflow/complete.json")
    _require(completion.get("status") == "complete", "Workflow completion receipt is incomplete")
    _require(completion.get("scope") == status, "Completion receipt scope differs from live status")
    return {
        "status": "complete",
        "expected": expected_tasks,
        "observed_complete": expected_tasks,
        "cache_issues": 0,
    }


def _audit_fixed(
    root: Path,
    *,
    target: str,
    roi: str,
    expected_changed: int,
) -> dict[str, Any]:
    config_path = root / "_pipeline/configs/fixed_mesh_spherical_median.json"
    config = load_experiment_config(config_path, validate_paths=False)
    _require(config.experiment_root == root.resolve(), "Fixed config root mismatch")
    _require(config.analysis.roi_preset == target, "Fixed target mismatch")
    _require(len(config.subjects) == 10, "Fixed subject count is not 10")
    _require(len(config.conditions) == 1, "Fixed condition count is not one")
    condition = config.conditions[0]
    _require(condition.name == "fixed_mesh", "Fixed condition is not fixed_mesh")
    _require(condition.mesh_mode == "fixed_mesh", "Fixed mesh mode mismatch")
    _require(condition.repeat_count == 40, "Fixed repeat count is not 40")

    experiment_manifest = _load_json(root / "_pipeline/experiment_manifest.json")
    selection_path = root / "_pipeline/spherical_median_selection.csv"
    seed_path = root / "_pipeline/cache_seed_manifest.csv"
    _require_manifest_hash(experiment_manifest, "config_sha256", config_path)
    _require_manifest_hash(experiment_manifest, "selection_csv_sha256", selection_path)
    _require_manifest_hash(experiment_manifest, "cache_seed_manifest_sha256", seed_path)
    _require(experiment_manifest.get("source_outputs_modified") is False, "Source mutation policy mismatch")

    selections = _read_csv(selection_path)
    seeds = _read_csv(seed_path)
    subjects = set(config.subjects)
    _require(len(selections) == 10, "Spherical selection row count is not 10")
    _require({row["subject"] for row in selections} == subjects, "Spherical selection subjects mismatch")
    _require({row["selection_status"] for row in selections} == {"selected"}, "A spherical selection is not ready")
    _require({row["metric"] for row in selections} == {SPHERICAL_SELECTION_METRIC}, "Selection metric is not spherical ROI median")
    selected_repeats = {
        row["subject"]: row["selected_repeat_tag"] for row in selections
    }
    _require(
        selected_repeats == EXPECTED_FIXED_SELECTIONS[target],
        f"Persisted spherical-median selections differ for {target}",
    )
    changed = sum(row["selection_changed"].lower() == "true" for row in selections)
    _require(changed == expected_changed, f"Expected {expected_changed} changed selections, got {changed}")
    _require(len(seeds) == 10, "Fixed cache-seed row count is not 10")
    seed_by_subject = {row["subject"]: row for row in seeds}
    for selection in selections:
        seed = seed_by_subject.get(selection["subject"])
        _require(seed is not None, f"Missing seed row for {selection['subject']}")
        _require(seed["condition"] == "fixed_mesh", "Fixed seed condition mismatch")
        _require(seed["seed_status"] == "ready", "Fixed seed is not ready")
        _require(seed["source_repeat_tag"] == selection["selected_repeat_tag"], "Selected repeat was not used to seed the fixed mesh")
        _require(seed["mesh_checksum"] == selection["mesh_checksum"], "Selected and seeded mesh hashes differ")

    simulation = _audit_simulations(root, config_path, 400)
    metrics = _audit_optimizer_metrics(
        output_root=root / "_post_processing/optimizer_roi_metrics_v1",
        expected_subjects=subjects,
        expected_conditions={"fixed_mesh"},
        expected_repeats=40,
        expected_roi=roi,
    )
    return {
        "status": "complete",
        "target": target,
        "subjects": 10,
        "selected_by": SPHERICAL_SELECTION_METRIC,
        "changed_from_anatomical_selection": changed,
        "selected_repeats": selected_repeats,
        "simulations": simulation,
        "optimizer_roi_metrics": metrics,
    }


def _audit_nested_analysis(root: Path) -> dict[str, Any]:
    output_dir = root / "_analysis/nested_variance"
    result_path = output_dir / "nested_variance_components.json"
    mesh_path = output_dir / "nested_mesh_summary.csv"
    variance_path = output_dir / "nested_variance_components.csv"
    result = _load_json(result_path)
    mesh_rows = _read_csv(mesh_path)
    variance_rows = _read_csv(variance_path)
    _require(result.get("status") == "complete", "Nested analysis is incomplete")
    _require(result.get("selected_subject") == EXPECTED_NESTED_SUBJECT, "Nested analysis subject mismatch")
    _require(result.get("selected_target") == EXPECTED_NESTED_TARGET, "Nested analysis target mismatch")
    _require(result.get("metric") == SPHERICAL_SELECTION_METRIC, "Nested analysis metric mismatch")
    _require(int(result.get("outer_meshes", -1)) == 40, "Nested outer-mesh count mismatch")
    _require(int(result.get("inner_repeats_per_mesh", -1)) == 40, "Nested inner-repeat count mismatch")
    _require(int(result.get("observations", -1)) == 1600, "Nested observation count mismatch")
    expected_conditions = {f"mesh_{index:03d}" for index in range(1, 41)}
    _require(len(mesh_rows) == 40, "Nested mesh-summary row count mismatch")
    _require({row["condition"] for row in mesh_rows} == expected_conditions, "Nested mesh-summary conditions mismatch")
    _require({int(row["n_inner_repeats"]) for row in mesh_rows} == {40}, "Nested mesh-summary repeat counts mismatch")
    _require(len(variance_rows) == 3, "Nested variance-component row count mismatch")
    _require(
        {row["component"] for row in variance_rows}
        == {"between_mesh", "within_mesh_solver_pipeline", "total"},
        "Nested variance components mismatch",
    )
    components = result.get("variance_components")
    _require(isinstance(components, dict), "Nested JSON lacks variance components")
    for component in ("between_mesh", "within_mesh_solver_pipeline", "total"):
        values = components.get(component)
        _require(isinstance(values, dict), f"Nested JSON lacks {component}")
        for field in ("variance", "sd"):
            _require(math.isfinite(float(values[field])), f"Non-finite {component} {field}")
    return {
        "status": "complete",
        "observations": 1600,
        "outer_meshes": 40,
        "inner_repeats_per_mesh": 40,
        "grand_mean": float(result["grand_mean"]),
        "intraclass_correlation_mesh": result.get("intraclass_correlation_mesh"),
        "variance_components": components,
    }


def _audit_nested(root: Path) -> dict[str, Any]:
    selection_path = root / "_pipeline/nested_case_selection.json"
    config_path = root / "_pipeline/configs/nested_40x40.json"
    seed_path = root / "_pipeline/cache_seed_manifest.csv"
    experiment_manifest = _load_json(root / "_pipeline/experiment_manifest.json")
    selection = _load_json(selection_path)
    config = load_experiment_config(config_path, validate_paths=False)
    _require(selection.get("status") == "selected", "Nested selection is incomplete")
    _require(selection.get("selected_subject") == EXPECTED_NESTED_SUBJECT, "Persisted nested subject changed")
    _require(selection.get("selected_target") == EXPECTED_NESTED_TARGET, "Persisted nested target changed")
    _require(selection.get("candidate_pool_sha256") == EXPECTED_NESTED_POOL_SHA256, "Nested candidate pool changed")
    _require(int(selection.get("selection_seed", -1)) == 20260831, "Nested selection seed changed")
    _require(config.experiment_root == root.resolve(), "Nested config root mismatch")
    _require(config.subjects == [EXPECTED_NESTED_SUBJECT], "Nested config subject mismatch")
    _require(config.analysis.roi_preset == EXPECTED_NESTED_TARGET, "Nested config target mismatch")
    expected_conditions = {f"mesh_{index:03d}" for index in range(1, 41)}
    _require({item.name for item in config.conditions} == expected_conditions, "Nested config mesh conditions mismatch")
    _require({item.mesh_mode for item in config.conditions} == {"fixed_mesh"}, "Nested mesh modes are not fixed")
    _require({item.repeat_count for item in config.conditions} == {40}, "Nested repeat counts mismatch")
    _require_manifest_hash(experiment_manifest, "selection_manifest_sha256", selection_path)
    _require_manifest_hash(experiment_manifest, "config_sha256", config_path)
    _require_manifest_hash(experiment_manifest, "cache_seed_manifest_sha256", seed_path)
    _require(experiment_manifest.get("source_outputs_modified") is False, "Nested source mutation policy mismatch")
    seeds = _read_csv(seed_path)
    _require(len(seeds) == 40, "Nested cache-seed row count is not 40")
    _require({row["condition"] for row in seeds} == expected_conditions, "Nested seed conditions mismatch")
    for index in range(1, 41):
        condition = f"mesh_{index:03d}"
        seed = next(row for row in seeds if row["condition"] == condition)
        _require(seed["subject"] == EXPECTED_NESTED_SUBJECT, "Nested seed subject mismatch")
        _require(seed["source_repeat_tag"] == f"repeat_{index:03d}", "Nested mesh-to-source-repeat mapping mismatch")
        _require(seed["seed_status"] == "ready", "Nested mesh seed is not ready")

    simulation = _audit_simulations(root, config_path, 1600)
    metrics = _audit_optimizer_metrics(
        output_root=root / "_post_processing/optimizer_roi_metrics_v1",
        expected_subjects={EXPECTED_NESTED_SUBJECT},
        expected_conditions=expected_conditions,
        expected_repeats=40,
        expected_roi="Left_Hippocampus",
    )
    analysis = _audit_nested_analysis(root)
    return {
        "status": "complete",
        "selected_subject": EXPECTED_NESTED_SUBJECT,
        "selected_target": EXPECTED_NESTED_TARGET,
        "selection_seed": 20260831,
        "simulations": simulation,
        "optimizer_roi_metrics": metrics,
        "nested_analysis": analysis,
    }


def audit_all(*, left_root: Path, right_root: Path, nested_root: Path) -> dict[str, Any]:
    checks = (
        (
            "left_hippocampus_spherical_fixed",
            lambda: _audit_fixed(
                left_root.resolve(),
                target="left-hippocampus",
                roi="Left_Hippocampus",
                expected_changed=8,
            ),
        ),
        (
            "right_m1_spherical_fixed",
            lambda: _audit_fixed(
                right_root.resolve(),
                target="right-m1",
                roi="Right_M1",
                expected_changed=10,
            ),
        ),
        ("nested_40x40", lambda: _audit_nested(nested_root.resolve())),
    )
    results: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for name, check in checks:
        try:
            results[name] = check()
        except Exception as exc:
            failures.append(
                {
                    "check": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            results[name] = {"status": "failed", "error": str(exc)}
    complete = not failures
    return {
        "status": "complete" if complete else "failed",
        "study": "spherical fixed-mesh correction plus nested 40x40 repeatability",
        "expected_total_simulations": 2400,
        "observed_complete_simulations": 2400 if complete else None,
        "source_outputs_modified": False,
        "checks": results,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", type=Path, default=DEFAULT_LEFT_ROOT)
    parser.add_argument("--right-root", type=Path, default=DEFAULT_RIGHT_ROOT)
    parser.add_argument("--nested-root", type=Path, default=DEFAULT_NESTED_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_all(
        left_root=args.left_root.expanduser(),
        right_root=args.right_root.expanduser(),
        nested_root=args.nested_root.expanduser(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
