from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "post"
    / "audit_spherical_fixed_nested_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_spherical_fixed_nested_results", SCRIPT
)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_optimizer_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    metrics = root / "optimizer_roi_metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "subject",
                "condition",
                "repeat_tag",
                "roi",
                "roi_median_v_per_m",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "subject": "sub-test",
                    "condition": "fixed_mesh",
                    "repeat_tag": "repeat_001",
                    "roi": "Left_Hippocampus",
                    "roi_median_v_per_m": "0.20",
                },
                {
                    "subject": "sub-test",
                    "condition": "fixed_mesh",
                    "repeat_tag": "repeat_002",
                    "roi": "Left_Hippocampus",
                    "roi_median_v_per_m": "0.21",
                },
            ]
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 2,
                "subject_count": 1,
                "expected_fields": 2,
                "combined_csv_sha256": _sha256(metrics),
                "validation": {
                    "rows": 2,
                    "all_runs_have_finite_roi_values": True,
                    "runs_with_nonfinite_roi_values": 0,
                    "minimum_finite_roi_fraction": 1.0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "checksums.sha256").write_text(
        f"{_sha256(metrics)}  {metrics.name}\n"
        f"{_sha256(manifest)}  {manifest.name}\n",
        encoding="utf-8",
    )


def test_optimizer_metric_audit_accepts_balanced_checksummed_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "optimizer"
    _write_optimizer_fixture(output)

    result = audit._audit_optimizer_metrics(
        output_root=output,
        expected_subjects={"sub-test"},
        expected_conditions={"fixed_mesh"},
        expected_repeats=2,
        expected_roi="Left_Hippocampus",
    )

    assert result["status"] == "complete"
    assert result["rows"] == 2
    assert result["checksummed_files"] == 2


def test_optimizer_metric_audit_rejects_tampered_csv(tmp_path: Path) -> None:
    output = tmp_path / "optimizer"
    _write_optimizer_fixture(output)
    with (output / "optimizer_roi_metrics.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("tampered\n")

    with pytest.raises(audit.AuditError):
        audit._audit_optimizer_metrics(
            output_root=output,
            expected_subjects={"sub-test"},
            expected_conditions={"fixed_mesh"},
            expected_repeats=2,
            expected_roi="Left_Hippocampus",
        )


def test_full_audit_reports_all_missing_studies(tmp_path: Path) -> None:
    result = audit.audit_all(
        left_root=tmp_path / "left",
        right_root=tmp_path / "right",
        nested_root=tmp_path / "nested",
    )

    assert result["status"] == "failed"
    assert result["observed_complete_simulations"] is None
    assert {failure["check"] for failure in result["failures"]} == {
        "left_hippocampus_spherical_fixed",
        "right_m1_spherical_fixed",
        "nested_40x40",
    }
