from __future__ import annotations

import csv
import json

from simnibs_326_finite_support_audit.collect import collect
from simnibs_326_finite_support_audit.config import CONDITIONS, REPEAT_COUNT, SUBJECTS, target_specs
from simnibs_326_finite_support_audit.io_utils import sha256_file


FIELDS = [
    "schema_version",
    "target",
    "subject",
    "condition",
    "repeat_number",
    "repeat_tag",
    "v3_finite_roi_fraction",
    "v4_finite_roi_fraction",
    "v3_raw_nonfinite_voxels",
    "v3_mask_excluded_voxels",
    "v3_unexplained_nonfinite_voxels",
    "v3_finite_outside_brain_mask_voxels",
    "v3_final_finite_raw_nonfinite_voxels",
    "v3_finite_value_mismatch_voxels",
    "v4_raw_nonfinite_voxels",
    "v4_mask_excluded_voxels",
    "v4_unexplained_nonfinite_voxels",
    "v4_finite_outside_brain_mask_voxels",
    "v4_final_finite_raw_nonfinite_voxels",
    "v4_finite_value_mismatch_voxels",
    "v3_mesh_roi_nonfinite_elements",
    "v3_mesh_reader_simnibs_version",
    "v3_cause",
    "v4_cause",
    "production_log_status",
    "log_failed_command_result_events",
]


def _write_cell(root, target, subject):
    cell = root / "cells" / target / subject
    cell.mkdir(parents=True)
    rows = []
    for condition in CONDITIONS:
        for repeat in range(1, REPEAT_COUNT + 1):
            rows.append(
                {
                    "schema_version": 1,
                    "target": target,
                    "subject": subject,
                    "condition": condition,
                    "repeat_number": repeat,
                    "repeat_tag": f"repeat_{repeat:03d}",
                    "v3_finite_roi_fraction": 0.6,
                    "v4_finite_roi_fraction": 1.0,
                    "v3_raw_nonfinite_voxels": 0,
                    "v3_mask_excluded_voxels": 80,
                    "v3_unexplained_nonfinite_voxels": 0,
                    "v3_finite_outside_brain_mask_voxels": 0,
                    "v3_final_finite_raw_nonfinite_voxels": 0,
                    "v3_finite_value_mismatch_voxels": 0,
                    "v4_raw_nonfinite_voxels": 0,
                    "v4_mask_excluded_voxels": 0,
                    "v4_unexplained_nonfinite_voxels": 0,
                    "v4_finite_outside_brain_mask_voxels": 0,
                    "v4_final_finite_raw_nonfinite_voxels": 0,
                    "v4_finite_value_mismatch_voxels": 0,
                    "v3_mesh_roi_nonfinite_elements": 0,
                    "v3_mesh_reader_simnibs_version": "3.2.6",
                    "v3_cause": "brain_mask_exclusion",
                    "v4_cause": "complete_support",
                    "production_log_status": "successful_final_attempt",
                    "log_failed_command_result_events": 0,
                }
            )
    runs = cell / "runs.csv"
    with runs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "status": "complete",
        "target": target,
        "subject": subject,
        "runs_csv_sha256": sha256_file(runs),
    }
    (cell / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")
    (cell / "cell_summary.json").write_text(json.dumps(receipt), encoding="utf-8")
    figures = cell / "figures"
    figures.mkdir()
    for condition in CONDITIONS:
        (figures / f"{condition}.png").write_bytes(b"png")


def test_collect_validates_full_1600_run_scope_and_builds_archive(tmp_path):
    for target in target_specs():
        for subject in SUBJECTS:
            _write_cell(tmp_path, target, subject)

    result = collect(output_root=tmp_path)

    assert result["paired_rows"] == 1600
    assert result["direct_v3_mesh_fields_audited"] == 1600
    assert result["diagnostic_panels"] == 40
    assert (tmp_path / "all_runs.csv").is_file()
    assert (tmp_path / "audit_summary.csv").is_file()
    report = (tmp_path / "audit_report.md").read_text(encoding="utf-8")
    assert "It is not the fraction above 0.2 V/m" in report
    assert "Finite raw values excluded by the WM/GM mask: **1,600 runs**" in report
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "finite_support_forensics_v1.tar.gz").is_file()
