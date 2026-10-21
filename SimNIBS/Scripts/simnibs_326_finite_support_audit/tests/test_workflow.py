from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simnibs_326_finite_support_audit import workflow


def test_audit_cell_emits_80_paired_rows_two_panels_and_complete_receipt(
    tmp_path, monkeypatch
):
    target = SimpleNamespace(
        key="left-hippocampus",
        roi="Left_Hippocampus",
        atlas_labels=(17,),
        v3_root=Path("/v3"),
        v4_root=Path("/v4"),
        remesh_job_id="111",
        fixed_job_id="222",
    )
    monkeypatch.setattr(workflow, "cell_from_index", lambda index: (target, 0, "sub-test"))
    monkeypatch.setattr(workflow, "_atlas_path", lambda subject: Path("/atlas.nii.gz"))
    monkeypatch.setattr(workflow.nib, "load", lambda path: object())
    monkeypatch.setattr(
        workflow,
        "resolve_run_artifacts",
        lambda root, subject: {
            "t1": root / subject / "T1.nii",
            "raw_field": root / "raw.nii.gz",
            "tissue_labels": root / "labels.nii.gz",
            "masked_field": root / "masked.nii.gz",
            "mesh": root / "TI.msh",
        },
    )
    roi = np.ones((2, 2, 2), dtype=bool)

    def fake_nifti(**kwargs):
        return {
            "support": {
                "roi_voxels": 8,
                "final_finite_voxels": 8,
                "raw_nonfinite_voxels": 0,
                "mask_excluded_voxels": 0,
                "unexplained_nonfinite_voxels": 0,
                "finite_outside_brain_mask_voxels": 0,
                "final_finite_raw_nonfinite_voxels": 0,
                "finite_value_mismatch_voxels": 0,
                "finite_roi_fraction": 1.0,
                "tissue_label_counts": {"2": 8},
            },
            "roi_metadata": {"radius_mm": 3.0},
            "grid": {"shape": [2, 2, 2]},
            "diagnostic": {
                "roi_mask": roi,
                "reference_img": SimpleNamespace(affine=np.eye(4)),
            },
        }

    monkeypatch.setattr(workflow, "audit_nifti_stages", fake_nifti)
    monkeypatch.setattr(
        workflow,
        "audit_mesh_field",
        lambda *args, **kwargs: {
            "mesh_field_elements": 10,
            "mesh_field_nonfinite_elements": 0,
            "mesh_tetrahedra": 10,
            "mesh_roi_elements": 2,
            "mesh_roi_finite_elements": 2,
            "mesh_roi_nonfinite_elements": 0,
            "mesh_roi_min_v_per_m": 0.1,
            "mesh_roi_max_v_per_m": 0.2,
            "mesh_size_bytes": 123,
            "mesh_field_name": "TImax",
            "simnibs_version": "3.2.6",
        },
    )
    monkeypatch.setattr(workflow, "select_task_logs", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        workflow,
        "summarize_task_logs",
        lambda paths: {
            "attempt_count": 1,
            "maximum_retry_index": 0,
            "command_result_events": 3,
            "failed_command_result_events": 0,
            "error_line_count": 0,
            "grid_check_events": 1,
            "all_grid_checks_passed": True,
            "simulation_done_observed": True,
            "validation_passed_observed": True,
            "final_attempt_success": True,
            "log_paths": ["task.log"],
        },
    )
    monkeypatch.setattr(
        workflow,
        "render_paired_diagnostic",
        lambda *, output_path, **kwargs: (
            output_path.parent.mkdir(parents=True, exist_ok=True),
            output_path.write_bytes(b"png"),
        ),
    )

    result = workflow.audit_cell(cell_index=0, output_root=tmp_path)

    assert result["paired_rows"] == 80
    assert result["direct_v3_mesh_fields_audited"] == 80
    cell = tmp_path / "cells" / target.key / "sub-test"
    with (cell / "runs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 80
    assert {row["condition"] for row in rows} == {"remesh", "fixed_mesh"}
    assert rows[0]["v3_mesh_reader_simnibs_version"] == "3.2.6"
    assert len(list((cell / "figures").glob("*.png"))) == 2
    assert (cell / "complete.json").is_file()
