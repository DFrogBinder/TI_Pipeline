from __future__ import annotations

import importlib

import pytest


def _logs_module():
    try:
        return importlib.import_module("simnibs_326_finite_support_audit.logs")
    except ModuleNotFoundError:
        pytest.fail("finite-support log auditor has not been implemented")


def test_select_and_summarize_task_logs_preserves_failed_attempt_evidence(tmp_path):
    """Catches treating a successful retry as if the first failed attempt never occurred."""

    logs = _logs_module()
    first = tmp_path / "task_11789553_7_retry_0_20261011T100000Z.log"
    first.write_text(
        '\n'.join(
            [
                '{"event":"cmd_result","label":"msh2nii_volume","returncode":1}',
                "[ERROR] first attempt failed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    second = tmp_path / "task_11789553_7_retry_1_20261011T101000Z.log"
    second.write_text(
        '\n'.join(
            [
                '{"event":"cmd_result","label":"msh2nii_volume","returncode":0}',
                '{"event":"grid_check","same_shape":true,"same_affine":true}',
                '{"event":"simulation_done","subject":"sub-01"}',
                "[INFO] Repeatability task validation passed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "task_11789553_70_retry_0_20261011T100000Z.log").write_text(
        "unrelated\n", encoding="utf-8"
    )

    selected = logs.select_task_logs(tmp_path, job_id="11789553", task_id=7)
    assert selected == [first, second]
    assert logs.summarize_task_logs(selected) == {
        "attempt_count": 2,
        "maximum_retry_index": 1,
        "command_result_events": 2,
        "failed_command_result_events": 1,
        "error_line_count": 1,
        "grid_check_events": 1,
        "all_grid_checks_passed": True,
        "simulation_done_observed": True,
        "validation_passed_observed": True,
        "final_attempt_success": True,
        "log_paths": [str(first), str(second)],
    }
