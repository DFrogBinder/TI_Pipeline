import csv
from pathlib import Path

from simulation.plan_simulation_repair import (
    scan_repair_needs,
    submit_repair_jobs,
    write_repair_outputs,
)


def _write_file(path: Path, payload=b"ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_runner_inputs(dataset_root: Path, subject: str) -> None:
    anat_dir = dataset_root / subject / "anat"
    _write_file(anat_dir / f"{subject}_T1w.nii")
    _write_file(anat_dir / f"{subject}_T2w.nii")
    _write_file(anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii")


def _write_complete_outputs(dataset_root: Path, subject: str) -> None:
    output_dir = dataset_root / subject / "anat" / "SimNIBS" / "Output" / subject
    _write_file(output_dir / "TI.msh")
    _write_file(output_dir / "Volume_Base" / "TI_Volumetric_Base.nii.gz")
    _write_file(output_dir / "Volume_Labels" / "TI_Volumetric_Labels.nii.gz")
    _write_file(dataset_root / subject / "anat" / "SimNIBS" / "ti_brain_only.nii.gz")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_scan_repair_needs_builds_plan_and_subject_counts(tmp_path):
    run_01 = tmp_path / "Left_Hippocampus_Data_01"
    run_02 = tmp_path / "Left_Hippocampus_Data_02"

    for run_root in (run_01, run_02):
        for subject in ("sub-01", "sub-02"):
            _write_runner_inputs(run_root, subject)

    _write_complete_outputs(run_01, "sub-01")
    _write_complete_outputs(run_01, "sub-02")
    _write_complete_outputs(run_02, "sub-01")

    result = scan_repair_needs(
        batch_root=tmp_path,
        check_nifti=False,
    )

    assert len(result.complete_statuses) == 3
    assert len(result.repairable_statuses) == 1
    assert result.repairable_statuses[0].subject == "sub-02"
    assert result.repairable_statuses[0].dataset_name == "Left_Hippocampus_Data_02"
    assert set(result.repairable_statuses[0].missing_outputs) == {
        "ti_mesh",
        "ti_volume",
        "ti_labels",
        "ti_brain_only",
    }

    paths = write_repair_outputs(result, out_dir=tmp_path / "plan")
    plan_rows = _read_tsv(Path(paths["repair_plan"]))
    count_rows = _read_tsv(Path(paths["repair_subject_counts"]))
    per_repeat_dir = Path(paths["per_repeat_repair_plans_dir"])
    per_repeat_plan = per_repeat_dir / "Left_Hippocampus_Data_02_repair_plan.tsv"

    assert plan_rows == [
        {
            "task_id": "0",
            "dataset_name": "Left_Hippocampus_Data_02",
            "repeat_id": "02",
            "dataset_root": str(run_02.resolve()),
            "subject": "sub-02",
            "missing_outputs": "ti_mesh;ti_volume;ti_labels;ti_brain_only",
            "failure_reasons": (
                "ti_mesh:missing;ti_volume:missing;ti_labels:missing;"
                "ti_brain_only:missing"
            ),
        }
    ]
    assert count_rows[0]["subject"] == "sub-02"
    assert count_rows[0]["n_missing_runs"] == "1"
    assert count_rows[0]["n_repairable_runs"] == "1"
    assert count_rows[0]["repairable_repeats"] == "02"
    assert _read_tsv(per_repeat_plan) == plan_rows


def test_scan_repair_needs_blocks_incomplete_subject_without_inputs(tmp_path):
    run_01 = tmp_path / "Left_Hippocampus_Data_01"
    run_02 = tmp_path / "Left_Hippocampus_Data_02"

    run_02.mkdir()
    _write_runner_inputs(run_01, "sub-01")
    _write_complete_outputs(run_01, "sub-01")

    result = scan_repair_needs(
        batch_root=tmp_path,
        check_nifti=False,
    )

    assert len(result.repairable_statuses) == 0
    assert len(result.blocked_statuses) == 1
    blocked = result.blocked_statuses[0]
    assert blocked.subject == "sub-01"
    assert blocked.dataset_name == "Left_Hippocampus_Data_02"
    assert any(path.endswith("sub-01/anat/sub-01_T1w.nii") for path in blocked.missing_inputs)


def test_submit_repair_jobs_dry_run_uses_one_plan_per_repeat(tmp_path):
    run_01 = tmp_path / "Left_Hippocampus_Data_01"
    run_02 = tmp_path / "Left_Hippocampus_Data_02"

    for run_root, subject in ((run_01, "sub-01"), (run_02, "sub-02")):
        _write_runner_inputs(run_root, subject)

    result = scan_repair_needs(
        batch_root=tmp_path,
        check_nifti=False,
    )
    paths = write_repair_outputs(result, out_dir=tmp_path / "plan")
    submit_script = tmp_path / "submit_repair_jobArray.sh"
    submit_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    submissions = submit_repair_jobs(
        output_dir=paths["output_dir"],
        submit_script=submit_script,
        dry_run=True,
    )

    assert [item["dataset_name"] for item in submissions] == [
        "Left_Hippocampus_Data_01",
        "Left_Hippocampus_Data_02",
    ]
    assert all(item["status"] == "dry_run" for item in submissions)
    assert all(Path(str(item["repair_plan"])).is_file() for item in submissions)
    assert (Path(paths["output_dir"]) / "submission_summary.json").is_file()
