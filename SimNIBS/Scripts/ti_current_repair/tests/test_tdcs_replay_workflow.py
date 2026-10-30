import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


CURRENT_REPAIR_ROOT = Path(__file__).resolve().parents[1]
if str(CURRENT_REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPAIR_ROOT))

import tdcs_replay
from pipeline import audit_existing_right_m1_meshes as mesh_audit
from pipeline import tdcs_replay_workflow as workflow
from simulation_runners import tdcs_replay as task_runner


def _base_plan(tmp_path: Path) -> dict:
    """Build a complete test-only plan without a repository-local study config."""
    subjects = [f"sub-test{index:02d}" for index in range(1, 11)]
    return {
        "schema_version": 1,
        "study": "test-only right-M1 tDCS replay",
        "status": "blocked_pending_stimulation_parameters_and_hpc_mesh_audit",
        "submission_allowed": False,
        "target": {
            "roi_preset": "right-m1",
            "optimizer_roi_name": "Right_M1",
            "roi_definition": "optimizer-matched-spherical",
        },
        "cohort": {
            "subjects": subjects,
            "subject_count": len(subjects),
            "repeats_per_arm": 40,
        },
        "design": {
            "arms": [
                {"name": "simnibs_v4_remesh_heads", "tdcs_solves": 400},
                {"name": "simnibs_v4_fixed_head_control", "tdcs_solves": 400},
                {"name": "simnibs_v3_deterministic_head_control", "tdcs_solves": 400},
            ],
            "total_tdcs_solves": 1200,
        },
        "existing_inputs": {
            "policy": "read_only",
            "simnibs_v4_root": str(tmp_path / "inputs" / "v4"),
            "simnibs_v3_root": str(tmp_path / "inputs" / "v3"),
        },
        "proposed_output_roots": {
            "simnibs_v4": str(tmp_path / "outputs" / "v4"),
            "simnibs_v3": str(tmp_path / "outputs" / "v3"),
        },
        "analysis": {
            "atlas_dir": str(tmp_path / "atlases"),
            "primary_metric": "mean_roi",
        },
        "stimulation_parameters": {
            "montage_preset": None,
            "anode": None,
            "cathode": None,
            "current_ma": None,
            "electrode_radius_mm": None,
            "electrode_thickness_mm": None,
            "electrode_shape": None,
            "electrode_conductivity_s_per_m": None,
        },
        "required_gates": [
            "approved_stimulation_parameters",
            "passing_mesh_audit",
            "explicit_submission_release",
        ],
    }


def _ready_plan(tmp_path: Path, *, submission_allowed: bool = False) -> dict:
    plan = _base_plan(tmp_path)
    plan["status"] = "ready_for_submission" if submission_allowed else "ready_for_preflight"
    plan["submission_allowed"] = submission_allowed
    plan["existing_inputs"]["simnibs_v4_root"] = str(tmp_path / "inputs" / "v4")
    plan["existing_inputs"]["simnibs_v3_root"] = str(tmp_path / "inputs" / "v3")
    plan["proposed_output_roots"]["simnibs_v4"] = str(tmp_path / "outputs" / "v4")
    plan["proposed_output_roots"]["simnibs_v3"] = str(tmp_path / "outputs" / "v3")
    plan["analysis"]["atlas_dir"] = str(tmp_path / "atlases")
    plan["stimulation_parameters"] = {
        "montage_preset": "test-only-right-m1",
        "anode": "TEST_ANODE",
        "cathode": "TEST_CATHODE",
        "current_ma": 1.5,
        "electrode_radius_mm": 9.0,
        "electrode_thickness_mm": 1.5,
        "electrode_shape": "test-shape",
        "electrode_conductivity_s_per_m": 1.2,
    }
    return plan


def _write_mesh(path: Path, payload: bytes, *, subject: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    anat = path.parent.parent
    (anat / f"{subject}_T1w.nii").write_bytes(b"t1")


def _build_inputs_and_audit(tmp_path: Path, plan: dict) -> tuple[Path, Path]:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    subjects = tuple(plan["cohort"]["subjects"])
    v4_root = Path(plan["existing_inputs"]["simnibs_v4_root"])
    v3_root = Path(plan["existing_inputs"]["simnibs_v3_root"])
    atlas_dir = Path(plan["analysis"]["atlas_dir"])
    atlas_dir.mkdir(parents=True)
    for subject in subjects:
        (atlas_dir / f"{subject}.nii.gz").write_bytes(b"atlas")
        for index in range(1, 41):
            _write_mesh(
                mesh_audit.remesh_path(v4_root, subject, index),
                f"{subject}-v4-{index}".encode(),
                subject=subject,
            )
            _write_mesh(
                mesh_audit.remesh_path(v3_root, subject, index),
                f"{subject}-v3-deterministic".encode(),
                subject=subject,
            )
        _write_mesh(
            mesh_audit.fixed_cache_path(v4_root, subject),
            f"{subject}-v4-2".encode(),
            subject=subject,
        )
        _write_mesh(
            mesh_audit.fixed_cache_path(v3_root, subject),
            f"{subject}-v3-deterministic".encode(),
            subject=subject,
        )

    stdout = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
        exit_code = mesh_audit.main(
            [
                "--v4-root",
                str(v4_root),
                "--v3-root",
                str(v3_root),
                "--subjects",
                ",".join(subjects),
            ]
        )
    assert exit_code == 0
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(stdout.getvalue(), encoding="utf-8")
    return plan_path, audit_path


def test_pending_plan_resolves_full_scope_but_remains_blocked(tmp_path, capsys):
    plan_path = tmp_path / "pending-plan.json"
    plan_path.write_text(json.dumps(_base_plan(tmp_path)), encoding="utf-8")
    assert workflow.main(["show-plan", "--plan", str(plan_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["scope"]["total_simulation_tasks"] == 1200
    assert payload["scope"]["v4_tasks"] == 800
    assert payload["scope"]["v3_tasks"] == 400
    assert any("current_ma is pending" in item for item in payload["blockers"])
    assert "passing mesh audit receipt is unavailable" in payload["blockers"]


def test_task_mapping_is_full_scope_and_uses_isolated_outputs(tmp_path):
    plan = _ready_plan(tmp_path)
    v4 = tdcs_replay.build_tasks(plan, "v4")
    v3 = tdcs_replay.build_tasks(plan, "v3")
    assert len(v4) == 800
    assert len(v3) == 400
    assert (v4[0].condition, v4[0].repeat_tag) == ("remesh", "repeat_001")
    assert (v4[399].condition, v4[399].repeat_tag) == ("remesh", "repeat_040")
    assert (v4[400].condition, v4[400].repeat_tag) == ("fixed_mesh", "repeat_001")
    assert (v4[799].condition, v4[799].repeat_tag) == ("fixed_mesh", "repeat_040")
    assert {task.condition for task in v3} == {"fixed_mesh"}
    assert v4[400].source_mesh == v4[439].source_mesh
    assert v3[0].source_mesh == v3[39].source_mesh
    assert Path(plan["proposed_output_roots"]["simnibs_v4"]) in v4[0].output_root.parents
    assert Path(plan["existing_inputs"]["simnibs_v4_root"]) not in v4[0].output_root.parents


def test_full_preflight_and_prepare_validate_all_audited_sources(tmp_path):
    plan = _ready_plan(tmp_path)
    plan_path, audit_path = _build_inputs_and_audit(tmp_path, plan)
    receipt = workflow.preflight(
        plan_path=plan_path,
        audit_path=audit_path,
        require_submission=False,
    )
    assert receipt["status"] == "ready"
    assert receipt["scope"]["total_simulation_tasks"] == 1200
    assert receipt["unique_source_meshes_checked"] == 420
    assert receipt["existing_completed_tasks"] == 0
    configs = workflow.prepare_outputs(plan_path, audit_path)
    assert set(configs) == {"v4", "v3"}
    v4_config = json.loads(Path(configs["v4"]).read_text(encoding="utf-8"))
    v3_config = json.loads(Path(configs["v3"]).read_text(encoding="utf-8"))
    assert [item["name"] for item in v4_config["conditions"]] == ["remesh", "fixed_mesh"]
    assert [item["name"] for item in v3_config["conditions"]] == ["fixed_mesh"]
    assert v4_config["stimulation"]["target_roi"] == "Right_M1"
    assert v4_config["tdcs_replay"]["source_policy"] == "read_only"


def test_runner_dry_run_and_mock_execution_preserve_source_mesh(
    tmp_path,
    monkeypatch,
    capsys,
):
    plan = _ready_plan(tmp_path)
    plan_path, audit_path = _build_inputs_and_audit(tmp_path, plan)
    task = tdcs_replay.task_for_index(plan, "v4", 0)
    source_hash = tdcs_replay.file_sha256(task.source_mesh)

    assert task_runner.main(
        [
            "run-task",
            "--plan",
            str(plan_path),
            "--audit",
            str(audit_path),
            "--version",
            "v4",
            "--task-index",
            "0",
            "--dry-run",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned"

    monkeypatch.setattr(task_runner, "install_backend", lambda version: "4.0.1-test")

    def fake_tdcs_pipeline(**kwargs):
        output_root = Path(kwargs["output_root"])
        subject = kwargs["subject"]
        field_root = output_root / "Output" / subject
        field_root.mkdir(parents=True, exist_ok=True)
        (field_root / "TDCS.msh").write_bytes(b"field")
        (field_root / f"{subject}_TDCS_1_scalar.msh").write_bytes(b"raw")
        (output_root / "tdcs_brain_only.nii.gz").write_bytes(b"nifti")
        return 0.5

    monkeypatch.setattr(task_runner.field_runner, "_run_tdcs_pipeline", fake_tdcs_pipeline)
    run_args = [
        "run-task",
        "--plan",
        str(plan_path),
        "--audit",
        str(audit_path),
        "--version",
        "v4",
        "--task-index",
        "0",
    ]
    assert task_runner.main(run_args) == 0
    assert task_runner.main(run_args) == 0
    assert tdcs_replay.file_sha256(task.source_mesh) == source_hash
    manifest = json.loads(task.task_manifest.read_text(encoding="utf-8"))
    assert manifest["source_policy"] == "read_only"
    assert manifest["runtime"]["simnibs_module"] == "SimNIBS/4.0.1-foss-2023a"
    assert manifest["outputs"]["field_mesh"]["sha256"] == tdcs_replay.file_sha256(
        tdcs_replay.required_output_paths(task)["field_mesh"]
    )


def test_submission_remains_fail_closed_without_release(tmp_path, capsys):
    plan = _ready_plan(tmp_path, submission_allowed=False)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert workflow.main(
        [
            "submit",
            "--plan",
            str(plan_path),
            "--audit",
            str(tmp_path / "missing-audit.json"),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Submission: NOT REQUESTED" in output
    assert "plan.submission_allowed must be true" in output


def test_released_submission_builds_full_dependency_gated_job_chain(
    tmp_path,
    monkeypatch,
    capsys,
):
    plan = _ready_plan(tmp_path, submission_allowed=True)
    plan_path, audit_path = _build_inputs_and_audit(tmp_path, plan)
    commands = []

    def fake_submit(command):
        commands.append(command)
        return str(1000 + len(commands))

    monkeypatch.setattr(workflow, "_submit", fake_submit)
    assert workflow.main(
        [
            "submit",
            "--plan",
            str(plan_path),
            "--audit",
            str(audit_path),
            "--submit",
        ]
    ) == 0
    capsys.readouterr()
    assert len(commands) == 7
    assert any("--array=0-799%50" in item for item in commands[2])
    assert any("--array=0-399%50" in item for item in commands[4])
    assert "--dependency=afterok:1002" in commands[2]
    assert "--dependency=afterok:1003" in commands[3]
    assert "--dependency=afterok:1004" in commands[4]
    assert "--dependency=afterok:1005" in commands[5]
    assert "--dependency=afterok:1006" in commands[6]
    submission = json.loads(
        (
            Path(plan["proposed_output_roots"]["simnibs_v4"])
            / "_pipeline"
            / "workflow"
            / "submission.json"
        ).read_text(encoding="utf-8")
    )
    assert submission["scope"]["total_simulation_tasks"] == 1200
    assert submission["jobs"]["all_finalize"] == "1007"


def test_exact_array_element_retry_is_encoded():
    script = (
        CURRENT_REPAIR_ROOT / "hpc_scripts" / "tdcs_replay_array.slurm"
    ).read_text(encoding="utf-8")
    assert 'REQUEUE_JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"' in script
    assert 'scontrol requeue "$REQUEUE_JOB_ID"' in script
    assert "SLURM_JOB_ID\"" not in script.split("scontrol requeue", maxsplit=1)[1]


def test_submission_wrapper_requires_an_explicit_plan():
    script = (
        CURRENT_REPAIR_ROOT
        / "hpc_scripts"
        / "submit_final132_right_m1_tdcs_replay.sh"
    ).read_text(encoding="utf-8")
    assert "final132_right_m1_tdcs_repeatability.pending.json" not in script
    assert "tdcs_replay_workflow.py\" submit" in script
    assert '"$@"' in script
    assert '--plan "$PLAN"' not in script
