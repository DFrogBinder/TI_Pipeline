import os
import subprocess
from pathlib import Path

import pytest

from approved_wave import workflow
from utils.camcan_dataset import sha256_file


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _build_campaign(tmp_path: Path, repeats=("01", "02")):
    subject = "sub-CC000001"
    subjects_file = _write(tmp_path / "subjects.txt", f"{subject}\n")
    source_root = tmp_path / "source"
    anat = source_root / subject / "anat"
    _write(anat / f"{subject}_T1w.nii", "t1")
    _write(anat / f"{subject}_T2w.nii", "t2")
    map_root = tmp_path / "maps"
    _write(map_root / f"{subject}{workflow.MAP_SUFFIX}", "approved-label")
    campaign = tmp_path / "study" / "campaign"
    prep_manifest = campaign / "prep.tsv"
    simulation_manifest = campaign / "simulation.tsv"
    summary = campaign / "preflight.json"
    payload = workflow.build_manifests(
        subjects_file=subjects_file,
        source_roots=(source_root,),
        map_root=map_root,
        output_root=tmp_path / "study" / "Left_Hippocampus_Runs",
        prep_result_dir=campaign / "prep-results",
        simulation_result_dir=campaign / "simulation-results",
        prep_manifest=prep_manifest,
        simulation_manifest=simulation_manifest,
        summary=summary,
        dataset_prefix="Left_Hippocampus",
        repeats=repeats,
        expected_subjects=1,
        expected_prep_tasks=1,
        expected_simulation_tasks=len(repeats),
    )
    assert payload["status"] == "ready"
    return (
        subject,
        prep_manifest,
        simulation_manifest,
        workflow.read_tsv(prep_manifest)[0],
        workflow.read_tsv(simulation_manifest),
        payload,
    )


def _run_preparation(tmp_path: Path):
    campaign = _build_campaign(tmp_path)
    subject, prep_manifest, _, prep_row, _, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    commands = []

    def fake_command(command, *, cwd, env=None):
        commands.append((list(command), Path(cwd), env))
        if "--segment" in command:
            _write(
                Path(prep_row["m2m_dir"])
                / "label_prep"
                / workflow.MAP_BASENAME,
                "generated-label",
            )
        elif command[-1:] == ["--mesh"]:
            _write(Path(prep_row["mesh_path"]), "mesh")
            _write(
                Path(prep_row["m2m_dir"])
                / "eeg_positions"
                / workflow.CAP_BASENAME,
                "Electrode,0,0,0,F8\n"
                "Electrode,0,0,0,P8\n"
                "Electrode,0,0,0,T7\n"
                "Electrode,0,0,0,P7\n"
                "Electrode,0,0,0,F1\n"
                "Electrode,0,0,0,F2\n"
                "Electrode,0,0,0,C3\n"
                "Electrode,0,0,0,CP3\n"
                "Electrode,0,0,0,AF4\n"
                "Electrode,0,0,0,F4\n"
                "Electrode,0,0,0,FC2\n"
                "Electrode,0,0,0,C2\n"
                "Electrode,0,0,0,F5\n"
                "Electrode,0,0,0,TP7\n"
                "Electrode,0,0,0,FT8\n",
            )

    def fake_mesh_validator(mesh_path):
        return {
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "tetrahedra": 123,
            "tissue_tags": [1, 2, 3, 5],
        }

    payload = workflow.run_prep_task(
        manifest=prep_manifest,
        task_index=0,
        montage_preset="left-hippocampus",
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        command_runner=fake_command,
        mesh_validator=fake_mesh_validator,
    )
    return campaign, payload, commands


def test_preflight_builds_one_prep_and_all_repeat_simulation_tasks(tmp_path):
    subject, _, _, prep, simulations, payload = _build_campaign(tmp_path)

    assert payload["prep_tasks_found"] == 1
    assert payload["simulation_tasks_found"] == 2
    assert payload["charm_segmentation_runs_expected"] == 1
    assert payload["meshes_expected"] == 2
    assert payload["repeat_scaffold_copies_expected"] == 1
    assert payload["fem_simulations_expected"] == 2
    assert payload["roast_involvement"] is False
    assert prep["task_id"] == "0"
    assert prep["dataset_name"] == "Left_Hippocampus_Data_01"
    assert prep["subject"] == subject
    assert prep["source_t1_sha256"] == sha256_file(Path(prep["source_t1"]))
    assert prep["approved_label_sha256"] == sha256_file(
        Path(prep["approved_label"])
    )
    assert [row["repeat_id"] for row in simulations] == ["01", "02"]
    assert all(row["canonical_m2m_dir"] == prep["m2m_dir"] for row in simulations)


def test_preflight_blocks_subject_present_in_multiple_complete_sources(tmp_path):
    subject = "sub-CC000001"
    subjects_file = _write(tmp_path / "subjects.txt", f"{subject}\n")
    roots = []
    for name in ("source-a", "source-b"):
        root = tmp_path / name
        anat = root / subject / "anat"
        _write(anat / f"{subject}_T1w.nii", f"{name}-t1")
        _write(anat / f"{subject}_T2w.nii", f"{name}-t2")
        roots.append(root)
    map_root = tmp_path / "maps"
    _write(map_root / f"{subject}{workflow.MAP_SUFFIX}", "approved-label")
    prep_manifest = tmp_path / "prep.tsv"

    payload = workflow.build_manifests(
        subjects_file=subjects_file,
        source_roots=roots,
        map_root=map_root,
        output_root=tmp_path / "output",
        prep_result_dir=tmp_path / "prep-results",
        simulation_result_dir=tmp_path / "simulation-results",
        prep_manifest=prep_manifest,
        simulation_manifest=tmp_path / "simulation.tsv",
        summary=tmp_path / "preflight.json",
        dataset_prefix="Left_Hippocampus",
        repeats=("01",),
        expected_subjects=1,
        expected_prep_tasks=1,
        expected_simulation_tasks=1,
    )

    assert payload["status"] == "blocked"
    assert "multiple source roots" in workflow.read_tsv(prep_manifest)[0]["message"]


def test_prep_runs_charm_once_then_meshes_the_exact_approved_label(tmp_path):
    campaign, payload, commands = _run_preparation(tmp_path)
    _, _, _, prep, _, _ = campaign

    assert payload["status"] == "complete"
    assert len(commands) == 2
    assert "--segment" in commands[0][0]
    assert commands[1][0][-1] == "--mesh"
    installed = Path(prep["m2m_dir"]) / "label_prep" / workflow.MAP_BASENAME
    assert installed.read_text(encoding="utf-8") == "approved-label"
    assert payload["installed_label_sha256"] == prep["approved_label_sha256"]
    assert payload["generated_label_sha256_before_approved_install"] != prep[
        "approved_label_sha256"
    ]
    assert payload["segmentation_runs_for_subject"] == 1
    assert payload["roast_involvement"] is False


def test_later_repeat_physically_copies_scaffold_remeshes_and_restores_cap(tmp_path):
    campaign, prep_payload, _ = _run_preparation(tmp_path)
    subject, _, simulation_manifest, prep, simulations, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    runner = _write(tmp_path / "runner.py", "# runner")
    validator = _write(tmp_path / "validator.py", "# validator")
    commands = []

    def fake_command(command, *, cwd, env=None):
        commands.append((list(command), Path(cwd), env))
        if command[-1:] == ["--mesh"]:
            m2m = Path(cwd) / f"m2m_{subject}"
            _write(m2m / f"{subject}.msh", "repeat-02-mesh")
            _write(
                m2m / "eeg_positions" / workflow.CAP_BASENAME,
                "generated-cap-that-must-be-replaced\n",
            )

    def fake_mesh_validator(mesh_path):
        return {
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "tetrahedra": 456,
            "tissue_tags": [1, 2, 3, 5],
        }

    payload = workflow.run_simulation_task(
        manifest=simulation_manifest,
        task_index=1,
        montage_preset="left-hippocampus",
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        simulation_runner=runner,
        simulation_validator=validator,
        command_runner=fake_command,
        mesh_validator=fake_mesh_validator,
    )

    assert payload["status"] == "complete"
    assert payload["mesh_sha256"] != prep_payload["mesh_sha256"]
    assert payload["segmentation_in_simulation_task"] is False
    assert payload["independent_repeat_mesh"] is True
    assert len(commands) == 3
    assert commands[0][0] == ["charm", subject, "--mesh"]
    assert "--reuse-existing-mesh" in commands[1][0]
    assert "--segment" not in commands[0][0]
    repeat_m2m = Path(simulations[1]["anat_dir"]) / f"m2m_{subject}"
    assert repeat_m2m.is_dir()
    assert not repeat_m2m.is_symlink()
    assert repeat_m2m.resolve() != Path(prep["m2m_dir"]).resolve()
    canonical_cap = Path(prep["m2m_dir"]) / "eeg_positions" / workflow.CAP_BASENAME
    repeat_cap = repeat_m2m / "eeg_positions" / workflow.CAP_BASENAME
    assert repeat_cap.read_bytes() == canonical_cap.read_bytes()
    assert commands[1][2]["TI_SIM_ROOT"] == simulations[1]["dataset_root"]
    mesh_marker = workflow._load_json(Path(simulations[1]["mesh_result_path"]))
    assert mesh_marker["scaffold_copy_mode"] == "physical"
    assert mesh_marker["independent_repeat_mesh"] is True


@pytest.mark.parametrize(
    ("dataset_prefix", "montage_preset"),
    (
        ("Left_M1", "left-m1"),
        ("Right_DLPC", "right-dlpfc"),
        ("Right_Thalamus", "right-thalamus"),
    ),
)
def test_external_scaffold_preflight_builds_all_repeat_remesh_tasks(
    tmp_path,
    dataset_prefix,
    montage_preset,
):
    campaign, _, _ = _run_preparation(tmp_path)
    subject, prep_manifest, _, prep, _, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    target_campaign = tmp_path / "study" / "campaign" / dataset_prefix
    manifest = target_campaign / "simulation.tsv"

    payload = workflow.build_scaffold_reuse_manifest(
        subjects_file=tmp_path / "subjects.txt",
        canonical_prep_manifest=prep_manifest,
        canonical_dataset_name="Left_Hippocampus_Data_01",
        output_root=tmp_path / "study" / f"{dataset_prefix}_Runs",
        simulation_result_dir=target_campaign / "simulation-results",
        simulation_manifest=manifest,
        summary=target_campaign / "preflight.json",
        dataset_prefix=dataset_prefix,
        montage_preset=montage_preset,
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        repeats=("01", "02"),
        expected_subjects=1,
        expected_simulation_tasks=2,
    )

    rows = workflow.read_tsv(manifest)
    assert payload["status"] == "ready"
    assert payload["charm_segmentation_runs_expected"] == 0
    assert payload["physical_scaffold_copies_expected"] == 2
    assert payload["independent_meshes_expected"] == 2
    assert payload["fem_simulations_expected"] == 2
    assert payload["roast_involvement"] is False
    assert [row["repeat_id"] for row in rows] == ["01", "02"]
    assert all(row["status"] == "ready" for row in rows)
    assert all(row["subject"] == subject for row in rows)
    assert all(row["canonical_m2m_dir"] == prep["m2m_dir"] for row in rows)
    assert all(Path(row["anat_dir"]) != Path(prep["anat_dir"]) for row in rows)
    assert all(row["mesh_result_path"] != prep["result_path"] for row in rows)


def test_external_roi_repeat01_copies_scaffold_and_remeshes(tmp_path):
    campaign, prep_payload, _ = _run_preparation(tmp_path)
    subject, prep_manifest, _, prep, _, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    target_campaign = tmp_path / "study" / "campaign" / "Left_M1"
    manifest = target_campaign / "simulation.tsv"
    runner = _write(tmp_path / "runner.py", "# runner")
    validator = _write(tmp_path / "validator.py", "# validator")
    commands = []

    workflow.build_scaffold_reuse_manifest(
        subjects_file=tmp_path / "subjects.txt",
        canonical_prep_manifest=prep_manifest,
        canonical_dataset_name="Left_Hippocampus_Data_01",
        output_root=tmp_path / "study" / "Left_M1_Runs",
        simulation_result_dir=target_campaign / "simulation-results",
        simulation_manifest=manifest,
        summary=target_campaign / "preflight.json",
        dataset_prefix="Left_M1",
        montage_preset="left-m1",
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        repeats=("01",),
        expected_subjects=1,
        expected_simulation_tasks=1,
    )
    row = workflow.read_tsv(manifest)[0]

    def fake_command(command, *, cwd, env=None):
        commands.append((list(command), Path(cwd), env))
        if command[-1:] == ["--mesh"]:
            m2m = Path(cwd) / f"m2m_{subject}"
            _write(m2m / f"{subject}.msh", "left-m1-independent-mesh")
            _write(
                m2m / "eeg_positions" / workflow.CAP_BASENAME,
                "generated-cap-that-must-be-replaced\n",
            )

    def fake_mesh_validator(mesh_path):
        return {
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "tetrahedra": 789,
            "tissue_tags": [1, 2, 3, 5],
        }

    payload = workflow.run_simulation_task(
        manifest=manifest,
        task_index=0,
        montage_preset="left-m1",
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        simulation_runner=runner,
        simulation_validator=validator,
        command_runner=fake_command,
        mesh_validator=fake_mesh_validator,
    )

    assert payload["status"] == "complete"
    assert payload["mesh_sha256"] != prep_payload["mesh_sha256"]
    assert payload["segmentation_in_simulation_task"] is False
    assert commands[0][0] == ["charm", subject, "--mesh"]
    assert all("--segment" not in command for command, _, _ in commands)
    target_m2m = Path(row["anat_dir"]) / f"m2m_{subject}"
    assert target_m2m.is_dir()
    assert not target_m2m.is_symlink()
    assert target_m2m.resolve() != Path(prep["m2m_dir"]).resolve()
    canonical_cap = Path(prep["m2m_dir"]) / "eeg_positions" / workflow.CAP_BASENAME
    target_cap = target_m2m / "eeg_positions" / workflow.CAP_BASENAME
    assert target_cap.read_bytes() == canonical_cap.read_bytes()


def test_external_scaffold_preflight_blocks_changed_reviewed_label(tmp_path):
    campaign, _, _ = _run_preparation(tmp_path)
    subject, prep_manifest, _, prep, _, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    installed_label = (
        Path(prep["m2m_dir"]) / "label_prep" / workflow.MAP_BASENAME
    )
    installed_label.write_text("changed-after-review", encoding="utf-8")
    target_campaign = tmp_path / "study" / "campaign" / "Left_M1"
    manifest = target_campaign / "simulation.tsv"

    payload = workflow.build_scaffold_reuse_manifest(
        subjects_file=tmp_path / "subjects.txt",
        canonical_prep_manifest=prep_manifest,
        canonical_dataset_name="Left_Hippocampus_Data_01",
        output_root=tmp_path / "study" / "Left_M1_Runs",
        simulation_result_dir=target_campaign / "simulation-results",
        simulation_manifest=manifest,
        summary=target_campaign / "preflight.json",
        dataset_prefix="Left_M1",
        montage_preset="left-m1",
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        repeats=("01", "02"),
        expected_subjects=1,
        expected_simulation_tasks=2,
    )

    rows = workflow.read_tsv(manifest)
    assert payload["status"] == "blocked"
    assert payload["blocked_subjects"] == 1
    assert payload["simulation_ready"] == 0
    assert all(row["status"] == "blocked" for row in rows)
    assert all("installed label hash mismatch" in row["message"] for row in rows)


def test_completed_simulation_marker_is_validated_and_reused(tmp_path):
    campaign, _, _ = _run_preparation(tmp_path)
    _, _, simulation_manifest, _, _, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    runner = _write(tmp_path / "runner.py", "# runner")
    validator = _write(tmp_path / "validator.py", "# validator")
    commands = []

    def fake_command(command, *, cwd, env=None):
        commands.append(list(command))

    kwargs = {
        "manifest": simulation_manifest,
        "task_index": 0,
        "montage_preset": "left-hippocampus",
        "targets_csv": targets_csv,
        "expected_targets_sha256": sha256_file(targets_csv),
        "simulation_runner": runner,
        "simulation_validator": validator,
        "command_runner": fake_command,
    }
    workflow.run_simulation_task(**kwargs)
    commands.clear()
    reused = workflow.run_simulation_task(**kwargs)

    assert reused["status"] == "already_complete"
    assert len(commands) == 1
    assert commands[0][-2:] == ["--subject", "sub-CC000001"]


def test_later_repeat_reuses_completed_repeat_mesh_after_simulation_retry(tmp_path):
    campaign, _, _ = _run_preparation(tmp_path)
    subject, _, simulation_manifest, _, simulations, _ = campaign
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    runner = _write(tmp_path / "runner.py", "# runner")
    validator = _write(tmp_path / "validator.py", "# validator")
    commands = []

    def fake_command(command, *, cwd, env=None):
        commands.append(list(command))
        if command[-1:] == ["--mesh"]:
            _write(
                Path(cwd) / f"m2m_{subject}" / f"{subject}.msh",
                "repeat-02-mesh",
            )

    def fake_mesh_validator(mesh_path):
        return {
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "tetrahedra": 456,
            "tissue_tags": [1, 2, 3, 5],
        }

    kwargs = {
        "manifest": simulation_manifest,
        "task_index": 1,
        "montage_preset": "left-hippocampus",
        "targets_csv": targets_csv,
        "expected_targets_sha256": sha256_file(targets_csv),
        "simulation_runner": runner,
        "simulation_validator": validator,
        "command_runner": fake_command,
        "mesh_validator": fake_mesh_validator,
    }
    workflow.run_simulation_task(**kwargs)
    Path(simulations[1]["result_path"]).unlink()
    commands.clear()

    workflow.run_simulation_task(**kwargs)

    assert len(commands) == 2
    assert all(command[-1:] != ["--mesh"] for command in commands)
    assert "--reuse-existing-mesh" in commands[0]


def test_submitter_submits_dependent_full_scope_arrays(tmp_path):
    subject = "sub-CC000001"
    subjects_file = _write(tmp_path / "subjects.txt", f"{subject}\n")
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_b.mkdir()
    anat = source_a / subject / "anat"
    _write(anat / f"{subject}_T1w.nii", "t1")
    _write(anat / f"{subject}_T2w.nii", "t2")
    maps = tmp_path / "maps"
    _write(maps / f"{subject}{workflow.MAP_SUFFIX}", "approved-label")
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_count = tmp_path / "sbatch.count"
    fake_sbatch = _write(
        tmp_path / "sbatch",
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "count=0\n"
        "test -f \"$FAKE_SBATCH_COUNT\" && count=$(cat \"$FAKE_SBATCH_COUNT\")\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$FAKE_SBATCH_COUNT\"\n"
        "test \"$count\" -eq 1 && echo 12001 || echo 12002\n",
    )
    fake_scontrol = _write(
        tmp_path / "scontrol",
        "#!/bin/bash\n"
        "echo 'MaxArraySize = 1001'\n"
        "for value in $(seq 1 20000); do echo \"OtherSetting${value} = ${value}\"; done\n",
    )
    fake_scancel = _write(tmp_path / "scancel", "#!/bin/bash\nexit 0\n")
    for executable in (fake_sbatch, fake_scontrol, fake_scancel):
        executable.chmod(0o755)
    script = (
        Path(__file__).resolve().parents[1]
        / "HPC_scripts"
        / "submit_approved_wave_mesh_sim.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "SUBJECTS_FILE": str(subjects_file),
            "SOURCE_ROOT_1": str(source_a),
            "SOURCE_ROOT_2": str(source_b),
            "MAP_ROOT": str(maps),
            "STUDY_ROOT": str(tmp_path / "study"),
            "EXPECTED_SUBJECTS": "1",
            "EXPECTED_PREP_TASKS": "1",
            "EXPECTED_SIMULATION_TASKS": "10",
            "MAX_CONCURRENT_TASKS": "10",
            "SBATCH_BIN": str(fake_sbatch),
            "SCONTROL_BIN": str(fake_scontrol),
            "SCANCEL_BIN": str(fake_scancel),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "FAKE_SBATCH_COUNT": str(sbatch_count),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "preparation tasks: 1" in completed.stdout
    assert "simulation tasks: 10" in completed.stdout
    assert "preparation array: 0-0%10" in completed.stdout
    assert "simulation array: 0-9%10 (afterok preparation)" in completed.stdout
    assert "expected independent meshes: 10" in completed.stdout
    assert "expected validated FEM simulations: 10" in completed.stdout
    assert "execution: full requested wave; not a smoke or subset" in completed.stdout
    assert "Submitted preparation array job: 12001" in completed.stdout
    assert "Submitted dependent simulation array job: 12002" in completed.stdout
    submissions = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(submissions) == 2
    assert "--array=0-0%10" in submissions[0]
    assert "TI_APPROVED_WAVE_STAGE=prepare" in submissions[0]
    assert "--array=0-9%10" in submissions[1]
    assert "--dependency=afterok:12001" in submissions[1]
    assert "TI_APPROVED_WAVE_STAGE=simulate" in submissions[1]


def test_external_scaffold_submitter_submits_one_full_roi_array(tmp_path):
    campaign, _, _ = _run_preparation(tmp_path)
    _, prep_manifest, _, _, _, _ = campaign
    sbatch_log = tmp_path / "external-sbatch.log"
    fake_sbatch = _write(
        tmp_path / "external-sbatch",
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "echo 13001\n",
    )
    fake_scontrol = _write(
        tmp_path / "external-scontrol",
        "#!/bin/bash\n"
        "echo 'MaxArraySize = 1001'\n"
        "for value in $(seq 1 20000); do echo \"OtherSetting${value} = ${value}\"; done\n",
    )
    for executable in (fake_sbatch, fake_scontrol):
        executable.chmod(0o755)

    script = (
        Path(__file__).resolve().parents[1]
        / "HPC_scripts"
        / "submit_approved_wave_roi_from_scaffold.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "SUBJECTS_FILE": str(tmp_path / "subjects.txt"),
            "STUDY_ROOT": str(tmp_path / "study"),
            "CANONICAL_PREP_MANIFEST": str(prep_manifest),
            "EXPECTED_SUBJECTS": "1",
            "EXPECTED_SIMULATION_TASKS": "10",
            "MAX_CONCURRENT_TASKS": "10",
            "SBATCH_BIN": str(fake_sbatch),
            "SCONTROL_BIN": str(fake_scontrol),
            "FAKE_SBATCH_LOG": str(sbatch_log),
        }
    )

    completed = subprocess.run(
        ["bash", str(script), "Left_M1"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "dataset/ROI: Left_M1 approved wave 1" in completed.stdout
    assert "canonical subjects/scaffolds: 1" in completed.stdout
    assert "simulation tasks: 10" in completed.stdout
    assert "array: 0-9%10" in completed.stdout
    assert "expected CHARM segmentation runs: 0" in completed.stdout
    assert "expected physical scaffold copies: 10" in completed.stdout
    assert "expected independent meshes: 10" in completed.stdout
    assert "expected validated FEM simulations: 10" in completed.stdout
    assert "Retry limit:          unlimited" in completed.stdout
    assert "Submitted Left_M1 independent-remesh/FEM array job: 13001" in completed.stdout
    submission = sbatch_log.read_text(encoding="utf-8").strip()
    assert "--array=0-9%10" in submission
    assert "--dependency" not in submission
    assert "TI_APPROVED_WAVE_STAGE=simulate" in submission
    assert "TI_MONTAGE_PRESET=left-m1" in submission
    assert "TI_APPROVED_WAVE_MAX_RETRIES=unlimited" in submission


def test_external_scaffold_retry_policy_accepts_unlimited_and_finite_override():
    scripts = Path(__file__).resolve().parents[1] / "HPC_scripts"
    submitter = (scripts / "submit_approved_wave_roi_from_scaffold.sh").read_text(
        encoding="utf-8"
    )
    worker = (scripts / "approved_wave_mesh_sim_array.slurm").read_text(
        encoding="utf-8"
    )

    assert 'MAX_RETRIES="${TI_APPROVED_WAVE_MAX_RETRIES:-unlimited}"' in submitter
    assert '[ "${MAX_RETRIES}" != "unlimited" ]' in submitter
    assert '[ "${MAX_RETRIES}" != "unlimited" ]' in worker
    assert '[ "${MAX_RETRIES}" = "unlimited" ] ||' in worker
