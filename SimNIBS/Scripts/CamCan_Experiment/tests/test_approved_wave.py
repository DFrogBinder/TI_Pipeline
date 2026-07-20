import json
import os
import subprocess
from pathlib import Path

from approved_wave import workflow
from utils.camcan_dataset import sha256_file


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _build_one_task(tmp_path: Path):
    subject = "sub-CC000001"
    subjects_file = _write(tmp_path / "subjects.txt", f"{subject}\n")
    source_root = tmp_path / "source"
    anat = source_root / subject / "anat"
    _write(anat / f"{subject}_T1w.nii", "t1")
    _write(anat / f"{subject}_T2w.nii", "t2")
    map_root = tmp_path / "maps"
    _write(map_root / f"{subject}{workflow.MAP_SUFFIX}", "approved-label")
    output_root = tmp_path / "study" / "Left_Hippocampus_Runs"
    result_dir = tmp_path / "study" / "campaign" / "results"
    manifest = tmp_path / "study" / "campaign" / "tasks.tsv"
    summary = tmp_path / "study" / "campaign" / "preflight.json"
    payload = workflow.build_manifest(
        subjects_file=subjects_file,
        source_roots=(source_root,),
        map_root=map_root,
        output_root=output_root,
        result_dir=result_dir,
        manifest=manifest,
        summary=summary,
        dataset_prefix="Left_Hippocampus",
        repeats=("01",),
        expected_subjects=1,
        expected_tasks=1,
    )
    assert payload["status"] == "ready"
    return subject, manifest, workflow.read_tsv(manifest)[0]


def test_preflight_builds_repeat_major_full_scope_and_hashes_inputs(tmp_path):
    subject, manifest, first = _build_one_task(tmp_path)
    rows = workflow.read_tsv(manifest)

    assert len(rows) == 1
    assert first["task_id"] == "0"
    assert first["dataset_name"] == "Left_Hippocampus_Data_01"
    assert first["subject"] == subject
    assert first["status"] == "ready"
    assert first["source_t1_sha256"] == sha256_file(Path(first["source_t1"]))
    assert first["approved_label_sha256"] == sha256_file(
        Path(first["approved_label"])
    )


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
    manifest = tmp_path / "tasks.tsv"

    payload = workflow.build_manifest(
        subjects_file=subjects_file,
        source_roots=roots,
        map_root=map_root,
        output_root=tmp_path / "output",
        result_dir=tmp_path / "results",
        manifest=manifest,
        summary=tmp_path / "preflight.json",
        dataset_prefix="Left_Hippocampus",
        repeats=("01",),
        expected_subjects=1,
        expected_tasks=1,
    )

    assert payload["status"] == "blocked"
    assert "multiple source roots" in workflow.read_tsv(manifest)[0]["message"]


def test_run_task_installs_exact_label_meshes_simulates_and_reuses_completion(tmp_path):
    subject, manifest, row = _build_one_task(tmp_path)
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    targets_hash = sha256_file(targets_csv)
    simulation_runner = _write(tmp_path / "runner.py", "# runner")
    simulation_validator = _write(tmp_path / "validator.py", "# validator")
    commands = []

    def fake_command(command, *, cwd, env=None):
        commands.append((list(command), Path(cwd), env))
        if "--segment" in command:
            _write(
                Path(row["m2m_dir"])
                / "label_prep"
                / workflow.MAP_BASENAME,
                "generated-label",
            )
        elif command[-1:] == ["--mesh"]:
            _write(Path(row["mesh_path"]), "mesh")
            _write(
                Path(row["m2m_dir"])
                / "eeg_positions"
                / workflow.CAP_BASENAME,
                "Electrode,0,0,0,F8\n"
                "Electrode,0,0,0,P8\n"
                "Electrode,0,0,0,T7\n"
                "Electrode,0,0,0,P7\n",
            )

    def fake_mesh_validator(mesh_path):
        return {
            "mesh_path": str(mesh_path),
            "mesh_sha256": sha256_file(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "tetrahedra": 123,
            "tissue_tags": [1, 2, 3, 4, 5],
        }

    payload = workflow.run_task(
        manifest=manifest,
        task_index=0,
        montage_preset="left-hippocampus",
        targets_csv=targets_csv,
        expected_targets_sha256=targets_hash,
        simulation_runner=simulation_runner,
        simulation_validator=simulation_validator,
        command_runner=fake_command,
        mesh_validator=fake_mesh_validator,
    )

    assert payload["status"] == "complete"
    assert len(commands) == 4
    assert "--segment" in commands[0][0]
    assert commands[1][0][-1] == "--mesh"
    assert "--reuse-existing-mesh" in commands[2][0]
    assert commands[3][0][-2:] == ["--subject", subject]
    installed = Path(row["m2m_dir"]) / "label_prep" / workflow.MAP_BASENAME
    assert installed.read_text(encoding="utf-8") == "approved-label"
    mesh_marker = json.loads(Path(row["mesh_result_path"]).read_text())
    assert mesh_marker["approved_label_sha256"] == row["approved_label_sha256"]
    assert mesh_marker["generated_label_sha256_before_approved_install"] != row[
        "approved_label_sha256"
    ]

    commands.clear()
    reused = workflow.run_task(
        manifest=manifest,
        task_index=0,
        montage_preset="left-hippocampus",
        targets_csv=targets_csv,
        expected_targets_sha256=targets_hash,
        simulation_runner=simulation_runner,
        simulation_validator=simulation_validator,
        command_runner=fake_command,
        mesh_validator=fake_mesh_validator,
    )

    assert reused["status"] == "already_complete"
    assert len(commands) == 1
    assert commands[0][0][-2:] == ["--subject", subject]


def test_run_task_reuses_completed_mesh_when_simulation_marker_is_absent(tmp_path):
    _, manifest, row = _build_one_task(tmp_path)
    targets_csv = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    simulation_runner = _write(tmp_path / "runner.py", "# runner")
    simulation_validator = _write(tmp_path / "validator.py", "# validator")
    label = _write(
        Path(row["m2m_dir"]) / "label_prep" / workflow.MAP_BASENAME,
        "approved-label",
    )
    mesh = _write(Path(row["mesh_path"]), "mesh")
    cap = _write(
        Path(row["m2m_dir"]) / "eeg_positions" / workflow.CAP_BASENAME,
        "Electrode,0,0,0,F8\n"
        "Electrode,0,0,0,P8\n"
        "Electrode,0,0,0,T7\n"
        "Electrode,0,0,0,P7\n",
    )
    workflow.write_json_atomic(
        Path(row["mesh_result_path"]),
        {
            "status": "complete",
            "mesh_sha256": sha256_file(mesh),
            "eeg_cap_sha256": sha256_file(cap),
            "approved_label_sha256": sha256_file(label),
        },
    )
    commands = []

    def fake_command(command, *, cwd, env=None):
        commands.append(list(command))

    payload = workflow.run_task(
        manifest=manifest,
        task_index=0,
        montage_preset="left-hippocampus",
        targets_csv=targets_csv,
        expected_targets_sha256=sha256_file(targets_csv),
        simulation_runner=simulation_runner,
        simulation_validator=simulation_validator,
        command_runner=fake_command,
        mesh_validator=lambda path: {},
    )

    assert payload["status"] == "complete"
    assert len(commands) == 2
    assert "--reuse-existing-mesh" in commands[0]
    assert "--segment" not in commands[0]


def test_submitter_preflights_full_requested_scope_before_fake_sbatch(tmp_path):
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
    fake_sbatch = _write(tmp_path / "sbatch", "#!/bin/bash\necho 12345\n")
    fake_scontrol = _write(
        tmp_path / "scontrol",
        "#!/bin/bash\n"
        "echo 'MaxArraySize = 1001'\n"
        "for value in $(seq 1 20000); do echo \"OtherSetting${value} = ${value}\"; done\n",
    )
    fake_sbatch.chmod(0o755)
    fake_scontrol.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "HPC_scripts" / "submit_approved_wave_mesh_sim.sh"
    env = os.environ.copy()
    env.update(
        {
            "SUBJECTS_FILE": str(subjects_file),
            "SOURCE_ROOT_1": str(source_a),
            "SOURCE_ROOT_2": str(source_b),
            "MAP_ROOT": str(maps),
            "STUDY_ROOT": str(tmp_path / "study"),
            "EXPECTED_SUBJECTS": "1",
            "EXPECTED_TASKS": "10",
            "MAX_CONCURRENT_TASKS": "10",
            "SBATCH_BIN": str(fake_sbatch),
            "SCONTROL_BIN": str(fake_scontrol),
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

    assert "tasks: 10" in completed.stdout
    assert "array: 0-9%10" in completed.stdout
    assert "execution: full requested wave; not a smoke or subset" in completed.stdout
    assert "Submitted approved wave-1 array job: 12345" in completed.stdout
