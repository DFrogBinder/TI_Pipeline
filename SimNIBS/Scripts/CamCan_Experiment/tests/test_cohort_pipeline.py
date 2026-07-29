import json
import os
import subprocess
from pathlib import Path

from cohort_pipeline import workflow
from utils.camcan_dataset import sha256_file


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, subjects=("sub-CC000001",)):
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_b.mkdir()
    maps = tmp_path / "corrected" / "maps"
    collection_rows = []
    for subject in subjects:
        anat = source_a / subject / "anat"
        _write(anat / f"{subject}_T1w.nii", f"{subject}-t1")
        _write(anat / f"{subject}_T2w.nii", f"{subject}-t2")
        label = _write(
            maps / f"{subject}{workflow.MAP_SUFFIX}",
            f"{subject}-corrected-v4",
        )
        collection_rows.append(
            "\t".join(
                (
                    subject,
                    "complete",
                    str(label),
                    str(label),
                    sha256_file(label),
                    str(label.stat().st_size),
                    "complete",
                )
            )
        )
    collection = _write(
        tmp_path / "corrected" / "collection.tsv",
        "subject\tstatus\tsource_map\tcollected_map\tsha256\tbytes\tmessage\n"
        + "\n".join(collection_rows)
        + "\n",
    )
    cohort_dir = tmp_path / "cohorts" / "test"
    subject_file = _write(
        cohort_dir / "subjects.txt", "\n".join(subjects) + "\n"
    )
    cohort = {
        "schema_version": 1,
        "cohort_id": "test",
        "status": "provisional",
        "subjects_file": "subjects.txt",
        "expected_subjects": len(subjects),
    }
    cohort_config = _write(
        cohort_dir / "cohort.json", json.dumps(cohort)
    )
    study_root = tmp_path / "study"
    scaffold_root = tmp_path / "scaffolds"
    study = {
        "schema_version": 1,
        "study_id": "test-study",
        "corrected_map_set": "test-v4",
        "hpc_corrected_map_manifest": str(collection),
        "hpc_study_root": str(study_root),
        "hpc_scaffold_root": str(scaffold_root),
        "hpc_source_roots": [str(source_a), str(source_b)],
        "hpc_legacy_scaffold_manifest": str(tmp_path / "missing-legacy.tsv"),
        "targets_csv_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
        ),
        "rois": list(workflow.DEFAULT_ROIS),
        "repeats": list(workflow.DEFAULT_REPEATS),
    }
    study_config = _write(tmp_path / "study.json", json.dumps(study))
    return {
        "subjects": subjects,
        "subjects_file": subject_file,
        "source_a": source_a,
        "source_b": source_b,
        "collection": collection,
        "cohort_config": cohort_config,
        "study_config": study_config,
        "study_root": study_root,
        "scaffold_root": scaffold_root,
        "campaign": study_root / "campaigns" / "test",
        "targets": Path(__file__).resolve().parents[2] / "utils" / "targets.csv",
    }


def _preflight(fixture, **overrides):
    return workflow.build_manifests(
        study_config=fixture["study_config"],
        cohort_config=fixture["cohort_config"],
        campaign_root=fixture["campaign"],
        targets_csv=fixture["targets"],
        max_array_elements=overrides.get("max_array_elements", 1000),
        mesh_workers=overrides.get("mesh_workers", 2),
        legacy_scaffold_manifest=overrides.get("legacy_scaffold_manifest"),
    )


def _write_cap(path: Path):
    names = {
        name
        for config in workflow.CAMCAN_ROI_CONFIGS
        for name in workflow.electrode_names_for_config(
            config, Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
        )
    }
    _write(
        path,
        "".join(f"Electrode,0,0,0,{name}\n" for name in sorted(names)),
    )


def _bootstrap_command_runner(scaffold_row, commands=None):
    def run(command, *, cwd, env=None):
        if commands is not None:
            commands.append(command)
        m2m = Path(scaffold_row["scaffold_m2m_dir"])
        if "--segment" in command:
            _write(
                m2m / "label_prep" / workflow.MAP_BASENAME,
                "generated",
            )
        elif "--mesh" in command:
            assert (
                m2m / "label_prep" / workflow.MAP_BASENAME
            ).read_text(encoding="utf-8") == (
                f"{scaffold_row['subject']}-corrected-v4"
            )
            _write(m2m / f"{scaffold_row['subject']}.msh", "temporary-mesh")
            _write_cap(m2m / "eeg_positions" / workflow.CAP_BASENAME)
        else:
            raise AssertionError(f"unexpected CHARM command: {command}")

    return run


def test_mesh_validation_skip_hashes_performs_no_content_hashing(
    tmp_path, monkeypatch
):
    subject = "sub-CC000001"
    m2m = tmp_path / "m2m"
    mesh = _write(m2m / f"{subject}.msh", "mesh")
    label = _write(
        m2m / "label_prep" / workflow.MAP_BASENAME,
        "label",
    )
    cap = _write(
        m2m / "eeg_positions" / workflow.CAP_BASENAME,
        "cap",
    )
    scaffold_result = _write(
        tmp_path / "scaffold.json",
        json.dumps({"status": "complete", "eeg_cap_sha256": "cap-hash"}),
    )
    mesh_result = _write(
        tmp_path / "mesh.json",
        json.dumps(
            {
                "status": "complete",
                "subject": subject,
                "roi": "Left_M1",
                "repeat_id": "01",
                "dataset_name": "Left_M1_Data_01",
                "label_sha256_after": "label-hash",
                "independent_repeat_mesh": True,
                "scaffold_copy_mode": "physical",
                "roast_involvement": False,
                "mesh_bytes": mesh.stat().st_size,
                "mesh_sha256": "mesh-hash",
            }
        ),
    )
    row = {
        "subject": subject,
        "roi": "Left_M1",
        "repeat_id": "01",
        "dataset_name": "Left_M1_Data_01",
        "corrected_label_sha256": "label-hash",
        "m2m_dir": str(m2m),
        "mesh_path": str(mesh),
        "mesh_result_path": str(mesh_result),
        "scaffold_result_path": str(scaffold_result),
    }

    def unexpected_hash(_path):
        raise AssertionError("skip-hashes attempted content hashing")

    monkeypatch.setattr(workflow, "sha256_file", unexpected_hash)
    result = workflow.mesh_result_is_current(row, verify_hash=False)

    assert result is not None
    assert label.is_file()
    assert cap.is_file()


def test_preflight_scales_four_rois_and_chunks_without_hardcoded_subject_count(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    payload = _preflight(
        fixture,
        max_array_elements=7,
        mesh_workers=2,
    )

    assert payload["status"] == "ready"
    assert payload["subjects"] == 1
    assert payload["scaffold_bootstrap"] == 1
    assert payload["full_charm_segmentations_expected"] == 1
    assert payload["temporary_scaffold_mesh_runs_expected"] == 1
    assert payload["mesh_tasks"] == 40
    assert payload["simulation_tasks"] == 40
    assert payload["mesh_array_elements"] == 20
    assert payload["mesh_array_chunks"] == 3
    assert payload["simulation_array_chunks"] == 6
    rows = workflow.approved.read_tsv(fixture["campaign"] / "mesh_tasks.tsv")
    assert len(rows) == 40
    assert {row["roi"] for row in rows} == set(workflow.DEFAULT_ROIS)
    assert {row["repeat_id"] for row in rows} == set(workflow.DEFAULT_REPEATS)


def test_preflight_chunks_maximum_200_subject_cohort(tmp_path):
    subjects = tuple(f"sub-CC{index:06d}" for index in range(1, 201))
    fixture = _fixture(tmp_path, subjects=subjects)
    payload = _preflight(
        fixture,
        max_array_elements=1000,
        mesh_workers=2,
    )

    assert payload["subjects"] == 200
    assert payload["mesh_tasks"] == 8000
    assert payload["mesh_array_elements"] == 4000
    assert payload["mesh_array_chunks"] == 4
    assert payload["simulation_tasks"] == 8000
    assert payload["simulation_array_chunks"] == 8


def test_preflight_supports_complete_subject_specific_pareto_table(tmp_path):
    fixture = _fixture(tmp_path)
    subject = fixture["subjects"][0]
    individualized = _write(
        fixture["cohort_config"].parent / "individualized_targets.csv",
        "subject,dataset_roi,roi,montage_preset,E_target,stimulated_volume,"
        "configuration,pair1,pair2,current1,current2,pareto_selection\n"
        f"{subject},Left_M1,ctx_lh_G_precentral,left-m1,0.2,1,1,"
        "F1-F2,C3-CP3,2,1,TI_free.Emin\n"
        f"{subject},Left_Hippocampus,Left_Hippocampus,left-hippocampus,"
        "0.2,1,2,F8-P8,T7-P7,2,1,TI_free.Emin\n"
        f"{subject},Right_DLPC,ctx_rh_G_front_middle,right-dlpfc,0.2,1,3,"
        "AF4-F4,FC2-C2,1,2,TI_free.Emin\n"
        f"{subject},Right_Thalamus,Right_Thalamus,right-thalamus,0.2,1,4,"
        "F5-TP7,FT8-P8,2,2,TI_free.Emin\n",
    )
    cohort = json.loads(fixture["cohort_config"].read_text())
    cohort["individualized_targets_csv"] = individualized.name
    cohort["individualized_targets_csv_sha256"] = sha256_file(individualized)
    fixture["cohort_config"].write_text(json.dumps(cohort))

    payload = _preflight(fixture)

    assert payload["montage_mode"] == "subject_roi_individualized"
    assert payload["individualized_target_rows"] == 4
    assert payload["individualized_targets_csv_sha256"] == sha256_file(
        individualized
    )
    rows = workflow.approved.read_tsv(
        fixture["campaign"] / "simulation_tasks.tsv"
    )
    hippocampus = next(
        row
        for row in rows
        if row["roi"] == "Left_Hippocampus"
        and row["repeat_id"] == "01"
    )
    assert hippocampus["required_electrodes"] == "F8,P8,T7,P7"


def test_release_job_submits_one_array_and_one_dependent_releaser(tmp_path):
    pipeline_dir = Path(__file__).resolve().parents[1] / "cohort_pipeline"
    release_script = pipeline_dir / "cohort_pipeline_release.sh"
    array_script = pipeline_dir / "cohort_pipeline_array.slurm"
    workflow_script = pipeline_dir / "workflow.py"
    camcan_dir = pipeline_dir.parent
    targets = Path(__file__).resolve().parents[2] / "utils" / "targets.csv"
    plan = _write(
        tmp_path / "release_plan.tsv",
        "step\tstage\tchunk_index\toffset\tcount\n"
        "0\tmesh\t0\t0\t2\n",
    )
    mesh_manifest = _write(tmp_path / "mesh.tsv", "task_id\n")
    simulation_manifest = _write(tmp_path / "simulation.tsv", "task_id\n")
    job_ids = _write(tmp_path / "submitted_job_ids.txt", "100\n")
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_counter = tmp_path / "sbatch.counter"
    fake_sbatch = _write(
        tmp_path / "fake_sbatch.sh",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> {sbatch_log}\n"
        f"count=$(cat {sbatch_counter} 2>/dev/null || printf '0')\n"
        "count=$((count + 1))\n"
        f"printf '%s\\n' \"$count\" > {sbatch_counter}\n"
        "printf '%s\\n' \"$((20000 + count))\"\n",
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "TI_COHORT_RELEASE_PLAN": str(plan),
        "TI_COHORT_RELEASE_STEP": "0",
        "TI_COHORT_RELEASE_SCRIPT": str(release_script),
        "TI_COHORT_ARRAY_SCRIPT": str(array_script),
        "TI_COHORT_WORKFLOW_PY": str(workflow_script),
        "TI_COHORT_LOG_DIR": str(tmp_path / "logs"),
        "TI_COHORT_JOB_ID_FILE": str(job_ids),
        "TI_COHORT_RELEASE_STATE_DIR": str(tmp_path / "release_state"),
        "TI_COHORT_MESH_MANIFEST": str(mesh_manifest),
        "TI_COHORT_SIMULATION_MANIFEST": str(simulation_manifest),
        "TI_TARGETS_CSV": str(targets),
        "TI_EXPECTED_TARGETS_SHA256": sha256_file(targets),
        "TI_SIM_RUNNER_PY": str(
            camcan_dir / "simulation" / "TI_runner_multi-core.py"
        ),
        "TI_COMPLETION_CHECK_PY": str(
            camcan_dir / "simulation" / "validate_simulation_outputs.py"
        ),
        "SBATCH_BIN": str(fake_sbatch),
        "SCANCEL_BIN": "/bin/true",
        "TI_COHORT_RELEASE_RETRY_DELAY": "0",
        "SLURM_JOB_ID": "555",
    }

    completed = subprocess.run(
        ["bash", str(release_script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Submitted mesh chunk 0" in completed.stdout
    assert job_ids.read_text(encoding="utf-8").splitlines() == [
        "100",
        "20001",
        "20002",
    ]
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--array=0-1%50" in calls[0]
    assert "TI_COHORT_STAGE=mesh" in calls[0]
    assert "ELEMENT_OFFSET=0" in calls[0]
    assert "--dependency=afterok:20001" in calls[1]
    assert "TI_COHORT_RELEASE_STEP=1" in calls[1]
    assert (
        tmp_path / "release_state" / "release_step_0.tsv"
    ).is_file()

    environment["TI_COHORT_RELEASE_STEP"] = "1"
    finalizer = subprocess.run(
        ["bash", str(release_script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "All planned scaffold, mesh, and FEM arrays completed" in finalizer.stdout
    assert (
        tmp_path / "release_state" / "chain_complete.tsv"
    ).is_file()


def test_legacy_scaffold_is_imported_without_segmentation_or_mesh(tmp_path):
    fixture = _fixture(tmp_path)
    subject = fixture["subjects"][0]
    legacy_anat = tmp_path / "legacy" / subject / "anat"
    legacy_m2m = legacy_anat / f"m2m_{subject}"
    cap = legacy_m2m / "eeg_positions" / workflow.CAP_BASENAME
    _write_cap(cap)
    _write(legacy_m2m / "label_prep" / workflow.MAP_BASENAME, "old-label")
    _write(legacy_m2m / f"{subject}.msh", "old-mesh")
    marker = {
        "status": "complete",
        "subject": subject,
        "eeg_cap_sha256": sha256_file(cap),
    }
    marker_path = _write(
        tmp_path / "legacy" / "results" / f"{subject}.json",
        json.dumps(marker),
    )
    legacy_manifest = _write(
        tmp_path / "legacy.tsv",
        "task_id\tsubject\tanat_dir\tm2m_dir\tresult_path\n"
        f"0\t{subject}\t{legacy_anat}\t{legacy_m2m}\t{marker_path}\n",
    )

    payload = _preflight(
        fixture,
        legacy_scaffold_manifest=legacy_manifest,
    )
    row = workflow.approved.read_tsv(
        fixture["campaign"] / "scaffold_tasks.tsv"
    )[0]
    commands = []

    def command_runner(command, *, cwd, env=None):
        commands.append(command)

    result = workflow.run_scaffold_task(
        manifest=fixture["campaign"] / "scaffold_tasks.tsv",
        task_index=0,
        command_runner=command_runner,
    )

    assert payload["scaffold_import"] == 1
    assert payload["full_charm_segmentations_expected"] == 0
    assert commands == []
    assert result["scaffold_mode"] == "import"
    assert result["segmentation_runs_for_scaffold"] == 0
    assert result["temporary_mesh_runs_for_eeg_cap"] == 0
    target_m2m = Path(row["scaffold_m2m_dir"])
    assert not (target_m2m / f"{subject}.msh").exists()
    assert (
        target_m2m / "label_prep" / workflow.MAP_BASENAME
    ).read_text(encoding="utf-8") == f"{subject}-corrected-v4"


def test_packed_mesh_task_physically_copies_scaffold_and_uses_direct_mesher(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    payload = _preflight(fixture)
    scaffold_manifest = fixture["campaign"] / "scaffold_tasks.tsv"
    scaffold_row = workflow.approved.read_tsv(scaffold_manifest)[0]
    scaffold_commands = []

    scaffold_result = workflow.run_scaffold_task(
        manifest=scaffold_manifest,
        task_index=0,
        command_runner=_bootstrap_command_runner(
            scaffold_row,
            scaffold_commands,
        ),
    )
    scaffold_m2m = Path(scaffold_row["scaffold_m2m_dir"])
    assert "--segment" in scaffold_commands[0]
    assert scaffold_commands[1] == [
        "charm",
        scaffold_row["subject"],
        "--mesh",
    ]
    assert scaffold_result["completed_segmentation_recovered"] is False
    assert scaffold_result["segmentation_runs_for_scaffold"] == 1
    assert scaffold_result["temporary_mesh_runs_for_eeg_cap"] == 1
    assert scaffold_result["temporary_mesh_removed"] is True
    assert not (scaffold_m2m / f"{scaffold_row['subject']}.msh").exists()
    captured = {}

    def fake_direct_mesh(**kwargs):
        captured.update(kwargs)
        mesh_path = _write(Path(kwargs["mesh_path"]), "independent-mesh")
        result = {
            "status": "complete",
            "subject": kwargs["subject"],
            "mesh_path": str(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "mesh_sha256": sha256_file(mesh_path),
            "label_sha256_after": kwargs["expected_label_hash"],
            **kwargs["provenance"],
        }
        workflow.approved.write_json_atomic(Path(kwargs["result_path"]), result)
        return result

    monkeypatch.setattr(
        workflow.direct_mesh, "create_mesh_from_label", fake_direct_mesh
    )
    result = workflow.run_mesh_task(
        manifest=fixture["campaign"] / "mesh_tasks.tsv",
        task_index=0,
        staging_root=tmp_path / "local-stage",
    )
    row = workflow.approved.read_tsv(
        fixture["campaign"] / "mesh_tasks.tsv"
    )[0]

    assert payload["mesh_workers_per_array_element"] == 2
    assert result["independent_repeat_mesh"] is True
    assert result["scaffold_copy_mode"] == "physical"
    assert Path(row["m2m_dir"]).is_dir()
    assert not Path(row["m2m_dir"]).is_symlink()
    assert captured["staging_root"] == tmp_path / "local-stage"
    assert "--mesh" not in json.dumps(result)


def test_bootstrap_retry_recovers_completed_segmentation_and_only_builds_cap(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    _preflight(fixture)
    manifest = fixture["campaign"] / "scaffold_tasks.tsv"
    row = workflow.approved.read_tsv(manifest)[0]
    anat = Path(row["scaffold_anat_dir"])
    m2m = Path(row["scaffold_m2m_dir"])
    task_t1 = anat / Path(row["source_t1"]).name
    task_t2 = anat / Path(row["source_t2"]).name
    workflow.approved._copy_verified(
        Path(row["source_t1"]),
        task_t1,
        row["source_t1_sha256"],
    )
    workflow.approved._copy_verified(
        Path(row["source_t2"]),
        task_t2,
        row["source_t2_sha256"],
    )
    workflow.approved._copy_verified(
        Path(row["corrected_label"]),
        m2m / "label_prep" / workflow.MAP_BASENAME,
        row["corrected_label_sha256"],
    )
    commands = []

    result = workflow.run_scaffold_task(
        manifest=manifest,
        task_index=0,
        command_runner=_bootstrap_command_runner(row, commands),
    )

    assert commands == [["charm", row["subject"], "--mesh"]]
    assert result["completed_segmentation_recovered"] is True
    assert result["segmentation_runs_for_scaffold"] == 0
    assert result["temporary_mesh_runs_for_eeg_cap"] == 1
    assert result["temporary_mesh_removed"] is True
    assert not (m2m / f"{row['subject']}.msh").exists()
    assert workflow.scaffold_result_is_current(row) == result


def test_simulation_uses_existing_mesh_and_reruns_if_outputs_are_missing(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    _preflight(fixture)
    scaffold_manifest = fixture["campaign"] / "scaffold_tasks.tsv"
    scaffold_row = workflow.approved.read_tsv(scaffold_manifest)[0]

    workflow.run_scaffold_task(
        manifest=scaffold_manifest,
        task_index=0,
        command_runner=_bootstrap_command_runner(scaffold_row),
    )

    def fake_direct_mesh(**kwargs):
        mesh_path = _write(Path(kwargs["mesh_path"]), "independent-mesh")
        result = {
            "status": "complete",
            "subject": kwargs["subject"],
            "mesh_path": str(mesh_path),
            "mesh_bytes": mesh_path.stat().st_size,
            "mesh_sha256": sha256_file(mesh_path),
            "label_sha256_after": kwargs["expected_label_hash"],
            **kwargs["provenance"],
        }
        workflow.approved.write_json_atomic(Path(kwargs["result_path"]), result)
        return result

    monkeypatch.setattr(
        workflow.direct_mesh, "create_mesh_from_label", fake_direct_mesh
    )
    manifest = fixture["campaign"] / "simulation_tasks.tsv"
    workflow.run_mesh_task(manifest=manifest, task_index=0)
    runner = _write(tmp_path / "runner.py", "")
    validator = _write(tmp_path / "validator.py", "")
    commands = []

    def capture_command(command, *, cwd, env=None):
        commands.append((command, cwd, env))

    result = workflow.run_simulation_task(
        manifest=manifest,
        task_index=0,
        targets_csv=fixture["targets"],
        expected_targets_sha256=sha256_file(fixture["targets"]),
        simulation_runner=runner,
        simulation_validator=validator,
        command_runner=capture_command,
    )

    assert result["segmentation_in_simulation_task"] is False
    assert result["mesh_created_in_simulation_task"] is False
    assert commands[0][0].count("--montage-preset") == 1
    assert "--reuse-existing-mesh" in commands[0][0]
    assert commands[1][0] == [
        "python",
        "-u",
        str(validator),
        "--root",
        workflow.approved.read_tsv(manifest)[0]["dataset_root"],
        "--subject",
        fixture["subjects"][0],
    ]
    assert commands[0][2]["TI_SIM_ROOT"] == commands[1][2]["TI_SIM_ROOT"]

    workflow.run_simulation_task(
        manifest=manifest,
        task_index=0,
        targets_csv=fixture["targets"],
        expected_targets_sha256=sha256_file(fixture["targets"]),
        simulation_runner=runner,
        simulation_validator=validator,
        command_runner=capture_command,
    )
    assert len(commands) == 4


def test_submitter_writes_full_plan_and_releases_only_first_stage(tmp_path):
    fixture = _fixture(tmp_path)
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_counter = tmp_path / "sbatch.counter"
    fake_sbatch = _write(
        tmp_path / "sbatch",
        "#!/bin/bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "count=0\n"
        "test -f \"$FAKE_SBATCH_COUNTER\" && count=$(cat \"$FAKE_SBATCH_COUNTER\")\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$FAKE_SBATCH_COUNTER\"\n"
        "echo $((21000 + count))\n",
    )
    fake_scontrol = _write(
        tmp_path / "scontrol",
        "#!/bin/bash\n"
        "echo 'MaxArraySize = 1001'\n",
    )
    fake_scancel = _write(tmp_path / "scancel", "#!/bin/bash\nexit 0\n")
    for executable in (fake_sbatch, fake_scontrol, fake_scancel):
        executable.chmod(0o755)
    submitter = (
        Path(__file__).resolve().parents[1]
        / "cohort_pipeline"
        / "submit_cohort_pipeline.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "STUDY_CONFIG": str(fixture["study_config"]),
            "COHORT_CONFIG": str(fixture["cohort_config"]),
            "STUDY_ROOT": str(fixture["study_root"]),
            "SCAFFOLD_ROOT": str(fixture["scaffold_root"]),
            "MAP_MANIFEST": str(fixture["collection"]),
            "SOURCE_ROOT_1": str(fixture["source_a"]),
            "SOURCE_ROOT_2": str(fixture["source_b"]),
            "LEGACY_SCAFFOLD_MANIFEST": str(tmp_path / "missing.tsv"),
            "MAX_ARRAY_ELEMENTS": "7",
            "MAX_CONCURRENT_TASKS": "5",
            "SBATCH_BIN": str(fake_sbatch),
            "SCONTROL_BIN": str(fake_scontrol),
            "SCANCEL_BIN": str(fake_scancel),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "FAKE_SBATCH_COUNTER": str(sbatch_counter),
        }
    )

    completed = subprocess.run(
        ["bash", str(submitter), "test"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "subjects: 1" in completed.stdout
    assert "independent mesh tasks: 40" in completed.stdout
    assert "packed mesh array elements: 20 in 3 sequential chunk(s)" in completed.stdout
    assert "FEM tasks: 40 in 6 sequential chunk(s)" in completed.stdout
    assert "automatic release steps: 9" in completed.stdout
    assert "execution: full requested cohort; not a smoke or subset" in completed.stdout
    submissions = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(submissions) == 2
    assert "--array=0-0%5" in submissions[0]
    assert "--dependency=afterok:21001" in submissions[1]
    assert "TI_COHORT_RELEASE_STEP=0" in submissions[1]
    assert "--array=" not in submissions[1]

    release_plan = (
        fixture["campaign"] / "release_plan.tsv"
    ).read_text(encoding="utf-8").splitlines()
    assert release_plan == [
        "step\tstage\tchunk_index\toffset\tcount",
        "0\tmesh\t0\t0\t7",
        "1\tmesh\t1\t7\t7",
        "2\tmesh\t2\t14\t6",
        "3\tsimulate\t0\t0\t7",
        "4\tsimulate\t1\t7\t7",
        "5\tsimulate\t2\t14\t7",
        "6\tsimulate\t3\t21\t7",
        "7\tsimulate\t4\t28\t7",
        "8\tsimulate\t5\t35\t5",
    ]


def test_progress_report_is_bounded_and_summarizes_campaign(tmp_path):
    fixture = _fixture(tmp_path)
    _preflight(fixture, max_array_elements=20, mesh_workers=2)
    subject = fixture["subjects"][0]
    scaffold_marker = (
        fixture["scaffold_root"] / "results" / f"{subject}.json"
    )
    mesh_marker = (
        fixture["study_root"]
        / "results"
        / "meshes"
        / "Left_Hippocampus"
        / "01"
        / f"{subject}.json"
    )
    simulation_marker = (
        fixture["study_root"]
        / "results"
        / "simulations"
        / "Left_Hippocampus"
        / "01"
        / f"{subject}.json"
    )
    for marker in (scaffold_marker, mesh_marker, simulation_marker):
        _write(marker, '{"status": "complete"}\n')

    campaign = fixture["campaign"]
    _write(
        campaign / "release_plan.tsv",
        "step\tstage\tchunk_index\toffset\tcount\n"
        "0\tmesh\t0\t0\t20\n"
        "1\tsimulate\t0\t0\t20\n",
    )
    _write(campaign / "submitted_job_ids.txt", "100\n101\n")
    _write(
        campaign / "release_state" / "release_step_0.tsv",
        "step\tstage\n0\tmesh\n",
    )
    _write(campaign / "logs" / "retry_state" / "mesh_100_0.retry", "1\n")
    _write(
        campaign / "logs" / "release-101.out",
        "[INFO] Submitted mesh chunk 0: job=100\n",
    )

    fake_squeue = _write(
        tmp_path / "squeue",
        "#!/bin/bash\n"
        "echo '100_0|cohort_test_mesh0|RUNNING|00:05|node001|2026-01-01T00:00:00'\n"
        "echo '101|cohort_test_release1|PENDING|0:00|Dependency|N/A'\n",
    )
    fake_sacct = _write(
        tmp_path / "sacct",
        "#!/bin/bash\n"
        "echo '100_0|cohort_test_mesh0|COMPLETED|0:0|300'\n"
        "echo '101|cohort_test_release1|PENDING|0:0|0'\n",
    )
    for executable in (fake_squeue, fake_sacct):
        executable.chmod(0o755)

    progress_script = (
        Path(__file__).resolve().parents[1]
        / "cohort_pipeline"
        / "check_cohort_progress.bash"
    )
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "STUDY_ROOT": str(fixture["study_root"]),
        "SCAFFOLD_ROOT": str(fixture["scaffold_root"]),
        "CAMPAIGN_ROOT": str(campaign),
        "TI_COHORT_PROGRESS_COMMAND_TIMEOUT_SECONDS": "2",
        "TI_COHORT_PROGRESS_SCAN_TIMEOUT_SECONDS": "2",
    }

    completed = subprocess.run(
        ["bash", str(progress_script), "test"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )

    assert "CAMCAN_COHORT_PROGRESS_V1" in completed.stdout
    assert "scaffolds_expected=1" in completed.stdout
    assert "meshes_expected=40" in completed.stdout
    assert "simulations_expected=40" in completed.stdout
    assert "scaffolds_complete=1 scan=complete" in completed.stdout
    assert "meshes_complete=1 scan=complete" in completed.stdout
    assert "simulations_complete=1 scan=complete" in completed.stdout
    assert "release_steps_recorded=1 scan=complete" in completed.stdout
    assert "active_retry_files=1 scan=complete" in completed.stdout
    assert "terminal_problem_records=0" in completed.stdout
    assert "report_read_only=true" in completed.stdout
    assert completed.returncode == 0
