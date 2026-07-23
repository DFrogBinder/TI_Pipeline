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

    def fake_charm(command, *, cwd, env=None):
        m2m = Path(scaffold_row["scaffold_m2m_dir"])
        _write(m2m / "label_prep" / workflow.MAP_BASENAME, "generated")
        _write_cap(m2m / "eeg_positions" / workflow.CAP_BASENAME)

    workflow.run_scaffold_task(
        manifest=scaffold_manifest,
        task_index=0,
        command_runner=fake_charm,
    )
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


def test_simulation_uses_existing_mesh_and_reruns_if_outputs_are_missing(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    _preflight(fixture)
    scaffold_manifest = fixture["campaign"] / "scaffold_tasks.tsv"
    scaffold_row = workflow.approved.read_tsv(scaffold_manifest)[0]

    def fake_charm(command, *, cwd, env=None):
        m2m = Path(scaffold_row["scaffold_m2m_dir"])
        _write(m2m / "label_prep" / workflow.MAP_BASENAME, "generated")
        _write_cap(m2m / "eeg_positions" / workflow.CAP_BASENAME)

    workflow.run_scaffold_task(
        manifest=scaffold_manifest,
        task_index=0,
        command_runner=fake_charm,
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


def test_submitter_chunks_full_scope_and_chains_every_array(tmp_path):
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
    assert "execution: full requested cohort; not a smoke or subset" in completed.stdout
    submissions = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(submissions) == 10
    assert "--array=0-0%5" in submissions[0]
    assert "--array=0-6%5" in submissions[1]
    assert "ELEMENT_OFFSET=0" in submissions[1]
    assert "ELEMENT_OFFSET=7" in submissions[2]
    assert "ELEMENT_OFFSET=14" in submissions[3]
    assert "TASK_OFFSET=0" in submissions[4]
    assert "TASK_OFFSET=35" in submissions[-1]
    for index, submission in enumerate(submissions[1:], start=1):
        assert f"--dependency=afterok:{21000 + index}" in submission
