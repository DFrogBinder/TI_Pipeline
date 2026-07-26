import json
import os
import subprocess
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "cohort_pipeline"


def _write_base_inputs(tmp_path: Path, subjects: list[str]) -> dict[str, Path]:
    study_root = tmp_path / "study"
    study_config = tmp_path / "study.json"
    study_config.write_text(
        json.dumps({"hpc_study_root": str(study_root)}),
        encoding="utf-8",
    )
    cohort_config = tmp_path / "cohort.json"
    cohort_config.write_text(
        json.dumps({"expected_subjects": len(subjects)}),
        encoding="utf-8",
    )
    subjects_file = tmp_path / "subjects.txt"
    subjects_file.write_text("\n".join(subjects) + "\n", encoding="utf-8")
    scaffold_root = tmp_path / "scaffolds"
    for subject in subjects:
        t1 = (
            scaffold_root
            / "subjects"
            / subject
            / "anat"
            / f"{subject}_T1w.nii"
        )
        t1.parent.mkdir(parents=True)
        t1.write_bytes(b"t1")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")
    return {
        "study_root": study_root,
        "study_config": study_config,
        "cohort_config": cohort_config,
        "subjects_file": subjects_file,
        "scaffold_root": scaffold_root,
        "license_path": license_path,
    }


def test_atlas_repair_preflight_keeps_flat_imports_nested_and_reconstructs(
    tmp_path: Path,
) -> None:
    subjects = ["sub-CC000001", "sub-CC000002", "sub-CC000003"]
    inputs = _write_base_inputs(tmp_path, subjects)
    flat_root = tmp_path / "flat"
    flat_root.mkdir()
    (flat_root / f"{subjects[0]}.nii.gz").write_bytes(b"existing")
    source_root = tmp_path / "nested"
    source_atlas = (
        source_root
        / subjects[1]
        / "mri"
        / "aparc.a2009s+aseg.nii.gz"
    )
    source_atlas.parent.mkdir(parents=True)
    source_atlas.write_bytes(b"nested")
    atlas_campaign = tmp_path / "atlas_campaign"
    atlas_work = tmp_path / "atlas_work"

    completed = subprocess.run(
        [
            "bash",
            str(PIPELINE_DIR / "submit_cohort_atlas_repair.sh"),
            "test",
            "--preflight",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "STUDY_CONFIG": str(inputs["study_config"]),
            "COHORT_CONFIG": str(inputs["cohort_config"]),
            "STUDY_ROOT": str(inputs["study_root"]),
            "ATLAS_CAMPAIGN_ROOT": str(atlas_campaign),
            "SUBJECTS_FILE": str(inputs["subjects_file"]),
            "SCAFFOLD_ROOT": str(inputs["scaffold_root"]),
            "FLAT_ATLAS_ROOT": str(flat_root),
            "ATLAS_WORK_ROOT": str(atlas_work),
            "ATLAS_SOURCE_ROOTS": str(source_root),
            "FS_LICENSE_FILE": str(inputs["license_path"]),
        },
    )

    payload = json.loads((atlas_campaign / "preflight.json").read_text())
    rows = (atlas_campaign / "atlas_tasks.tsv").read_text().splitlines()
    assert payload["status"] == "ready"
    assert payload["subjects"] == 3
    assert payload["flat_atlases_present"] == 1
    assert payload["tasks"] == 2
    assert payload["import_nifti"] == 1
    assert payload["reconstruct"] == 1
    assert len(rows) == 3
    assert f"\t{subjects[1]}\timport_nifti\t" in rows[1]
    assert f"\t{subjects[2]}\treconstruct\t" in rows[2]
    assert "Preflight passed without submitting jobs." in completed.stdout


def test_atlas_repair_submitter_uses_array_and_afterany_collector(
    tmp_path: Path,
) -> None:
    subjects = ["sub-CC000001", "sub-CC000002"]
    inputs = _write_base_inputs(tmp_path, subjects)
    flat_root = tmp_path / "flat"
    atlas_campaign = tmp_path / "atlas_campaign"
    atlas_work = tmp_path / "atlas_work"
    calls = tmp_path / "sbatch_calls.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$*\" >> {calls!s}\n"
        f"count=$(wc -l < {calls!s})\n"
        "if [ \"$count\" -eq 1 ]; then printf '12345\\n'; else printf '12346\\n'; fi\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(PIPELINE_DIR / "submit_cohort_atlas_repair.sh"),
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "STUDY_CONFIG": str(inputs["study_config"]),
            "COHORT_CONFIG": str(inputs["cohort_config"]),
            "STUDY_ROOT": str(inputs["study_root"]),
            "ATLAS_CAMPAIGN_ROOT": str(atlas_campaign),
            "SUBJECTS_FILE": str(inputs["subjects_file"]),
            "SCAFFOLD_ROOT": str(inputs["scaffold_root"]),
            "FLAT_ATLAS_ROOT": str(flat_root),
            "ATLAS_WORK_ROOT": str(atlas_work),
            "FS_LICENSE_FILE": str(inputs["license_path"]),
            "SBATCH_BIN": str(fake_sbatch),
            "SQUEUE_BIN": "/bin/true",
        },
    )

    submitted = calls.read_text(encoding="utf-8")
    assert "--array=0-1%50" in submitted
    assert "--dependency=afterany:12345" in submitted
    assert "Submitted missing-atlas array job: 12345" in completed.stdout
    assert "Submitted afterany atlas collector: 12346" in completed.stdout


def test_atlas_repair_slurm_is_resumable_and_destrieux_specific() -> None:
    text = (PIPELINE_DIR / "cohort_atlas_repair.slurm").read_text(
        encoding="utf-8"
    )
    assert "aparc.a2009s+aseg.mgz" in text
    assert "Resuming existing FreeSurfer subject state." in text
    assert 'scontrol requeue "${SLURM_JOB_ID}"' in text
    assert 'mri_info "${FLAT_ATLAS}"' in text


def test_atlas_repair_slurm_imports_nested_nifti_and_writes_receipt(
    tmp_path: Path,
) -> None:
    subject = "sub-CC000001"
    source = tmp_path / "source" / subject / "mri" / "aparc.a2009s+aseg.nii.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"destrieux")
    flat = tmp_path / "flat" / f"{subject}.nii.gz"
    fs_subject = tmp_path / "work" / "subjects" / subject
    result = tmp_path / "results" / f"{subject}.tsv"
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "task_id\tsubject\tmode\tt1_path\tsource_atlas\tflat_atlas\t"
        "freesurfer_subject_dir\tresult_path\n"
        f"0\t{subject}\timport_nifti\t-\t{source}\t{flat}\t"
        f"{fs_subject}\t{result}\n",
        encoding="utf-8",
    )
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("module", "recon-all", "mri_convert", "mri_info"):
        executable = bin_dir / command
        executable.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(PIPELINE_DIR / "cohort_atlas_repair.slurm")],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_JOB_ID": "12347",
            "SLURM_CPUS_PER_TASK": "16",
            "TI_COHORT_ATLAS_MANIFEST": str(manifest),
            "TI_COHORT_ATLAS_LOG_DIR": str(tmp_path / "logs"),
            "TI_COHORT_ATLAS_STATE_DIR": str(tmp_path / "retry"),
            "TI_COHORT_ATLAS_FS_LICENSE": str(license_path),
            "TI_COHORT_ATLAS_FREESURFER_MODULE": "FreeSurfer/test",
        },
    )

    assert flat.read_bytes() == b"destrieux"
    assert result.read_text(encoding="utf-8").splitlines()[1].startswith(
        f"complete\t{subject}\timport_nifti\t"
    )
    assert f"Atlas repair complete: {subject}" in completed.stdout
