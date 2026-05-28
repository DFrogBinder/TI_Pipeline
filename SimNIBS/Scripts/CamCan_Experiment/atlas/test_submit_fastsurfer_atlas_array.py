import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("submit_fastsurfer_atlas_array.sh")


def make_subject(data_root: Path, subject: str, *, with_t1: bool = True) -> None:
    anat_dir = data_root / subject / "anat"
    anat_dir.mkdir(parents=True)
    if with_t1:
        (anat_dir / f"{subject}_T1w.nii.gz").write_text("fake nifti")


def run_submitter(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_dry_run_discovers_subjects_and_writes_array_job(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "fastsurfer_out"
    license_path = tmp_path / "license.txt"
    license_path.write_text("license")
    make_subject(data_root, "sub-B")
    make_subject(data_root, "sub-A")
    make_subject(data_root, "sub-missing", with_t1=False)

    result = run_submitter(
        str(data_root),
        str(output_root),
        "--dry-run",
        "--partition",
        "night",
        "--cpus",
        "20",
        "--mem",
        "96G",
        "--max-concurrent",
        "25",
        "--fastsurfer-module",
        "FastSurfer/2.3.0",
        "--freesurfer-module",
        "FreeSurfer/7.4.1",
        "--license",
        str(license_path),
    )

    assert result.returncode == 0, result.stderr
    assert "[DRY-RUN]" in result.stdout

    subjects_file = output_root / "slurm" / "subjects.txt"
    assert subjects_file.read_text().splitlines() == ["sub-A", "sub-B"]

    batch_script = output_root / "slurm" / "fastsurfer_atlas_subject.slurm"
    batch_text = batch_script.read_text()
    assert "#SBATCH --partition=night" in batch_text
    assert "#SBATCH --time=08:00:00" in batch_text
    assert "#SBATCH --cpus-per-task=20" in batch_text
    assert "#SBATCH --mem=96G" in batch_text
    assert 'module load "${FASTSURFER_MODULE}"' in batch_text
    assert 'module load "${FREESURFER_MODULE}"' in batch_text
    assert "run_fastsurfer.sh" in batch_text
    assert "mri_convert" in batch_text
    assert 'ATLAS_FLAT="${OUTPUT_ROOT}/${SUBJECT}.nii.gz"' in batch_text

    submit_command = (output_root / "slurm" / "submit_command.txt").read_text()
    assert "--array=0-1%25" in submit_command
    assert "FASTSURFER_MODULE=FastSurfer/2.3.0" in submit_command
    assert "FREESURFER_MODULE=FreeSurfer/7.4.1" in submit_command


def test_no_valid_subjects_fails_before_submission(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "fastsurfer_out"
    make_subject(data_root, "sub-missing", with_t1=False)

    result = run_submitter(str(data_root), str(output_root), "--dry-run")

    assert result.returncode == 1
    assert "No subjects with T1 input found" in result.stderr


def test_dry_run_does_not_clear_site_license_when_license_is_omitted(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "fastsurfer_out"
    make_subject(data_root, "sub-A")

    result = run_submitter(str(data_root), str(output_root), "--dry-run")

    assert result.returncode == 0, result.stderr
    submit_command = (output_root / "slurm" / "submit_command.txt").read_text()
    assert "FS_LICENSE=" not in submit_command
