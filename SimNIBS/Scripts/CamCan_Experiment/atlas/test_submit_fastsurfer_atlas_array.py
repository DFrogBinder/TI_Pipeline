import os
import shutil
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("submit_fastsurfer_atlas_array.sh")


def make_subject(data_root: Path, subject: str, *, with_t1: bool = True) -> None:
    anat_dir = data_root / subject / "anat"
    anat_dir.mkdir(parents=True)
    if with_t1:
        (anat_dir / f"{subject}_T1w.nii.gz").write_text("fake nifti")


def configured_script(
    tmp_path: Path,
    *,
    data_root: Path,
    output_root: Path,
    license_path: Path | None = None,
    array_spec: str = "0-999%64",
) -> Path:
    script_copy = tmp_path / "submit_fastsurfer_atlas_array.sh"
    shutil.copy(SCRIPT, script_copy)
    text = script_copy.read_text()
    replacements = {
        'DATA_ROOT="/path/to/CamCan_Data"': f'DATA_ROOT="{data_root}"',
        'OUTPUT_ROOT="/path/to/FastSurfer_atlases"': f'OUTPUT_ROOT="{output_root}"',
        'FASTSURFER_MODULE="FastSurfer"': 'FASTSURFER_MODULE="FastSurfer/2.3.0"',
        'FREESURFER_MODULE="FreeSurfer"': 'FREESURFER_MODULE="FreeSurfer/7.4.1"',
        'FS_LICENSE_FILE=""': f'FS_LICENSE_FILE="{license_path or ""}"',
        'DRY_RUN="0"': 'DRY_RUN="1"',
        "#SBATCH --array=0-999%64": f"#SBATCH --array={array_spec}",
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    script_copy.write_text(text)
    script_copy.chmod(0o755)
    return script_copy


def run_script(script: Path, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", str(script)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=merged_env,
    )


def test_local_dry_run_uses_top_of_file_config_without_args(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "fastsurfer_out"
    license_path = tmp_path / "license.txt"
    license_path.write_text("license")
    make_subject(data_root, "sub-B")
    make_subject(data_root, "sub-A")
    make_subject(data_root, "sub-missing", with_t1=False)
    script = configured_script(
        tmp_path,
        data_root=data_root,
        output_root=output_root,
        license_path=license_path,
    )

    result = run_script(script)

    assert result.returncode == 0, result.stderr
    assert "[DRY-RUN]" in result.stdout
    assert "Subject count:   2" in result.stdout
    assert "Submit with:     sbatch" in result.stdout
    assert str(data_root) in result.stdout
    assert str(output_root) in result.stdout

    subjects_file = output_root / "slurm" / "subjects.txt"
    assert subjects_file.read_text().splitlines() == ["sub-A", "sub-B"]


def test_array_task_selects_subject_and_skips_external_commands_in_dry_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "fastsurfer_out"
    make_subject(data_root, "sub-A")
    make_subject(data_root, "sub-B")
    script = configured_script(tmp_path, data_root=data_root, output_root=output_root)

    result = run_script(
        script,
        env={
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_CPUS_PER_TASK": "20",
            "SLURM_JOB_ID": "123",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Subject:     sub-B" in result.stdout
    assert "[DRY-RUN] Would run FastSurfer command:" in result.stdout
    assert f"--t1 {data_root}/sub-B/anat/sub-B_T1w.nii.gz" in result.stdout
    assert f"--sd {output_root}/subjects" in result.stdout
    assert f"mri_convert {output_root}/subjects/sub-B/mri/aparc.DKTatlas+aseg.deep.mgz {output_root}/sub-B.nii.gz" in result.stdout


def test_local_validation_warns_when_configured_array_is_too_short(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "fastsurfer_out"
    make_subject(data_root, "sub-A")
    make_subject(data_root, "sub-B")
    script = configured_script(
        tmp_path,
        data_root=data_root,
        output_root=output_root,
        array_spec="0-0%64",
    )

    result = run_script(script)

    assert result.returncode == 0, result.stderr
    assert "[WARN] Current #SBATCH --array=0-0%64 only covers through index 0; edit it to at least 0-1." in result.stdout


def test_script_has_no_argument_or_sbatch_export_configuration_path() -> None:
    text = SCRIPT.read_text()
    assert "--export" not in text
    assert "DATA_ARG" not in text
    assert "OUTPUT_ARG" not in text
    assert "Usage:" not in text
    assert "cat >" not in text
