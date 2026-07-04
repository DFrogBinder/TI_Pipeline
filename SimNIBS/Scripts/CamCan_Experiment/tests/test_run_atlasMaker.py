import os
import stat
import subprocess
import sys
from pathlib import Path

from atlas.run_atlasMaker import already_processed, input_available, t1_input_path
import atlas.run_atlasMaker as runner


def test_already_processed_requires_destrieux_nifti(tmp_path: Path):
    output_dir = tmp_path / "FastSurfer_out"
    mri_dir = output_dir / "sub-01" / "mri"
    mri_dir.mkdir(parents=True)
    (mri_dir / "aparc.DKTatlas+aseg.deep.nii.gz").write_text("dkt", encoding="utf-8")

    assert already_processed(output_dir, "sub-01") is False

    (mri_dir / "aparc.a2009s+aseg.nii.gz").write_text("destrieux", encoding="utf-8")

    assert already_processed(output_dir, "sub-01") is True


def test_t1_input_path_accepts_uncompressed_nifti(tmp_path: Path):
    subject = "sub-01"
    anat_dir = tmp_path / subject / "anat"
    anat_dir.mkdir(parents=True)
    t1_path = anat_dir / f"{subject}_T1w.nii"
    t1_path.write_text("t1", encoding="utf-8")

    assert t1_input_path(tmp_path, subject) == t1_path
    assert input_available(tmp_path, subject) is True


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_atlasmaker_avoids_new_union_annotation_syntax_for_hpc_python():
    source = (Path(__file__).resolve().parents[1] / "atlas" / "run_atlasMaker.py").read_text(
        encoding="utf-8"
    )

    assert "| None" not in source
    assert "None |" not in source


def test_main_reports_subject_failure_without_python_traceback(monkeypatch, tmp_path: Path, capsys):
    subject = "sub-01"
    data_dir = tmp_path / "CamCan_Data"
    anat_dir = data_dir / subject / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_T1w.nii").write_text("t1", encoding="utf-8")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")

    def fail_process_subject(*_args, **_kwargs):
        raise subprocess.CalledProcessError(7, ["make_atlas.sh", subject])

    monkeypatch.setattr(runner, "process_subject", fail_process_subject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_atlasMaker.py",
            "--data-dir",
            str(data_dir),
            "--license-path",
            str(license_path),
            "--subjects",
            subject,
            "--max-parallel-jobs",
            "1",
        ],
    )

    result = runner.main()
    captured = capsys.readouterr()

    assert result == 1
    assert "[error] sub-01 failed with exit code 7" in captured.out
    assert "Traceback" not in captured.err
    assert "FastSurfer_out/logs/make_atlas_*.log" in captured.out


def test_make_atlas_native_backend_uses_freesurfer_tools_without_docker(tmp_path: Path):
    data_dir = tmp_path / "CamCan_Data"
    subject = "sub-01"
    anat_dir = data_dir / subject / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_T1w.nii").write_text("t1", encoding="utf-8")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recon_log = tmp_path / "recon-all.log"
    convert_log = tmp_path / "mri_convert.log"

    write_executable(
        bin_dir / "recon-all",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{recon_log}"
sid=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -s) sid="$2"; shift 2 ;;
    *) shift ;;
  esac
done
subject_dir="${{SUBJECTS_DIR}}/${{sid}}"
mkdir -p "${{subject_dir}}/mri" "${{subject_dir}}/surf" "${{subject_dir}}/scripts"
printf 't1' > "${{subject_dir}}/mri/T1.mgz"
printf 'destrieux' > "${{subject_dir}}/mri/aparc.a2009s+aseg.mgz"
touch "${{subject_dir}}/surf/lh.white" "${{subject_dir}}/surf/rh.white"
touch "${{subject_dir}}/scripts/recon-all.done"
""",
    )
    write_executable(
        bin_dir / "mri_convert",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s -> %s\\n' "$1" "$2" >> "{convert_log}"
cp "$1" "$2"
""",
    )

    script = Path(__file__).resolve().parents[1] / "atlas" / "make_atlas.sh"
    env = os.environ.copy()
    env["ATLAS_BACKEND"] = "native"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(script), str(data_dir), "2", str(license_path), subject],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (data_dir / "FastSurfer_out" / subject / "mri" / "aparc.a2009s+aseg.nii.gz").is_file()
    assert "-s sub-01" in recon_log.read_text(encoding="utf-8")
    assert f"{subject}_T1w.nii" in recon_log.read_text(encoding="utf-8")
    assert "aparc.a2009s+aseg.mgz" in convert_log.read_text(encoding="utf-8")


def test_make_atlas_native_backend_does_not_precreate_subject_before_input_recon(tmp_path: Path):
    data_dir = tmp_path / "CamCan_Data"
    subject = "sub-01"
    anat_dir = data_dir / subject / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_T1w.nii.gz").write_text("t1", encoding="utf-8")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recon_log = tmp_path / "recon-all.log"

    write_executable(
        bin_dir / "recon-all",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{recon_log}"
sid=""
has_input=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -s) sid="$2"; shift 2 ;;
    -i) has_input=1; shift 2 ;;
    *) shift ;;
  esac
done
subject_dir="${{SUBJECTS_DIR}}/${{sid}}"
if [[ "${{has_input}}" == "1" && -d "${{subject_dir}}" ]]; then
  echo "recon-all refuses -i for an existing subject directory" >&2
  exit 64
fi
mkdir -p "${{subject_dir}}/mri" "${{subject_dir}}/surf" "${{subject_dir}}/scripts"
printf 't1' > "${{subject_dir}}/mri/T1.mgz"
printf 'destrieux' > "${{subject_dir}}/mri/aparc.a2009s+aseg.mgz"
touch "${{subject_dir}}/surf/lh.white" "${{subject_dir}}/surf/rh.white"
touch "${{subject_dir}}/scripts/recon-all.done"
""",
    )
    write_executable(
        bin_dir / "mri_convert",
        """#!/usr/bin/env bash
set -euo pipefail
cp "$1" "$2"
""",
    )

    script = Path(__file__).resolve().parents[1] / "atlas" / "make_atlas.sh"
    env = os.environ.copy()
    env["ATLAS_BACKEND"] = "native"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(script), str(data_dir), "2", str(license_path), subject],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (data_dir / "FastSurfer_out" / subject / "mri" / "aparc.a2009s+aseg.nii.gz").is_file()
    assert f"-s {subject}" in recon_log.read_text(encoding="utf-8")


def test_make_atlas_native_backend_removes_stale_empty_subject_before_input_recon(tmp_path: Path):
    data_dir = tmp_path / "CamCan_Data"
    subject = "sub-01"
    anat_dir = data_dir / subject / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_T1w.nii.gz").write_text("t1", encoding="utf-8")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")
    (data_dir / "FastSurfer_out" / subject).mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recon_log = tmp_path / "recon-all.log"

    write_executable(
        bin_dir / "recon-all",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{recon_log}"
sid=""
has_input=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -s) sid="$2"; shift 2 ;;
    -i) has_input=1; shift 2 ;;
    *) shift ;;
  esac
done
subject_dir="${{SUBJECTS_DIR}}/${{sid}}"
if [[ "${{has_input}}" == "1" && -d "${{subject_dir}}" ]]; then
  echo "recon-all refuses -i for an existing subject directory" >&2
  exit 64
fi
mkdir -p "${{subject_dir}}/mri" "${{subject_dir}}/surf" "${{subject_dir}}/scripts"
printf 't1' > "${{subject_dir}}/mri/T1.mgz"
printf 'destrieux' > "${{subject_dir}}/mri/aparc.a2009s+aseg.mgz"
touch "${{subject_dir}}/surf/lh.white" "${{subject_dir}}/surf/rh.white"
touch "${{subject_dir}}/scripts/recon-all.done"
""",
    )
    write_executable(
        bin_dir / "mri_convert",
        """#!/usr/bin/env bash
set -euo pipefail
cp "$1" "$2"
""",
    )

    script = Path(__file__).resolve().parents[1] / "atlas" / "make_atlas.sh"
    env = os.environ.copy()
    env["ATLAS_BACKEND"] = "native"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(script), str(data_dir), "2", str(license_path), subject],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (data_dir / "FastSurfer_out" / subject / "mri" / "aparc.a2009s+aseg.nii.gz").is_file()
    assert f"-s {subject}" in recon_log.read_text(encoding="utf-8")


def test_make_atlas_native_backend_converts_existing_destrieux_mgz_without_recon(tmp_path: Path):
    data_dir = tmp_path / "CamCan_Data"
    subject = "sub-01"
    anat_dir = data_dir / subject / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_T1w.nii.gz").write_text("t1", encoding="utf-8")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")

    mri_dir = data_dir / "FastSurfer_out" / subject / "mri"
    mri_dir.mkdir(parents=True)
    (mri_dir / "T1.mgz").write_text("t1 mgz", encoding="utf-8")
    (mri_dir / "aparc.a2009s+aseg.mgz").write_text("destrieux mgz", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    convert_log = tmp_path / "mri_convert.log"
    write_executable(
        bin_dir / "recon-all",
        """#!/usr/bin/env bash
echo "recon-all should not run" >&2
exit 99
""",
    )
    write_executable(
        bin_dir / "mri_convert",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s -> %s\\n' "$1" "$2" >> "{convert_log}"
cp "$1" "$2"
""",
    )

    script = Path(__file__).resolve().parents[1] / "atlas" / "make_atlas.sh"
    env = os.environ.copy()
    env["ATLAS_BACKEND"] = "native"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(script), str(data_dir), "2", str(license_path), subject],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (mri_dir / "aparc.a2009s+aseg.nii.gz").is_file()
    assert "aparc.a2009s+aseg.mgz" in convert_log.read_text(encoding="utf-8")


def test_make_atlas_native_backend_resumes_nonempty_subject_without_input_flag(tmp_path: Path):
    data_dir = tmp_path / "CamCan_Data"
    subject = "sub-01"
    anat_dir = data_dir / subject / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_T1w.nii").write_text("t1", encoding="utf-8")
    license_path = tmp_path / "license.txt"
    license_path.write_text("license", encoding="utf-8")

    existing_mri_dir = data_dir / "FastSurfer_out" / subject / "mri"
    existing_mri_dir.mkdir(parents=True)
    (existing_mri_dir / "mri_nu_correct.mni.log").write_text("partial recon", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recon_log = tmp_path / "recon-all.log"
    convert_log = tmp_path / "mri_convert.log"
    write_executable(
        bin_dir / "recon-all",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{recon_log}"
for arg in "$@"; do
  if [[ "$arg" == "-i" ]]; then
    echo "unexpected -i for existing subject" >&2
    exit 88
  fi
done
sid=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -s) sid="$2"; shift 2 ;;
    *) shift ;;
  esac
done
subject_dir="${{SUBJECTS_DIR}}/${{sid}}"
mkdir -p "${{subject_dir}}/mri" "${{subject_dir}}/surf" "${{subject_dir}}/scripts"
printf 't1' > "${{subject_dir}}/mri/T1.mgz"
printf 'destrieux' > "${{subject_dir}}/mri/aparc.a2009s+aseg.mgz"
touch "${{subject_dir}}/surf/lh.white" "${{subject_dir}}/surf/rh.white"
touch "${{subject_dir}}/scripts/recon-all.done"
""",
    )
    write_executable(
        bin_dir / "mri_convert",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s -> %s\\n' "$1" "$2" >> "{convert_log}"
cp "$1" "$2"
""",
    )

    script = Path(__file__).resolve().parents[1] / "atlas" / "make_atlas.sh"
    env = os.environ.copy()
    env["ATLAS_BACKEND"] = "native"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(script), str(data_dir), "2", str(license_path), subject],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "-i" not in recon_log.read_text(encoding="utf-8")
    assert (existing_mri_dir / "aparc.a2009s+aseg.nii.gz").is_file()
