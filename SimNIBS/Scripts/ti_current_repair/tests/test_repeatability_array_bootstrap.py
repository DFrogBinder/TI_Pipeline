from __future__ import annotations

import os
import subprocess
from pathlib import Path


ARRAY_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "hpc_scripts"
    / "repeatability_experiment_array.slurm"
)
SUBMITTER = (
    Path(__file__).resolve().parents[1]
    / "hpc_scripts"
    / "submit_repeatability_experiment.sh"
)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_retry_limit_is_four_in_submitter_and_array_fallback() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    array_runner = ARRAY_RUNNER.read_text(encoding="utf-8")

    assert 'MESH_MAX_RETRIES_CONFIG="4"' in submitter
    assert 'MESH_MAX_RETRIES="${TI_MESH_MAX_RETRIES:-4}"' in array_runner


def test_array_stops_before_task_setup_when_module_load_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "module",
        "#!/bin/bash\n"
        'if [ "${1:-}" = "load" ]; then exit 23; fi\n'
        "exit 0\n",
    )
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    slurm_tmp = tmp_path / "slurm-tmp"
    slurm_tmp.mkdir()
    log_dir = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PIPELINE_DIR": str(pipeline_dir),
            "EXPERIMENT_CONFIG": str(tmp_path / "config.json"),
            "LOG_DIR": str(log_dir),
            "SLURM_TMPDIR": str(slurm_tmp),
            "SLURM_JOB_ID": "12345",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_CPUS_PER_TASK": "8",
            "USER": "test-user",
        }
    )

    completed = subprocess.run(
        ["bash", str(ARRAY_RUNNER)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 23
    assert not (slurm_tmp / "test-user_simnibs_shim_12345_7").exists()
    assert not log_dir.exists()


def test_array_stops_before_task_setup_when_scientific_runtime_is_missing(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "module", "#!/bin/bash\nexit 0\n")
    _write_executable(
        fake_bin / "python",
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-c" ]; then exit 24; fi\n'
        "exit 0\n",
    )
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    slurm_tmp = tmp_path / "slurm-tmp"
    slurm_tmp.mkdir()
    log_dir = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PIPELINE_DIR": str(pipeline_dir),
            "EXPERIMENT_CONFIG": str(tmp_path / "config.json"),
            "LOG_DIR": str(log_dir),
            "SLURM_TMPDIR": str(slurm_tmp),
            "SLURM_JOB_ID": "12345",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_CPUS_PER_TASK": "8",
            "USER": "test-user",
        }
    )

    completed = subprocess.run(
        ["bash", str(ARRAY_RUNNER)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 24
    assert not (slurm_tmp / "test-user_simnibs_shim_12345_7").exists()
    assert not log_dir.exists()
