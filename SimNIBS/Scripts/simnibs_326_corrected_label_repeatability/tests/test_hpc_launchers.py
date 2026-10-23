from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SUBMIT = PACKAGE / "hpc" / "submit.sh"
HEADMODEL_ARRAY = PACKAGE / "hpc" / "headmodel_array.slurm"
EXPECTED = {
    "activate_compat.sh",
    "module_preflight.slurm",
    "reference_stage_array.slurm",
    "headmodel_array.slurm",
    "headmodel_collect.slurm",
    "simulation_array.slurm",
    "control.slurm",
    "optimizer_roi_array.slurm",
    "optimizer_roi_collect.slurm",
    "analyze_optimizer_roi.slurm",
    "submit.sh",
}


@pytest.fixture
def launcher_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "sbatch.counter"
    fake = bin_dir / "sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "$FAKE_SBATCH_CALLS"\n'
        'n=0; [ ! -f "$FAKE_SBATCH_COUNTER" ] || n="$(cat "$FAKE_SBATCH_COUNTER")"\n'
        'n=$((n+1)); printf "%s\\n" "$n" > "$FAKE_SBATCH_COUNTER"\n'
        'printf "10%03d\\n" "$n"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    head = tmp_path / "headmodels"
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (head, left, right):
        root.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SBATCH_BIN": str(fake),
            "FAKE_SBATCH_CALLS": str(calls),
            "FAKE_SBATCH_COUNTER": str(counter),
            "PYTHON_BIN": "/bin/true",
            "PIPELINE_DIR": str(PACKAGE),
            "SIMNIBS326H_HEADMODEL_ROOT": str(head),
            "SIMNIBS326H_LEFT_ROOT": str(left),
            "SIMNIBS326H_RIGHT_ROOT": str(right),
            "SIMNIBS326H_RENDER_PYTHON": "/bin/true",
        }
    )
    return env, calls


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMIT), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _calls(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _ready_prerequisites(env: dict[str, str]) -> None:
    head = Path(env["SIMNIBS326H_HEADMODEL_ROOT"])
    module = head / "_pipeline/module_preflight.json"
    references = head / "_pipeline/reference_stage_collection.json"
    promotion = head / "_pipeline/headmodels/promoted.json"
    for path, payload in (
        (module, {"status": "ready"}),
        (references, {"status": "complete", "subject_count": 10}),
        (promotion, {"status": "promoted", "subjects": [str(i) for i in range(10)]}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def _configs(env: dict[str, str]) -> None:
    for key in ("SIMNIBS326H_LEFT_ROOT", "SIMNIBS326H_RIGHT_ROOT"):
        config_dir = Path(env[key]) / "_pipeline/configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "remesh_only.json", "fixed_mesh_spherical_median.json", "paired_analysis.json"
        ):
            (config_dir / name).write_text("{}", encoding="utf-8")


def test_all_expected_launchers_exist() -> None:
    hpc = PACKAGE / "hpc"
    assert {path.name for path in hpc.iterdir() if path.is_file()} == EXPECTED


@pytest.mark.parametrize("mode", ["preflight", "status"])
def test_read_only_modes_submit_no_jobs_and_print_scope_first(
    launcher_env: tuple[dict[str, str], Path], mode: str
) -> None:
    env, calls = launcher_env
    result = _run(env, mode)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Scope:\n")
    assert _calls(calls) == []


def test_reference_staging_is_ten_bounded_tasks(
    launcher_env: tuple[dict[str, str], Path]
) -> None:
    env, calls = launcher_env
    result = _run(env, "stage-references")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Scope:\n")
    submitted = _calls(calls)
    assert len(submitted) == 2
    assert "--array=0-9%10" in submitted[0]
    assert "reference_stage_array.slurm" in submitted[0]
    assert "--dependency=afterok:10001" in submitted[1]
    receipt = Path(env["SIMNIBS326H_HEADMODEL_ROOT"]) / "_pipeline/submissions.tsv"
    assert receipt.is_file()
    assert "stage-references" in receipt.read_text(encoding="utf-8")
    assert "10001,10002" in receipt.read_text(encoding="utf-8")


def test_headmodels_require_receipts_then_submit_array_and_collector(
    launcher_env: tuple[dict[str, str], Path]
) -> None:
    env, calls = launcher_env
    failed = _run(env, "headmodels")
    assert failed.returncode != 0
    assert _calls(calls) == []

    _ready_prerequisites(env)
    result = _run(env, "headmodels")
    assert result.returncode == 0, result.stderr
    submitted = _calls(calls)
    assert len(submitted) == 2
    assert "--array=0-9%10" in submitted[0]
    assert "headmodel_array.slurm" in submitted[0]
    assert "--dependency=afterok:10001" in submitted[1]
    assert "headmodel_collect.slurm" in submitted[1]


def test_simulation_array_is_exact_400_by_50(
    launcher_env: tuple[dict[str, str], Path]
) -> None:
    env, calls = launcher_env
    _ready_prerequisites(env)
    _configs(env)
    result = _run(env, "remesh", "left-hippocampus")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Scope:\n")
    submitted = _calls(calls)
    assert len(submitted) == 1
    assert "--array=0-399%50" in submitted[0]
    assert "--cpus-per-task=8" in submitted[0]
    assert "--mem=32G" in submitted[0]
    assert "--time=08:00:00" in submitted[0]
    assert "--open-mode=append" in submitted[0]


def test_all_remesh_and_all_fixed_keep_target_arrays_sequential(
    launcher_env: tuple[dict[str, str], Path]
) -> None:
    env, calls = launcher_env
    _ready_prerequisites(env)
    _configs(env)
    remesh = _run(env, "all-remesh")
    assert remesh.returncode == 0, remesh.stderr
    remesh_calls = _calls(calls)
    simulations = [line for line in remesh_calls if "simulation_array.slurm" in line]
    assert len(simulations) == 2
    assert "--array=0-399%50" in simulations[0]
    assert "--dependency=afterok:" in simulations[1]

    Path(calls).unlink()
    Path(env["FAKE_SBATCH_COUNTER"]).unlink()
    fixed = _run(env, "all-fixed")
    assert fixed.returncode == 0, fixed.stderr
    fixed_calls = _calls(calls)
    fixed_simulations = [line for line in fixed_calls if "simulation_array.slurm" in line]
    assert len(fixed_simulations) == 2
    assert "--dependency=afterok:" in fixed_simulations[1]


def test_launchers_remove_matlab_cat12_and_preserve_retry_contract() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in (PACKAGE / "hpc").glob("*")
        if path.is_file()
    )
    assert "module load SimNIBS/3.2.6-foss-2023a" in text
    assert "module load MATLAB" not in text
    assert "cat12_compat" not in text.lower()
    assert '${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}' in text
    assert "SIMNIBS326H_MAX_RETRIES" in text
    assert "--open-mode=append" in text
    assert "validate_repeatability_task.py" in text


def test_headmodel_worker_isolates_gmsh_workdir_and_writes_diagnostic_receipt(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("module", "#!/bin/bash\nexit 0\n"),
        ("python", "#!/bin/bash\nexit 0\n"),
        ("scontrol", "#!/bin/bash\nexit 99\n"),
        (
            "timeout",
            "#!/bin/bash\n"
            'printf "%s\\n" "$PWD" > "$OBSERVED_HEADMODEL_CWD"\n'
            'printf "isolated gmsh log\\n" > gmsh_log.txt\n'
            "exit 1\n",
        ),
    ):
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    log_dir = tmp_path / "logs"
    preflight = tmp_path / "module_preflight.json"
    preflight.write_text("{}", encoding="utf-8")
    observed_cwd = tmp_path / "observed-cwd.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "HOME": str(tmp_path / "home"),
            "USER": "test-user",
            "PIPELINE_DIR": str(PACKAGE),
            "PREFLIGHT_RECEIPT": str(preflight),
            "LOG_DIR": str(log_dir),
            "SLURM_ARRAY_TASK_ID": "6",
            "SLURM_ARRAY_JOB_ID": "119",
            "SLURM_JOB_ID": "119_6",
            "SLURM_CPUS_PER_TASK": "8",
            "SIMNIBS326H_MAX_RETRIES": "0",
            "SIMNIBS326H_DIAGNOSTIC": "1",
            "OBSERVED_HEADMODEL_CWD": str(observed_cwd),
        }
    )

    completed = subprocess.run(
        ["bash", str(HEADMODEL_ARRAY)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    workdir = Path(observed_cwd.read_text(encoding="utf-8").strip())
    assert workdir.parent == log_dir / "work" / "119_6"
    assert workdir.name.startswith("retry_0_")
    assert (workdir / "gmsh_log.txt").read_text(encoding="utf-8") == "isolated gmsh log\n"
    receipts = list((log_dir / "diagnostic_receipts").glob("diagnostic_119_6_*.tsv"))
    assert len(receipts) == 1
    row = receipts[0].read_text(encoding="utf-8")
    assert "job_key\tsubject_index\tretry\texit_code\twork_dir\tattempt_log" in row
    assert "119_6\t6\t0\t1\t" in row


def test_default_preflight_does_not_hide_the_audit_plan() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    assert (
        '"$PYTHON_BIN" -m simnibs_326_corrected_label_repeatability preflight\n'
        in submit
    )
