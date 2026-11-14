from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
HPC = PACKAGE / "hpc"
SUBMITTER = HPC / "submit.sh"


def _text(name):
    return (HPC / name).read_text(encoding="utf-8")


def test_static_slurm_scope_resources_modules_and_stage_boundaries():
    headreco = _text("headreco_array.slurm")
    charm = _text("charm_array.slurm")
    preflight326 = _text("module_preflight_326.slurm")
    preflight401 = _text("module_preflight_401.slurm")
    collect = _text("collect.slurm")
    submitter = _text("submit.sh")

    assert "#SBATCH --partition=sheffield" in headreco
    assert "#SBATCH --cpus-per-task=8" in headreco
    assert "#SBATCH --mem=32G" in headreco
    assert "#SBATCH --time=08:00:00" in headreco
    assert "#SBATCH --array=0-9%10" in headreco
    assert "#SBATCH --array=0-9%10" in charm
    assert "--open-mode=append" in submitter
    assert "SimNIBS/3.2.6-foss-2023a" in preflight326 + headreco
    assert "MATLAB/2023b" in preflight326 + headreco
    assert "SimNIBS/4.0.1-foss-2023a" in preflight401 + charm + collect
    assert "activate_compat.sh" in headreco
    assert 'MAX_RETRIES="${SEGDET_MAX_RETRIES:-1}"' in headreco
    assert 'MAX_RETRIES="${SEGDET_MAX_RETRIES:-1}"' in charm
    assert 'scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in headreco
    assert 'scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in charm
    assert "afterany:$array326:$array401" in submitter
    assert "afterok:$preflight326" in submitter
    assert "afterok:$preflight401" in submitter
    assert "scheduler tasks: 23" in submitter
    assert "runtime-preflight" in preflight326
    assert "runtime-preflight" in preflight401
    assert (
        'DATASET_ROOT="${SEGDET_DATASET_ROOT:-/mnt/parscratch/users/cop23bi/'
        'final_132_repeatability_combined_dataset_v1}"'
    ) in submitter
    assert (
        'STAGING_ROOT="${SEGDET_STAGING_ROOT:-/mnt/parscratch/users/cop23bi/'
        'segmentation_determinism_staging_v1}"'
    ) in submitter
    assert 'OUTPUT_ROOT="$STAGING_ROOT"' in submitter
    assert (
        'SOURCE_ROOT="/mnt/parscratch/users/cop23bi/'
        'ti_dataset_final_132_balanced_10_corrected"'
    ) in submitter
    assert 'export SEGDET_DATASET_ROOT="$DATASET_ROOT"' in submitter
    assert 'export SEGDET_STAGING_ROOT="$STAGING_ROOT"' in submitter
    assert 'export SEGDET_OUTPUT_ROOT="$OUTPUT_ROOT"' in submitter
    assert "submission.lock" in submitter

    forbidden_headreco = re.compile(
        r"headreco\s+(?:all|preparecat|cleanvols|surfacemesh|volumemesh)\b"
    )
    assert not forbidden_headreco.search(headreco)
    assert not re.search(r"charm\b[^\n]*(?:--surfaces?|--mesh)\b", charm)


def _fake_scheduler(tmp_path: Path):
    log = tmp_path / "sbatch.log"
    state = tmp_path / "sbatch.count"
    fake = tmp_path / "fake-sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "$FAKE_SBATCH_LOG"\n'
        'count=0; [[ -f "$FAKE_SBATCH_STATE" ]] && count=$(cat "$FAKE_SBATCH_STATE")\n'
        'count=$((count + 1)); printf "%s\\n" "$count" > "$FAKE_SBATCH_STATE"\n'
        'printf "%s\\n" "$((910000 + count))"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake, log, state


def _fake_python(tmp_path: Path) -> Path:
    fake = tmp_path / "fake-python"
    fake.write_text(
        "#!/bin/bash\n"
        "if [[ \"${1:-}\" == \"-\" ]]; then exec python3 \"$@\"; fi\n"
        "cat >/dev/null || true\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _run_submitter(command, tmp_path, fake, log, state, output=None):
    staging = output or tmp_path / "staging"
    dataset = tmp_path / "dataset"
    env = {
        **os.environ,
        "SBATCH_BIN": str(fake),
        "FAKE_SBATCH_LOG": str(log),
        "FAKE_SBATCH_STATE": str(state),
        "SEGDET_DATASET_ROOT": str(dataset),
        "SEGDET_STAGING_ROOT": str(staging),
        "PYTHON_BIN": str(_fake_python(tmp_path)),
    }
    return subprocess.run(
        ["bash", str(SUBMITTER), command],
        cwd=str(PACKAGE.parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_only_run_calls_sbatch_and_run_submits_guarded_dependency_graph(tmp_path):
    fake, log, state = _fake_scheduler(tmp_path)

    for index, command in enumerate(("scope", "preflight", "init", "status"), start=1):
        result = _run_submitter(
            command,
            tmp_path,
            fake,
            log,
            state,
            output=tmp_path / f"nonrun-{index}",
        )
        assert result.returncode == 0, result.stderr
    analysis_output = tmp_path / "analysis-nonrun"
    assert _run_submitter("init", tmp_path, fake, log, state, output=analysis_output).returncode == 0
    assert _run_submitter("analyze", tmp_path, fake, log, state, output=analysis_output).returncode == 0
    assert not log.exists()

    result = _run_submitter("run", tmp_path, fake, log, state, output=tmp_path / "run-output")
    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 5
    assert "module_preflight_326.slurm" in calls[0]
    assert "module_preflight_401.slurm" in calls[1]
    assert "headreco_array.slurm" in calls[2]
    assert "--dependency=afterok:910001" in calls[2]
    assert "--array=0-9%10" in calls[2]
    assert "charm_array.slurm" in calls[3]
    assert "--dependency=afterok:910002" in calls[3]
    assert "collect.slurm" in calls[4]
    assert "--dependency=afterany:910003:910004" in calls[4]
    assert all("--open-mode=append" in call for call in calls)
    ledger = (
        tmp_path
        / "run-output/_pipeline/job_ids.tsv"
    )
    assert ledger.is_file()
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 6


def test_failed_module_preflight_submission_never_submits_an_array(tmp_path):
    log = tmp_path / "sbatch.log"
    fake = tmp_path / "failing-sbatch"
    fake.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$FAKE_SBATCH_LOG"\n'
        "exit 9\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    result = _run_submitter(
        "run", tmp_path, fake, log, tmp_path / "state", output=tmp_path / "failed-output"
    )

    assert result.returncode != 0
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert "module_preflight_326.slurm" in calls[0]
    assert not any("array.slurm" in call for call in calls)


def test_second_run_is_rejected_before_duplicate_submission(tmp_path):
    fake, log, state = _fake_scheduler(tmp_path)
    output = tmp_path / "run-output"
    first = _run_submitter("run", tmp_path, fake, log, state, output=output)
    second = _run_submitter("run", tmp_path, fake, log, state, output=output)

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert len(log.read_text(encoding="utf-8").splitlines()) == 5


def test_repeat_scope_override_is_rejected_before_submission(tmp_path):
    fake, log, state = _fake_scheduler(tmp_path)
    dataset = tmp_path / "dataset"
    env = {
        **os.environ,
        "SBATCH_BIN": str(fake),
        "FAKE_SBATCH_LOG": str(log),
        "FAKE_SBATCH_STATE": str(state),
        "SEGDET_DATASET_ROOT": str(dataset),
        "SEGDET_REPEATS_PER_ARM": "7",
        "SEGDET_STAGING_ROOT": str(tmp_path / "staging"),
        "PYTHON_BIN": str(_fake_python(tmp_path)),
    }
    result = subprocess.run(
        ["bash", str(SUBMITTER), "run"],
        cwd=str(PACKAGE.parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not log.exists()


def test_install_modes_never_submit_jobs_and_confirmation_is_exact(tmp_path):
    fake, log, state = _fake_scheduler(tmp_path)
    dataset = tmp_path / "dataset"

    result = _run_submitter(
        "install-preflight", tmp_path, fake, log, state, output=tmp_path / "staging"
    )
    assert result.returncode == 0

    rejected = subprocess.run(
        ["bash", str(SUBMITTER), "install-extension", "--confirm-dataset-root=/wrong"],
        cwd=str(PACKAGE.parent),
        env={
            **os.environ,
            "SBATCH_BIN": str(fake),
            "FAKE_SBATCH_LOG": str(log),
            "FAKE_SBATCH_STATE": str(state),
            "SEGDET_DATASET_ROOT": str(dataset),
            "SEGDET_STAGING_ROOT": str(tmp_path / "staging"),
            "PYTHON_BIN": str(_fake_python(tmp_path)),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0

    confirmed = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "install-extension",
            f"--confirm-dataset-root={dataset}",
        ],
        cwd=str(PACKAGE.parent),
        env={
            **os.environ,
            "SBATCH_BIN": str(fake),
            "FAKE_SBATCH_LOG": str(log),
            "FAKE_SBATCH_STATE": str(state),
            "SEGDET_DATASET_ROOT": str(dataset),
            "SEGDET_STAGING_ROOT": str(tmp_path / "staging"),
            "PYTHON_BIN": str(_fake_python(tmp_path)),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert confirmed.returncode == 0
    assert not log.exists()
