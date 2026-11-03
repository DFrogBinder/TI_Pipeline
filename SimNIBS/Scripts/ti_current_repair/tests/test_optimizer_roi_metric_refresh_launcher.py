from __future__ import annotations

import os
from pathlib import Path
import subprocess


PIPELINE = Path(__file__).resolve().parents[1]
SUBMITTER = (
    PIPELINE
    / "hpc_scripts"
    / "submit_repeatability_optimizer_roi_metric_refresh.sh"
)


def _fake_python(tmp_path: Path) -> Path:
    path = tmp_path / "fake-python"
    path.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_sbatch(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "sbatch.log"
    counter = tmp_path / "counter"
    counter.write_text("8000\n", encoding="utf-8")
    path = tmp_path / "fake-sbatch"
    path.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'value=$(cat "$FAKE_SBATCH_COUNTER")\n'
        "value=$((value + 1))\n"
        'printf "%s\\n" "$value" > "$FAKE_SBATCH_COUNTER"\n'
        'printf "CALL" >> "$FAKE_SBATCH_LOG"\n'
        'for argument in "$@"; do printf "\\t%s" "$argument" >> "$FAKE_SBATCH_LOG"; done\n'
        'printf "\\n" >> "$FAKE_SBATCH_LOG"\n'
        'printf "%s\\n" "$value"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path, log


def test_default_preflight_retains_full_ten_subject_scope(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_DIR": str(PIPELINE),
            "PYTHON_BIN": str(_fake_python(tmp_path)),
            "LEFT_ROOT": str(tmp_path / "left-source"),
            "RIGHT_ROOT": str(tmp_path / "right-source"),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER), "all", "--preflight"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "subjects per target: 10" in result.stdout
    assert "existing TI NIfTIs total: 1600" in result.stdout
    assert "extraction arrays: 2 x 0-9%10" in result.stdout


def test_single_subject_refresh_uses_isolated_outputs_and_exact_arrays(
    tmp_path: Path,
) -> None:
    fake_sbatch, sbatch_log = _fake_sbatch(tmp_path)
    left_source = tmp_path / "left-source"
    right_source = tmp_path / "right-source"
    left_output = tmp_path / "isolated" / "left_hippocampus"
    right_output = tmp_path / "isolated" / "right_m1"
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_DIR": str(PIPELINE),
            "PYTHON_BIN": str(_fake_python(tmp_path)),
            "SBATCH_BIN": str(fake_sbatch),
            "FAKE_SBATCH_COUNTER": str(tmp_path / "counter"),
            "FAKE_SBATCH_LOG": str(sbatch_log),
            "SUBJECT_COUNT": "1",
            "LEFT_ROOT": str(left_source),
            "RIGHT_ROOT": str(right_source),
            "LEFT_OUTPUT_ROOT": str(left_output),
            "RIGHT_OUTPUT_ROOT": str(right_output),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER), "all"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "subjects per target: 1" in result.stdout
    assert "existing TI NIfTIs total: 160" in result.stdout
    assert "extraction arrays: 2 x 0-0%1" in result.stdout
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert "--array=0-0%1" in calls[0]
    assert "--array=0-0%1" in calls[2]
    assert f"OUTPUT_ROOT={left_output}" in calls[0]
    assert f"OUTPUT_ROOT={right_output}" in calls[2]
    assert str(left_source / "_post_processing") not in "\n".join(calls)
    assert str(right_source / "_post_processing") not in "\n".join(calls)
    assert (left_output / "submitted_job_ids.tsv").is_file()
    assert (right_output / "submitted_job_ids.tsv").is_file()


def test_subject_count_must_be_positive_integer(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_DIR": str(PIPELINE),
            "PYTHON_BIN": str(_fake_python(tmp_path)),
            "SUBJECT_COUNT": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER), "all", "--preflight"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "SUBJECT_COUNT must be a positive integer" in result.stderr


def test_two_target_refresh_rejects_colliding_output_roots(tmp_path: Path) -> None:
    output = tmp_path / "same-output"
    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_DIR": str(PIPELINE),
            "PYTHON_BIN": str(_fake_python(tmp_path)),
            "LEFT_OUTPUT_ROOT": str(output),
            "RIGHT_OUTPUT_ROOT": str(output),
        }
    )

    result = subprocess.run(
        ["bash", str(SUBMITTER), "all", "--preflight"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "output roots must be different" in result.stderr
