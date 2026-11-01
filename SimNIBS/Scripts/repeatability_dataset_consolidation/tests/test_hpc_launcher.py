from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "hpc/submit.sh"


def test_launcher_has_valid_bash_and_no_simulation_submission() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "sbatch" not in text
    assert "SimNIBS/" not in text
    assert "SLURM_JOB_ID" in text


def test_launcher_passes_modes_and_arguments_as_an_array(tmp_path: Path) -> None:
    fake_python = tmp_path / "python"
    arguments = tmp_path / "arguments.txt"
    fake_python.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$FAKE_ARGUMENTS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({"PYTHON_BIN": str(fake_python), "FAKE_ARGUMENTS": str(arguments)})

    completed = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "copy",
            "--confirm-output-root=/path with spaces/dataset",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert arguments.read_text().splitlines() == [
        "-m",
        "repeatability_dataset_consolidation",
        "copy",
        "--confirm-output-root=/path with spaces/dataset",
    ]


def test_launcher_refuses_compute_node_and_unknown_mode(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["SLURM_JOB_ID"] = "123"
    compute = subprocess.run(
        ["bash", str(LAUNCHER), "preflight"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compute.returncode == 2
    assert "login node" in compute.stderr

    unknown = subprocess.run(
        ["bash", str(LAUNCHER), "not-a-mode"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert unknown.returncode == 2
    assert "Unsupported mode" in unknown.stderr
