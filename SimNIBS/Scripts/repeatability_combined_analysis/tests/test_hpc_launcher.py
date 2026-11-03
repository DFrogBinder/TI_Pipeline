from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SUBMIT = PACKAGE / "hpc/submit.sh"
SLURM = PACKAGE / "hpc/analysis.slurm"


@pytest.fixture
def fake_hpc(tmp_path: Path) -> dict[str, object]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    root = tmp_path / "analysis"
    python_log = tmp_path / "python.log"
    sbatch_log = tmp_path / "sbatch.log"
    python = bin_dir / "python"
    python.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_PY_LOG"
if [[ "${1:-}" == "-c" && "${2:-}" == *production_config* ]]; then
    printf '%s\n' "$FAKE_ROOT"
    exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-m" && "${3:-}" == "prepare" ]]; then
    mkdir -p "$FAKE_ROOT/_workflow"
    printf '{"plan_sha256":"%064d"}\n' 0 > "$FAKE_ROOT/_workflow/input_plan.json"
    printf '{"workflow":"repeatability_combined_analysis"}\n' > "$FAKE_ROOT/_workflow/owner.json"
fi
if [[ "${1:-}" == "-m" && "${3:-}" == "scope" ]]; then
    printf '{"status":"scope"}\n'
elif [[ "${1:-}" == "-m" && "${3:-}" == "status" ]]; then
    printf '{"status":"ok","state":"prepared"}\n'
else
    printf '{"status":"ready"}\n'
fi
""",
        encoding="utf-8",
    )
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        """#!/bin/bash
printf '%s\n' "$*" >> "$SBATCH_LOG"
printf '123456\n'
""",
        encoding="utf-8",
    )
    sacct = bin_dir / "sacct"
    sacct.write_text(
        """#!/bin/bash
printf 'JobIDRaw|JobName|State|ExitCode\n123456|ti_repeat_combined|COMPLETED|0:0\n'
""",
        encoding="utf-8",
    )
    for path in (python, sbatch, sacct):
        path.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": str(python),
            "ANALYSIS_PYTHON": str(python),
            "FAKE_ROOT": str(root),
            "FAKE_PY_LOG": str(python_log),
            "SBATCH_LOG": str(sbatch_log),
        }
    )
    return {
        "env": env,
        "root": root,
        "python_log": python_log,
        "sbatch_log": sbatch_log,
    }


def run_launcher(fake_hpc: dict[str, object], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SUBMIT), *args],
        cwd=PACKAGE.parent,
        env=fake_hpc["env"],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("mode", ("scope", "preflight"))
def test_read_only_scope_modes_print_exact_scope_and_do_not_submit(
    fake_hpc: dict[str, object], mode: str
) -> None:
    result = run_launcher(fake_hpc, mode)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    for expected in (
        "subjects: 10",
        "versions: 2",
        "targets: 2",
        "conditions: 2",
        "repeats per cell: 40",
        "participant cells: 80",
        "run-level rows: 3200",
        "meshing tasks: 0",
        "FEM simulation tasks: 0",
        "jobs submitted by run: 1",
        "resources: sheffield; 4 CPU; 16G; 02:00:00",
        "source campaigns: read-only",
        "output: isolated analysis root",
    ):
        assert expected in output
    assert not Path(fake_hpc["sbatch_log"]).exists()


def test_run_preflights_prepares_and_submits_exactly_one_job(
    fake_hpc: dict[str, object]
) -> None:
    result = run_launcher(fake_hpc, "run")
    assert result.returncode == 0, result.stderr
    root = Path(fake_hpc["root"])
    python_calls = Path(fake_hpc["python_log"]).read_text()
    assert "-m repeatability_combined_analysis preflight" in python_calls
    assert "-m repeatability_combined_analysis prepare" in python_calls
    assert f"--confirm-output-root={root}" in python_calls
    sbatch_calls = Path(fake_hpc["sbatch_log"]).read_text().splitlines()
    assert len(sbatch_calls) == 1
    assert str(SLURM) in sbatch_calls[0]
    assert "--parsable" in sbatch_calls[0]
    submission = json.loads((root / "_workflow/submission.json").read_text())
    assert submission["status"] == "submitted"
    assert submission["job_id"] == "123456"
    assert submission["output_root"] == str(root)
    assert len(submission["plan_sha256"]) == 64
    assert len(submission["git_commit"]) == 40


def test_status_combines_sacct_and_independent_package_status(
    fake_hpc: dict[str, object]
) -> None:
    assert run_launcher(fake_hpc, "run").returncode == 0
    result = run_launcher(fake_hpc, "status")
    assert result.returncode == 0, result.stderr
    assert "SLURM ACCOUNTING" in result.stdout
    assert "123456|ti_repeat_combined|COMPLETED|0:0" in result.stdout
    assert "PACKAGE STATUS" in result.stdout
    calls = Path(fake_hpc["python_log"]).read_text()
    assert "-m repeatability_combined_analysis status" in calls


def test_environment_and_resource_contract_is_explicit() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=sheffield",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=16G",
        "#SBATCH --time=02:00:00",
    ):
        assert directive in slurm
    assert "set -euo pipefail" in slurm
    assert "PYTHONNOUSERSITE=1" in slurm
    assert "MPLCONFIGDIR" in slurm
    assert "pandas" in slurm and "numpy" in slurm
    assert "scipy" in slurm and "matplotlib" in slurm
    assert "PREPARED_PLAN_SHA256" in slurm
    assert "REPOSITORY_COMMIT" in slurm
    assert "python -m repeatability_combined_analysis execute" not in slurm
    assert '"$ANALYSIS_PYTHON" -m repeatability_combined_analysis execute' in slurm
    assert "production_config" in submit
    assert "rev-parse HEAD" in submit


def test_launcher_refuses_non_code_owned_output_override(
    fake_hpc: dict[str, object]
) -> None:
    result = run_launcher(fake_hpc, "run", "--output-root=/tmp/wrong")
    assert result.returncode == 2
    assert "code-owned output root" in result.stderr
    assert not Path(fake_hpc["sbatch_log"]).exists()


def test_rebind_mode_requires_and_forwards_dataset_root(
    fake_hpc: dict[str, object], tmp_path: Path
) -> None:
    missing = run_launcher(fake_hpc, "rebind-preflight")
    assert missing.returncode == 2
    dataset = tmp_path / "dataset"
    result = run_launcher(
        fake_hpc, "rebind-preflight", f"--dataset-root={dataset}"
    )
    assert result.returncode == 0
    calls = Path(fake_hpc["python_log"]).read_text()
    assert (
        f"-m repeatability_combined_analysis rebind-preflight "
        f"--dataset-root={dataset}" in calls
    )
    assert not Path(fake_hpc["sbatch_log"]).exists()
