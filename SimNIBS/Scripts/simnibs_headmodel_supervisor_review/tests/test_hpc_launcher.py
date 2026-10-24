from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from simnibs_headmodel_supervisor_review.settings import ReviewPaths


PACKAGE = Path(__file__).resolve().parents[1]
SUBMIT = PACKAGE / "hpc" / "submit.sh"
WORKER = PACKAGE / "hpc" / "build.slurm"
README = PACKAGE / "README.md"


def _args(paths: ReviewPaths) -> list[str]:
    return [
        f"--original-charm-root={paths.original_charm_root}",
        f"--corrected-v4-scaffold-root={paths.corrected_v4_scaffold_root}",
        f"--legacy-v3-root={paths.legacy_v3_root}",
        f"--harmonized-v3-root={paths.harmonized_v3_root}",
        f"--atlas-root={paths.atlas_root}",
        f"--completed-v4-left-root={paths.completed_v4_left_root}",
        f"--output-root={paths.output_root}",
    ]


def _run(
    env: dict[str, str], paths: ReviewPaths, mode: str, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMIT), mode, *_args(paths), *extra],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _fake_sbatch(tmp_path: Path, response: str) -> tuple[Path, Path]:
    fake = tmp_path / "sbatch"
    calls = tmp_path / "sbatch.calls"
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        f"printf '%s\\n' '{response}'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake, calls


def test_launcher_contract_is_one_pinned_simnibs326_job() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    combined = submit + worker

    assert "SimNIBS/3.2.6-foss-2023a" in worker
    assert "SimNIBS/4" not in combined
    assert "#SBATCH --partition=sheffield" in worker
    assert "#SBATCH --cpus-per-task=8" in worker
    assert "#SBATCH --mem=32G" in worker
    assert "#SBATCH --time=08:00:00" in worker
    assert "#SBATCH --array" not in worker
    assert "timeout --kill-after=30s 7h" in worker
    assert 'export PIPELINE_DIR="$SCRIPTS_ROOT/simnibs_326_repeatability"' in worker
    assert "--confirm-output-root" in submit
    assert "sub-CC" not in combined


def test_preflight_is_read_only_and_reports_exact_scope(
    review_layout: tuple[ReviewPaths, tuple[str, ...]], tmp_path: Path
) -> None:
    paths, _ = review_layout
    fake, calls = _fake_sbatch(tmp_path, "41001")
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    result = _run(env, paths, "preflight")

    assert result.returncode == 0, result.stderr
    assert "candidates: 9" in result.stdout
    assert "selected subjects: 1" in result.stdout
    assert "scientific artifacts: 8" in result.stdout
    assert "jobs submitted: 0" in result.stdout
    assert "SimNIBS/3.2.6-foss-2023a" in result.stdout
    assert not calls.exists()
    assert not paths.output_root.exists()


def test_submit_requires_exact_output_confirmation(
    review_layout: tuple[ReviewPaths, tuple[str, ...]], tmp_path: Path
) -> None:
    paths, _ = review_layout
    fake, calls = _fake_sbatch(tmp_path, "42001")
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    missing = _run(env, paths, "submit")
    wrong = _run(
        env,
        paths,
        "submit",
        f"--confirm-output-root={paths.output_root / 'wrong'}",
    )

    assert missing.returncode == 2
    assert "--confirm-output-root" in missing.stderr
    assert wrong.returncode == 2
    assert "exactly match" in wrong.stderr
    assert not calls.exists()
    assert not paths.output_root.exists()


def test_submit_selects_once_and_submits_one_non_array_job(
    review_layout: tuple[ReviewPaths, tuple[str, ...]], tmp_path: Path
) -> None:
    paths, subjects = review_layout
    fake, calls = _fake_sbatch(tmp_path, "43001;stanage")
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    result = _run(
        env,
        paths,
        "submit",
        f"--confirm-output-root={paths.output_root}",
    )

    assert result.returncode == 0, result.stderr
    assert "Review-package job: 43001" in result.stdout
    selection = json.loads(paths.selection_receipt.read_text(encoding="utf-8"))
    assert selection["selected_subject"] in subjects
    assert selection["candidate_count"] == 9
    assert selection["artifact_count"] == 8
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "--array" not in lines[0]
    assert "--partition=sheffield" in lines[0]
    assert "--cpus-per-task=8" in lines[0]
    assert "--mem=32G" in lines[0]
    assert "--time=08:00:00" in lines[0]
    assert f"SIMNIBS_REVIEW_OUTPUT_ROOT={paths.output_root}" in lines[0]
    assert "build.slurm" in lines[0]


def test_malformed_scheduler_response_fails_closed(
    review_layout: tuple[ReviewPaths, tuple[str, ...]], tmp_path: Path
) -> None:
    paths, _ = review_layout
    fake, _ = _fake_sbatch(tmp_path, "not-a-job")
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    result = _run(
        env,
        paths,
        "submit",
        f"--confirm-output-root={paths.output_root}",
    )

    assert result.returncode == 2
    assert "parse Slurm job ID" in result.stderr


def test_readme_documents_all_eight_artifacts_and_provenance_boundary() -> None:
    text = README.read_text(encoding="utf-8")
    assert "SimNIBS/3.2.6-foss-2023a" in text
    assert "eight scientific artifacts" in text
    assert "original headreco" in text
    assert "harmonised" in text
    assert "median representative" in text
    assert "not a raw CAT12" in text
    assert "Scheduler submission is not completion" in text
    assert "source roots are read-only" in text
