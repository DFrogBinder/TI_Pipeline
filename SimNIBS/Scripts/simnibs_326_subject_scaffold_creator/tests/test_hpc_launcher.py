from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from simnibs_326_subject_scaffold_creator.settings import CreatorConfig


PACKAGE = Path(__file__).resolve().parents[1]
SUBMIT = PACKAGE / "hpc" / "submit.sh"
WORKER = PACKAGE / "hpc" / "scaffold_array.slurm"
README = PACKAGE / "README.md"


def _config(layout: dict[str, Path | str]) -> CreatorConfig:
    return CreatorConfig(
        subject=str(layout["subject"]),
        corrected_source_root=Path(layout["source"]),
        output_root=Path(layout["output"]),
        final132_subjects_file=Path(layout["subjects"]),
    )


def _args(config: CreatorConfig) -> list[str]:
    return [
        f"--subject={config.subject}",
        f"--corrected-source-root={config.corrected_source_root}",
        f"--output-root={config.output_root}",
        f"--final132-subjects={config.final132_subjects_file}",
    ]


def _run(
    env: dict[str, str], config: CreatorConfig, mode: str, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SUBMIT), mode, *_args(config), *extra],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _fake_sbatch(
    tmp_path: Path, outputs: list[str]
) -> tuple[Path, Path]:
    fake = tmp_path / "sbatch"
    calls = tmp_path / "sbatch.calls"
    counter = tmp_path / "sbatch.counter"
    encoded = json.dumps(outputs)
    fake.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> {calls}\n"
        f"counter={counter}\n"
        "index=0\n"
        "if [ -f \"$counter\" ]; then index=$(cat \"$counter\"); fi\n"
        "printf '%s\\n' $((index + 1)) > \"$counter\"\n"
        f"python3 -c 'import json,sys; values=json.loads(sys.argv[1]); i=int(sys.argv[2]); print(values[i] if i < len(values) else \"\")' '{encoded}' \"$index\"\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake, calls


def test_launcher_contract_is_exactly_simnibs_326_singleton() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    combined = submit + worker

    assert "SimNIBS/3.2.6-foss-2023a" in worker
    assert "MATLAB/2023b" in worker
    assert "SimNIBS/4" not in combined
    assert "#SBATCH --partition=sheffield" in worker
    assert "#SBATCH --cpus-per-task=8" in worker
    assert "#SBATCH --mem=32G" in worker
    assert "#SBATCH --time=08:00:00" in worker
    assert "#SBATCH --array=0-0%1" in worker
    assert "timeout --kill-after=30s 7h" in worker
    assert 'MAX_RETRIES="${SIMNIBS326_MAX_RETRIES:-1}"' in worker
    assert 'REQUEUE_JOB_ID="${ARRAY_JOB_ID}_${TASK_ID}"' in worker
    assert "scontrol requeue \"$REQUEUE_JOB_ID\"" in worker
    assert "#SBATCH --open-mode=append" in worker
    assert "retry_${RETRY_COUNT}_${ATTEMPT_STAMP}.log" in worker
    assert "--confirm-output-root" in submit
    assert "sub-CC510415" not in combined


def test_worker_maps_repeatability_directory_to_compatibility_contract() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    mapping = 'export PIPELINE_DIR="$REPEATABILITY_DIR"'
    activation = 'source "$REPEATABILITY_DIR/hpc/activate_compat.sh"'

    assert mapping in worker
    assert worker.index(mapping) < worker.index(activation)


def test_wrapper_requires_runtime_subject_and_exact_confirmation(
    creator_layout: dict[str, Path | str],
) -> None:
    missing = subprocess.run(
        ["bash", str(SUBMIT), "preflight"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "--subject" in missing.stderr

    config = _config(creator_layout)
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    wrong = _run(
        env,
        config,
        "submit",
        f"--confirm-output-root={config.output_root / 'wrong'}",
    )
    assert wrong.returncode == 2
    assert "exactly match" in wrong.stderr


def test_read_only_modes_never_call_sbatch(
    creator_layout: dict[str, Path | str], tmp_path: Path
) -> None:
    config = _config(creator_layout)
    fake, calls = _fake_sbatch(tmp_path, ["999"])
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    preflight = _run(env, config, "preflight")
    status = _run(env, config, "status")
    validate = _run(env, config, "validate")

    assert preflight.returncode == 0, preflight.stderr
    assert status.returncode == 1
    assert validate.returncode != 0
    assert not calls.exists()
    assert not config.output_root.exists()


def test_module_check_accepts_parsable_cluster_suffix(
    creator_layout: dict[str, Path | str], tmp_path: Path
) -> None:
    config = _config(creator_layout)
    fake, calls = _fake_sbatch(tmp_path, ["31001;stanage"])
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    result = _run(
        env,
        config,
        "module-check",
        f"--confirm-output-root={config.output_root}",
    )

    assert result.returncode == 0, result.stderr
    assert "31001" in result.stdout
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "module_preflight.slurm" in lines[0]


def test_submit_wires_module_check_before_singleton_scaffold(
    creator_layout: dict[str, Path | str], tmp_path: Path
) -> None:
    config = _config(creator_layout)
    fake, calls = _fake_sbatch(tmp_path, ["32001;stanage", "32002"])
    env = os.environ.copy()
    env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})

    result = _run(
        env,
        config,
        "submit",
        f"--confirm-output-root={config.output_root}",
    )

    assert result.returncode == 0, result.stderr
    assert "module=32001" in result.stdout
    assert "scaffold=32002" in result.stdout
    lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "module_preflight.slurm" in lines[0]
    assert "--dependency=afterok:32001" in lines[1]
    assert "--array=0-0%1" in lines[1]
    assert f"SUBJECT={config.subject}" in lines[1]
    assert "scaffold_array.slurm" in lines[1]


def test_malformed_or_empty_scheduler_response_fails_closed(
    creator_layout: dict[str, Path | str], tmp_path: Path
) -> None:
    config = _config(creator_layout)
    for index, response in enumerate(("not-a-job", "")):
        run_root = tmp_path / str(index)
        run_root.mkdir()
        fake, _ = _fake_sbatch(run_root, [response])
        env = os.environ.copy()
        env.update({"PYTHON_BIN": sys.executable, "SBATCH_BIN": str(fake)})
        result = _run(
            env,
            config,
            "module-check",
            f"--confirm-output-root={config.output_root}",
        )
        assert result.returncode == 2
        assert "parse Slurm job ID" in result.stderr


def test_runbook_documents_version_retention_and_replacement_handoff() -> None:
    text = README.read_text(encoding="utf-8")
    assert "SimNIBS/3.2.6-foss-2023a" in text
    assert "MATLAB/2023b" in text
    assert "headreco all --noclean" in text
    assert "complete diagnostic" in text
    assert "does not submit FEM" in text
    assert "/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1" in text
    assert "--legacy-v3-root" in text
    assert "sub-CC510415" in text
    assert "Scheduler submission is not completion" in text
