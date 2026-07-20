import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "HPC_scripts" / "submit_delete_directory.sh"
WORKER = ROOT / "HPC_scripts" / "delete_directory.slurm"


def test_worker_deletes_only_matching_directory_and_writes_receipt(tmp_path):
    allowed_root = tmp_path / "scratch"
    target = allowed_root / "obsolete"
    log_dir = allowed_root / "logs"
    target.mkdir(parents=True)
    log_dir.mkdir()
    (target / "nested").mkdir()
    (target / "nested" / "payload.bin").write_bytes(b"payload")
    target_id = f"{target.stat().st_dev}:{target.stat().st_ino}"

    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": "12345",
            "TI_DELETE_TARGET": str(target.resolve()),
            "TI_DELETE_TARGET_ID": target_id,
            "TI_DELETE_ALLOWED_ROOT": str(allowed_root.resolve()),
            "TI_DELETE_LOG_DIR": str(log_dir.resolve()),
        }
    )
    completed = subprocess.run(
        ["bash", str(WORKER)], env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert not target.exists()
    receipt = (log_dir / "delete-directory-12345.receipt.tsv").read_text()
    assert "\tcomplete\t" in receipt
    assert "target deleted" in receipt


def test_worker_refuses_changed_target_identity(tmp_path):
    allowed_root = tmp_path / "scratch"
    target = allowed_root / "keep"
    log_dir = allowed_root / "logs"
    target.mkdir(parents=True)
    log_dir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": "12346",
            "TI_DELETE_TARGET": str(target.resolve()),
            "TI_DELETE_TARGET_ID": "0:0",
            "TI_DELETE_ALLOWED_ROOT": str(allowed_root.resolve()),
            "TI_DELETE_LOG_DIR": str(log_dir.resolve()),
        }
    )
    completed = subprocess.run(
        ["bash", str(WORKER)], env=env, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 2
    assert target.is_dir()
    assert "device/inode changed" in completed.stderr
    receipt = (log_dir / "delete-directory-12346.receipt.tsv").read_text()
    assert "\tfailed\t" in receipt


def test_submitter_requires_exact_confirmation(tmp_path):
    allowed_root = tmp_path / "scratch"
    target = allowed_root / "obsolete"
    target.mkdir(parents=True)

    completed = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--target",
            str(target),
            "--confirm",
            str(target) + "-typo",
            "--allowed-root",
            str(allowed_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert target.is_dir()
    assert "must exactly match" in completed.stderr


def test_submitter_submits_one_task_without_deleting_locally(tmp_path):
    allowed_root = tmp_path / "scratch"
    target = allowed_root / "obsolete"
    log_dir = allowed_root / "logs"
    target.mkdir(parents=True)
    fake_sbatch = tmp_path / "sbatch"
    fake_args = tmp_path / "sbatch.args"
    fake_sbatch.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$FAKE_SBATCH_ARGS"\nprintf \'24680\\n\'\n'
    )
    fake_sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update({"SBATCH_BIN": str(fake_sbatch), "FAKE_SBATCH_ARGS": str(fake_args)})
    completed = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--target",
            str(target),
            "--confirm",
            str(target),
            "--allowed-root",
            str(allowed_root),
            "--log-dir",
            str(log_dir),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert target.is_dir()
    assert "tasks: 1" in completed.stdout
    assert "Submitted deletion job: 24680" in completed.stdout
    assert "--nodes=1" in fake_args.read_text().splitlines()
    request = (log_dir / "delete-directory-24680.request.tsv").read_text()
    assert "\tsubmitted\t" in request
