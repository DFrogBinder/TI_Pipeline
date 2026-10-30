#!/usr/bin/env python3
"""Download one provenance-verified corrected-v4 mesh per selected subject.

The default source is repeat 01 of the completed Left_Hippocampus final-132
campaign. The head mesh is suitable for subsequent montage optimization and is
not itself target-specific. The script audits every remote mesh against its
completion marker before transfer and compares local and remote SHA-256 hashes
after transfer.

Audit only (default):

    python3 optimized_vs_unoptimized_age_cohort/download_hpc_meshes.py

Download all 14 meshes:

    python3 optimized_vs_unoptimized_age_cohort/download_hpc_meshes.py --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBJECTS_FILE = (
    REPO_ROOT
    / "CamCan_Experiment"
    / "cohort_pipeline"
    / "cohorts"
    / "optimized_vs_unoptimized_age_balanced_14"
    / "subjects.txt"
)
DEFAULT_SELECTION_MANIFEST = DEFAULT_SUBJECTS_FILE.with_name("selection_manifest.csv")
DEFAULT_LOCAL_ROOT = (
    REPO_ROOT
    / "optimized_vs_unoptimized_age_cohort"
    / "downloads"
    / "optimized_vs_unoptimized_age_balanced_14"
)
DEFAULT_SSH_TARGET = "cop23bi@stanage.shef.ac.uk"
DEFAULT_STUDY_ROOT = "/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Four_ROI"
DEFAULT_ROI = "Left_Hippocampus"
DEFAULT_REPEAT = "01"
EXPECTED_SUBJECTS = 14
SSH_CONTROL_PERSIST = "4h"
SUBJECT_PATTERN = re.compile(r"sub-CC[0-9]+\Z")
ROI_PATTERN = re.compile(r"[A-Za-z0-9_]+\Z")
REPEAT_PATTERN = re.compile(r"[0-9]{2}\Z")


REMOTE_AUDIT_PROGRAM = r"""
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


study_root = Path(sys.argv[1])
roi = sys.argv[2]
repeat_id = sys.argv[3]
subjects = sys.argv[4:]
dataset_name = f"{roi}_Data_{repeat_id}"
had_error = False

for subject in subjects:
    mesh = (
        study_root
        / "runs"
        / f"{roi}_Runs"
        / dataset_name
        / subject
        / "anat"
        / f"m2m_{subject}"
        / f"{subject}.msh"
    )
    marker_path = study_root / "results" / "meshes" / roi / repeat_id / f"{subject}.json"
    record = {
        "subject": subject,
        "roi": roi,
        "repeat_id": repeat_id,
        "dataset_name": dataset_name,
        "mesh_path": str(mesh),
        "marker_path": str(marker_path),
        "status": "error",
        "errors": [],
    }
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        mesh_bytes = mesh.stat().st_size
        mesh_sha256 = sha256_file(mesh)
        expected = {
            "status": "complete",
            "subject": subject,
            "roi": roi,
            "repeat_id": repeat_id,
            "dataset_name": dataset_name,
            "mesh_path": str(mesh),
            "independent_repeat_mesh": True,
            "scaffold_copy_mode": "physical",
            "roast_involvement": False,
        }
        for key, value in expected.items():
            if marker.get(key) != value:
                record["errors"].append(
                    f"marker {key}={marker.get(key)!r}; expected {value!r}"
                )
        if mesh_bytes <= 0:
            record["errors"].append("mesh is empty")
        if marker.get("mesh_bytes") != mesh_bytes:
            record["errors"].append(
                f"marker mesh_bytes={marker.get('mesh_bytes')!r}; actual={mesh_bytes}"
            )
        if marker.get("mesh_sha256") != mesh_sha256:
            record["errors"].append("mesh SHA-256 does not match completion marker")
        label_sha256 = marker.get("label_sha256_after")
        if not isinstance(label_sha256, str) or len(label_sha256) != 64:
            record["errors"].append("marker has no valid corrected-label SHA-256")
        record.update(
            {
                "mesh_bytes": mesh_bytes,
                "mesh_sha256": mesh_sha256,
                "corrected_label_sha256": label_sha256,
            }
        )
        if not record["errors"]:
            record["status"] = "ready"
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    if record["status"] != "ready":
        had_error = True
    print(json.dumps(record, sort_keys=True), flush=True)

raise SystemExit(2 if had_error else 0)
""".lstrip()


@dataclass(frozen=True)
class RemoteMesh:
    subject: str
    roi: str
    repeat_id: str
    dataset_name: str
    mesh_path: str
    marker_path: str
    mesh_bytes: int
    mesh_sha256: str
    corrected_label_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_subjects(path: Path, expected_count: int = EXPECTED_SUBJECTS) -> list[str]:
    subjects = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(subjects) != expected_count:
        raise ValueError(f"{path} contains {len(subjects)} subjects; expected {expected_count}.")
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"{path} contains duplicate subject IDs.")
    invalid = [subject for subject in subjects if SUBJECT_PATTERN.fullmatch(subject) is None]
    if invalid:
        raise ValueError(f"Invalid subject IDs in {path}: {invalid}")
    return subjects


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.subjects_file.is_file():
        raise ValueError(f"Subjects file not found: {args.subjects_file}")
    if not args.study_root.startswith("/") or any(char in args.study_root for char in "\n\r\t"):
        raise ValueError("--study-root must be an absolute remote path without control characters.")
    if ROI_PATTERN.fullmatch(args.roi) is None:
        raise ValueError(f"Invalid --roi value: {args.roi!r}")
    if REPEAT_PATTERN.fullmatch(args.repeat_id) is None:
        raise ValueError(f"Invalid --repeat value: {args.repeat_id!r}; expected two digits.")
    if not args.ssh_target or any(char in args.ssh_target for char in "\n\r\t"):
        raise ValueError("--ssh-target must be non-empty and contain no control characters.")
    if args.local_root == Path("/"):
        raise ValueError("Refusing to use / as --local-root.")
    for executable in ("ssh", "rsync"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required executable is not on PATH: {executable}")


def parse_remote_records(stdout: str, subjects: Sequence[str]) -> list[RemoteMesh]:
    raw_records: list[dict[str, object]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Remote audit emitted invalid JSON on line {line_number}: {line!r}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Remote audit line {line_number} is not a JSON object.")
        raw_records.append(value)

    by_subject = {str(record.get("subject")): record for record in raw_records}
    if len(raw_records) != len(by_subject):
        raise ValueError("Remote audit returned duplicate subject records.")
    if set(by_subject) != set(subjects):
        raise ValueError(
            "Remote audit subject mismatch: "
            f"missing={sorted(set(subjects) - set(by_subject))}, "
            f"extra={sorted(set(by_subject) - set(subjects))}"
        )

    errors = []
    output: list[RemoteMesh] = []
    for subject in subjects:
        record = by_subject[subject]
        if record.get("status") != "ready":
            errors.append(f"{subject}: {record.get('errors')}")
            continue
        try:
            output.append(
                RemoteMesh(
                    subject=subject,
                    roi=str(record["roi"]),
                    repeat_id=str(record["repeat_id"]),
                    dataset_name=str(record["dataset_name"]),
                    mesh_path=str(record["mesh_path"]),
                    marker_path=str(record["marker_path"]),
                    mesh_bytes=int(record["mesh_bytes"]),
                    mesh_sha256=str(record["mesh_sha256"]),
                    corrected_label_sha256=str(record["corrected_label_sha256"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{subject}: malformed ready record ({exc})")
    if errors:
        raise ValueError("Remote mesh audit failed:\n  " + "\n  ".join(errors))
    return output


def audit_remote_meshes(
    *,
    ssh_target: str,
    study_root: str,
    roi: str,
    repeat_id: str,
    subjects: Sequence[str],
    ssh_options: Sequence[str] = (),
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[RemoteMesh]:
    remote_command = shlex.join(["python3", "-", study_root, roi, repeat_id, *subjects])
    completed = run(
        ["ssh", *ssh_options, ssh_target, remote_command],
        input=REMOTE_AUDIT_PROGRAM,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        records = parse_remote_records(completed.stdout, subjects)
    except ValueError as exc:
        stderr = completed.stderr.strip()
        suffix = f"\nRemote stderr:\n{stderr}" if stderr else ""
        raise ValueError(f"{exc}{suffix}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"Remote audit exited {completed.returncode} despite parseable records."
            + (f"\nRemote stderr:\n{stderr}" if stderr else "")
        )
    return records


@contextmanager
def ssh_control_master(
    ssh_target: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Iterator[tuple[str, ...]]:
    """Authenticate once, then provide non-interactive options for child SSH calls."""
    with tempfile.TemporaryDirectory(prefix="ti-mesh-ssh-") as temporary_dir:
        control_path = Path(temporary_dir) / "control.sock"
        control_option = f"ControlPath={control_path}"
        open_command = [
            "ssh",
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPersist={SSH_CONTROL_PERSIST}",
            "-o",
            control_option,
            "-Nf",
            ssh_target,
        ]
        print("\nSSH authentication")
        print("  Approve the Stanage login once; all audit and transfer channels will reuse it.")
        opened = run(open_command, text=True, capture_output=True, check=False)
        if opened.returncode != 0:
            detail = opened.stderr.strip() or opened.stdout.strip()
            raise RuntimeError(
                f"Could not establish the shared SSH session (exit {opened.returncode})."
                + (f"\nSSH output:\n{detail}" if detail else "")
            )

        try:
            checked = run(
                ["ssh", "-o", control_option, "-O", "check", ssh_target],
                text=True,
                capture_output=True,
                check=False,
            )
            if checked.returncode != 0:
                detail = checked.stderr.strip() or checked.stdout.strip()
                raise RuntimeError(
                    "The shared SSH session was created but failed its control-socket check."
                    + (f"\nSSH output:\n{detail}" if detail else "")
                )
            yield (
                "-o",
                control_option,
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
            )
        finally:
            closed = run(
                ["ssh", "-o", control_option, "-O", "exit", ssh_target],
                text=True,
                capture_output=True,
                check=False,
            )
            if closed.returncode != 0:
                detail = closed.stderr.strip() or closed.stdout.strip()
                print(
                    "[WARNING] Could not explicitly close the shared SSH session; "
                    f"it will expire after {SSH_CONTROL_PERSIST}."
                    + (f" SSH output: {detail}" if detail else ""),
                    file=sys.stderr,
                )


def print_scope(args: argparse.Namespace, subjects: Sequence[str]) -> None:
    mode = "FULL DOWNLOAD" if args.apply else "FULL REMOTE AUDIT ONLY"
    print("Scope")
    print("  Cohort:           optimized_vs_unoptimized_age_balanced_14")
    print(f"  Subjects:         {len(subjects)}")
    print(f"  Remote mesh set:  {args.roi}_Data_{args.repeat_id}")
    print(f"  Meshes audited:   {len(subjects)}")
    print(f"  Meshes downloaded:{len(subjects) if args.apply else 0}")
    print(f"  Expected outputs: {len(subjects)} .msh files")
    print(f"  Mode:             {mode}")
    print("  SSH approvals:    1 (shared ControlMaster session)")
    print(f"  SSH target:       {args.ssh_target}")
    print(f"  Study root:       {args.study_root}")
    if args.roi != DEFAULT_ROI or args.repeat_id != DEFAULT_REPEAT:
        print(
            "  SOURCE DELTA:     "
            f"default {DEFAULT_ROI}_Data_{DEFAULT_REPEAT} -> "
            f"{args.roi}_Data_{args.repeat_id} (user override)"
        )


def print_remote_summary(records: Sequence[RemoteMesh]) -> None:
    print("\nRemote audit")
    print("  subject         size_MiB  sha256")
    for record in records:
        print(
            f"  {record.subject:<15} {record.mesh_bytes / (1024 * 1024):>8.1f}  "
            f"{record.mesh_sha256}"
        )
    total_bytes = sum(record.mesh_bytes for record in records)
    print(f"  TOTAL           {total_bytes / (1024 ** 3):>8.2f} GiB")


def run_rsync(
    ssh_target: str,
    record: RemoteMesh,
    destination: Path,
    ssh_options: Sequence[str] = (),
) -> None:
    remote_shell = shlex.join(["ssh", *ssh_options])
    command = [
        "rsync",
        "-e",
        remote_shell,
        "--times",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--info=progress2",
        "--protect-args",
        f"{ssh_target}:{record.mesh_path}",
        str(destination),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"rsync failed for {record.subject} with exit code {completed.returncode}.")


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.incoming")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def download_meshes(
    args: argparse.Namespace,
    subjects: Sequence[str],
    records: Sequence[RemoteMesh],
    ssh_options: Sequence[str] = (),
) -> None:
    local_root = args.local_root.resolve()
    mesh_dir = local_root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    record_by_subject = {record.subject: record for record in records}
    statuses: dict[str, str] = {}
    backups: dict[str, str] = {}
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    for subject in subjects:
        record = record_by_subject[subject]
        destination = mesh_dir / f"{subject}.msh"
        if destination.exists():
            existing_hash = sha256_file(destination)
            if existing_hash == record.mesh_sha256 and destination.stat().st_size == record.mesh_bytes:
                statuses[subject] = "verified_existing"
                print(f"[verified existing] {subject}")
                continue
            if not args.replace_mismatch:
                raise RuntimeError(
                    f"Existing local mesh differs for {subject}: {destination}. "
                    "Rerun with --replace-mismatch to preserve it as a backup and install the verified copy."
                )

        incoming = mesh_dir / f".{subject}.msh.incoming"
        print(f"[download] {subject}: {record.mesh_path}")
        run_rsync(args.ssh_target, record, incoming, ssh_options)
        if not incoming.is_file():
            raise RuntimeError(f"rsync returned success but did not create {incoming}.")
        incoming_hash = sha256_file(incoming)
        incoming_size = incoming.stat().st_size
        if incoming_size != record.mesh_bytes or incoming_hash != record.mesh_sha256:
            raise RuntimeError(
                f"Downloaded mesh verification failed for {subject}: "
                f"bytes={incoming_size}/{record.mesh_bytes}, sha256={incoming_hash}/{record.mesh_sha256}. "
                f"Unverified file retained at {incoming}."
            )
        if destination.exists():
            backup = mesh_dir / f"{subject}.msh.preexisting-{run_timestamp}.bak"
            os.replace(destination, backup)
            backups[subject] = str(backup)
        os.replace(incoming, destination)
        statuses[subject] = "downloaded_verified"

    expected_names = {f"{subject}.msh" for subject in subjects}
    actual_names = {path.name for path in mesh_dir.glob("*.msh")}
    if actual_names != expected_names:
        raise RuntimeError(
            "Local mesh directory does not contain exactly the expected cohort: "
            f"missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )

    for record in records:
        local_path = mesh_dir / f"{record.subject}.msh"
        if local_path.stat().st_size != record.mesh_bytes or sha256_file(local_path) != record.mesh_sha256:
            raise RuntimeError(f"Final local verification failed for {record.subject}.")

    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path = local_root / "download_manifest.tsv"
    manifest_temporary = local_root / ".download_manifest.tsv.incoming"
    with manifest_temporary.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "subject",
            "status",
            "source_ssh_target",
            "source_mesh_path",
            "source_marker_path",
            "roi",
            "repeat_id",
            "dataset_name",
            "mesh_bytes",
            "mesh_sha256",
            "corrected_label_sha256",
            "local_mesh_path",
            "completed_at_utc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for subject in subjects:
            record = record_by_subject[subject]
            writer.writerow(
                {
                    "subject": subject,
                    "status": statuses[subject],
                    "source_ssh_target": args.ssh_target,
                    "source_mesh_path": record.mesh_path,
                    "source_marker_path": record.marker_path,
                    "roi": record.roi,
                    "repeat_id": record.repeat_id,
                    "dataset_name": record.dataset_name,
                    "mesh_bytes": record.mesh_bytes,
                    "mesh_sha256": record.mesh_sha256,
                    "corrected_label_sha256": record.corrected_label_sha256,
                    "local_mesh_path": str((mesh_dir / f"{subject}.msh").resolve()),
                    "completed_at_utc": completed_at,
                }
            )
    os.replace(manifest_temporary, manifest_path)

    subject_text = "".join(f"{subject}\n" for subject in subjects)
    atomic_write_text(local_root / "subjects.txt", subject_text)
    checksums = "".join(
        f"{record_by_subject[subject].mesh_sha256}  meshes/{subject}.msh\n"
        for subject in subjects
    )
    atomic_write_text(local_root / "SHA256SUMS", checksums)

    source_manifest_hash = (
        sha256_file(DEFAULT_SELECTION_MANIFEST)
        if DEFAULT_SELECTION_MANIFEST.is_file()
        else None
    )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "cohort_id": "optimized_vs_unoptimized_age_balanced_14",
        "completed_at_utc": completed_at,
        "scope": {
            "subjects": len(subjects),
            "remote_meshes": len(records),
            "local_meshes": len(actual_names),
            "one_mesh_per_subject": True,
        },
        "source": {
            "ssh_target": args.ssh_target,
            "ssh_authentication": "temporary_control_master",
            "expected_interactive_approvals": 1,
            "study_root": args.study_root,
            "roi": args.roi,
            "repeat_id": args.repeat_id,
            "dataset_name": f"{args.roi}_Data_{args.repeat_id}",
            "subjects_file": str(args.subjects_file.resolve()),
            "subjects_file_sha256": sha256_file(args.subjects_file),
            "selection_manifest": (
                str(DEFAULT_SELECTION_MANIFEST.resolve())
                if DEFAULT_SELECTION_MANIFEST.is_file()
                else None
            ),
            "selection_manifest_sha256": source_manifest_hash,
        },
        "validation": {
            "all_remote_markers_complete": True,
            "all_remote_mesh_hashes_match_markers": True,
            "all_local_mesh_hashes_match_remote": True,
            "unexpected_local_meshes": [],
        },
        "replacement_backups": backups,
        "meshes": [
            {
                "subject": record.subject,
                "status": statuses[record.subject],
                "bytes": record.mesh_bytes,
                "sha256": record.mesh_sha256,
                "remote_path": record.mesh_path,
                "remote_marker": record.marker_path,
                "local_path": str((mesh_dir / f"{record.subject}.msh").resolve()),
            }
            for record in records
        ],
    }
    atomic_write_text(local_root / "download_receipt.json", json.dumps(receipt, indent=2) + "\n")

    print("\nDownload complete")
    print(f"  Mesh directory: {mesh_dir}")
    print(f"  Verified files: {len(actual_names)}")
    print(f"  Manifest:       {manifest_path}")
    print(f"  Checksums:      {local_root / 'SHA256SUMS'}")
    print(f"  Receipt:        {local_root / 'download_receipt.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and download one corrected-v4 final-132 mesh per subject for the "
            "optimized-vs-unoptimized age-balanced cohort. Default mode is read-only audit."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Download all 14 meshes after remote validation.")
    parser.add_argument(
        "--replace-mismatch",
        action="store_true",
        help="Preserve a differing local mesh as a timestamped backup, then install the verified remote copy.",
    )
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--study-root", default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--roi", default=DEFAULT_ROI)
    parser.add_argument("--repeat", dest="repeat_id", default=DEFAULT_REPEAT)
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS_FILE)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_inputs(args)
        subjects = read_subjects(args.subjects_file)
        print_scope(args, subjects)
        with ssh_control_master(args.ssh_target) as ssh_options:
            records = audit_remote_meshes(
                ssh_target=args.ssh_target,
                study_root=args.study_root,
                roi=args.roi,
                repeat_id=args.repeat_id,
                subjects=subjects,
                ssh_options=ssh_options,
            )
            print_remote_summary(records)
            if not args.apply:
                print("\nAudit passed. Rerun with --apply to download all 14 verified meshes.")
                return 0
            download_meshes(args, subjects, records, ssh_options)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
