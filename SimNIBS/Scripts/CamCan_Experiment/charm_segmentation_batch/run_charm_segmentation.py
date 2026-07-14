#!/usr/bin/env python3
"""Run CHARM segmentation only for one CamCan subject.

The runner deliberately stops before CHARM meshing and never invokes a SimNIBS
simulation. The generated tissue map is copied byte-for-byte into ``maps/``
and accompanied by a SHA-256 provenance record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIMEOUT_EXIT_CODE = 124
INCOMPLETE_EXIT_CODE = 125
INPUT_EXIT_CODE = 126
MAP_BASENAME = "tissue_labeling_upsampled.nii.gz"
SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9][A-Za-z0-9._-]*$")


class MissingInput(RuntimeError):
    """Raised when a subject's required source data are absent."""


class CharmTimeout(RuntimeError):
    """Raised when CHARM exceeds the configured wall-clock deadline."""


@dataclass(frozen=True)
class SubjectInputs:
    subject: str
    anat_dir: Path
    t1: Path
    t2: Path


def log_event(event: str, **fields: object) -> None:
    print(
        json.dumps({"event": event, **fields}, default=str, sort_keys=True), flush=True
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def copy_unchanged_atomic(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        source_hash = sha256_file(source)
        copied_hash = sha256_file(temporary)
        if copied_hash != source_hash:
            raise IOError(
                f"SHA-256 mismatch while copying {source} to {destination}: "
                f"{source_hash} != {copied_hash}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def read_subjects_file(path: Path) -> tuple[str, ...]:
    subjects: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            subject = raw_line.strip()
            if not subject or subject.startswith("#"):
                continue
            if not SUBJECT_PATTERN.fullmatch(subject):
                raise ValueError(
                    f"Invalid subject ID at {path}:{line_number}: {subject!r}"
                )
            if subject in seen:
                raise ValueError(
                    f"Duplicate subject ID at {path}:{line_number}: {subject}"
                )
            seen.add(subject)
            subjects.append(subject)
    return tuple(subjects)


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def resolve_subject_inputs(source_root: Path, subject: str) -> SubjectInputs:
    if not SUBJECT_PATTERN.fullmatch(subject):
        raise MissingInput(f"Invalid subject ID: {subject!r}")
    anat_dir = source_root / subject / "anat"
    t1 = first_existing(
        (anat_dir / f"{subject}_T1w.nii", anat_dir / f"{subject}_T1w.nii.gz")
    )
    t2 = first_existing(
        (anat_dir / f"{subject}_T2w.nii", anat_dir / f"{subject}_T2w.nii.gz")
    )
    missing: list[str] = []
    if t1 is None:
        missing.append(str(anat_dir / f"{subject}_T1w.nii[.gz]"))
    if t2 is None:
        missing.append(str(anat_dir / f"{subject}_T2w.nii[.gz]"))
    if missing:
        raise MissingInput("Missing required input(s): " + ", ".join(missing))
    return SubjectInputs(subject=subject, anat_dir=anat_dir, t1=t1, t2=t2)


def map_destination(out_root: Path, subject: str) -> Path:
    return out_root / "maps" / f"{subject}_CHARM_{MAP_BASENAME}"


def metadata_path(out_root: Path, subject: str) -> Path:
    return out_root / "metadata" / f"{subject}.json"


def find_charm_map(work_anat: Path, subject: str) -> Path:
    exact = work_anat / f"m2m_{subject}" / "label_prep" / MAP_BASENAME
    if exact.is_file():
        return exact
    matches = sorted(work_anat.glob(f"m2m_*/label_prep/{MAP_BASENAME}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"CHARM did not create {MAP_BASENAME} under {work_anat}"
        )
    raise RuntimeError(f"Multiple CHARM tissue maps found under {work_anat}: {matches}")


def completed_output_is_valid(out_root: Path, subject: str) -> bool:
    marker = metadata_path(out_root, subject)
    destination = map_destination(out_root, subject)
    if (
        not marker.is_file()
        or not destination.is_file()
        or destination.stat().st_size <= 0
    ):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and payload.get("subject") == subject
            and payload.get("archived_map_sha256") == sha256_file(destination)
        )
    except (OSError, ValueError, TypeError):
        return False


def run_with_timeout(command: list[str], *, cwd: Path, timeout_seconds: float) -> None:
    log_event("charm_start", command=command, cwd=cwd, timeout_seconds=timeout_seconds)
    process = subprocess.Popen(command, cwd=str(cwd), start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        log_event("charm_timeout", pid=process.pid, timeout_seconds=timeout_seconds)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise CharmTimeout(
            f"CHARM timed out after {timeout_seconds / 3600:.2f} hours"
        ) from exc
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    log_event("charm_exit", return_code=return_code)


def run_subject(
    *,
    source_root: Path,
    out_root: Path,
    subject: str,
    timeout_hours: float,
    charm_bin: str,
    force: bool,
    work_root: Path | None = None,
) -> Path:
    if source_root == out_root:
        raise ValueError("Source and output roots must be different directories")
    inputs = resolve_subject_inputs(source_root, subject)
    destination = map_destination(out_root, subject)
    if not force and completed_output_is_valid(out_root, subject):
        log_event("reuse_complete", subject=subject, map=destination)
        return destination

    # Once a regeneration starts, do not leave an older map/marker pair that a
    # dependent collector could mistake for the result of this attempt.
    destination.unlink(missing_ok=True)
    metadata_path(out_root, subject).unlink(missing_ok=True)

    temporary_root = work_root or out_root / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    subject_work = Path(
        tempfile.mkdtemp(prefix=f"charm_{subject}_", dir=temporary_root)
    )

    # CHARM segmentation is atlas-based. Registration and atlas initialization
    # are prerequisites used only inside this temporary directory. Neither
    # surfaces nor a mesh are requested, and only the final tissue NIfTI is
    # copied out before this directory is deleted.
    command = [
        charm_bin,
        subject,
        str(inputs.t1),
        str(inputs.t2),
        "--registerT2",
        "--initatlas",
        "--segment",
        "--forceqform",
    ]
    started_at = time.time()
    payload: dict[str, object]
    try:
        run_with_timeout(
            command, cwd=subject_work, timeout_seconds=timeout_hours * 3600
        )

        generated_map = find_charm_map(subject_work, subject)
        if generated_map.stat().st_size <= 0:
            raise RuntimeError(f"CHARM tissue map is empty: {generated_map}")
        generated_hash = copy_unchanged_atomic(generated_map, destination)
        archived_hash = sha256_file(destination)
        if archived_hash != generated_hash:
            raise RuntimeError(
                "Archived tissue map does not match CHARM's generated map"
            )

        payload = {
            "status": "complete",
            "subject": subject,
            "source_root": str(source_root),
            "source_anat": str(inputs.anat_dir),
            "source_t1": str(inputs.t1),
            "source_t2": str(inputs.t2),
            "charm_output_relative_path": f"m2m_{subject}/label_prep/{MAP_BASENAME}",
            "generated_map_sha256": generated_hash,
            "archived_map": str(destination),
            "archived_map_sha256": archived_hash,
            "archived_map_bytes": destination.stat().st_size,
            "charm_command": command,
            "surfaces_requested": False,
            "mesh_requested": False,
            "simulation_requested": False,
            "started_at_epoch": started_at,
            "completed_at_epoch": time.time(),
        }
    finally:
        shutil.rmtree(subject_work)

    payload["temporary_charm_outputs_deleted"] = not subject_work.exists()
    write_json_atomic(metadata_path(out_root, subject), payload)
    log_event(
        "segmentation_complete",
        subject=subject,
        map=destination,
        sha256=archived_hash,
        bytes=destination.stat().st_size,
    )
    return destination


def discover_subjects(
    *,
    source_root: Path,
    out_root: Path,
    subjects_file: Path,
    report_path: Path | None,
) -> int:
    if not source_root.is_dir():
        print(f"[ERROR] Source root not found: {source_root}", file=sys.stderr)
        return INPUT_EXIT_CODE
    if source_root == out_root:
        print("[ERROR] Source and output roots must be different.", file=sys.stderr)
        return INPUT_EXIT_CODE
    rows: list[dict[str, str]] = []
    ready_subjects: list[str] = []
    blocked = 0
    subject_dirs = sorted(
        path
        for path in source_root.glob("sub-*")
        if path.is_dir() and SUBJECT_PATTERN.fullmatch(path.name)
    )
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        try:
            inputs = resolve_subject_inputs(source_root, subject)
            ready_subjects.append(subject)
            rows.append(
                {
                    "subject": subject,
                    "t1": str(inputs.t1),
                    "t2": str(inputs.t2),
                    "status": "ready",
                    "message": "ready",
                }
            )
        except MissingInput as exc:
            blocked += 1
            rows.append(
                {
                    "subject": subject,
                    "t1": "",
                    "t2": "",
                    "status": "blocked",
                    "message": str(exc),
                }
            )

    if not ready_subjects:
        print(
            f"[ERROR] No subjects with both T1 and T2 inputs were found under {source_root}.",
            file=sys.stderr,
        )
        return INPUT_EXIT_CODE

    subjects_file.parent.mkdir(parents=True, exist_ok=True)
    subjects_file.write_text(
        "".join(f"{subject}\n" for subject in ready_subjects), encoding="utf-8"
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("subject", "t1", "t2", "status", "message"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"[INFO] CHARM discovery: candidate_directories={len(subject_dirs)} "
        f"ready={len(ready_subjects)} blocked={blocked} source={source_root} "
        f"output={out_root} subjects_file={subjects_file}"
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--subject")
    parser.add_argument("--subjects-file")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--preflight-report")
    parser.add_argument("--timeout-hours", type=float, default=7.5)
    parser.add_argument("--charm-bin", default="charm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--work-root")
    args = parser.parse_args(argv)
    if args.discover_only and not args.subjects_file:
        parser.error("--discover-only requires --subjects-file")
    if not args.discover_only and not args.subject:
        parser.error("--subject is required unless --discover-only is used")
    if args.timeout_hours <= 0:
        parser.error("--timeout-hours must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    try:
        if args.discover_only:
            return discover_subjects(
                source_root=source_root,
                out_root=out_root,
                subjects_file=Path(args.subjects_file).expanduser().resolve(),
                report_path=(
                    Path(args.preflight_report).expanduser().resolve()
                    if args.preflight_report
                    else None
                ),
            )
        run_subject(
            source_root=source_root,
            out_root=out_root,
            subject=args.subject,
            timeout_hours=args.timeout_hours,
            charm_bin=args.charm_bin,
            force=args.force,
            work_root=(
                Path(args.work_root).expanduser().resolve() if args.work_root else None
            ),
        )
        return 0
    except MissingInput as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return INPUT_EXIT_CODE
    except CharmTimeout as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return TIMEOUT_EXIT_CODE
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return INCOMPLETE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
