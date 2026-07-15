#!/usr/bin/env python3
"""Audit and install subject-matched CHARM tissue maps into repeat datasets.

The default mode is read-only with respect to the dataset tree.  ``--apply``
is accepted only after a complete 4-ROI/repeat/subject preflight and requires a
separate backup root.  Existing maps are backed up before atomic replacement.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SOURCE_SUFFIX = "_CHARM_tissue_labeling_upsampled.nii.gz"
TARGET_BASENAME = "tissue_labeling_upsampled.nii.gz"
DEFAULT_ROIS = (
    "Left_Hippocampus",
    "Left_M1",
    "Right_DLPC",
    "Right_Thalamus",
)
DEFAULT_REPEATS = tuple(range(1, 11))
DEFAULT_EXPECTED_SUBJECTS = 175
SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class InstallTask:
    dataset_name: str
    dataset_root: Path
    subject: str
    source: Path
    destination: Path


@dataclass(frozen=True)
class PreflightPlan:
    maps_root: Path
    roi_root: Path
    source_maps: tuple[tuple[str, Path], ...]
    subjects: tuple[str, ...]
    datasets: tuple[Path, ...]
    tasks: tuple[InstallTask, ...]
    issues: tuple[dict[str, str], ...]
    expected_dataset_count: int
    expected_task_count: int

    @property
    def ready(self) -> bool:
        return not self.issues and len(self.tasks) == self.expected_task_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_tsv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_repeats(value: str) -> tuple[int, ...]:
    repeats: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid repeat range: {part}")
            repeats.update(range(start, end + 1))
        else:
            repeats.add(int(part))
    if not repeats or min(repeats) < 0:
        raise ValueError("Repeats must be a non-empty set of non-negative integers")
    return tuple(sorted(repeats))


def expected_dataset_names(
    rois: Sequence[str], repeats: Sequence[int]
) -> tuple[str, ...]:
    return tuple(f"{roi}_Data_{repeat:02d}" for roi in rois for repeat in repeats)


def discover_source_maps(
    maps_root: Path,
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    sources: dict[str, Path] = {}
    issues: list[dict[str, str]] = []
    for path in sorted(maps_root.rglob(f"*{SOURCE_SUFFIX}")):
        if path.is_symlink() or not path.is_file():
            issues.append(
                {
                    "kind": "invalid_source",
                    "dataset": "",
                    "subject": "",
                    "path": str(path),
                    "message": "source must be a regular, non-symlink file",
                }
            )
            continue
        subject = path.name.removesuffix(SOURCE_SUFFIX)
        if not SUBJECT_PATTERN.fullmatch(subject):
            issues.append(
                {
                    "kind": "invalid_source_name",
                    "dataset": "",
                    "subject": subject,
                    "path": str(path),
                    "message": f"expected <subject>{SOURCE_SUFFIX}",
                }
            )
            continue
        if path.stat().st_size <= 0:
            issues.append(
                {
                    "kind": "empty_source",
                    "dataset": "",
                    "subject": subject,
                    "path": str(path),
                    "message": "source map is empty",
                }
            )
            continue
        if subject in sources:
            issues.append(
                {
                    "kind": "duplicate_source",
                    "dataset": "",
                    "subject": subject,
                    "path": str(path),
                    "message": f"also found {sources[subject]}",
                }
            )
            continue
        sources[subject] = path.resolve()
    return sources, issues


def discover_dataset_dirs(
    roi_root: Path, names: Sequence[str]
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    wanted = set(names)
    matches: dict[str, list[Path]] = {name: [] for name in names}
    for path in roi_root.rglob("*_Data_*"):
        if path.is_dir() and path.name in wanted:
            matches[path.name].append(path.resolve())

    datasets: dict[str, Path] = {}
    issues: list[dict[str, str]] = []
    for name in names:
        found = sorted(set(matches[name]))
        if len(found) == 1:
            datasets[name] = found[0]
        elif not found:
            issues.append(
                {
                    "kind": "missing_dataset",
                    "dataset": name,
                    "subject": "",
                    "path": str(roi_root),
                    "message": "expected dataset directory was not found recursively",
                }
            )
        else:
            issues.append(
                {
                    "kind": "ambiguous_dataset",
                    "dataset": name,
                    "subject": "",
                    "path": ";".join(str(path) for path in found),
                    "message": "multiple directories have the expected dataset name",
                }
            )
    return datasets, issues


def find_target_label(anat_dir: Path, subject: str) -> tuple[Path | None, str]:
    suffix = subject.split("-", 1)[-1].upper()
    candidates = {
        anat_dir / f"m2m_{subject}" / "label_prep" / TARGET_BASENAME,
        anat_dir / f"m2m_sub-{suffix}" / "label_prep" / TARGET_BASENAME,
    }
    exact = sorted(path for path in candidates if path.is_file())
    if len(exact) == 1:
        return exact[0], ""
    if len(exact) > 1:
        return None, "multiple subject-specific m2m label maps were found"

    fallback = sorted(anat_dir.glob(f"m2m_*/label_prep/{TARGET_BASENAME}"))
    if len(fallback) == 1 and fallback[0].is_file():
        return fallback[0], ""
    if not fallback:
        return None, f"no m2m_*/label_prep/{TARGET_BASENAME} was found"
    return None, "multiple fallback m2m label maps were found"


def build_preflight_plan(
    *,
    maps_root: Path,
    roi_root: Path,
    rois: Sequence[str] = DEFAULT_ROIS,
    repeats: Sequence[int] = DEFAULT_REPEATS,
    expected_subjects: int = DEFAULT_EXPECTED_SUBJECTS,
) -> PreflightPlan:
    maps_root = maps_root.expanduser().resolve()
    roi_root = roi_root.expanduser().resolve()
    issues: list[dict[str, str]] = []
    if not maps_root.is_dir():
        issues.append(
            {
                "kind": "missing_maps_root",
                "dataset": "",
                "subject": "",
                "path": str(maps_root),
                "message": "maps root is not a directory",
            }
        )
        sources: dict[str, Path] = {}
    else:
        sources, source_issues = discover_source_maps(maps_root)
        issues.extend(source_issues)
    if len(sources) != expected_subjects:
        issues.append(
            {
                "kind": "source_count_mismatch",
                "dataset": "",
                "subject": "",
                "path": str(maps_root),
                "message": f"found {len(sources)} source maps; expected {expected_subjects}",
            }
        )

    names = expected_dataset_names(rois, repeats)
    if not roi_root.is_dir():
        issues.append(
            {
                "kind": "missing_roi_root",
                "dataset": "",
                "subject": "",
                "path": str(roi_root),
                "message": "ROI root is not a directory",
            }
        )
        datasets: dict[str, Path] = {}
    else:
        datasets, dataset_issues = discover_dataset_dirs(roi_root, names)
        issues.extend(dataset_issues)

    source_subjects = set(sources)
    tasks: list[InstallTask] = []
    for name in names:
        dataset = datasets.get(name)
        if dataset is None:
            continue
        dataset_subjects = {
            path.name
            for path in dataset.glob("sub-*")
            if path.is_dir() and SUBJECT_PATTERN.fullmatch(path.name)
        }
        missing = sorted(source_subjects - dataset_subjects)
        extra = sorted(dataset_subjects - source_subjects)
        if missing:
            issues.append(
                {
                    "kind": "subjects_missing_from_dataset",
                    "dataset": name,
                    "subject": ";".join(missing),
                    "path": str(dataset),
                    "message": f"{len(missing)} source subject(s) are absent",
                }
            )
        if extra:
            issues.append(
                {
                    "kind": "subjects_missing_source_map",
                    "dataset": name,
                    "subject": ";".join(extra),
                    "path": str(dataset),
                    "message": f"{len(extra)} dataset subject(s) have no source map",
                }
            )
        for subject in sorted(source_subjects & dataset_subjects):
            anat_dir = dataset / subject / "anat"
            destination, message = find_target_label(anat_dir, subject)
            if destination is None:
                issues.append(
                    {
                        "kind": "missing_or_ambiguous_target",
                        "dataset": name,
                        "subject": subject,
                        "path": str(anat_dir),
                        "message": message,
                    }
                )
                continue
            if destination.is_symlink() or destination.stat().st_size <= 0:
                issues.append(
                    {
                        "kind": "invalid_target",
                        "dataset": name,
                        "subject": subject,
                        "path": str(destination),
                        "message": "target must be a non-empty, non-symlink file",
                    }
                )
                continue
            tasks.append(
                InstallTask(
                    dataset_name=name,
                    dataset_root=dataset,
                    subject=subject,
                    source=sources[subject],
                    destination=destination.resolve(),
                )
            )

    return PreflightPlan(
        maps_root=maps_root,
        roi_root=roi_root,
        source_maps=tuple(sorted(sources.items())),
        subjects=tuple(sorted(sources)),
        datasets=tuple(datasets[name] for name in names if name in datasets),
        tasks=tuple(tasks),
        issues=tuple(issues),
        expected_dataset_count=len(names),
        expected_task_count=len(names) * expected_subjects,
    )


def source_hashes(plan: PreflightPlan) -> dict[str, str]:
    return {subject: sha256_file(path) for subject, path in plan.source_maps}


def write_preflight_reports(
    plan: PreflightPlan,
    report_dir: Path,
    hashes: dict[str, str],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        report_dir / "sources.tsv",
        ("subject", "source", "bytes", "sha256"),
        (
            {
                "subject": subject,
                "source": source,
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
            for subject, source in plan.source_maps
            for digest in (hashes.get(subject, ""),)
        ),
    )
    write_tsv(
        report_dir / "targets.tsv",
        ("dataset", "subject", "source", "destination", "source_sha256", "status"),
        (
            {
                "dataset": task.dataset_name,
                "subject": task.subject,
                "source": task.source,
                "destination": task.destination,
                "source_sha256": hashes.get(task.subject, ""),
                "status": "ready" if plan.ready else "blocked",
            }
            for task in plan.tasks
        ),
    )
    write_tsv(
        report_dir / "issues.tsv",
        ("kind", "dataset", "subject", "path", "message"),
        plan.issues,
    )
    write_json_atomic(
        report_dir / "preflight_summary.json",
        {
            "status": "ready" if plan.ready else "blocked",
            "maps_root": plan.maps_root,
            "roi_root": plan.roi_root,
            "source_subjects": len(plan.subjects),
            "datasets_found": len(plan.datasets),
            "datasets_expected": plan.expected_dataset_count,
            "targets_ready": len(plan.tasks),
            "targets_expected": plan.expected_task_count,
            "issues": len(plan.issues),
            "meshes_changed": False,
            "simulation_outputs_changed": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def backup_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
            raise
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise IOError(f"backup verification failed for {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "copy"


def atomic_install(source: Path, destination: Path, expected_hash: str) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.charm-install-",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        installed_hash = sha256_file(temporary)
        if installed_hash != expected_hash:
            raise IOError(
                f"temporary install hash mismatch: {installed_hash} != {expected_hash}"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class ApplyLock:
    def __init__(self, roi_root: Path):
        self.path = roi_root / ".charm-segmentation-install.lock"

    def __enter__(self) -> "ApplyLock":
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                f"another apply may be active (or left a stale lock): {self.path}"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"host={socket.gethostname()} pid={os.getpid()}\n")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.path.unlink(missing_ok=True)


def apply_plan(
    plan: PreflightPlan,
    *,
    hashes: dict[str, str],
    backup_root: Path,
    report_dir: Path,
) -> dict[str, int | str]:
    if not plan.ready:
        raise ValueError("refusing --apply because preflight is blocked")
    backup_root = backup_root.expanduser().resolve()
    if backup_root == plan.roi_root or backup_root.is_relative_to(plan.roi_root):
        raise ValueError(
            "backup root must be outside the ROI root so backups cannot be "
            "rediscovered as duplicate datasets"
        )
    backup_root.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "apply_manifest.tsv"
    if manifest_path.exists():
        raise FileExistsError(
            f"apply manifest already exists; use a new report directory: {manifest_path}"
        )

    fieldnames = (
        "dataset",
        "subject",
        "source",
        "destination",
        "backup",
        "backup_mode",
        "before_sha256",
        "installed_sha256",
        "status",
        "message",
    )
    installed = 0
    already_current = 0
    failed = 0
    with (
        ApplyLock(plan.roi_root),
        manifest_path.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        for index, task in enumerate(plan.tasks, start=1):
            expected_hash = hashes[task.subject]
            before_hash = ""
            backup_path = backup_root / task.destination.relative_to(plan.roi_root)
            backup_mode = ""
            status = "failed"
            message = ""
            try:
                before_hash = sha256_file(task.destination)
                if before_hash == expected_hash:
                    status = "already_current"
                    already_current += 1
                    backup_path_text = ""
                else:
                    if backup_path.exists():
                        if sha256_file(backup_path) != before_hash:
                            raise IOError(
                                f"existing backup does not match current target: {backup_path}"
                            )
                        backup_mode = "existing"
                    else:
                        backup_mode = backup_file(task.destination, backup_path)
                    if sha256_file(backup_path) != before_hash:
                        raise IOError(f"backup hash mismatch: {backup_path}")
                    atomic_install(task.source, task.destination, expected_hash)
                    if sha256_file(task.destination) != expected_hash:
                        raise IOError(
                            f"installed target hash mismatch: {task.destination}"
                        )
                    status = "installed"
                    installed += 1
                    backup_path_text = str(backup_path)
            except Exception as exc:
                failed += 1
                message = str(exc)
                backup_path_text = str(backup_path) if backup_path.exists() else ""
            writer.writerow(
                {
                    "dataset": task.dataset_name,
                    "subject": task.subject,
                    "source": task.source,
                    "destination": task.destination,
                    "backup": backup_path_text,
                    "backup_mode": backup_mode,
                    "before_sha256": before_hash,
                    "installed_sha256": expected_hash if status != "failed" else "",
                    "status": status,
                    "message": message,
                }
            )
            handle.flush()
            os.fsync(handle.fileno())
            if index % 100 == 0 or status == "failed":
                print(
                    f"[INFO] apply progress {index}/{len(plan.tasks)}: "
                    f"installed={installed} current={already_current} failed={failed}",
                    flush=True,
                )
            if status == "failed":
                break

    summary: dict[str, int | str] = {
        "status": "complete" if failed == 0 else "failed",
        "targets_expected": len(plan.tasks),
        "installed": installed,
        "already_current": already_current,
        "failed": failed,
        "backup_root": str(backup_root),
        "manifest": str(manifest_path),
    }
    write_json_atomic(report_dir / "apply_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument(
        "--roi-root",
        type=Path,
        required=True,
        help="Root searched recursively for the 40 <ROI>_Data_<repeat> directories.",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--expected-subjects", type=int, default=DEFAULT_EXPECTED_SUBJECTS
    )
    parser.add_argument("--repeats", default="1-10")
    parser.add_argument("--rois", nargs="+", default=list(DEFAULT_ROIS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_subjects < 1:
        print("[ERROR] --expected-subjects must be positive", file=sys.stderr)
        return 2
    try:
        repeats = parse_repeats(args.repeats)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.apply and args.backup_root is None:
        print("[ERROR] --backup-root is required with --apply", file=sys.stderr)
        return 2

    report_dir = args.report_dir.expanduser().resolve()
    plan = build_preflight_plan(
        maps_root=args.maps_root,
        roi_root=args.roi_root,
        rois=args.rois,
        repeats=repeats,
        expected_subjects=args.expected_subjects,
    )
    hashes = source_hashes(plan) if plan.source_maps else {}
    write_preflight_reports(plan, report_dir, hashes)
    print(
        f"[INFO] preflight status={'ready' if plan.ready else 'blocked'} "
        f"sources={len(plan.subjects)}/{args.expected_subjects} "
        f"datasets={len(plan.datasets)}/{plan.expected_dataset_count} "
        f"targets={len(plan.tasks)}/{plan.expected_task_count} "
        f"issues={len(plan.issues)}",
        flush=True,
    )
    print(f"[INFO] reports: {report_dir}", flush=True)
    if not plan.ready:
        print(
            f"[ERROR] Refusing installation; inspect {report_dir / 'issues.tsv'}",
            file=sys.stderr,
        )
        return 2
    if not args.apply:
        print(
            "[INFO] Audit only: no dataset maps, meshes, or simulations were changed."
        )
        return 0

    try:
        result = apply_plan(
            plan,
            hashes=hashes,
            backup_root=args.backup_root,
            report_dir=report_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(
        f"[INFO] apply status={result['status']} installed={result['installed']} "
        f"already_current={result['already_current']} failed={result['failed']}",
        flush=True,
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
