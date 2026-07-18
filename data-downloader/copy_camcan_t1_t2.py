#!/usr/bin/env python3
"""Copy only original CamCAN T1w and T2w NIfTI images.

Expected source layout::

    SOURCE/
      sub-CC110033/
        anat/
          sub-CC110033_T1w.nii
          sub-CC110033_T2w.nii

The same subject/anat layout is created below DESTINATION.  Source files are
never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import stat
import sys
import threading
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:
    print(
        "ERROR: tqdm is required for progress tracking. "
        "Install it with: python3 -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(2)


MODALITIES = ("T1w", "T2w")


@dataclass(frozen=True)
class CopyTask:
    subject: str
    source: Path
    destination: Path
    size: int


@dataclass(frozen=True)
class CopyResult:
    task: CopyTask
    status: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy exact <subjectID>_T1w.nii and <subjectID>_T2w.nii files "
            "from a CamCAN directory into a local destination."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="CamCAN base directory containing sub-* subject directories",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Local destination directory (must not be inside source)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show exactly what would be copied without creating any files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace destination files that already exist",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="number of simultaneous copies (default: 4; use 1 for sequential)",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=32,
        metavar="N",
        help="number of simultaneous VPN metadata checks (default: 32)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="suppress per-image output while retaining summaries and warnings",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the interactive tqdm progress bar",
    )
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="return a nonzero exit status if any subject lacks T1w or T2w",
    )
    return parser.parse_args()


def is_relative_to(path: Path, possible_parent: Path) -> bool:
    """Compatibility helper for Python versions before Path.is_relative_to."""
    try:
        path.relative_to(possible_parent)
    except ValueError:
        return False
    return True


def validate_paths(
    source: Path, destination: Path, workers: int, scan_workers: int
) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()

    if not source.is_dir():
        raise ValueError(f"source is not a readable directory: {source}")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if scan_workers < 1:
        raise ValueError("--scan-workers must be at least 1")
    if source == destination or is_relative_to(destination, source):
        raise ValueError(
            "destination must not be the source directory or anywhere inside it"
        )
    return source, destination


def discover_subject(
    subject_directory: Path, destination: Path
) -> tuple[list[CopyTask], list[str]]:
    subject_tasks: list[CopyTask] = []
    missing: list[str] = []
    subject = subject_directory.name
    anat_directory = subject_directory / "anat"

    for modality in MODALITIES:
        filename = f"{subject}_{modality}.nii"
        image = anat_directory / filename
        try:
            image_stat = image.stat()
        except FileNotFoundError:
            missing.append(f"{subject}: {filename}")
            continue
        if not stat.S_ISREG(image_stat.st_mode):
            missing.append(f"{subject}: {filename} (not a regular file)")
            continue

        subject_tasks.append(
            CopyTask(
                subject=subject,
                source=image,
                destination=destination / subject / "anat" / filename,
                size=image_stat.st_size,
            )
        )

    return subject_tasks, missing


def discover_tasks(
    source: Path, destination: Path, scan_workers: int
) -> tuple[list[CopyTask], list[str], int]:
    with os.scandir(source) as entries:
        subject_directories = sorted(
            (
                Path(entry.path)
                for entry in entries
                if entry.name.startswith("sub-") and entry.is_dir()
            ),
            key=lambda entry: entry.name,
        )

    tasks: list[CopyTask] = []
    missing: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=scan_workers) as executor:
        futures = [
            executor.submit(discover_subject, subject_directory, destination)
            for subject_directory in subject_directories
        ]
        for future in concurrent.futures.as_completed(futures):
            subject_tasks, subject_missing = future.result()
            tasks.extend(subject_tasks)
            missing.extend(subject_missing)

    tasks.sort(key=lambda task: (task.subject, task.source.name))
    missing.sort()
    return tasks, missing, len(subject_directories)


def human_size(number_of_bytes: int) -> str:
    value = float(number_of_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def inspect_existing(task: CopyTask, overwrite: bool) -> CopyResult | None:
    destination = task.destination
    if not destination.exists():
        return None
    if not destination.is_file():
        return CopyResult(task, "failed", "destination exists but is not a file")
    if overwrite:
        return None
    try:
        destination_size = destination.stat().st_size
    except OSError as error:
        return CopyResult(task, "failed", str(error))
    if destination_size == task.size:
        return CopyResult(task, "skipped", "already exists with the same size")
    return CopyResult(
        task,
        "failed",
        (
            f"destination exists with size {destination_size}, expected {task.size}; "
            "use --overwrite to replace it"
        ),
    )


def copy_one(task: CopyTask, overwrite: bool) -> CopyResult:
    existing = inspect_existing(task, overwrite)
    if existing is not None:
        return existing

    temporary = task.destination.with_name(
        f".{task.destination.name}.part-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        task.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task.source, temporary)
        if temporary.stat().st_size != task.size:
            raise OSError(
                f"incomplete copy: wrote {temporary.stat().st_size} of {task.size} bytes"
            )
        os.replace(temporary, task.destination)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return CopyResult(task, "failed", str(error))

    return CopyResult(task, "copied")


def print_plan(tasks: Iterable[CopyTask], source: Path) -> None:
    for task in tasks:
        relative_source = task.source.relative_to(source)
        print(
            f"WOULD COPY  {relative_source} -> {task.destination} "
            f"({human_size(task.size)})"
        )


def print_missing(missing: list[str]) -> None:
    if not missing:
        return
    print(f"\nMissing expected images ({len(missing)}):", file=sys.stderr)
    for item in missing:
        print(f"  {item}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        source, destination = validate_paths(
            args.source, args.destination, args.workers, args.scan_workers
        )
        tasks, missing, subject_count = discover_tasks(
            source, destination, args.scan_workers
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    total_bytes = sum(task.size for task in tasks)
    print(f"Source:      {source}")
    print(f"Destination: {destination}")
    print(f"Subjects:    {subject_count}")
    print(f"Images:      {len(tasks)} ({human_size(total_bytes)})")

    if subject_count == 0:
        print("ERROR: no sub-* subject directories were found", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\nDry run; no directories or files will be created.\n")
        if not args.summary_only:
            print_plan(tasks, source)
        print_missing(missing)
        return 1 if args.require_both and missing else 0

    copied = 0
    skipped = 0
    failures: list[CopyResult] = []
    processed_bytes = 0
    show_progress = not args.no_progress and sys.stderr.isatty()
    with tqdm(
        total=len(tasks),
        desc="T1/T2 progress",
        unit="image",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(copy_one, task, args.overwrite) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                processed_bytes += result.task.size
                if result.status == "copied":
                    copied += 1
                    if not args.summary_only and not show_progress:
                        print(f"COPIED   {result.task.destination}")
                elif result.status == "skipped":
                    skipped += 1
                    if not args.summary_only and not show_progress:
                        print(
                            f"SKIPPED  {result.task.destination} ({result.detail})"
                        )
                else:
                    failures.append(result)
                    failure_message = (
                        f"FAILED   {result.task.destination} ({result.detail})"
                    )
                    if show_progress:
                        tqdm.write(failure_message, file=sys.stderr)
                    else:
                        print(failure_message, file=sys.stderr)

                progress.update(1)
                progress.set_postfix(
                    copied=copied,
                    skipped=skipped,
                    failed=len(failures),
                    data=f"{human_size(processed_bytes)}/{human_size(total_bytes)}",
                    refresh=False,
                )

    print_missing(missing)
    print(
        f"\nFinished: {copied} copied, {skipped} skipped, "
        f"{len(failures)} failed, {len(missing)} missing."
    )

    if failures or (args.require_both and missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
