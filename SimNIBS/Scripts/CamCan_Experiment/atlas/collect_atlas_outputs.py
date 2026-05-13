#!/usr/bin/env python3
"""
Flatten atlas-maker FastSurfer outputs into a post-processing atlas directory.

The atlas-maker wrappers write subject outputs as:

  <data_dir>/FastSurfer_out/<subject>/mri/aparc.DKTatlas+aseg.deep.nii.gz

The post-processing pipeline's default FastSurfer lookup expects:

  <atlas_root>/<subject>.nii.gz
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


DEFAULT_FASTSURFER_OUT_DIR_NAME = "FastSurfer_out"
DEFAULT_DEST_DIR_NAME = "atlases"
DEFAULT_SOURCE_RELATIVE = Path("mri/aparc.DKTatlas+aseg.deep.nii.gz")
SKIP_DIR_NAMES = {"logs", "fsaverage"}
console = Console(markup=False)


@dataclass(frozen=True)
class AtlasExportItem:
    subject: str
    src: Path
    dst: Path
    size_bytes: int


@dataclass(frozen=True)
class MissingAtlas:
    subject: str
    expected: Path
    reason: str


def parse_subjects(raw: str) -> list[str]:
    return [item for item in raw.replace(",", " ").split() if item]


def load_subjects_file(path: Path) -> list[str]:
    subjects: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            subjects.extend(parse_subjects(stripped))
    return subjects


def unique_sorted(subjects: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(subjects))


def resolve_fastsurfer_out_root(root: Path, output_dir_name: str = DEFAULT_FASTSURFER_OUT_DIR_NAME) -> Path:
    """
    Accept either the FastSurfer_out directory itself or its parent data directory.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"Source directory not found: {root}")

    nested = root / output_dir_name
    if nested.is_dir():
        return nested

    return root


def default_dest_for(fastsurfer_out: Path, dest_name: str = DEFAULT_DEST_DIR_NAME) -> Path:
    return fastsurfer_out.parent / dest_name


def output_suffix_for(source_relative: Path, output_suffix: str | None = None) -> str:
    if output_suffix:
        return output_suffix if output_suffix.startswith(".") else f".{output_suffix}"

    name = source_relative.name
    if name.endswith(".nii.gz"):
        return ".nii.gz"

    suffixes = source_relative.suffixes
    if suffixes:
        return "".join(suffixes)
    return ""


def discover_subject_dirs(fastsurfer_out: Path) -> list[Path]:
    return sorted(
        path
        for path in fastsurfer_out.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in SKIP_DIR_NAMES
    )


def _subject_dirs_for_request(fastsurfer_out: Path, subjects: Sequence[str] | None) -> list[Path]:
    if subjects:
        return [fastsurfer_out / subject for subject in unique_sorted(subjects)]
    return discover_subject_dirs(fastsurfer_out)


def build_atlas_export_plan(
    *,
    fastsurfer_out: Path,
    dest: Path,
    source_relative: Path = DEFAULT_SOURCE_RELATIVE,
    subjects: Sequence[str] | None = None,
    output_suffix: str | None = None,
    on_subject_processed: Callable[[], None] | None = None,
) -> tuple[list[AtlasExportItem], list[MissingAtlas]]:
    suffix = output_suffix_for(source_relative, output_suffix)
    items: list[AtlasExportItem] = []
    missing: list[MissingAtlas] = []

    for subject_dir in _subject_dirs_for_request(fastsurfer_out, subjects):
        subject = subject_dir.name
        src = subject_dir / source_relative
        dst = dest / f"{subject}{suffix}"

        if not subject_dir.is_dir():
            missing.append(
                MissingAtlas(subject=subject, expected=src, reason="subject directory is missing")
            )
            if on_subject_processed is not None:
                on_subject_processed()
            continue
        if not src.is_file():
            missing.append(MissingAtlas(subject=subject, expected=src, reason="atlas file is missing"))
            if on_subject_processed is not None:
                on_subject_processed()
            continue

        items.append(
            AtlasExportItem(
                subject=subject,
                src=src,
                dst=dst,
                size_bytes=src.stat().st_size,
            )
        )
        if on_subject_processed is not None:
            on_subject_processed()

    items.sort(key=lambda item: item.subject)
    missing.sort(key=lambda item: item.subject)
    return items, missing


def copy_atlas_outputs(
    items: Sequence[AtlasExportItem],
    *,
    dest: Path,
    dry_run: bool = False,
    overwrite: bool = True,
) -> None:
    if dry_run:
        for item in items:
            console.log(f"{item.src} -> {item.dst}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Copying atlas outputs", total=len(items))
        for item in items:
            if item.dst.exists() and not overwrite:
                raise FileExistsError(f"Destination exists: {item.dst}")
            shutil.copy2(item.src, item.dst)
            progress.advance(task)


def write_manifest(path: Path, items: Sequence[AtlasExportItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "source", "destination", "size_bytes"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "subject": item.subject,
                    "source": str(item.src),
                    "destination": str(item.dst),
                    "size_bytes": item.size_bytes,
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one atlas output per subject from atlas-maker/FastSurfer output "
            "folders into a flat atlas root named <subject>.nii.gz."
        )
    )
    parser.add_argument(
        "root",
        help=(
            "FastSurfer_out directory, or the parent data directory containing "
            "a FastSurfer_out child."
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help=(
            "Destination atlas directory. Default: create an 'atlases' sibling "
            "next to FastSurfer_out."
        ),
    )
    parser.add_argument(
        "--dest-name",
        default=DEFAULT_DEST_DIR_NAME,
        help="Destination directory name used when --dest is omitted.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_FASTSURFER_OUT_DIR_NAME,
        help="FastSurfer output directory name to detect when root is the parent data directory.",
    )
    parser.add_argument(
        "--source-relative",
        type=Path,
        default=DEFAULT_SOURCE_RELATIVE,
        help=(
            "Relative atlas path inside each subject directory. "
            "Default: mri/aparc.DKTatlas+aseg.deep.nii.gz"
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Suffix for destination files. Default: infer from --source-relative "
            "(.nii.gz for the default atlas)."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to collect. Accepts space-separated values.",
    )
    parser.add_argument(
        "--subjects-file",
        type=Path,
        default=None,
        help="Optional text file of subject IDs, one or more per line.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit with an error if any requested/discovered subject is missing the atlas file.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if a destination subject atlas already exists.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write atlas_export_manifest.csv in the destination directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without creating the destination directory or copying files.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    input_root = Path(args.root).expanduser().resolve()
    fastsurfer_out = resolve_fastsurfer_out_root(input_root, args.output_dir_name)
    dest = (
        Path(args.dest).expanduser().resolve()
        if args.dest
        else default_dest_for(fastsurfer_out, args.dest_name).resolve()
    )

    subjects: list[str] = []
    if args.subjects:
        subjects.extend(args.subjects)
    if args.subjects_file:
        subjects.extend(load_subjects_file(args.subjects_file.expanduser()))
    subjects = unique_sorted(subjects)

    subject_dirs = _subject_dirs_for_request(fastsurfer_out, subjects or None)
    scan_progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with scan_progress:
        scan_task = scan_progress.add_task(
            "Scanning subject atlases",
            total=max(1, len(subject_dirs)),
        )
        items, missing = build_atlas_export_plan(
            fastsurfer_out=fastsurfer_out,
            dest=dest,
            source_relative=args.source_relative,
            subjects=subjects or None,
            output_suffix=args.output_suffix,
            on_subject_processed=lambda: scan_progress.advance(scan_task),
        )

    console.log(f"[INFO] FastSurfer output root: {fastsurfer_out}")
    console.log(f"[INFO] Destination atlas root: {dest}")
    console.log(f"[INFO] Source relative path:   {args.source_relative}")
    console.log(f"[INFO] Atlas files found:      {len(items)}")
    console.log(f"[INFO] Missing atlas files:    {len(missing)}")

    for entry in missing[:20]:
        console.log(f"[WARN] {entry.subject}: {entry.reason}: {entry.expected}")
    if len(missing) > 20:
        console.log(f"[WARN] ... {len(missing) - 20} additional missing atlas file(s).")

    if missing and args.require_all:
        return 1
    if not items:
        return 1

    copy_atlas_outputs(
        items,
        dest=dest,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
    )

    if args.write_manifest and not args.dry_run:
        write_manifest(dest / "atlas_export_manifest.csv", items)

    action = "Planned" if args.dry_run else "Copied"
    console.log(f"[INFO] {action} {len(items)} atlas file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
