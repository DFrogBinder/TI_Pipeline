#!/usr/bin/env python3
"""
Flatten atlas-maker FastSurfer outputs into a post-processing atlas directory.

The atlas-maker wrappers write subject outputs as:

  <data_dir>/FastSurfer_out/<subject>/mri/aparc.a2009s+aseg.nii.gz

The post-processing pipeline's default FastSurfer lookup expects:

  <atlas_root>/<subject>.nii.gz
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
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
DEFAULT_SOURCE_RELATIVE = Path("mri/aparc.a2009s+aseg.nii.gz")
DEFAULT_SOURCE_CANDIDATES = (
    Path("mri/aparc.a2009s+aseg.nii.gz"),
    Path("mri/aparc.DKTatlas+aseg.deep.nii.gz"),
    Path("mri/aparc.DKTatlas+aseg.nii.gz"),
    Path("mri/aparc+aseg.nii.gz"),
    Path("mri/aseg.nii.gz"),
)
DEFAULT_MGZ_SOURCE_CANDIDATES = (
    Path("mri/aparc.a2009s+aseg.mgz"),
    Path("mri/aparc.DKTatlas+aseg.mgz"),
    Path("mri/aparc+aseg.mgz"),
    Path("mri/aseg.mgz"),
)
SKIP_DIR_NAMES = {"logs", "fsaverage"}
console = Console(markup=False)


@dataclass(frozen=True)
class AtlasExportItem:
    subject: str
    src: Path
    dst: Path
    size_bytes: int
    action: str = "copy"


@dataclass(frozen=True)
class MissingAtlas:
    subject: str
    expected: str
    reason: str
    available: str = ""


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


def output_suffix_for_candidates(
    source_candidates: Sequence[Path],
    output_suffix: str | None = None,
) -> str:
    if output_suffix:
        return output_suffix_for(source_candidates[0], output_suffix)
    return ".nii.gz"


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


def normalize_source_candidates(source_relative: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(source_relative, Path):
        return (source_relative,)
    return tuple(source_relative)


def normalize_optional_source_candidates(source_relative: Sequence[Path] | None) -> tuple[Path, ...]:
    return tuple(source_relative or ())


def find_first_existing_source(subject_dir: Path, source_candidates: Sequence[Path]) -> Path | None:
    for source_relative in source_candidates:
        candidate = subject_dir / source_relative
        if candidate.is_file():
            return candidate
    return None


def describe_expected_sources(subject_dir: Path, source_candidates: Sequence[Path]) -> str:
    return ", ".join(str(subject_dir / source_relative) for source_relative in source_candidates)


def is_mgz_path(path: Path) -> bool:
    return path.name.endswith(".mgz")


def describe_available_mri_files(subject_dir: Path, limit: int = 12) -> str:
    mri_dir = subject_dir / "mri"
    if not mri_dir.is_dir():
        return "mri directory is missing"

    files = sorted(path.name for path in mri_dir.iterdir() if path.is_file())
    if not files:
        return "mri directory contains no files"

    displayed = files[:limit]
    suffix = f"; ... {len(files) - limit} more" if len(files) > limit else ""
    return ", ".join(displayed) + suffix


def mri_dir_has_mgz_atlas(subject_dir: Path) -> bool:
    mri_dir = subject_dir / "mri"
    if not mri_dir.is_dir():
        return False
    return any(path.is_file() and "aseg" in path.name for path in mri_dir.glob("*.mgz"))


def build_atlas_export_plan(
    *,
    fastsurfer_out: Path,
    dest: Path,
    source_relative: Path | Sequence[Path] = DEFAULT_SOURCE_RELATIVE,
    mgz_source_relative: Sequence[Path] | None = None,
    convert_mgz: bool = True,
    subjects: Sequence[str] | None = None,
    output_suffix: str | None = None,
    on_subject_processed: Callable[[], None] | None = None,
) -> tuple[list[AtlasExportItem], list[MissingAtlas]]:
    source_candidates = normalize_source_candidates(source_relative)
    mgz_source_candidates = normalize_optional_source_candidates(mgz_source_relative)
    suffix = output_suffix_for_candidates(source_candidates, output_suffix)
    items: list[AtlasExportItem] = []
    missing: list[MissingAtlas] = []

    for subject_dir in _subject_dirs_for_request(fastsurfer_out, subjects):
        subject = subject_dir.name
        dst = dest / f"{subject}{suffix}"

        if not subject_dir.is_dir():
            missing.append(
                MissingAtlas(
                    subject=subject,
                    expected=describe_expected_sources(subject_dir, source_candidates),
                    reason="subject directory is missing",
                )
            )
            if on_subject_processed is not None:
                on_subject_processed()
            continue
        src = find_first_existing_source(subject_dir, source_candidates)
        action = "copy"
        if src is not None and is_mgz_path(src):
            action = "convert" if convert_mgz else "copy"
        if src is None and convert_mgz:
            src = find_first_existing_source(subject_dir, mgz_source_candidates)
            if src is not None:
                action = "convert"
        if src is None:
            available = describe_available_mri_files(subject_dir)
            if mri_dir_has_mgz_atlas(subject_dir):
                reason = (
                    "MGZ atlas exists but no matching NIfTI atlas was found; use "
                    "--source-relative for that MGZ file or enable --convert-mgz"
                )
            elif "aparc" not in available and "aseg" not in available:
                reason = "no atlas segmentation file found; recon/segmentation appears incomplete"
            else:
                reason = "atlas file is missing"
            missing.append(
                MissingAtlas(
                    subject=subject,
                    expected=describe_expected_sources(subject_dir, source_candidates),
                    reason=reason,
                    available=available,
                )
            )
            if on_subject_processed is not None:
                on_subject_processed()
            continue

        if not src.is_file():
            missing.append(
                MissingAtlas(
                    subject=subject,
                    expected=describe_expected_sources(subject_dir, source_candidates),
                    reason="atlas file is missing",
                    available=describe_available_mri_files(subject_dir),
                )
            )
            if on_subject_processed is not None:
                on_subject_processed()
            continue

        items.append(
            AtlasExportItem(
                subject=subject,
                src=src,
                dst=dst,
                size_bytes=src.stat().st_size,
                action=action,
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
    mri_convert: str = "mri_convert",
) -> None:
    if dry_run:
        for item in items:
            console.log(f"{item.action}: {item.src} -> {item.dst}")
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
        task = progress.add_task("Collecting atlas outputs", total=len(items))
        for item in items:
            if item.dst.exists() and not overwrite:
                raise FileExistsError(f"Destination exists: {item.dst}")
            if item.action == "convert":
                subprocess.run([mri_convert, str(item.src), str(item.dst)], check=True)
            else:
                shutil.copy2(item.src, item.dst)
            progress.advance(task)


def any_conversion_needed(items: Sequence[AtlasExportItem]) -> bool:
    return any(item.action == "convert" for item in items)


def resolve_executable(command: str) -> str | None:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") and candidate.is_file():
        return str(candidate)
    return shutil.which(command)


def write_manifest(path: Path, items: Sequence[AtlasExportItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "source", "destination", "size_bytes", "action"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "subject": item.subject,
                    "source": str(item.src),
                    "destination": str(item.dst),
                    "size_bytes": item.size_bytes,
                    "action": item.action,
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one atlas output per subject from FreeSurfer/FastSurfer output "
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
            "Relative atlas path inside each subject directory. If omitted, the "
            "extractor tries common DKT, aparc+aseg, a2009s, and aseg outputs."
        ),
    )
    parser.add_argument(
        "--try-default-atlas-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --source-relative is not set, try common FreeSurfer/FastSurfer "
            "atlas filenames instead of one hardcoded path."
        ),
    )
    parser.add_argument(
        "--convert-mgz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Convert common MGZ atlas outputs to flat .nii.gz files with mri_convert "
            "when matching NIfTI atlas files are absent."
        ),
    )
    parser.add_argument(
        "--mri-convert",
        default="mri_convert",
        help="mri_convert executable to use when --convert-mgz is enabled.",
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
            source_relative=(
                args.source_relative
                if args.source_relative != DEFAULT_SOURCE_RELATIVE or not args.try_default_atlas_candidates
                else DEFAULT_SOURCE_CANDIDATES
            ),
            mgz_source_relative=(
                None
                if args.source_relative != DEFAULT_SOURCE_RELATIVE or not args.try_default_atlas_candidates
                else DEFAULT_MGZ_SOURCE_CANDIDATES
            ),
            convert_mgz=args.convert_mgz,
            subjects=subjects or None,
            output_suffix=args.output_suffix,
            on_subject_processed=lambda: scan_progress.advance(scan_task),
        )

    console.log(f"[INFO] FastSurfer output root: {fastsurfer_out}")
    console.log(f"[INFO] Destination atlas root: {dest}")
    if args.source_relative != DEFAULT_SOURCE_RELATIVE or not args.try_default_atlas_candidates:
        console.log(f"[INFO] Source relative path:   {args.source_relative}")
    else:
        console.log("[INFO] Source candidates:      common FreeSurfer/FastSurfer atlas outputs")
        if args.convert_mgz:
            console.log("[INFO] MGZ conversion:        enabled for common atlas outputs")
    console.log(f"[INFO] Atlas files found:      {len(items)}")
    console.log(f"[INFO] Missing atlas files:    {len(missing)}")

    for entry in missing[:20]:
        console.log(f"[WARN] {entry.subject}: {entry.reason}")
        console.log(f"[WARN] Expected one of: {entry.expected}")
        if entry.available:
            console.log(f"[WARN] Available MRI files: {entry.available}")
    if len(missing) > 20:
        console.log(f"[WARN] ... {len(missing) - 20} additional missing atlas file(s).")

    if missing and args.require_all:
        return 1
    if not items:
        return 1
    if any_conversion_needed(items) and not args.dry_run:
        resolved_mri_convert = resolve_executable(args.mri_convert)
        if resolved_mri_convert is None:
            console.log(
                "[ERROR] MGZ conversion is required, but mri_convert was not found. "
                "Load FreeSurfer first or pass --mri-convert /path/to/mri_convert."
            )
            console.log(
                "[ERROR] On Stanage this is typically fixed with: "
                "module load FreeSurfer/7.4.1-centos7_x86_64"
            )
            return 2
        args.mri_convert = resolved_mri_convert
        console.log(f"[INFO] mri_convert:           {args.mri_convert}")

    copy_atlas_outputs(
        items,
        dest=dest,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
        mri_convert=args.mri_convert,
    )

    if args.write_manifest and not args.dry_run:
        write_manifest(dest / "atlas_export_manifest.csv", items)

    action = "Planned" if args.dry_run else "Copied"
    console.log(f"[INFO] {action} {len(items)} atlas file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
