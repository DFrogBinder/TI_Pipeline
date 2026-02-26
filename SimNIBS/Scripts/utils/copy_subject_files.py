#!/usr/bin/env python3
"""Copy selected files for subjects listed in a CSV/XLSX file.

Example:
    python copy_subject_files.py \
      --input subjects.xlsx \
      --root /data/source_subjects \
      --dest /data/selected_files
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from tqdm import tqdm

DEFAULT_PATTERNS = [
    "anat/{id}_T2w.nii",
    "anat/{id}_T1w.nii",
    "anat/{id}__T1w_ras_1mm_T1andT2_masks.nii",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read subject IDs from column 1 of CSV/XLS/XLSX and copy exactly "
            "three files per subject from root/<subject_id>/ to dest/<subject_id>/ "
            "using default BIDS-style anat file patterns."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV/XLS/XLSX file path")
    parser.add_argument("--root", required=True, type=Path, help="Root directory containing subject folders")
    parser.add_argument("--dest", required=True, type=Path, help="Destination directory")
    parser.add_argument(
        "--patterns",
        nargs=3,
        default=DEFAULT_PATTERNS,
        metavar=("PATTERN1", "PATTERN2", "PATTERN3"),
        help=(
            "Exactly three relative file-name patterns. Defaults to: "
            "'anat/{id}_T2w.nii', 'anat/{id}_T1w.nii', "
            "'anat/{id}__T1w_ras_1mm_T1andT2_masks.nii'. "
            "Use {id} or {subject_id} placeholders."
        ),
    )
    return parser.parse_args()


def normalize_subject_id(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    # Pandas may coerce integer-like IDs to "123.0"; normalize those.
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def load_subject_ids(file_path: Path) -> list[str]:
    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(file_path, header=None, usecols=[0], dtype=str, sep=sep)
    elif suffix in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(file_path, header=None, usecols=[0], dtype=str)
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Use CSV/TSV/XLS/XLSX.")

    raw_ids = [normalize_subject_id(v) for v in df.iloc[:, 0].tolist()]
    ids = [sid for sid in raw_ids if sid]

    if ids and ids[0].lower() in {"subject_id", "subjectid", "subject", "id"}:
        ids = ids[1:]

    # Deduplicate while preserving order.
    return list(dict.fromkeys(ids))


def render_file_names(subject_id: str, patterns: list[str]) -> list[str]:
    rendered: list[str] = []
    for pattern in patterns:
        try:
            file_name = pattern.format(id=subject_id, subject_id=subject_id).strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid pattern '{pattern}'. Use only {{id}} or {{subject_id}} placeholders."
            ) from exc

        if not file_name:
            raise ValueError(f"Pattern rendered to an empty file name: '{pattern}'")

        rel_path = Path(file_name)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(
                f"Pattern '{pattern}' rendered to an invalid relative path: '{file_name}'"
            )

        rendered.append(file_name)

    return rendered


def main() -> None:
    args = parse_args()
    console = Console()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if not args.root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {args.root}")

    args.dest.mkdir(parents=True, exist_ok=True)

    subject_ids = load_subject_ids(args.input)
    if not subject_ids:
        console.print("[yellow]No subject IDs found in first column.[/yellow]")
        return

    missing_subjects: list[str] = []
    missing_files: list[tuple[str, str]] = []
    copied_count = 0
    found_subject_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        subject_task = progress.add_task("Checking subjects", total=len(subject_ids))

        for subject_id in subject_ids:
            src_subject_dir = args.root / subject_id
            dest_subject_dir = args.dest / subject_id

            if not src_subject_dir.is_dir():
                missing_subjects.append(subject_id)
                progress.advance(subject_task)
                continue

            found_subject_count += 1
            dest_subject_dir.mkdir(parents=True, exist_ok=True)
            target_files = render_file_names(subject_id, args.patterns)

            # tqdm for the per-subject file copy loop (3 files exactly).
            for file_name in tqdm(
                target_files,
                desc=f"{subject_id}",
                unit="file",
                leave=False,
            ):
                src_file = src_subject_dir / file_name
                dest_file = dest_subject_dir / file_name

                if not src_file.is_file():
                    missing_files.append((subject_id, file_name))
                    continue

                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
                copied_count += 1

            progress.advance(subject_task)

    table = Table(title="Copy Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="white")
    table.add_row("Input subjects", str(len(subject_ids)))
    table.add_row("Subjects found", str(found_subject_count))
    table.add_row("Subjects missing", str(len(missing_subjects)))
    table.add_row("Files copied", str(copied_count))
    table.add_row("Files missing", str(len(missing_files)))
    console.print(table)

    if missing_subjects:
        console.print("\n[bold yellow]Missing subject directories:[/bold yellow]")
        for sid in missing_subjects:
            console.print(f"- {sid}")

    if missing_files:
        console.print("\n[bold yellow]Missing files in existing subjects:[/bold yellow]")
        for sid, file_name in missing_files:
            console.print(f"- {sid}: {file_name}")


if __name__ == "__main__":
    main()
