#!/usr/bin/env python3
"""
Collect batch post-processing outputs into a single export directory.

The collector preserves paths relative to the supplied root so the export
directory can later be compressed and downloaded from HPC without losing the
dataset / subject structure.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.run_post_processing_batch import discover_repeat_datasets

console = Console(markup=False)


@dataclass(frozen=True)
class ExportItem:
    src: Path
    rel_path: Path
    kind: str  # "dir" | "file"


def find_post_dirs(root: Path) -> List[Path]:
    post_dirs: List[Path] = []
    for subj_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        post_dir = subj_dir / "anat" / "post"
        if post_dir.is_dir():
            post_dirs.append(post_dir)
    return post_dirs


def _find_single_dataset_root(root: Path) -> Optional[Path]:
    if find_post_dirs(root):
        return root
    return None


def discover_dataset_roots(
    root: Path,
    dataset_glob: str,
    repeats: Optional[Iterable[str]],
) -> List[Path]:
    datasets = discover_repeat_datasets(root, dataset_glob=dataset_glob, repeats=repeats)
    if datasets:
        return [dataset.root for dataset in datasets]

    single_dataset_root = _find_single_dataset_root(root)
    if single_dataset_root is not None:
        return [single_dataset_root.resolve()]

    return []


def build_export_plan(
    *,
    root: Path,
    dataset_glob: str = "*_Data_*",
    repeats: Optional[Iterable[str]] = None,
    include_population: bool = True,
    summary_glob: Optional[str] = "post_processing_batch_summary*.json",
) -> List[ExportItem]:
    items: List[ExportItem] = []
    seen_paths: set[Path] = set()

    dataset_roots = discover_dataset_roots(root, dataset_glob=dataset_glob, repeats=repeats)
    for dataset_root in dataset_roots:
        for post_dir in find_post_dirs(dataset_root):
            rel_path = post_dir.relative_to(root)
            if post_dir not in seen_paths:
                items.append(ExportItem(src=post_dir, rel_path=rel_path, kind="dir"))
                seen_paths.add(post_dir)

        if include_population:
            population_dir = dataset_root / "population_analysis"
            if population_dir.is_dir() and population_dir not in seen_paths:
                items.append(
                    ExportItem(
                        src=population_dir,
                        rel_path=population_dir.relative_to(root),
                        kind="dir",
                    )
                )
                seen_paths.add(population_dir)

    if summary_glob:
        for summary_file in sorted(root.glob(summary_glob)):
            if summary_file.is_file() and summary_file not in seen_paths:
                items.append(
                    ExportItem(
                        src=summary_file,
                        rel_path=summary_file.relative_to(root),
                        kind="file",
                    )
                )
                seen_paths.add(summary_file)

    items.sort(key=lambda item: item.rel_path.as_posix())
    return items


def _expand_export_item(
    item: ExportItem,
    *,
    dest: Path,
) -> tuple[set[Path], List[tuple[Path, Path]]]:
    directories: set[Path] = set()
    file_copies: List[tuple[Path, Path]] = []

    target = dest / item.rel_path
    if item.kind == "file":
        directories.add(target.parent)
        file_copies.append((item.src, target))
        return directories, file_copies

    directories.add(target)
    for src_path in sorted(item.src.rglob("*")):
        rel_child = src_path.relative_to(item.src)
        dst_path = target / rel_child
        if src_path.is_dir():
            directories.add(dst_path)
        elif src_path.is_file():
            directories.add(dst_path.parent)
            file_copies.append((src_path, dst_path))

    return directories, file_copies


def build_copy_manifest(
    items: Sequence[ExportItem],
    *,
    dest: Path,
    on_item_processed: Optional[Callable[[], None]] = None,
) -> tuple[List[Path], List[tuple[Path, Path]]]:
    directories: set[Path] = set()
    file_copies: List[tuple[Path, Path]] = []

    for item in items:
        item_directories, item_file_copies = _expand_export_item(item, dest=dest)
        directories.update(item_directories)
        file_copies.extend(item_file_copies)
        if on_item_processed is not None:
            on_item_processed()

    return sorted(directories), file_copies


def execute_export_plan(
    items: Sequence[ExportItem],
    *,
    dest: Path,
    dry_run: bool = False,
) -> None:
    if not items:
        console.log("[INFO] No post-processing outputs found.")
        return

    console.log(f"[INFO] Found {len(items)} top-level export item(s).")
    console.log("[INFO] Building file manifest...")

    manifest_progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with manifest_progress:
        manifest_task = manifest_progress.add_task(
            "Scanning export directories",
            total=len(items),
        )
        directories, file_copies = build_copy_manifest(
            items,
            dest=dest,
            on_item_processed=lambda: manifest_progress.advance(manifest_task),
        )

    console.log(
        f"[INFO] Planned {len(file_copies)} file copy operation(s) "
        f"across {len(directories)} directorie(s)."
    )

    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    if not file_copies:
        console.log(f"[INFO] Created {len(directories)} directorie(s) in {dest}; no files to copy.")
        return

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
        task = progress.add_task("Collecting post outputs", total=len(file_copies))
        for src, target in file_copies:
            if dry_run:
                console.log(f"{src} -> {target}")
            else:
                shutil.copy2(src, target)
            progress.advance(task)

    console.log(
        f"[INFO] Collected {len(file_copies)} file(s) "
        f"across {len(directories)} directorie(s) into {dest}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy anat/post directories and related batch post-processing outputs "
            "into a single export directory."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Batch root containing repeated dataset directories, or a single "
            "dataset root containing subject folders."
        ),
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory for the collected post-processing outputs.",
    )
    parser.add_argument(
        "--dataset-glob",
        default="*_Data_*",
        help="Glob used to find repeated dataset directories under --root.",
    )
    parser.add_argument(
        "--repeats",
        nargs="*",
        default=None,
        help="Optional repeat identifiers to include, for example: --repeats 01 02 10",
    )
    parser.add_argument(
        "--summary-glob",
        default="post_processing_batch_summary*.json",
        help="Glob for batch summary JSON files at the supplied root. Use an empty string to disable.",
    )
    parser.add_argument(
        "--no-population",
        action="store_true",
        help="Do not copy dataset-level population_analysis directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the paths that would be copied without copying them.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    plan = build_export_plan(
        root=root,
        dataset_glob=args.dataset_glob,
        repeats=args.repeats,
        include_population=not args.no_population,
        summary_glob=args.summary_glob or None,
    )
    execute_export_plan(plan, dest=dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
