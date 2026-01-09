#!/usr/bin/env python3
"""
Remove anat/post directories for all subjects under a root directory.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def find_post_dirs(root: Path) -> List[Path]:
    post_dirs: List[Path] = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        post_dir = subj_dir / "anat" / "post"
        if post_dir.is_dir():
            post_dirs.append(post_dir)
    return post_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete anat/post directories for all subjects under a root folder."
    )
    parser.add_argument("--root", required=True, help="Root directory containing subject folders.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories that would be deleted without removing them.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    post_dirs = find_post_dirs(root)
    if not post_dirs:
        print("[INFO] No anat/post directories found.")
        return

    for d in post_dirs:
        print(d)

    if args.dry_run:
        print(f"[INFO] Dry run: {len(post_dirs)} directories listed.")
        return

    if not args.yes:
        resp = input(f"Delete {len(post_dirs)} directories listed above? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("[INFO] Aborted.")
            return

    for d in post_dirs:
        shutil.rmtree(d)

    print(f"[INFO] Deleted {len(post_dirs)} directories.")


if __name__ == "__main__":
    main()
