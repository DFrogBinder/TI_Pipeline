#!/usr/bin/env python3
"""
Collect overlay PNGs from each subject and copy them into a destination folder.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def find_overlay_pngs(root: Path, pattern: str) -> List[Path]:
    pngs: List[Path] = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        post_dir = subj_dir / "anat" / "post"
        if not post_dir.is_dir():
            continue
        pngs.extend(sorted(post_dir.glob(pattern)))
    return pngs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy overlay PNGs from each subject's anat/post into a destination folder."
    )
    parser.add_argument("--root", required=True, help="Root directory containing subject folders.")
    parser.add_argument("--dest", required=True, help="Destination directory for copied PNGs.")
    parser.add_argument(
        "--pattern",
        default="*_TI_overlay_*.png",
        help="Glob pattern to match overlay PNGs within each subject's anat/post.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be copied without copying them.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    pngs = find_overlay_pngs(root, args.pattern)
    if not pngs:
        print("[INFO] No overlay PNGs found.")
        return

    dest.mkdir(parents=True, exist_ok=True)

    for src in pngs:
        subject_id = src.parents[2].name  # <root>/<subject>/anat/post/<file>
        dst_name = f"{subject_id}__{src.name}"
        dst = dest / dst_name
        print(f"{src} -> {dst}")
        if not args.dry_run:
            shutil.copy2(src, dst)

    print(f"[INFO] Copied {len(pngs)} file(s) to {dest}.")


if __name__ == "__main__":
    main()
