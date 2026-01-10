#!/usr/bin/env python3
"""
Find subject IDs missing a specific post-processing file.

Expected structure:
  ProjectDir/[subjectID]/anat/post/M1_TI_overlay_[subjectID]_above200.00.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def find_missing_subjects(
    project_dir: Path,
    rel_target: str,
    require_post_dir: bool,
    roi: Optional[str],
) -> List[str]:
    missing: List[str] = []

    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
    if not project_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {project_dir}")

    # iterate over immediate children in ProjectDir (subject folders)
    for subj_dir in sorted([p for p in project_dir.iterdir() if p.is_dir()]):
        subject_id = subj_dir.name

        post_dir = subj_dir / "anat" / "post"
        rel_path = Path(rel_target.format(subject=subject_id, roi=roi or ""))
        target_path = subj_dir / rel_path

        if require_post_dir and not post_dir.is_dir():
            # If you only care about cases where post/ exists, skip subjects without it.
            continue

        if not target_path.is_file():
            missing.append(subject_id)

    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate subject IDs missing a post-processing output file."
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=Path,
        help="Path to ProjectDir (contains subjectID subfolders).",
    )
    parser.add_argument(
        "--roi",
        default="M1",
        help="ROI name used in the overlay filename (e.g., M1, Hippocampus).",
    )
    parser.add_argument(
        "--target",
        default="anat/post/{roi}_TI_overlay_{subject}_above200.00.png",
        help=(
            "Relative path (from each subject folder) to the required file. "
            "Use '{subject}' and '{roi}' placeholders."
        ),
    )
    parser.add_argument(
        "--require-post-dir",
        action="store_true",
        help="Only flag subjects where anat/post exists but the target file is missing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output text file path (one subject ID per line).",
    )
    args = parser.parse_args()

    missing = find_missing_subjects(
        project_dir=args.project_dir.resolve(),
        rel_target=args.target,
        require_post_dir=args.require_post_dir,
        roi=args.roi,
    )

    # Print to stdout
    for sid in missing:
        print(sid)

    # Optional save
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
