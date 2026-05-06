#!/usr/bin/env python3
"""Create left-right flipped NIfTI segmentation maps.

Supports:
1) Single-file mode: pass a .nii/.nii.gz file path.
2) Batch mode: pass a root directory with layout <root>/<subject_id>/anat/ and
   expected file <subject_id>_T1w_ras_1mm_T1andT2_masks.nii(.gz) in each anat dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


def temp_nifti_output_path(target_path: Path) -> Path:
    """Build a temporary NIfTI path in the same directory as target_path."""
    name = target_path.name
    if name.endswith(".nii.gz"):
        base = name[:-7]
        return target_path.with_name(f".{base}.tmp.nii.gz")
    if name.endswith(".nii"):
        base = name[:-4]
        return target_path.with_name(f".{base}.tmp.nii")
    return target_path.with_name(f".{name}.tmp.nii")


def find_nifti(stem: Path) -> Path | None:
    nii_path = stem.with_suffix(".nii")
    if nii_path.is_file():
        return nii_path

    niigz_path = stem.with_suffix(".nii.gz")
    if niigz_path.is_file():
        return niigz_path

    return None


def flip_lr(input_path: Path, output_path: Path, validate: bool = True) -> None:
    img = nib.load(str(input_path))

    # Use dataobj directly to avoid any floating-point scaling that can alter labels.
    data = np.asanyarray(img.dataobj)
    flipped = np.flip(data, axis=0)

    header = img.header.copy()
    header.set_data_dtype(data.dtype)
    out_img = nib.Nifti1Image(flipped, img.affine, header=header)
    out_img.header.set_slope_inter(1, 0)

    # Save to a temporary file in the destination directory, then atomically
    # replace the destination. This keeps in-place replacement safe.
    tmp_output = temp_nifti_output_path(output_path)
    try:
        tmp_output.unlink(missing_ok=True)
        nib.save(out_img, str(tmp_output))
        tmp_output.replace(output_path)
    finally:
        tmp_output.unlink(missing_ok=True)

    if validate:
        new = np.asanyarray(nib.load(str(output_path)).dataobj)
        if not np.array_equal(new, np.flip(data, axis=0)):
            raise RuntimeError("Validation failed: output is not an exact axis-0 mirror.")


def run_batch(root_dir: Path, validate: bool = True) -> int:
    anat_dirs = sorted(p for p in root_dir.glob("*/anat") if p.is_dir())
    if not anat_dirs:
        print(f"No subject anat directories found under: {root_dir}")
        return 1

    total = 0
    success = 0
    failed = 0
    skipped = 0

    for anat_dir in anat_dirs:
        total += 1
        subject_id = anat_dir.parent.name
        seg_stem = anat_dir / f"{subject_id}_T1w_ras_1mm_T1andT2_masks"
        seg_file = find_nifti(seg_stem)

        if seg_file is None:
            print(f"[{subject_id}] Skipping: required segmentation not found.")
            print(f"  Expected: {subject_id}_T1w_ras_1mm_T1andT2_masks.nii(.gz)")
            skipped += 1
            continue

        out_file = seg_file
        print(f"[{subject_id}] Flipping: {seg_file.name}")
        try:
            flip_lr(seg_file, out_file, validate=validate)
            print(f"[{subject_id}] Replaced: {out_file}")
            success += 1
        except Exception as exc:
            print(f"[{subject_id}] Failed: {exc}")
            failed += 1

    print()
    print("Batch complete.")
    print(f"Total subjects found: {total}")
    print(f"Successful:           {success}")
    print(f"Skipped:              {skipped}")
    print(f"Failed:               {failed}")

    return 1 if failed > 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create exact left-right (axis 0) flipped NIfTI segmentations. "
            "Input can be a single file or a root directory for batch processing."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Input .nii/.nii.gz file, or root directory containing subject folders with "
            "anat subdirectories."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for single-file mode (default: replace input in-place). "
            "Not used in directory batch mode."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip exact output-vs-input flip validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()

    if input_path.is_dir():
        if args.output is not None:
            raise ValueError("--output is only supported in single-file mode.")
        exit_code = run_batch(input_path, validate=not args.no_validate)
        sys.exit(exit_code)

    output_path = args.output.resolve() if args.output else input_path
    flip_lr(input_path=input_path, output_path=output_path, validate=not args.no_validate)
    print(f"Replaced: {output_path}")


if __name__ == "__main__":
    main()
