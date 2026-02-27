#!/usr/bin/env python3
"""Create a left-right (horizontal) flipped copy of a NIfTI segmentation map.

This mirrors voxel data along axis 0, preserving integer labels exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".nii.gz"):
        base = name[:-7]
        return input_path.with_name(f"{base}_hflip.nii.gz")
    if name.endswith(".nii"):
        base = name[:-4]
        return input_path.with_name(f"{base}_hflip.nii")
    return input_path.with_name(f"{name}_hflip.nii")


def flip_lr(input_path: Path, output_path: Path, validate: bool = True) -> None:
    img = nib.load(str(input_path))

    # Use dataobj directly to avoid any floating-point scaling that can alter labels.
    data = np.asanyarray(img.dataobj)
    flipped = np.flip(data, axis=0)

    header = img.header.copy()
    header.set_data_dtype(data.dtype)
    out_img = nib.Nifti1Image(flipped, img.affine, header=header)
    out_img.header.set_slope_inter(1, 0)

    nib.save(out_img, str(output_path))

    if validate:
        orig = np.asanyarray(nib.load(str(input_path)).dataobj)
        new = np.asanyarray(nib.load(str(output_path)).dataobj)
        if not np.array_equal(new, np.flip(orig, axis=0)):
            raise RuntimeError("Validation failed: output is not an exact axis-0 mirror.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an exact left-right (axis 0) flipped copy of a NIfTI segmentation."
    )
    parser.add_argument("input", type=Path, help="Path to input .nii or .nii.gz")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: input with _hflip suffix).",
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
    output_path = args.output.resolve() if args.output else default_output_path(input_path)

    flip_lr(input_path=input_path, output_path=output_path, validate=not args.no_validate)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
