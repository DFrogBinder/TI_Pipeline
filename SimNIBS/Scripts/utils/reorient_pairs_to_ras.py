#!/usr/bin/env python3
"""Check T1/mask orientation pairs and reorient mismatched pairs to RAS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np

NIFTI_EXT_PATTERN = r"(?:\.nii(?:\.gz)?)"
T1_PATTERN = re.compile(rf"^(?P<subject>.+)_T1w{NIFTI_EXT_PATTERN}$")
MASK_PATTERN = re.compile(
    rf"^(?P<subject>.+)_T1w_ras_1mm_T1andT2_masks{NIFTI_EXT_PATTERN}$"
)


def orientation_str(img: nib.spatialimages.SpatialImage) -> str:
    return "".join(nib.orientations.aff2axcodes(img.affine))


def find_pairs(root: Path) -> Tuple[List[Tuple[str, Path, Path]], List[str]]:
    t1_by_key: Dict[Tuple[Path, str], Path] = {}
    mask_by_key: Dict[Tuple[Path, str], Path] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        t1_match = T1_PATTERN.match(path.name)
        if t1_match:
            key = (path.parent.resolve(), t1_match.group("subject"))
            t1_by_key[key] = path
            continue

        mask_match = MASK_PATTERN.match(path.name)
        if mask_match:
            key = (path.parent.resolve(), mask_match.group("subject"))
            mask_by_key[key] = path

    missing: List[str] = []
    pairs: List[Tuple[str, Path, Path]] = []

    all_keys = sorted(set(t1_by_key) | set(mask_by_key), key=lambda x: (str(x[0]), x[1]))
    for key in all_keys:
        t1_path = t1_by_key.get(key)
        mask_path = mask_by_key.get(key)
        parent, subject = key
        if t1_path and mask_path:
            pairs.append((subject, t1_path, mask_path))
        elif t1_path and not mask_path:
            missing.append(f"{subject}: missing mask in {parent}")
        elif mask_path and not t1_path:
            missing.append(f"{subject}: missing T1 in {parent}")

    return pairs, missing


def backup_file(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists: {backup_path}")
    backup_path.write_bytes(path.read_bytes())


def load_nifti(path: Path) -> nib.spatialimages.SpatialImage:
    # Avoid memory-mapped proxies so rewritten files never invalidate readers.
    return nib.load(str(path), mmap=False)


def temp_nifti_path(path: Path) -> Path:
    ext = ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"
    stem = path.name[: -len(ext)]
    return path.with_name(f"{stem}.tmp{ext}")


def ras_image_from_path(path: Path) -> nib.spatialimages.SpatialImage:
    img = load_nifti(path)
    ras_img = nib.as_closest_canonical(img)

    # Force full read before writing; surfaces corruption before touching outputs.
    data = np.asanyarray(ras_img.dataobj)
    out = nib.Nifti1Image(data, ras_img.affine, ras_img.header.copy())

    # Keep both transforms valid for tools that require qform (e.g., SimNIBS --forceqform).
    s_aff, s_code = ras_img.get_sform(coded=True)
    q_aff, q_code = ras_img.get_qform(coded=True)
    if s_aff is None:
        s_aff = ras_img.affine
    if q_aff is None:
        q_aff = s_aff
    s_code = int(s_code) if s_code and int(s_code) > 0 else 1
    q_code = int(q_code) if q_code and int(q_code) > 0 else s_code
    out.set_sform(s_aff, code=s_code)
    out.set_qform(q_aff, code=q_code)

    return out


def save_nifti_atomic(img: nib.spatialimages.SpatialImage, path: Path) -> None:
    tmp_path = temp_nifti_path(path)
    try:
        nib.save(img, str(tmp_path))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def process_pairs(
    pairs: Iterable[Tuple[str, Path, Path]], dry_run: bool, make_backup: bool
) -> Tuple[int, int, int, int]:
    checked = 0
    already_same = 0
    fixed = 0
    failed = 0

    for subject, t1_path, mask_path in pairs:
        checked += 1
        t1_img = load_nifti(t1_path)
        mask_img = load_nifti(mask_path)

        t1_ori = orientation_str(t1_img)
        mask_ori = orientation_str(mask_img)

        if t1_ori == mask_ori:
            already_same += 1
            print(f"[OK]   {subject}: same orientation ({t1_ori})")
            continue

        print(
            f"[FIX]  {subject}: T1={t1_ori}, MASK={mask_ori} -> reorienting both to RAS"
        )
        if dry_run:
            continue

        try:
            t1_ras_img = ras_image_from_path(t1_path)
            mask_ras_img = ras_image_from_path(mask_path)

            if make_backup:
                backup_file(t1_path)
                backup_file(mask_path)

            save_nifti_atomic(t1_ras_img, t1_path)
            save_nifti_atomic(mask_ras_img, mask_path)

            fixed += 1

            t1_after = orientation_str(load_nifti(t1_path))
            mask_after = orientation_str(load_nifti(mask_path))
            print(f"      after: T1={t1_after}, MASK={mask_after}")
        except Exception as exc:
            failed += 1
            print(f"[ERR]  {subject}: {exc}")

    return checked, already_same, fixed, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively check T1/mask file orientation pairs and reorient "
            "mismatched pairs to RAS."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root folder containing subject files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without writing files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup files before overwriting originals.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    pairs, missing = find_pairs(root)
    print(f"Found {len(pairs)} complete T1/mask pair(s) under {root}")

    if missing:
        print("\nUnpaired files:")
        for item in missing:
            print(f"  - {item}")

    checked, already_same, fixed, failed = process_pairs(
        pairs=pairs, dry_run=args.dry_run, make_backup=args.backup
    )

    print("\nSummary:")
    print(f"  checked pairs: {checked}")
    print(f"  already same:  {already_same}")
    print(f"  reoriented:    {fixed}{' (dry-run)' if args.dry_run else ''}")
    print(f"  failed:        {failed}")


if __name__ == "__main__":
    main()
