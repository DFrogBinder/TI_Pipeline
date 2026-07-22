#!/usr/bin/env python3
"""Create before/after CSF-only NIfTIs for a bounded tuning cohort."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

try:
    from . import workflow
except ImportError:
    import workflow


def _save_nifti_atomic(
    path: Path,
    data: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.nii.gz")
    header = source.header.copy()
    header.set_data_dtype(dtype)
    output = nib.Nifti1Image(data.astype(dtype, copy=False), source.affine, header)
    qform, qcode = source.get_qform(coded=True)
    sform, scode = source.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    try:
        nib.save(output, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_subject(
    *,
    subject: str,
    maps_root: Path,
    output_root: Path,
    csf_closing_radius: int,
    csf_opening_radius: int,
) -> dict[str, object]:
    source_path = maps_root / f"{subject}{workflow.MAP_SUFFIX}"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha256_before = workflow.sha256_file(source_path)
    image = nib.load(str(source_path))
    source = np.asanyarray(image.dataobj)
    corrected, metrics = workflow.correct_label_array(
        source,
        csf_radius=csf_closing_radius,
        csf_opening_radius=csf_opening_radius,
        skin_radius=10,
        include_blood_in_csf=False,
        csf_component_policy="largest",
        component_policy="largest",
        connectivity=26,
    )

    before_exclusive = source == 3
    after_exclusive = corrected == 3
    before_cumulative = np.isin(source, (1, 2, 3))
    after_cumulative = np.isin(corrected, (1, 2, 3))
    difference = np.zeros(source.shape, dtype=np.int8)
    difference[~before_exclusive & after_exclusive] = 1
    difference[before_exclusive & ~after_exclusive] = -1

    subject_root = output_root / subject
    paths = {
        "before_csf_exclusive": subject_root / "before_csf_exclusive_label3.nii.gz",
        "after_csf_exclusive": subject_root / "after_csf_exclusive_label3.nii.gz",
        "before_csf_cumulative": subject_root / "before_csf_cumulative_123.nii.gz",
        "after_csf_cumulative": subject_root / "after_csf_cumulative_123.nii.gz",
        "csf_difference": subject_root / "csf_difference_added1_removed_minus1.nii.gz",
    }
    _save_nifti_atomic(paths["before_csf_exclusive"], before_exclusive, image, np.uint8)
    _save_nifti_atomic(paths["after_csf_exclusive"], after_exclusive, image, np.uint8)
    _save_nifti_atomic(paths["before_csf_cumulative"], before_cumulative, image, np.uint8)
    _save_nifti_atomic(paths["after_csf_cumulative"], after_cumulative, image, np.uint8)
    _save_nifti_atomic(paths["csf_difference"], difference, image, np.int8)

    blood = source == 9
    source_sha256_after = workflow.sha256_file(source_path)
    if source_sha256_after != source_sha256_before:
        raise RuntimeError(f"source map changed during comparison: {source_path}")
    result: dict[str, object] = {
        "subject": subject,
        "source_map": str(source_path),
        "source_sha256_before": source_sha256_before,
        "source_sha256_after": source_sha256_after,
        "source_modified": False,
        "parameters": metrics["parameters"],
        "before_exclusive_csf_voxels": int(before_exclusive.sum()),
        "after_exclusive_csf_voxels": int(after_exclusive.sum()),
        "exclusive_csf_added_voxels": int(
            np.count_nonzero(~before_exclusive & after_exclusive)
        ),
        "exclusive_csf_removed_voxels": int(
            np.count_nonzero(before_exclusive & ~after_exclusive)
        ),
        "original_blood_voxels": int(blood.sum()),
        "preserved_original_blood_voxels": int(np.count_nonzero(corrected[blood] == 9)),
        "metrics": metrics,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    workflow.write_json_atomic(subject_root / "comparison.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--csf-closing-radius", type=int, default=7)
    parser.add_argument("--csf-opening-radius", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    subjects = list(dict.fromkeys(args.subject))
    if len(subjects) != args.expected_subjects:
        raise ValueError(
            f"received {len(subjects)} unique subjects, expected {args.expected_subjects}"
        )
    if args.csf_closing_radius < 0 or args.csf_opening_radius < 0:
        raise ValueError("CSF radii must be non-negative")
    maps_root = Path(args.maps_root).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for subject in subjects:
        rows.append(
            _process_subject(
                subject=subject,
                maps_root=maps_root,
                output_root=output_root,
                csf_closing_radius=args.csf_closing_radius,
                csf_opening_radius=args.csf_opening_radius,
            )
        )
        gc.collect()

    summary = {
        "status": "complete",
        "subjects": len(rows),
        "subject_ids": subjects,
        "maps_root": str(maps_root),
        "output_root": str(output_root),
        "source_maps_modified": False,
        "csf_closing_radius_voxels": args.csf_closing_radius,
        "csf_opening_radius_voxels": args.csf_opening_radius,
        "results": rows,
    }
    workflow.write_json_atomic(output_root / "comparison_summary.json", summary)
    with (output_root / "comparison_summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = (
            "subject",
            "before_exclusive_csf_voxels",
            "after_exclusive_csf_voxels",
            "exclusive_csf_added_voxels",
            "exclusive_csf_removed_voxels",
            "original_blood_voxels",
            "preserved_original_blood_voxels",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
