#!/usr/bin/env python3
"""Compare current correction, blood-last correction, and blood overlay.

This is an isolated diagnostic utility. It imports the production correction
function but does not modify source maps, corrected maps, or production code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import workflow  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    rounded = np.rint(data)
    if data.ndim != 3 or not np.array_equal(data, rounded):
        raise ValueError(f"not a 3D integer label map: {path}")
    return image, rounded.astype(np.uint16)


def save_labels(path: Path, labels: np.ndarray, template: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = template.header.copy()
    header.set_data_dtype(np.uint16)
    output = nib.Nifti1Image(labels.astype(np.uint16), template.affine, header=header)
    output.set_qform(template.get_qform(), int(template.header["qform_code"]))
    output.set_sform(template.get_sform(), int(template.header["sform_code"]))
    nib.save(output, str(path))


def label_counts_at(mask: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[mask], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def compare_subject(
    subject: str,
    original_root: Path,
    corrected_root: Path,
    output_root: Path,
) -> dict[str, object]:
    filename = f"{subject}{workflow.MAP_SUFFIX}"
    original_path = original_root / filename
    corrected_path = corrected_root / filename
    if not original_path.is_file():
        raise FileNotFoundError(original_path)
    if not corrected_path.is_file():
        raise FileNotFoundError(corrected_path)

    original_image, original = load_labels(original_path)
    corrected_image, existing = load_labels(corrected_path)
    if original.shape != existing.shape:
        raise ValueError(f"shape mismatch for {subject}: {original.shape} != {existing.shape}")
    if not np.allclose(original_image.affine, corrected_image.affine, atol=1e-6, rtol=0):
        raise ValueError(f"affine mismatch for {subject}")

    # Reproduce the production correction, then make blood the final label
    # assignment. This is equivalent to moving the blood assignment to the
    # end of the reconstruction block without modifying production code.
    rerun_current, _ = workflow.correct_label_array(original)
    rerun_blood_last = rerun_current.copy()
    original_blood = original == 9
    rerun_blood_last[original_blood] = 9

    # Independent sanity variant: do no new correction. Start from today's
    # stored corrected map and paste the original blood mask over it.
    overlay_only = existing.copy()
    overlay_only[original_blood] = 9

    subject_root = output_root / subject
    blood_last_path = subject_root / f"{subject}_blood_last_correction.nii.gz"
    overlay_path = subject_root / f"{subject}_existing_plus_original_blood.nii.gz"
    save_labels(blood_last_path, rerun_blood_last, original_image)
    save_labels(overlay_path, overlay_only, corrected_image)

    return {
        "subject": subject,
        "original_path": str(original_path),
        "existing_corrected_path": str(corrected_path),
        "blood_last_path": str(blood_last_path),
        "overlay_only_path": str(overlay_path),
        "original_blood_voxels": int(original_blood.sum()),
        "existing_blood_voxels": int(np.count_nonzero(existing == 9)),
        "rerun_current_blood_voxels": int(np.count_nonzero(rerun_current == 9)),
        "blood_last_blood_voxels": int(np.count_nonzero(rerun_blood_last == 9)),
        "overlay_only_blood_voxels": int(np.count_nonzero(overlay_only == 9)),
        "original_blood_labels_in_existing": label_counts_at(original_blood, existing),
        "original_blood_overwritten_in_existing": int(
            np.count_nonzero(original_blood & (existing != 9))
        ),
        "original_blood_overwritten_by_skin_in_existing": int(
            np.count_nonzero(original_blood & (existing == 5))
        ),
        "blood_added_by_blood_last_vs_existing": int(
            np.count_nonzero((rerun_blood_last == 9) & (existing != 9))
        ),
        "blood_added_by_overlay_vs_existing": int(
            np.count_nonzero((overlay_only == 9) & (existing != 9))
        ),
        "different_voxels_existing_vs_rerun_current": int(
            np.count_nonzero(existing != rerun_current)
        ),
        "different_voxels_existing_vs_blood_last": int(
            np.count_nonzero(existing != rerun_blood_last)
        ),
        "different_voxels_existing_vs_overlay_only": int(
            np.count_nonzero(existing != overlay_only)
        ),
        "different_voxels_blood_last_vs_overlay_only": int(
            np.count_nonzero(rerun_blood_last != overlay_only)
        ),
        "existing_sha256": sha256_file(corrected_path),
        "blood_last_sha256": sha256_file(blood_last_path),
        "overlay_only_sha256": sha256_file(overlay_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        compare_subject(
            subject=subject,
            original_root=args.original_root,
            corrected_root=args.corrected_root,
            output_root=args.output_root,
        )
        for subject in args.subjects
    ]

    json_path = args.output_root / "comparison.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tsv_path = args.output_root / "comparison.tsv"
    scalar_fields = [key for key, value in rows[0].items() if not isinstance(value, dict)]
    with tsv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_fields})

    summary = {
        "subjects": len(rows),
        "original_blood_voxels": sum(int(row["original_blood_voxels"]) for row in rows),
        "existing_blood_voxels": sum(int(row["existing_blood_voxels"]) for row in rows),
        "blood_last_blood_voxels": sum(int(row["blood_last_blood_voxels"]) for row in rows),
        "overlay_only_blood_voxels": sum(int(row["overlay_only_blood_voxels"]) for row in rows),
        "original_blood_overwritten_in_existing": sum(
            int(row["original_blood_overwritten_in_existing"]) for row in rows
        ),
        "original_blood_overwritten_by_skin_in_existing": sum(
            int(row["original_blood_overwritten_by_skin_in_existing"]) for row in rows
        ),
        "blood_added_by_blood_last_vs_existing": sum(
            int(row["blood_added_by_blood_last_vs_existing"]) for row in rows
        ),
        "blood_added_by_overlay_vs_existing": sum(
            int(row["blood_added_by_overlay_vs_existing"]) for row in rows
        ),
        "different_voxels_existing_vs_rerun_current": sum(
            int(row["different_voxels_existing_vs_rerun_current"]) for row in rows
        ),
        "different_voxels_existing_vs_blood_last": sum(
            int(row["different_voxels_existing_vs_blood_last"]) for row in rows
        ),
        "different_voxels_existing_vs_overlay_only": sum(
            int(row["different_voxels_existing_vs_overlay_only"]) for row in rows
        ),
        "different_voxels_blood_last_vs_overlay_only": sum(
            int(row["different_voxels_blood_last_vs_overlay_only"]) for row in rows
        ),
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"comparison_json={json_path}")
    print(f"comparison_tsv={tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
