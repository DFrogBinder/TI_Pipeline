#!/usr/bin/env python3
"""Generate Candidate A merged segmentations from precomputed CHARM maps.

Candidate A starts from the complete CHARM tissue map and overwrites every
positive manual label except skin label 5. The runner is resumable per subject
and does not run CHARM, meshing, figures, or electric-field simulations.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


TRIAL_DIR = Path(__file__).resolve().parents[1]
if str(TRIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAL_DIR))

from run_merge_mesh_trial import (  # noqa: E402
    BACKGROUND_LABEL,
    SKIN_LABEL,
    merge_candidate_a,
    resample_manual_to_charm,
    save_nifti,
    to_int_labels,
)


DEFAULT_CHARM_ROOT = Path("/home/boyan/sandbox/Jake_Data/all-seg-maps")
DEFAULT_MANUAL_ROOT = Path("/home/boyan/sandbox/Jake_Data/Archive/ti_dataset")
DEFAULT_OUT_ROOT = Path("/home/boyan/sandbox/Jake_Data/segmentation-merge-all-175-A")
APPROACH_DIR = "candidate_A_full_charm_fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charm-root", type=Path, default=DEFAULT_CHARM_ROOT)
    parser.add_argument("--manual-root", type=Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--subjects-file", type=Path, help="Optional subject list, one ID per line.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected subjects.")
    parser.add_argument("--force", action="store_true", help="Regenerate completed outputs.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def charm_subjects(root: Path) -> dict[str, Path]:
    suffix = "_CHARM_tissue_labeling_upsampled.nii.gz"
    return {path.name.removesuffix(suffix): path for path in sorted(root.glob(f"sub-*{suffix}"))}


def manual_subjects(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob("sub-*/anat/sub-*_T1w_ras_1mm_T1andT2_masks.nii*")):
        subject = path.parents[1].name
        if path.name.startswith(f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"):
            result[subject] = path
    return result


def read_subjects(path: Path) -> list[str]:
    subjects: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            subjects.append(value)
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"Duplicate IDs in {path}")
    return subjects


def output_paths(out_root: Path, subject: str) -> dict[str, Path]:
    root = out_root / subject / APPROACH_DIR
    return {
        "root": root,
        "merged": root / f"{subject}_candidate_a_merged.nii.gz",
        "qc": root / "merge_qc.json",
        "complete": root / "COMPLETE.json",
        "failed": root / "FAILED.json",
    }


def existing_output_is_valid(merged: Path, charm_path: Path) -> bool:
    if not merged.is_file() or merged.stat().st_size == 0:
        return False
    try:
        merged_img = nib.load(str(merged))
        charm_img = nib.load(str(charm_path))
        return merged_img.shape == charm_img.shape and np.allclose(merged_img.affine, charm_img.affine, atol=1e-5)
    except Exception:
        return False


def label_counts(data: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(data, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def process_subject(
    *,
    subject: str,
    charm_path: Path,
    manual_path: Path,
    out_root: Path,
    force: bool,
) -> dict[str, Any]:
    paths = output_paths(out_root, subject)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if not force and paths["complete"].is_file() and existing_output_is_valid(paths["merged"], charm_path):
        print(f"[SKIP] {subject}: valid completed output", flush=True)
        return json.loads(paths["qc"].read_text(encoding="utf-8"))

    started = time.time()
    print(f"[RUN ] {subject}", flush=True)
    charm_img = nib.load(str(charm_path))
    manual_img = nib.load(str(manual_path))
    charm = to_int_labels(charm_img)
    manual_img_resampled = resample_manual_to_charm(manual_img, charm_img)
    manual = to_int_labels(manual_img_resampled)
    merged, debug = merge_candidate_a(manual, charm)
    overlay = (manual != BACKGROUND_LABEL) & (manual != SKIN_LABEL)
    overlay_mismatches = int((merged[overlay] != manual[overlay]).sum())
    if overlay_mismatches:
        raise RuntimeError(f"{subject}: {overlay_mismatches} manual non-skin overlay mismatches")

    save_nifti(merged, charm_img, paths["merged"], dtype="int16")
    elapsed = time.time() - started
    qc = {
        "subject": subject,
        "strategy": "candidate_a",
        "source_charm": str(charm_path),
        "source_manual": str(manual_path),
        "output_merged": str(paths["merged"]),
        "shape": list(charm_img.shape),
        "affine": np.asarray(charm_img.affine).tolist(),
        "manual_overlay_voxels": int(overlay.sum()),
        "manual_skin_voxels_ignored": int((manual == SKIN_LABEL).sum()),
        "manual_overlay_mismatches": overlay_mismatches,
        "final_label_counts": label_counts(merged),
        "elapsed_seconds": elapsed,
        "merge_debug": debug,
    }
    write_json(paths["qc"], qc)
    write_json(
        paths["complete"],
        {
            "subject": subject,
            "strategy": "candidate_a",
            "output_merged": str(paths["merged"]),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": elapsed,
        },
    )
    if paths["failed"].exists():
        paths["failed"].unlink()
    print(f"[DONE] {subject}: {elapsed:.1f}s -> {paths['merged']}", flush=True)
    del charm, manual, merged, overlay, manual_img_resampled
    gc.collect()
    return qc


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "subject",
        "strategy",
        "output_merged",
        "manual_overlay_voxels",
        "manual_skin_voxels_ignored",
        "manual_overlay_mismatches",
        "elapsed_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    charm = charm_subjects(args.charm_root)
    manual = manual_subjects(args.manual_root)
    eligible = sorted(set(charm) & set(manual))
    selected = read_subjects(args.subjects_file) if args.subjects_file else eligible
    missing = [subject for subject in selected if subject not in eligible]
    if missing:
        raise ValueError(f"Subjects missing a CHARM or manual map: {', '.join(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_root / "run_manifest.json",
        {
            "strategy": "candidate_a",
            "charm_root": str(args.charm_root),
            "manual_root": str(args.manual_root),
            "out_root": str(args.out_root),
            "eligible_subject_count": len(eligible),
            "selected_subject_count": len(selected),
            "subjects": selected,
        },
    )
    print(f"[INFO] Candidate A batch: {len(selected)} of {len(eligible)} eligible subjects", flush=True)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, subject in enumerate(selected, start=1):
        print(f"[INFO] Progress {index}/{len(selected)}", flush=True)
        try:
            completed.append(
                process_subject(
                    subject=subject,
                    charm_path=charm[subject],
                    manual_path=manual[subject],
                    out_root=args.out_root,
                    force=args.force,
                )
            )
        except Exception as exc:
            failure = {
                "subject": subject,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            failures.append(failure)
            write_json(output_paths(args.out_root, subject)["failed"], failure)
            print(f"[FAIL] {subject}: {exc}", flush=True)
        write_summary(args.out_root / "summary.csv", completed)
        write_json(args.out_root / "failures.json", failures)

    print(f"[INFO] Finished: {len(completed)} completed, {len(failures)} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
