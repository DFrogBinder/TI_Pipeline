#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeatability analysis for SimNIBS mesh outputs.

Compares per-repeat volumetric label maps derived from TI.msh, and links
mesh differences to peak TI within an ROI (e.g., FreeSurfer M1 label).
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


REPEAT_PREFIX = "repeat_"


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str))


def _find_repeat_dirs(repeats_root: Path, subject: str) -> list[Path]:
    repeat_dirs = []
    if not repeats_root.exists():
        return repeat_dirs
    for child in sorted(repeats_root.iterdir()):
        if child.is_dir() and child.name.startswith(REPEAT_PREFIX):
            anat_dir = child / subject / "anat"
            if anat_dir.is_dir():
                repeat_dirs.append(anat_dir)
    return repeat_dirs


def _find_volume_candidate(parent: Path, prefix: str) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted([p for p in parent.iterdir() if p.name.startswith(prefix)])
    return candidates[0] if candidates else None


def _run_msh2nii(ti_msh: Path, t1_path: Path, out_base: Path, *, mode: str) -> None:
    cmd = ["msh2nii", str(ti_msh), str(t1_path), str(out_base), mode]
    log_event("run_cmd", label="msh2nii", cmd=cmd, cwd=str(out_base.parent))
    result = subprocess.run(cmd, cwd=str(out_base.parent), capture_output=True, text=True)
    log_event(
        "cmd_result",
        label="msh2nii",
        returncode=result.returncode,
        stdout_tail=result.stdout[-2000:] if result.stdout else "",
        stderr_tail=result.stderr[-2000:] if result.stderr else "",
    )
    result.check_returncode()


def _load_or_create_volumes(anat_dir: Path, subject: str, t1_path: Path) -> tuple[Path, Path]:
    out_dir = anat_dir / "SimNIBS" / "Output" / subject
    labels_dir = out_dir / "Volume_Labels"
    base_dir = out_dir / "Volume_Base"

    label_file = _find_volume_candidate(labels_dir, "TI_Volumetric_")
    base_file = _find_volume_candidate(base_dir, "TI_Volumetric_")

    if label_file and base_file:
        return label_file, base_file

    ti_msh = out_dir / "TI.msh"
    if not ti_msh.exists():
        raise FileNotFoundError(f"Missing TI.msh at {ti_msh}")
    if not t1_path.exists():
        raise FileNotFoundError(f"Missing T1 at {t1_path}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    label_base = labels_dir / "TI_Volumetric_Labels"
    base_base = base_dir / "TI_Volumetric_Base"

    if not label_file:
        _run_msh2nii(ti_msh, t1_path, label_base, mode="--create_label")
        label_file = _find_volume_candidate(labels_dir, "TI_Volumetric_")
    if not base_file:
        _run_msh2nii(ti_msh, t1_path, base_base, mode="--create_base")
        base_file = _find_volume_candidate(base_dir, "TI_Volumetric_")

    if not label_file or not base_file:
        raise FileNotFoundError("Failed to generate label/base volumes with msh2nii.")

    return label_file, base_file


def _ensure_label_grid(label_img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    same_shape = label_img.shape == ref_img.shape
    same_affine = np.allclose(label_img.affine, ref_img.affine, atol=1e-4)
    if same_shape and same_affine:
        return label_img
    return resample_from_to(label_img, ref_img, order=0)


def _ensure_scalar_grid(scalar_img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    same_shape = scalar_img.shape == ref_img.shape
    same_affine = np.allclose(scalar_img.affine, ref_img.affine, atol=1e-4)
    if same_shape and same_affine:
        return scalar_img
    return resample_from_to(scalar_img, ref_img, order=1)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else np.nan


def _resolve_t1_path(subject: str, anat_dir: Path, t1_root: Path | None) -> Path:
    local_t1 = anat_dir / f"{subject}_T1w.nii"
    if local_t1.exists():
        return local_t1
    if t1_root:
        alt = t1_root / f"{subject}_repeatability" / subject / "anat" / f"{subject}_T1w.nii"
        if alt.exists():
            return alt
    return local_t1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze repeatability of SimNIBS mesh outputs."
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--rootdir",
        default="/media/boyan/main/PhD/CamCan-SimNIBS_Repeatability/simulation-data",
    )
    parser.add_argument(
        "--repeats-dir",
        default=None,
        help="Overrides rootdir/repeats if set.",
    )
    parser.add_argument(
        "--t1-root",
        default=None,
        help="Root folder containing simulation-data/<subject>_repeatability/<subject>/anat.",
    )
    parser.add_argument(
        "--atlas",
        required=True,
        help="FreeSurfer atlas NIfTI (subject space).",
    )
    parser.add_argument(
        "--m1-labels",
        required=True,
        help="Comma-separated label IDs for M1 (e.g., 1024).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports and plots.",
    )
    parser.add_argument(
        "--reference-repeat",
        default=None,
        help="Repeat tag to use as reference (e.g., repeat_001). Defaults to first repeat found.",
    )

    args = parser.parse_args()
    subject = args.subject.strip()
    rootdir = Path(args.rootdir)
    repeats_root = Path(args.repeats_dir) if args.repeats_dir else (
        rootdir / f"{subject}_repeatability" / "repeats"
    )

    repeat_anat_dirs = _find_repeat_dirs(repeats_root, subject)
    if not repeat_anat_dirs:
        raise SystemExit(f"No repeats found under {repeats_root}")

    # Pick reference repeat
    if args.reference_repeat:
        ref_anat = repeats_root / args.reference_repeat / subject / "anat"
        if ref_anat not in repeat_anat_dirs:
            raise SystemExit(f"Reference repeat not found: {ref_anat}")
    else:
        ref_anat = repeat_anat_dirs[0]

    output_dir = Path(args.output_dir) if args.output_dir else repeats_root / "_analysis" / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas and reference T1
    atlas_img = nib.load(args.atlas)
    t1_root = Path(args.t1_root) if args.t1_root else rootdir
    ref_t1_path = _resolve_t1_path(subject, ref_anat, t1_root)
    ref_t1 = nib.load(ref_t1_path)
    atlas_img = _ensure_label_grid(atlas_img, ref_t1)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32, copy=False)

    m1_labels = [int(x.strip()) for x in args.m1_labels.split(",") if x.strip()]
    m1_mask = np.isin(atlas_data, m1_labels)
    if not np.any(m1_mask):
        raise SystemExit("M1 mask is empty after resampling; check label IDs and atlas alignment.")

    # Load reference label volume
    ref_label_path, _ = _load_or_create_volumes(ref_anat, subject, ref_t1_path)
    ref_label_img = nib.load(ref_label_path)
    ref_label_img = _ensure_label_grid(ref_label_img, ref_t1)
    ref_labels = np.asarray(ref_label_img.dataobj).astype(np.int32, copy=False)

    # Aggregate metrics
    summary_rows = []
    diff_counts = np.zeros(ref_labels.shape, dtype=np.int32)
    label_set = sorted(set(np.unique(ref_labels)))

    for anat_dir in repeat_anat_dirs:
        repeat_tag = anat_dir.parent.parent.name  # repeat_###
        t1_path = _resolve_t1_path(subject, anat_dir, t1_root)
        label_path, base_path = _load_or_create_volumes(anat_dir, subject, t1_path)

        label_img = _ensure_label_grid(nib.load(label_path), ref_t1)
        labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)

        base_img = _ensure_scalar_grid(nib.load(base_path), ref_t1)
        base_data = np.asarray(base_img.dataobj, dtype=np.float32)

        diff = labels != ref_labels
        diff_counts += diff.astype(np.int32)

        diff_fraction = float(diff.mean())
        diff_fraction_m1 = float(diff[m1_mask].mean())

        m1_vals = base_data[m1_mask]
        peak_m1 = float(np.nanmax(m1_vals)) if m1_vals.size else float("nan")
        mean_m1 = float(np.nanmean(m1_vals)) if m1_vals.size else float("nan")

        # Per-label Dice vs reference
        dice_by_label = {}
        for lab in label_set:
            a = labels == lab
            b = ref_labels == lab
            dice_by_label[int(lab)] = _dice(a, b)

        summary_rows.append(
            {
                "repeat_tag": repeat_tag,
                "diff_fraction": diff_fraction,
                "diff_fraction_m1": diff_fraction_m1,
                "peak_m1": peak_m1,
                "mean_m1": mean_m1,
                "dice_by_label": dice_by_label,
            }
        )

    # Save summary JSON and CSV
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("repeat_tag,diff_fraction,diff_fraction_m1,peak_m1,mean_m1\n")
        for row in summary_rows:
            f.write(
                f"{row['repeat_tag']},{row['diff_fraction']},"
                f"{row['diff_fraction_m1']},{row['peak_m1']},{row['mean_m1']}\n"
            )

    # Save difference frequency map
    diff_freq = diff_counts.astype(np.float32) / max(1, len(repeat_anat_dirs))
    diff_img = nib.Nifti1Image(diff_freq, ref_t1.affine, ref_t1.header)
    diff_img.header.set_data_dtype(np.float32)
    nib.save(diff_img, output_dir / "label_diff_frequency.nii.gz")

    # Basic plots (matplotlib imported lazily)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = [r["repeat_tag"] for r in summary_rows]
        peak_vals = [r["peak_m1"] for r in summary_rows]
        diff_vals = [r["diff_fraction_m1"] for r in summary_rows]

        plt.figure(figsize=(8, 4))
        plt.plot(tags, peak_vals, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Peak TI in M1")
        plt.tight_layout()
        plt.savefig(output_dir / "peak_m1_by_repeat.png", dpi=150)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.scatter(diff_vals, peak_vals, alpha=0.8)
        plt.xlabel("M1 label diff fraction vs reference")
        plt.ylabel("Peak TI in M1")
        plt.tight_layout()
        plt.savefig(output_dir / "peak_m1_vs_mesh_diff.png", dpi=150)
        plt.close()
    except Exception as exc:
        log_event("plot_error", error=str(exc))

    log_event(
        "done",
        repeats=len(repeat_anat_dirs),
        output_dir=str(output_dir),
        reference_repeat=ref_anat.parent.parent.name,
    )


if __name__ == "__main__":
    main()
