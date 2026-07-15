#!/usr/bin/env python3
"""Run both segmentation merge candidates on a reproducible random cohort.

This consumes precomputed CHARM tissue maps and manually corrected segmentations.
It does not run CHARM segmentation, meshing, or electric-field simulations.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


TRIAL_DIR = Path(__file__).resolve().parents[1]
if str(TRIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAL_DIR))

from run_merge_mesh_trial import (  # noqa: E402
    BACKGROUND_LABEL,
    SKIN_LABEL,
    merge_candidate_a,
    merge_candidate_b,
    resample_manual_to_charm,
    save_nifti,
    to_int_labels,
)


DEFAULT_CHARM_ROOT = Path("/home/boyan/sandbox/Jake_Data/all-seg-maps")
DEFAULT_MANUAL_ROOT = Path("/home/boyan/sandbox/Jake_Data/Archive/ti_dataset")
DEFAULT_OUT_ROOT = Path("/home/boyan/sandbox/Jake_Data/segmentation-merge-random-10")
DEFAULT_SEED = 20260714

APPROACH_A_DIR = "candidate_A_full_charm_fallback"
APPROACH_B_DIR = "candidate_B_solid_charm_head_skin_base"

TISSUE_NAMES = {
    0: "Background",
    1: "White matter",
    2: "Grey matter",
    3: "CSF",
    4: "Bone",
    5: "Skin/scalp",
    6: "Eyes",
    7: "Compact bone",
    8: "Spongy bone",
    9: "Blood",
    10: "Muscle",
}

# Approximate SimNIBS tissue palette, kept stable across all figures and slides.
TISSUE_RGB = {
    0: (248, 249, 251),
    1: (232, 232, 232),
    2: (129, 129, 129),
    3: (104, 163, 255),
    4: (255, 239, 179),
    5: (226, 128, 100),
    6: (255, 224, 32),
    7: (230, 207, 143),
    8: (255, 138, 57),
    9: (32, 85, 146),
    10: (35, 133, 67),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charm-root", type=Path, default=DEFAULT_CHARM_ROOT)
    parser.add_argument("--manual-root", type=Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--subjects-file",
        type=Path,
        help="Optional fixed subject list. When omitted, a seeded random sample is used.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate completed subject outputs.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def label_counts(data: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(data, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def edge_voxel_count(mask: np.ndarray) -> int:
    return int(
        mask[0, :, :].sum()
        + mask[-1, :, :].sum()
        + mask[1:-1, 0, :].sum()
        + mask[1:-1, -1, :].sum()
        + mask[1:-1, 1:-1, 0].sum()
        + mask[1:-1, 1:-1, -1].sum()
    )


def charm_subjects(charm_root: Path) -> dict[str, Path]:
    suffix = "_CHARM_tissue_labeling_upsampled.nii.gz"
    result: dict[str, Path] = {}
    for path in sorted(charm_root.glob(f"sub-*{suffix}")):
        subject = path.name.removesuffix(suffix)
        result[subject] = path
    return result


def manual_subjects(manual_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(manual_root.glob("sub-*/anat/sub-*_T1w_ras_1mm_T1andT2_masks.nii*")):
        subject = path.parents[1].name
        expected_prefix = f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"
        if path.name.startswith(expected_prefix):
            result[subject] = path
    return result


def read_subject_file(path: Path) -> list[str]:
    subjects: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            subjects.append(value)
    return subjects


def select_subjects(
    *,
    charm: dict[str, Path],
    manual: dict[str, Path],
    subjects_file: Path | None,
    sample_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    eligible = sorted(set(charm) & set(manual))
    if subjects_file is not None:
        selected = read_subject_file(subjects_file)
        selection_rule = [f"fixed list: {subjects_file}"]
    else:
        if sample_size < 1:
            raise ValueError("--sample-size must be positive")
        if sample_size > len(eligible):
            raise ValueError(f"Requested {sample_size} subjects, but only {len(eligible)} are eligible")
        selected = random.Random(seed).sample(eligible, sample_size)
        selection_rule = [f"random.Random({seed}).sample(sorted(eligible), {sample_size})"]

    if len(selected) != len(set(selected)):
        raise ValueError("Subject list contains duplicate IDs")
    missing = [subject for subject in selected if subject not in eligible]
    if missing:
        raise ValueError(f"Selected subjects missing a CHARM or manual map: {', '.join(missing)}")
    return selected, selection_rule


def voxel_volume_mm3(img: nib.spatialimages.SpatialImage) -> float:
    return float(abs(np.linalg.det(np.asarray(img.affine)[:3, :3])))


def choose_slice(mask: np.ndarray, axis: int) -> int:
    collapse_axes = tuple(i for i in range(3) if i != axis)
    counts = mask.sum(axis=collapse_axes)
    if np.any(counts):
        return int(np.argmax(counts))
    return mask.shape[axis] // 2


def plane(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        result = data[index, :, :]
    elif axis == 1:
        result = data[:, index, :]
    else:
        result = data[:, :, index]
    return np.rot90(result)


def make_tissue_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    colors = [tuple(channel / 255 for channel in TISSUE_RGB[label]) for label in range(11)]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, 11.5, 1), 11)


def save_comparison_figure(
    *,
    subject: str,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    path: Path,
    voxel_mm3: float,
) -> dict[str, int]:
    changed = candidate_a != candidate_b
    indices = {axis: choose_slice(changed, axis) for axis in range(3)}
    cmap, norm = make_tissue_cmap()
    orientations = ((0, "Sagittal"), (1, "Coronal"), (2, "Axial"))

    fig, axes = plt.subplots(3, 3, figsize=(14.2, 10.2), facecolor="#f7f8fa")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.91, bottom=0.12, wspace=0.035, hspace=0.11)
    delta_cm3 = float(changed.sum()) * voxel_mm3 / 1000.0
    fig.suptitle(
        f"{subject}: Approach B relabels {delta_cm3:.1f} cm3 as skin",
        x=0.035,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#1c222a",
    )
    fig.text(
        0.035,
        0.935,
        "Approach A preserves CHARM fallback tissues; Approach B assigns unclaimed solid-head voxels to skin",
        ha="left",
        fontsize=10.5,
        color="#5b6571",
    )

    column_titles = (
        "Approach A: CHARM tissue fallback",
        "Approach B: residual head -> skin",
        "Voxels changed by B",
    )
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11.5, fontweight="bold", color="#1c222a", pad=10)

    for row, (axis, orientation) in enumerate(orientations):
        index = indices[axis]
        a_slice = plane(candidate_a, axis, index)
        b_slice = plane(candidate_b, axis, index)
        changed_slice = plane(changed, axis, index)
        head_slice = plane(candidate_b != BACKGROUND_LABEL, axis, index)

        axes[row, 0].imshow(a_slice, cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 1].imshow(b_slice, cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 2].imshow(head_slice, cmap=ListedColormap(["#f7f8fa", "#dfe4ea"]), interpolation="nearest")
        axes[row, 2].imshow(
            np.ma.masked_where(~changed_slice, changed_slice),
            cmap=ListedColormap(["#b64f39"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        axes[row, 0].set_ylabel(f"{orientation}\nslice {index}", fontsize=10.5, color="#5b6571")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_facecolor("#f7f8fa")
            for spine in axes[row, col].spines.values():
                spine.set_color("#d2dae3")
                spine.set_linewidth(0.8)

    legend_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = 0.035
    for label in legend_labels:
        fig.patches.append(
            plt.Rectangle(
                (x, 0.045),
                0.014,
                0.014,
                transform=fig.transFigure,
                facecolor=tuple(channel / 255 for channel in TISSUE_RGB[label]),
                edgecolor="#9da8b4",
                linewidth=0.4,
            )
        )
        fig.text(x + 0.018, 0.044, TISSUE_NAMES[label], fontsize=8.2, color="#434d59", va="bottom")
        x += 0.094 if label not in (1, 2, 7, 8) else 0.112

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {str(axis): index for axis, index in indices.items()}


def source_label_change_counts(candidate_a: np.ndarray, candidate_b: np.ndarray) -> dict[str, int]:
    changed = candidate_a != candidate_b
    labels, counts = np.unique(candidate_a[changed], return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def subject_paths(out_root: Path, subject: str) -> dict[str, Path]:
    root = out_root / "subjects" / subject
    return {
        "root": root,
        "manual_resampled": root / f"{subject}_manual_resampled_to_CHARM.nii.gz",
        "candidate_a": root / APPROACH_A_DIR / f"{subject}_candidate_a_merged.nii.gz",
        "candidate_b": root / APPROACH_B_DIR / f"{subject}_candidate_b_merged.nii.gz",
        "comparison": root / f"{subject}_candidate_A_vs_B.png",
        "qc": root / "merge_qc.json",
    }


def run_subject(
    *,
    subject: str,
    charm_path: Path,
    manual_path: Path,
    out_root: Path,
    force: bool,
) -> dict[str, Any]:
    paths = subject_paths(out_root, subject)
    if not force and all(paths[key].is_file() for key in ("candidate_a", "candidate_b", "comparison", "qc")):
        print(f"[INFO] Reusing completed outputs for {subject}", flush=True)
        return json.loads(paths["qc"].read_text(encoding="utf-8"))

    print(f"[INFO] Loading {subject}", flush=True)
    charm_img = nib.load(str(charm_path))
    manual_img = nib.load(str(manual_path))
    charm = to_int_labels(charm_img)
    manual_resampled_img = resample_manual_to_charm(manual_img, charm_img)
    manual = to_int_labels(manual_resampled_img)

    candidate_a, debug_a = merge_candidate_a(manual, charm)
    candidate_b, debug_b = merge_candidate_b(manual, charm)
    overlay = (manual != BACKGROUND_LABEL) & (manual != SKIN_LABEL)
    changed = candidate_a != candidate_b
    voxel_mm3 = voxel_volume_mm3(charm_img)

    paths["root"].mkdir(parents=True, exist_ok=True)
    save_nifti(manual, charm_img, paths["manual_resampled"], dtype="int16")
    save_nifti(candidate_a, charm_img, paths["candidate_a"], dtype="int16")
    save_nifti(candidate_b, charm_img, paths["candidate_b"], dtype="int16")
    slice_indices = save_comparison_figure(
        subject=subject,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        path=paths["comparison"],
        voxel_mm3=voxel_mm3,
    )

    skin_a = int((candidate_a == SKIN_LABEL).sum())
    skin_b = int((candidate_b == SKIN_LABEL).sum())
    head_a = int((candidate_a != BACKGROUND_LABEL).sum())
    head_b = int((candidate_b != BACKGROUND_LABEL).sum())
    changed_voxels = int(changed.sum())
    qc: dict[str, Any] = {
        "subject": subject,
        "source_charm": str(charm_path),
        "source_manual": str(manual_path),
        "output_candidate_a": str(paths["candidate_a"]),
        "output_candidate_b": str(paths["candidate_b"]),
        "output_manual_resampled": str(paths["manual_resampled"]),
        "comparison_figure": str(paths["comparison"]),
        "charm_shape": list(charm_img.shape),
        "manual_shape": list(manual_img.shape),
        "charm_affine": np.asarray(charm_img.affine).tolist(),
        "manual_affine": np.asarray(manual_img.affine).tolist(),
        "voxel_volume_mm3": voxel_mm3,
        "manual_overlay_voxels": int(overlay.sum()),
        "manual_skin_voxels_ignored": int((manual == SKIN_LABEL).sum()),
        "manual_overlay_voxels_outside_charm_head": int((overlay & (charm == BACKGROUND_LABEL)).sum()),
        "candidate_a_manual_overlay_mismatches": int((candidate_a[overlay] != manual[overlay]).sum()),
        "candidate_b_manual_overlay_mismatches": int((candidate_b[overlay] != manual[overlay]).sum()),
        "candidate_a_skin_voxels": skin_a,
        "candidate_b_skin_voxels": skin_b,
        "additional_skin_voxels_b_minus_a": skin_b - skin_a,
        "additional_skin_volume_cm3_b_minus_a": (skin_b - skin_a) * voxel_mm3 / 1000.0,
        "candidate_a_head_voxels": head_a,
        "candidate_b_head_voxels": head_b,
        "changed_voxels_a_vs_b": changed_voxels,
        "changed_volume_cm3_a_vs_b": changed_voxels * voxel_mm3 / 1000.0,
        "changed_percent_of_candidate_a_head": 100.0 * changed_voxels / head_a if head_a else 0.0,
        "candidate_a_edge_skin_voxels": edge_voxel_count(candidate_a == SKIN_LABEL),
        "candidate_b_edge_skin_voxels": edge_voxel_count(candidate_b == SKIN_LABEL),
        "candidate_a_label_counts": label_counts(candidate_a),
        "candidate_b_label_counts": label_counts(candidate_b),
        "candidate_a_labels_changed_by_b": source_label_change_counts(candidate_a, candidate_b),
        "slice_indices": slice_indices,
        "candidate_a_debug": debug_a,
        "candidate_b_debug": debug_b,
    }
    write_json(paths["qc"], qc)
    print(
        f"[INFO] Completed {subject}: B adds {qc['additional_skin_volume_cm3_b_minus_a']:.1f} cm3 skin; "
        f"{qc['changed_percent_of_candidate_a_head']:.1f}% of A head voxels changed",
        flush=True,
    )

    del charm, manual, candidate_a, candidate_b, changed, overlay
    gc.collect()
    return qc


SUMMARY_FIELDS = [
    "subject",
    "voxel_volume_mm3",
    "manual_overlay_voxels",
    "manual_skin_voxels_ignored",
    "manual_overlay_voxels_outside_charm_head",
    "candidate_a_manual_overlay_mismatches",
    "candidate_b_manual_overlay_mismatches",
    "candidate_a_skin_voxels",
    "candidate_b_skin_voxels",
    "additional_skin_voxels_b_minus_a",
    "additional_skin_volume_cm3_b_minus_a",
    "candidate_a_head_voxels",
    "candidate_b_head_voxels",
    "changed_voxels_a_vs_b",
    "changed_volume_cm3_a_vs_b",
    "changed_percent_of_candidate_a_head",
    "candidate_a_edge_skin_voxels",
    "candidate_b_edge_skin_voxels",
]


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    charm = charm_subjects(args.charm_root)
    manual = manual_subjects(args.manual_root)
    selected, selection_rule = select_subjects(
        charm=charm,
        manual=manual,
        subjects_file=args.subjects_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "sampled_subjects.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    write_json(
        args.out_root / "sample_manifest.json",
        {
            "seed": args.seed,
            "sample_size": len(selected),
            "selection_rule": selection_rule,
            "eligible_subject_count": len(set(charm) & set(manual)),
            "charm_subject_count": len(charm),
            "manual_subject_count": len(manual),
            "charm_root": str(args.charm_root),
            "manual_root": str(args.manual_root),
            "subjects": selected,
            "methods": {
                "candidate_a": "Full CHARM tissue map, overwritten by positive manual labels except label 5.",
                "candidate_b": "Solid CHARM head envelope labelled 5, overwritten by positive manual labels except label 5.",
            },
        },
    )

    print(f"[INFO] Selected {len(selected)} subjects (seed={args.seed}): {', '.join(selected)}", flush=True)
    rows = [
        run_subject(
            subject=subject,
            charm_path=charm[subject],
            manual_path=manual[subject],
            out_root=args.out_root,
            force=args.force,
        )
        for subject in selected
    ]
    write_json(args.out_root / "summary.json", rows)
    write_summary_csv(args.out_root / "summary.csv", rows)
    print(f"[INFO] Wrote cohort summary to {args.out_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
