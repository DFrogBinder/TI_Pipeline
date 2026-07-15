#!/usr/bin/env python3
"""Render large segmentation and voxel-change figures for the random-10 deck."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


TRIAL_DIR = Path(__file__).resolve().parents[1]
if str(TRIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAL_DIR))

from run_merge_mesh_trial import BACKGROUND_LABEL, save_preview_png, to_int_labels  # noqa: E402


DEFAULT_ROOT = Path("/home/boyan/sandbox/Jake_Data/segmentation-merge-random-10")
APPROACH_A_DIR = "candidate_A_full_charm_fallback"
APPROACH_B_DIR = "candidate_B_solid_charm_head_skin_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def plane(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        result = data[index, :, :]
    elif axis == 1:
        result = data[:, index, :]
    else:
        result = data[:, :, index]
    return np.rot90(result)


def save_voxel_change_figure(
    *,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    slice_indices: dict[str, int],
    path: Path,
) -> None:
    changed = candidate_a != candidate_b
    head = candidate_b != BACKGROUND_LABEL
    orientations = ((0, "Sagittal"), (1, "Coronal"), (2, "Axial"))

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.2), facecolor="#f7f8fa")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.91, bottom=0.10, wspace=0.08)
    for axis, (dimension, title) in zip(axes, orientations):
        index = int(slice_indices[str(dimension)])
        head_slice = plane(head, dimension, index)
        changed_slice = plane(changed, dimension, index)
        axis.imshow(
            head_slice,
            cmap=ListedColormap(["#f7f8fa", "#dfe4ea"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.imshow(
            np.ma.masked_where(~changed_slice, changed_slice),
            cmap=ListedColormap(["#b64f39"]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(f"{title} | slice {index}", fontsize=16, fontweight="bold", color="#1c222a", pad=12)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#f7f8fa")
        for spine in axis.spines.values():
            spine.set_color("#d2dae3")
            spine.set_linewidth(1.0)

    fig.text(
        0.5,
        0.025,
        "Red: label changed by Approach B    |    Grey: unchanged head envelope",
        ha="center",
        fontsize=12,
        color="#5b6571",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_subject(root: Path, subject: str) -> None:
    subject_root = root / "subjects" / subject
    qc = json.loads((subject_root / "merge_qc.json").read_text(encoding="utf-8"))
    manual_path = subject_root / f"{subject}_manual_resampled_to_CHARM.nii.gz"
    charm_path = Path(qc["source_charm"])
    candidate_a_path = subject_root / APPROACH_A_DIR / f"{subject}_candidate_a_merged.nii.gz"
    candidate_b_path = subject_root / APPROACH_B_DIR / f"{subject}_candidate_b_merged.nii.gz"

    print(f"[INFO] Rendering presentation assets for {subject}", flush=True)
    manual = to_int_labels(nib.load(str(manual_path)))
    charm = to_int_labels(nib.load(str(charm_path)))
    candidate_a = to_int_labels(nib.load(str(candidate_a_path)))
    candidate_b = to_int_labels(nib.load(str(candidate_b_path)))

    preview_a = subject_root / APPROACH_A_DIR / f"{subject}_candidate_a_preview.png"
    preview_b = subject_root / APPROACH_B_DIR / f"{subject}_candidate_b_preview.png"
    change_path = subject_root / f"{subject}_voxel_change_large.png"
    error_a = save_preview_png(
        manual=manual,
        charm=charm,
        final=candidate_a,
        path=preview_a,
        title=f"{subject} candidate A: manual vs CHARM vs merged",
    )
    error_b = save_preview_png(
        manual=manual,
        charm=charm,
        final=candidate_b,
        path=preview_b,
        title=f"{subject} candidate B: manual vs CHARM vs merged",
    )
    if error_a or error_b:
        raise RuntimeError(f"Preview rendering failed for {subject}: A={error_a}, B={error_b}")
    save_voxel_change_figure(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        slice_indices=qc["slice_indices"],
        path=change_path,
    )


def main() -> int:
    args = parse_args()
    with (args.root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        subjects = [row["subject"] for row in csv.DictReader(handle)]
    for subject in subjects:
        render_subject(args.root, subject)
    print(f"[INFO] Rendered {len(subjects) * 3} subject presentation assets", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
