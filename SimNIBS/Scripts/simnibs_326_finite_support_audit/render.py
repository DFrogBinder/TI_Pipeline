"""Paired v3/v4 cross-sections for finite-support forensic review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from nibabel.processing import resample_from_to  # noqa: E402


def _slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis).T


def _reference_t1(t1_path: Path, reference_img) -> np.ndarray:
    t1 = nib.load(str(t1_path))
    if t1.shape[:3] != reference_img.shape[:3] or not np.allclose(
        t1.affine, reference_img.affine, rtol=0.0, atol=1e-5
    ):
        t1 = resample_from_to(t1, reference_img, order=1)
    data = np.asarray(t1.dataobj, dtype=float)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"T1 must be 3D: {t1_path} has {data.shape}")
    finite = data[np.isfinite(data)]
    if finite.size:
        low, high = np.percentile(finite, [1.0, 99.0])
        if high > low:
            data = np.clip((data - low) / (high - low), 0.0, 1.0)
    return data


def render_paired_diagnostic(
    *,
    output_path: Path,
    target_label: str,
    subject: str,
    condition: str,
    versions: list[dict[str, Any]],
) -> None:
    """Render two version rows and three orthogonal views at each ROI centre."""

    if [item["name"] for item in versions] != ["v3", "v4"]:
        raise ValueError("Diagnostic versions must be ordered v3 then v4")
    roi_values = []
    for item in versions:
        diag = item["audit"]["diagnostic"]
        values = diag["masked_field"][diag["roi_mask"] & np.isfinite(diag["masked_field"])]
        roi_values.extend(np.asarray(values, dtype=float).tolist())
    vmax = float(np.percentile(roi_values, 99.0)) if roi_values else 1.0
    vmax = max(vmax, np.finfo(float).eps)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    orientation = ((2, "axial"), (1, "coronal"), (0, "sagittal"))
    field_artist = None
    for row, item in enumerate(versions):
        audit = item["audit"]
        diag = audit["diagnostic"]
        roi_indices = np.argwhere(diag["roi_mask"])
        centre = np.rint(roi_indices.mean(axis=0)).astype(int)
        background = _reference_t1(Path(item["t1_path"]), diag["reference_img"])
        for column, (axis, label) in enumerate(orientation):
            ax = axes[row, column]
            index = int(centre[axis])
            ax.imshow(_slice(background, axis, index), cmap="gray", origin="lower")
            field = np.ma.masked_invalid(_slice(diag["masked_field"], axis, index))
            field_artist = ax.imshow(
                field,
                cmap="viridis",
                norm=norm,
                alpha=0.68,
                origin="lower",
                interpolation="nearest",
            )
            roi_slice = _slice(diag["roi_mask"].astype(float), axis, index)
            if np.any(roi_slice):
                ax.contour(roi_slice, levels=[0.5], colors=["white"], linewidths=1.2)
            for key, colour in (
                ("raw_nonfinite_mask", "magenta"),
                ("mask_excluded_mask", "orange"),
                ("inconsistent_mask", "red"),
            ):
                points = np.argwhere(_slice(diag[key], axis, index))
                if points.size:
                    ax.scatter(points[:, 1], points[:, 0], s=18, c=colour, marker="x", linewidths=1.1)
            ax.set_title(f"{item['name']} {label}", fontsize=10)
            ax.set_axis_off()
        support = audit["support"]
        axes[row, 0].set_ylabel(
            f"{item['name']}\nfinite {support['final_finite_voxels']}/{support['roi_voxels']}\n"
            f"{audit['cause']}",
            fontsize=9,
        )

    fig.suptitle(
        f"{target_label} finite-support audit | {subject} | {condition} | repeat_001",
        fontsize=13,
    )
    if field_artist is not None:
        fig.colorbar(field_artist, ax=axes, shrink=0.72, label="Field magnitude (V/m)")
    fig.legend(
        handles=[
            Line2D([0], [0], color="white", lw=1.5, label="optimizer-matched ROI"),
            Line2D([0], [0], marker="x", color="magenta", lw=0, label="raw/interpolation non-finite"),
            Line2D([0], [0], marker="x", color="orange", lw=0, label="excluded by WM/GM mask"),
            Line2D([0], [0], marker="x", color="red", lw=0, label="final/raw/mask inconsistency"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary, dpi=170, facecolor="white")
    plt.close(fig)
    temporary.replace(output_path)
