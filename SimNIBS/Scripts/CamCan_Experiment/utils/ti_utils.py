"""
Shared utilities for the TI pipeline (ROI helpers, I/O helpers, summaries).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to

NiftiLike = Union[str, os.PathLike, nib.Nifti1Image]


# ------------------ Basics ------------------
def normalize_roi_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().replace(" ", "_"))


def ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def vol_mm3(img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms))


def load_ti_as_scalar(img: nib.Nifti1Image) -> np.ndarray:
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 3:
        data = np.linalg.norm(data, axis=3)
    elif data.ndim != 3:
        raise ValueError(f"Unexpected TI data shape: {data.shape}")
    return data


def save_masked_nii(data: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    masked_data = np.where(mask, data, 0)
    out_img = nib.Nifti1Image(masked_data, ref_img.affine, ref_img.header)
    nib.save(out_img, out_path)


# ------------------ Atlas helpers ------------------
def resample_atlas_to_ti_grid(atlas_img: nib.Nifti1Image, ti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Resample a label atlas to the TI grid with nearest-neighbor interpolation."""
    if (atlas_img.shape == ti_img.shape) and np.allclose(atlas_img.affine, ti_img.affine, atol=1e-5):
        return atlas_img
    resampled = resample_from_to(atlas_img, ti_img, order=0)
    data_int = np.asarray(resampled.dataobj).astype(np.int32)
    return nib.Nifti1Image(data_int, ti_img.affine, ti_img.header)


def summarize_atlas_regions(
    ti_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    label_map: Dict[int, str],
    *,
    percentile: float = 95.0,
    min_voxels: int = 1,
) -> pd.DataFrame:
    """
    Summarize TI values per atlas label on the TI grid.
    Returns a DataFrame with mean/median/max/pXX, voxel counts, and volumes.
    """
    ti_data = load_ti_as_scalar(ti_img)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
    finite = np.isfinite(ti_data)

    records: list[dict[str, Any]] = []
    for lab_id, lab_name in label_map.items():
        mask = atlas_data == lab_id
        if mask.sum() < min_voxels:
            continue

        vals = ti_data[mask & finite]
        if vals.size == 0:
            continue

        rec = {
            "label_id": int(lab_id),
            "label_name": lab_name,
            "voxels": int(mask.sum()),
            "volume_mm3": float(mask.sum() * vol_mm3(ti_img)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
            f"p{int(percentile)}": float(np.percentile(vals, percentile)),
            "std": float(np.std(vals)),
        }
        rec["cv"] = float(rec["std"] / rec["mean"]) if rec["mean"] else np.nan
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values("label_id", inplace=True)
    return df


# ------------------ Table helpers ------------------
def extract_table(mask: np.ndarray, ref_img: nib.Nifti1Image, data: np.ndarray):
    ijk = np.argwhere(mask)
    xyz = apply_affine(ref_img.affine, ijk)
    vals = data[mask]
    return ijk, xyz, vals
