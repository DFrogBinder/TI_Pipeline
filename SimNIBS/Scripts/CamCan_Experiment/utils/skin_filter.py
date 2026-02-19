"""
Skin-label post-processing utilities for deterministic segmentation cleanup.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np


def _load_nifti(x: Union[str, Path, nib.Nifti1Image]) -> nib.Nifti1Image:
    if isinstance(x, nib.Nifti1Image):
        return x
    return nib.load(str(x))


def smooth_skin_segmentation(
    segmentation: Union[str, Path, nib.Nifti1Image],
    *,
    skin_label: int = 5,
    background_label: int = 0,
    closing_voxels: int = 2,
    opening_voxels: int = 1,
    keep_largest_component: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """
    Deterministically smooth the skin label using binary morphology.

    Only skin/background labels are modified so inner tissues remain unchanged.
    """
    if closing_voxels < 0 or opening_voxels < 0:
        raise ValueError("closing_voxels and opening_voxels must be >= 0")

    from scipy.ndimage import (
        binary_closing,
        binary_opening,
        generate_binary_structure,
        label as cc_label,
    )

    seg_img = _load_nifti(segmentation)
    seg_arr = np.asanyarray(seg_img.dataobj).astype(np.int32, copy=False)

    skin_mask = seg_arr == skin_label
    initial_skin_voxels = int(skin_mask.sum())

    if initial_skin_voxels == 0:
        out_img = nib.Nifti1Image(seg_arr.astype(np.int16, copy=False), seg_img.affine, seg_img.header)
        out_img.header.set_data_dtype(np.int16)
        debug = {
            "initial_skin_voxels": 0,
            "final_skin_voxels": 0,
            "voxels_added": 0,
            "voxels_removed": 0,
            "closing_voxels": int(closing_voxels),
            "opening_voxels": int(opening_voxels),
            "keep_largest_component": bool(keep_largest_component),
            "skin_label": int(skin_label),
            "background_label": int(background_label),
        }
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(out_img, str(out_path))
        return out_img, debug

    structure = generate_binary_structure(3, 1)

    smoothed = skin_mask
    if closing_voxels > 0:
        smoothed = binary_closing(smoothed, structure=structure, iterations=int(closing_voxels))
    if opening_voxels > 0:
        smoothed = binary_opening(smoothed, structure=structure, iterations=int(opening_voxels))

    components_kept = 0
    if keep_largest_component and np.any(smoothed):
        labeled, n_components = cc_label(smoothed, structure=generate_binary_structure(3, 1))
        components_kept = int(n_components)
        if n_components > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            smoothed = labeled == counts.argmax()

    editable = (seg_arr == skin_label) | (seg_arr == background_label)
    add_mask = smoothed & editable & (seg_arr != skin_label)
    remove_mask = (seg_arr == skin_label) & (~smoothed)

    out_arr = np.array(seg_arr, copy=True)
    out_arr[remove_mask] = background_label
    out_arr[add_mask] = skin_label

    final_skin_voxels = int((out_arr == skin_label).sum())
    debug = {
        "initial_skin_voxels": initial_skin_voxels,
        "final_skin_voxels": final_skin_voxels,
        "voxels_added": int(add_mask.sum()),
        "voxels_removed": int(remove_mask.sum()),
        "closing_voxels": int(closing_voxels),
        "opening_voxels": int(opening_voxels),
        "keep_largest_component": bool(keep_largest_component),
        "connected_components_before_keep": int(components_kept),
        "skin_label": int(skin_label),
        "background_label": int(background_label),
    }

    out_img = nib.Nifti1Image(out_arr.astype(np.int16, copy=False), seg_img.affine, seg_img.header)
    out_img.header.set_data_dtype(np.int16)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(out_path))

    return out_img, debug
