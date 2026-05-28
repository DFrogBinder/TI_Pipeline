import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any, List, Literal, Sequence

import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion
from nibabel.processing import resample_from_to
from nilearn import datasets, plotting
from nilearn.image import resample_to_img
from nilearn.plotting import plot_anat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ti_utils import (
    normalize_roi_name,
    ensure_dir,
    vol_mm3,
    load_ti_as_scalar,
    save_masked_nii,
    resample_atlas_to_ti_grid,
    summarize_atlas_regions,
    extract_table,
)
from utils.paths import fastsurfer_atlas_path
from utils.roi_registry import (
    FASTSURFER_DKT_LABELS,
    fastsurfer_roi_component_names,
    resolve_fastsurfer_roi_label_ids,
    resolve_fastsurfer_roi_name,
)




NiftiLike = Union[str, os.PathLike, nib.Nifti1Image]
AtlasMode = Literal["auto", "mni", "fastsurfer"]

# ------------------ ROI dictionaries ------------------

# Harvard–Oxford (MNI) queries (substring matches over HO label list)
ROI_QUERIES_OXFORD = {
    "M1":          {"atlas": "cort-maxprob-thr25-2mm", "query": "precentral gyrus"},
    "Hippocampus": {"atlas": "sub-maxprob-thr25-2mm",  "query": "hippocampus"},
}

DEFAULT_FASTSURFER_ROI_NAMES = ("ctx-lh-precentral", "Left-Hippocampus")
FASTSURFER_CONTEXT_EXCLUDE_LABELS = (4, 5, 14, 15, 24, 43, 44)

EFIELD_PERCENTILE   = 95
WRITE_PER_VOXEL_CSV = True

# ------------------ FastSurfer DKT labels ------------------
fastsurfer_dkt_labels = FASTSURFER_DKT_LABELS

# ------------------ Overlay helpers ------------------

NEIGHBOR_OVERLAY_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
    "#9EDAE5",
)


def neighbor_overlay_color_map(label_ids: Sequence[int]) -> Dict[int, str]:
    return {
        int(label_id): NEIGHBOR_OVERLAY_COLORS[index % len(NEIGHBOR_OVERLAY_COLORS)]
        for index, label_id in enumerate(label_ids)
    }


def _ordered_positive_labels(
    data: np.ndarray,
    label_ids: Optional[Sequence[int]] = None,
) -> list[int]:
    present = {int(value) for value in np.unique(data) if int(value) > 0}
    ordered: list[int] = []
    if label_ids is not None:
        for label_id in label_ids:
            label_int = int(label_id)
            if label_int in present and label_int not in ordered:
                ordered.append(label_int)
    ordered.extend(label_id for label_id in sorted(present) if label_id not in ordered)
    return ordered


def _neighbor_index_data(
    categorical_data: np.ndarray,
    label_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, list[int], list[str]]:
    labels = _ordered_positive_labels(categorical_data, label_ids)
    index_data = np.zeros(categorical_data.shape, dtype=np.uint16)
    colors = neighbor_overlay_color_map(label_ids if label_ids is not None else labels)
    for index, label_id in enumerate(labels, start=1):
        index_data[categorical_data == label_id] = index
        if label_id not in colors:
            colors[label_id] = NEIGHBOR_OVERLAY_COLORS[(index - 1) % len(NEIGHBOR_OVERLAY_COLORS)]
    return index_data, labels, [colors[label_id] for label_id in labels]


def _neighbor_display_bounds(n_labels: int) -> tuple[float, float]:
    if n_labels <= 1:
        return 0.5, 1.5
    return 1.0, float(n_labels)


def _roi_or_mask_cut_coords(
    roi_img: nib.Nifti1Image,
    fallback_mask: Optional[np.ndarray],
    *,
    z_offset_mm: float,
) -> tuple[float, float, float]:
    roi_data = np.asarray(roi_img.dataobj) > 0
    coords = np.argwhere(roi_data)
    affine = roi_img.affine
    if not coords.size and fallback_mask is not None:
        coords = np.argwhere(fallback_mask)
    if coords.size:
        center_ijk = coords.mean(axis=0)
        center_xyz = np.asarray(nib.affines.apply_affine(affine, center_ijk), dtype=float)
        center_xyz[2] += float(z_offset_mm)
        return tuple(float(value) for value in center_xyz)
    return (0.0, 0.0, 0.0)


def _resample_neighbor_visualization_inputs(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    neighbor_mask_img: nib.Nifti1Image,
) -> tuple[np.ndarray, nib.Nifti1Image, nib.Nifti1Image, np.ndarray]:
    ti_arr = load_ti_as_scalar(ti_img)
    ti_scalar_img = nib.Nifti1Image(ti_arr, ti_img.affine, ti_img.header)
    t1_on_ti = resample_to_img(t1_img, ti_scalar_img, interpolation="continuous")
    roi_on_ti = resample_to_img(roi_mask_img, ti_scalar_img, interpolation="nearest")
    neighbor_on_ti = resample_to_img(neighbor_mask_img, ti_scalar_img, interpolation="nearest")
    neighbor_data = np.asarray(neighbor_on_ti.dataobj).astype(np.int32, copy=False)
    return ti_arr, t1_on_ti, roi_on_ti, neighbor_data


def _add_neighbor_region_layers(
    display,
    *,
    index_data: np.ndarray,
    colors: Sequence[str],
    affine: np.ndarray,
    fill_alpha: float,
    contour_linewidth: float,
) -> None:
    if not colors:
        return

    _add_neighbor_region_fill(
        display,
        index_data=index_data,
        colors=colors,
        affine=affine,
        fill_alpha=fill_alpha,
    )
    _add_neighbor_region_contours(
        display,
        index_data=index_data,
        colors=colors,
        affine=affine,
        contour_linewidth=contour_linewidth,
    )


def _add_neighbor_region_fill(
    display,
    *,
    index_data: np.ndarray,
    colors: Sequence[str],
    affine: np.ndarray,
    fill_alpha: float,
) -> None:
    if not colors or not np.any(index_data > 0):
        return

    n_labels = len(colors)
    vmin, vmax = _neighbor_display_bounds(n_labels)
    index_img = nib.Nifti1Image(index_data.astype(np.float32, copy=False), affine)
    display.add_overlay(
        index_img,
        threshold=0.5,
        cmap=ListedColormap(list(colors)),
        transparency=fill_alpha,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
    )


def _add_neighbor_region_contours(
    display,
    *,
    index_data: np.ndarray,
    colors: Sequence[str],
    affine: np.ndarray,
    contour_linewidth: float,
) -> None:
    if not colors:
        return

    for index, color in enumerate(colors, start=1):
        region_img = nib.Nifti1Image((index_data == index).astype(np.uint8), affine)
        display.add_contours(
            region_img,
            levels=[0.5],
            colors=[color],
            linewidths=contour_linewidth,
        )


def _sparse_cut_plane_neighbor_markers(
    index_data: np.ndarray,
    *,
    affine: np.ndarray,
    cut_coords: tuple[float, float, float],
    marker_spacing_voxels: int = 10,
    max_markers_per_region_per_plane: int = 5,
) -> np.ndarray:
    marker_data = np.zeros(index_data.shape, dtype=index_data.dtype)
    labels = [int(value) for value in np.unique(index_data) if int(value) > 0]
    if not labels:
        return marker_data

    spacing = max(1, int(marker_spacing_voxels))
    max_markers = max(1, int(max_markers_per_region_per_plane))
    cut_ijk = np.rint(
        nib.affines.apply_affine(np.linalg.inv(affine), np.asarray(cut_coords, dtype=float))
    ).astype(int)
    shape = np.asarray(index_data.shape, dtype=int)
    cut_ijk = np.clip(cut_ijk, 0, shape - 1)

    for label_index in labels:
        region = index_data == label_index

        for axis, plane_index in enumerate(cut_ijk):
            plane_slice = [slice(None), slice(None), slice(None)]
            plane_slice[axis] = int(plane_index)
            plane_slice_tuple = tuple(plane_slice)
            plane_region = region[plane_slice_tuple]
            plane_interior = binary_erosion(plane_region, iterations=1, border_value=0)
            coords_2d = np.argwhere(plane_interior if np.any(plane_interior) else plane_region)
            if coords_2d.size == 0:
                continue

            coords = np.zeros((coords_2d.shape[0], 3), dtype=int)
            other_axes = [dim for dim in range(3) if dim != axis]
            coords[:, axis] = int(plane_index)
            coords[:, other_axes[0]] = coords_2d[:, 0]
            coords[:, other_axes[1]] = coords_2d[:, 1]
            if coords.size == 0:
                continue

            plane_min = coords[:, other_axes].min(axis=0)
            spaced = (
                ((coords[:, other_axes[0]] - plane_min[0]) % spacing == 0)
                & ((coords[:, other_axes[1]] - plane_min[1]) % spacing == 0)
            )
            selected = coords[spaced]
            if selected.shape[0] == 0:
                step = max(1, coords.shape[0] // max_markers)
                selected = coords[::step]
            if selected.shape[0] > max_markers:
                selected = selected[
                    np.linspace(0, selected.shape[0] - 1, max_markers, dtype=int)
                ]
            marker_data[tuple(selected.T)] = label_index

    return marker_data


def try_fast_crop_to_target(atlas_img: nib.Nifti1Image, target_img: nib.Nifti1Image, mask_bool: np.ndarray):
    # Fast crop only if affines (orientation & voxel size) match exactly (within tolerance)
    if not np.allclose(atlas_img.affine, target_img.affine, atol=1e-5):
        return None  # not safe to crop

    # Map target voxel corners to atlas index space
    A = np.linalg.inv(atlas_img.affine)
    ti_shape = np.array(target_img.shape, int)

    corners_ijk = np.array([[0,0,0,1],
                            [ti_shape[0]-1,0,0,1],
                            [0,ti_shape[1]-1,0,1],
                            [0,0,ti_shape[2]-1,1],
                            [ti_shape[0]-1,ti_shape[1]-1,ti_shape[2]-1,1]], float)
    corners_xyz = (target_img.affine @ corners_ijk.T).T
    corners_atlas_ijk = (A @ corners_xyz.T).T[:, :3]

    # Expect integer-aligned grids; round safely
    idx_min = np.floor(corners_atlas_ijk.min(axis=0)).astype(int)
    idx_max = np.ceil( corners_atlas_ijk.max(axis=0)).astype(int) + 1

    # Clip to atlas bounds
    atlas_shape = np.array(atlas_img.shape, int)
    s0 = slice(max(0, idx_min[0]), min(atlas_shape[0], idx_max[0]))
    s1 = slice(max(0, idx_min[1]), min(atlas_shape[1], idx_max[1]))
    s2 = slice(max(0, idx_min[2]), min(atlas_shape[2], idx_max[2]))

    cropped = mask_bool[s0, s1, s2]
    # If shapes now match, we’re done; otherwise fall back to resample.
    return cropped if cropped.shape == tuple(ti_shape) else None


def make_outline(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Return a 1-voxel outline (contour) of a binary mask using erosion XOR.
    Increase `iterations` for a thicker outline.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)
    eroded = binary_erosion(mask, iterations=iterations, border_value=0)
    outline = mask & (~eroded)
    return outline.astype(np.uint8, copy=False)


def make_overlay_png(out_png, overlay_img, bg_img=None, title=None, roi_mask_img=None, abs_colour=False):
    data = overlay_img.get_fdata()
    finite = np.isfinite(data)
    if np.any(finite):
        if abs_colour:
            vmin = np.min(data[finite])
            vmax = np.max(data[finite])
        else:
            vmin = np.percentile(data[finite], 2)
            vmax = np.percentile(data[finite], 98)
        if vmin >= vmax:
            vmin, vmax = np.nanmin(data[finite]), np.nanmax(data[finite])
    else:
        vmin = vmax = None

    if bg_img is not None:
        disp = plotting.plot_anat(
            bg_img, annotate=False, draw_cross=False, black_bg=True, colorbar=False,cut_coords=(30, 0, 0)
        )
        disp.add_overlay(overlay_img, colorbar=True, vmin=vmin, vmax=vmax, cmap="viridis")
    else:
        disp = plotting.plot_img(
            overlay_img, annotate=False, draw_cross=False, black_bg=True,
            colorbar=True, vmin=vmin, vmax=vmax, cmap="viridis"
        )

    if roi_mask_img is not None:
        disp.add_contours(roi_mask_img, levels=[0.5], linewidths=1.5, colors="red")

    if title:
        disp.title(title)

    disp.savefig(out_png, dpi=300)
    disp.close()


def overlay_neighbor_union_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    neighbor_mask_img: nib.Nifti1Image,
    out_png: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    dpi: int = 180,
) -> str:
    return overlay_neighbor_regions_on_t1_with_roi(
        ti_img=ti_img,
        t1_img=t1_img,
        roi_mask_img=roi_mask_img,
        neighbor_mask_img=neighbor_mask_img,
        out_png=out_png,
        subject=subject,
        z_offset_mm=z_offset_mm,
        dpi=dpi,
    )


def overlay_neighbor_regions_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    neighbor_mask_img: nib.Nifti1Image,
    out_png: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    dpi: int = 180,
    neighbor_label_ids: Optional[Sequence[int]] = None,
    fill_alpha: float = 0.58,
) -> str:
    """
    Render fixed-template neighbor masks on the subject anatomy.

    Each positive value in ``neighbor_mask_img`` is treated as a separate
    neighbor region, so callers should pass the categorical neighbor mask when
    available. Binary masks still render as a single region for backwards
    compatibility.
    """
    _ti_arr, t1_on_ti, roi_on_ti, neighbor_data = _resample_neighbor_visualization_inputs(
        ti_img=ti_img,
        t1_img=t1_img,
        roi_mask_img=roi_mask_img,
        neighbor_mask_img=neighbor_mask_img,
    )
    index_data, _labels, colors = _neighbor_index_data(neighbor_data, neighbor_label_ids)
    cut_coords = _roi_or_mask_cut_coords(
        roi_on_ti,
        index_data > 0,
        z_offset_mm=z_offset_mm,
    )
    title_subject = f" ({subject})" if subject else ""
    display = plot_anat(
        t1_on_ti,
        display_mode="ortho",
        dim=0,
        annotate=True,
        draw_cross=True,
        colorbar=False,
        black_bg=True,
        cut_coords=cut_coords,
        title=f"Fixed neighbor region masks{title_subject}",
    )
    _add_neighbor_region_layers(
        display,
        index_data=index_data,
        colors=colors,
        affine=ti_img.affine,
        fill_alpha=fill_alpha,
        contour_linewidth=0.8,
    )
    display.add_contours(
        roi_on_ti,
        levels=[0.5],
        colors=["red"],
        linewidths=1.2,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    display.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    display.close()
    return out_png


def overlay_neighbor_regions_and_efield_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    neighbor_mask_img: nib.Nifti1Image,
    out_png: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    dpi: int = 180,
    neighbor_label_ids: Optional[Sequence[int]] = None,
    efield_cmap: str = "viridis",
    neighbor_fill_alpha: float = 0.04,
    neighbor_marker_spacing_voxels: int = 10,
    max_neighbor_markers_per_region_per_plane: int = 5,
    efield_upper_percentile: float = 99.5,
) -> str:
    """
    Render TI/e-field values cropped to fixed-template neighbor regions.

    The e-field keeps the main color scale. Neighbor regions are added as sparse,
    faint categorical markers plus colored contours so their boundaries remain
    visible without dominating the e-field overlay.
    """
    ti_arr, t1_on_ti, roi_on_ti, neighbor_data = _resample_neighbor_visualization_inputs(
        ti_img=ti_img,
        t1_img=t1_img,
        roi_mask_img=roi_mask_img,
        neighbor_mask_img=neighbor_mask_img,
    )
    index_data, _labels, colors = _neighbor_index_data(neighbor_data, neighbor_label_ids)
    neighbor_union = index_data > 0
    cut_coords = _roi_or_mask_cut_coords(
        roi_on_ti,
        neighbor_union,
        z_offset_mm=z_offset_mm,
    )

    arr = np.asarray(ti_arr, dtype=float)
    finite_neighbor = neighbor_union & np.isfinite(arr)
    if not np.any(finite_neighbor):
        raise ValueError("No finite TI/e-field voxels were found inside the neighbor regions.")

    positive_neighbor = finite_neighbor & (arr > 0)
    display_values = arr[positive_neighbor] if np.any(positive_neighbor) else arr[finite_neighbor]
    efield_data = np.where(finite_neighbor, arr, 0.0).astype(np.float32, copy=False)
    efield_img = nib.Nifti1Image(efield_data, ti_img.affine)
    vmin = float(np.nanmin(display_values))
    vmax = _robust_vmax(display_values, efield_upper_percentile)
    vmin, vmax = _coerce_display_bounds(vmin, vmax)

    title_subject = f" ({subject})" if subject else ""
    display = plot_anat(
        t1_on_ti,
        display_mode="ortho",
        dim=0,
        annotate=True,
        draw_cross=True,
        colorbar=False,
        black_bg=True,
        cut_coords=cut_coords,
        title=f"TI/e-field in fixed neighbor regions{title_subject}",
    )
    display.add_overlay(
        efield_img,
        threshold=1e-12,
        colorbar=True,
        vmin=vmin,
        vmax=vmax,
        cmap=efield_cmap,
        transparency=0.86,
    )
    marker_data = _sparse_cut_plane_neighbor_markers(
        index_data,
        affine=ti_img.affine,
        cut_coords=cut_coords,
        marker_spacing_voxels=neighbor_marker_spacing_voxels,
        max_markers_per_region_per_plane=max_neighbor_markers_per_region_per_plane,
    )
    _add_neighbor_region_fill(
        display,
        index_data=marker_data,
        colors=colors,
        affine=ti_img.affine,
        fill_alpha=neighbor_fill_alpha,
    )
    _add_neighbor_region_contours(
        display,
        index_data=index_data,
        colors=colors,
        affine=ti_img.affine,
        contour_linewidth=0.65,
    )
    display.add_contours(
        roi_on_ti,
        levels=[0.5],
        colors=["red"],
        linewidths=1.0,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    display.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    display.close()
    return out_png


def build_context_scale_mask_from_fastsurfer(
    atlas_img: nib.Nifti1Image,
    *,
    exclude_labels: tuple[int, ...] = FASTSURFER_CONTEXT_EXCLUDE_LABELS,
) -> nib.Nifti1Image:
    """
    Build a plotting scale mask that keeps labeled brain tissue while excluding
    CSF/ventricular labels that tend to dominate the colorbar.
    """
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
    include = (atlas_data > 0) & (~np.isin(atlas_data, exclude_labels))
    return nib.Nifti1Image(include.astype(np.uint8), atlas_img.affine, atlas_img.header)


def _robust_vmax(values: np.ndarray, upper_percentile: float) -> float:
    if values.size == 0:
        raise ValueError("Cannot compute display bounds from an empty array.")

    vmax = float(np.nanpercentile(values, upper_percentile))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(values))

    vmin = float(np.nanmin(values))
    if vmax <= vmin:
        vmax = float(np.nanmax(values))
    if vmax <= 0:
        vmax = float(np.nanmax(values))
    return vmax


def _coerce_display_bounds(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = vmin
    if vmax <= vmin:
        vmax = float(np.nextafter(vmin, np.inf))
    return vmin, vmax


def _prepare_overlay_data(arr: np.ndarray, thr_value: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(arr)
    if thr_value is None:
        overlay_data = np.where(finite, arr, 0.0)
    else:
        overlay_data = np.where(finite & (arr >= thr_value), arr, 0.0)

    subset = overlay_data[overlay_data > 0]
    return overlay_data, subset


def _overlay_ti_thresholds_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    out_prefix: str,
    scale_mode: str,
    scale_mask_img: Optional[nib.Nifti1Image] = None,
    scale_upper_percentile: float = 99.5,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    include_full_field: bool = False,
    percentile: float = 95.0,
    hard_threshold: float = 200.0,
    contour_color: str = "red",
    contour_linewidth: float = 0.5,
    cmap: str = "viridis",
    dpi: int = 150,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    ti_arr = load_ti_as_scalar(ti_img)
    ti_scalar_img = nib.Nifti1Image(ti_arr, ti_img.affine, ti_img.header)

    t1_on_ti = resample_to_img(t1_img, ti_scalar_img, interpolation="continuous")
    roi_on_ti = resample_to_img(roi_mask_img, ti_scalar_img, interpolation="nearest")
    scale_on_ti = (
        resample_to_img(scale_mask_img, ti_scalar_img, interpolation="nearest")
        if scale_mask_img is not None
        else None
    )

    arr = np.asarray(ti_arr, dtype=float)
    finite_pos = np.isfinite(arr) & (arr > 0)
    if not np.any(finite_pos):
        raise ValueError("TI has no positive finite voxels.")
    thr_percentile = float(np.percentile(arr[finite_pos], percentile))
    thr_fixed = float(hard_threshold)

    roi_data = np.asarray(roi_on_ti.dataobj) > 0
    roi_coords = np.argwhere(roi_data)
    if roi_coords.size:
        center_ijk = roi_coords.mean(axis=0)
        center_xyz = nib.affines.apply_affine(roi_on_ti.affine, center_ijk)
        center_xyz = np.asarray(center_xyz, dtype=float)
        center_xyz[2] += float(z_offset_mm)
        cut_coords = tuple(float(x) for x in center_xyz)
    else:
        cut_coords = (0.0, 0.0, 0.0)

    if scale_mode == "roi_focus":
        scale_mask = roi_data
        scale_title = "ROI focus"
    elif scale_mode == "whole_brain":
        scale_mask = np.ones(arr.shape, dtype=bool)
        scale_title = "Whole-brain"
    elif scale_on_ti is not None:
        scale_mask = np.asarray(scale_on_ti.dataobj) > 0
        scale_title = "Context"
    else:
        scale_mask = np.ones(arr.shape, dtype=bool)
        scale_title = "Context"

    scale_values = arr[finite_pos & scale_mask]
    if scale_values.size == 0:
        scale_values = arr[finite_pos]
    vmax = _robust_vmax(scale_values, scale_upper_percentile)

    def _plot_overlay(thr_value: Optional[float], label: str) -> Optional[str]:
        overlay_data, subset = _prepare_overlay_data(arr, thr_value)
        vmin = float(np.nanmin(subset)) if subset.size else 0.0
        local_vmax = max(vmax, float(np.nanmax(subset))) if subset.size else vmax
        vmin, local_vmax = _coerce_display_bounds(vmin, local_vmax)
        overlay_img = nib.Nifti1Image(overlay_data, ti_img.affine, ti_img.header)

        if subject:
            out_path = f"{out_prefix}_{subject}_{label}.png"
        else:
            out_path = f"{out_prefix}_{label}.png"

        display = None
        try:
            display = plot_anat(
                t1_on_ti,
                display_mode="ortho",
                dim=0,
                annotate=True,
                draw_cross=True,
                colorbar=False,
                black_bg=True,
                cut_coords=cut_coords,
                title=(
                    f"TI ≥ {thr_value:.3f} ({label}, {scale_title} scale)"
                    if thr_value is not None
                    else f"TI (full field, {scale_title} scale)"
                ),
            )
            display.add_overlay(
                overlay_img, colorbar=True, vmin=vmin, vmax=local_vmax, cmap=cmap
            )
            display.add_contours(
                roi_on_ti, levels=[0.5], colors=[contour_color], linewidths=contour_linewidth
            )
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            display.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        except Exception as exc:
            print(
                f"[WARN] Skipped overlay '{label}' for prefix '{out_prefix}': "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        finally:
            if display is not None:
                display.close()
        return out_path

    png_full = _plot_overlay(None, "full") if include_full_field else None
    png_percentile = _plot_overlay(thr_percentile, f"top{int(percentile)}")
    png_fixed = _plot_overlay(thr_fixed, f"above{hard_threshold:.2f}")
    return png_percentile, png_fixed, png_full

def resample_atlas_to_ti_grid(atlas_img: nib.Nifti1Image, ti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Resample a label atlas to the TI grid with nearest-neighbor interpolation.
    """
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
    Returns a DataFrame with mean/median/max/pXX, voxel counts and volumes.
    """
    ti_data = load_ti_as_scalar(ti_img)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
    finite = np.isfinite(ti_data)

    present_labels = sorted(int(value) for value in np.unique(atlas_data) if int(value) > 0)

    records: list[dict[str, Any]] = []
    for lab_id in present_labels:
        lab_name = label_map.get(lab_id, f"Label-{lab_id}")
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
        # Coefficient of variation (avoid divide-by-zero)
        rec["cv"] = float(rec["std"] / rec["mean"]) if rec["mean"] else np.nan
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values("label_id", inplace=True)
    return df

def overlay_ti_thresholds_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    out_prefix: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    include_full_field: bool = False,
    percentile: float = 95.0,
    hard_threshold: float = 200.0,
    contour_color: str = "red",
    contour_linewidth: float = 0.5,
    cmap: str = "viridis",
    dpi: int = 150,
    alpha: float = 0.85,
    scale_mask_img: Optional[nib.Nifti1Image] = None,
    scale_upper_percentile: float = 99.5,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    return _overlay_ti_thresholds_on_t1_with_roi(
        ti_img=ti_img,
        t1_img=t1_img,
        roi_mask_img=roi_mask_img,
        out_prefix=out_prefix,
        scale_mode="context",
        scale_mask_img=scale_mask_img,
        scale_upper_percentile=scale_upper_percentile,
        subject=subject,
        z_offset_mm=z_offset_mm,
        include_full_field=include_full_field,
        percentile=percentile,
        hard_threshold=hard_threshold,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
        cmap=cmap,
        dpi=dpi,
    )


def overlay_ti_thresholds_on_t1_with_roi_individual_scale(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    out_prefix: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    include_full_field: bool = False,
    percentile: float = 95.0,
    hard_threshold: float = 200.0,
    contour_color: str = "red",
    contour_linewidth: float = 0.5,
    cmap: str = "viridis",
    dpi: int = 150,
    scale_upper_percentile: float = 99.0,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Overlay TI on T1 with ROI contour using a robust ROI-focused display scale.
    """
    return _overlay_ti_thresholds_on_t1_with_roi(
        ti_img=ti_img,
        t1_img=t1_img,
        roi_mask_img=roi_mask_img,
        out_prefix=out_prefix,
        scale_mode="roi_focus",
        scale_upper_percentile=scale_upper_percentile,
        subject=subject,
        z_offset_mm=z_offset_mm,
        include_full_field=include_full_field,
        percentile=percentile,
        hard_threshold=hard_threshold,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
        cmap=cmap,
        dpi=dpi,
    )


def overlay_ti_thresholds_on_t1_with_roi_whole_brain_scale(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    out_prefix: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    include_full_field: bool = False,
    percentile: float = 95.0,
    hard_threshold: float = 200.0,
    contour_color: str = "red",
    contour_linewidth: float = 0.5,
    cmap: str = "viridis",
    dpi: int = 150,
    scale_upper_percentile: float = 99.5,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Overlay TI on T1 with ROI contour using a robust whole-brain display scale.
    """
    return _overlay_ti_thresholds_on_t1_with_roi(
        ti_img=ti_img,
        t1_img=t1_img,
        roi_mask_img=roi_mask_img,
        out_prefix=out_prefix,
        scale_mode="whole_brain",
        scale_upper_percentile=scale_upper_percentile,
        subject=subject,
        z_offset_mm=z_offset_mm,
        include_full_field=include_full_field,
        percentile=percentile,
        hard_threshold=hard_threshold,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
        cmap=cmap,
        dpi=dpi,
    )


def overlay_ti_full_field_true_vmax_reference_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
    roi_mask_img: nib.Nifti1Image,
    out_prefix: str,
    subject: Optional[str] = None,
    z_offset_mm: float = 0.0,
    contour_color: str = "red",
    contour_linewidth: float = 0.5,
    cmap: str = "viridis",
    dpi: int = 150,
) -> str:
    """
    Write one unscaled full-field reference overlay using the true whole-brain
    positive maximum as the colorbar upper bound.
    """
    ti_arr = load_ti_as_scalar(ti_img)
    ti_scalar_img = nib.Nifti1Image(ti_arr, ti_img.affine, ti_img.header)

    t1_on_ti = resample_to_img(t1_img, ti_scalar_img, interpolation="continuous")
    roi_on_ti = resample_to_img(roi_mask_img, ti_scalar_img, interpolation="nearest")

    arr = np.asarray(ti_arr, dtype=float)
    finite_pos = np.isfinite(arr) & (arr > 0)
    if not np.any(finite_pos):
        raise ValueError("TI has no positive finite voxels.")

    roi_data = np.asarray(roi_on_ti.dataobj) > 0
    roi_coords = np.argwhere(roi_data)
    if roi_coords.size:
        center_ijk = roi_coords.mean(axis=0)
        center_xyz = nib.affines.apply_affine(roi_on_ti.affine, center_ijk)
        center_xyz = np.asarray(center_xyz, dtype=float)
        center_xyz[2] += float(z_offset_mm)
        cut_coords = tuple(float(x) for x in center_xyz)
    else:
        cut_coords = (0.0, 0.0, 0.0)

    overlay_data, subset = _prepare_overlay_data(arr, None)
    overlay_img = nib.Nifti1Image(overlay_data, ti_img.affine, ti_img.header)
    vmin = float(np.nanmin(subset)) if subset.size else 0.0
    vmax = float(np.nanmax(subset)) if subset.size else 0.0
    vmin, vmax = _coerce_display_bounds(vmin, vmax)

    display = plot_anat(
        t1_on_ti,
        display_mode="ortho",
        dim=0,
        annotate=True,
        draw_cross=True,
        colorbar=False,
        black_bg=True,
        cut_coords=cut_coords,
        title="TI (full field, true whole-brain max reference)",
    )
    display.add_overlay(
        overlay_img, colorbar=True, vmin=vmin, vmax=vmax, cmap=cmap
    )
    display.add_contours(
        roi_on_ti, levels=[0.5], colors=[contour_color], linewidths=contour_linewidth
    )

    if subject:
        out_path = f"{out_prefix}_{subject}_full.png"
    else:
        out_path = f"{out_prefix}_full.png"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    display.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    display.close()
    return out_path

# ------------------ ROI extraction core ------------------

def load_custom_atlas(atlas_path: NiftiLike) -> nib.Nifti1Image:
    if isinstance(atlas_path, (str, os.PathLike)):
        return nib.load(str(atlas_path))
    if isinstance(atlas_path, nib.Nifti1Image):
        return atlas_path
    raise TypeError(f"Expected path or NIfTI image, got {type(atlas_path)}")

def _resolve_fastsurfer_atlas(subject: str, fastsurfer_root: Optional[str], explicit_path: Optional[str]) -> Optional[str]:
    """
    Resolve the subject FastSurfer atlas from the supported flat-file layout.
    Priority: explicit_path > {fastsurfer_root}/{subject}.nii > {fastsurfer_root}/{subject}.nii.gz
    """
    atlas_path = fastsurfer_atlas_path(fastsurfer_root, subject, explicit_path)
    return str(atlas_path) if atlas_path else None

def roi_masks_on_ti_grid(
    ti_img: nib.Nifti1Image,
    *,
    atlas_mode: AtlasMode = "auto",
    subject: Optional[str] = None,
    fastsurfer_root: Optional[str] = None,
    fastsurfer_atlas_path: Optional[str] = None,
    roi_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, nib.Nifti1Image]]:
    """
    Build ROI masks on the TI grid using either Harvard–Oxford (MNI) or subject FastSurfer atlas.

    atlas_mode:
      - "auto": use FastSurfer if a subject atlas is found (subject != 'MNI152'), else Harvard–Oxford
      - "fastsurfer": require a FastSurfer atlas (error if missing)
      - "mni": always use Harvard–Oxford

    FastSurfer search order: fastsurfer_atlas_path (explicit) >
                             {fastsurfer_root}/{subject}.nii >
                             {fastsurfer_root}/{subject}.nii.gz
    """
    # Decide mode
    chosen_mode: AtlasMode = atlas_mode
    fs_atlas: Optional[str] = None

    if atlas_mode == "auto":
        not_mni = subject is not None and subject.upper() != "MNI152"
        # not_mni = True #! DELETE LATER --- FORCE FASTSURFER FOR TESTING
        if not_mni:
            fs_atlas = _resolve_fastsurfer_atlas(subject, fastsurfer_root, fastsurfer_atlas_path)
        chosen_mode = "fastsurfer" if fs_atlas else "mni"

    elif atlas_mode == "fastsurfer":
        fs_atlas = _resolve_fastsurfer_atlas(subject or "", fastsurfer_root, fastsurfer_atlas_path)
        if not fs_atlas:
            raise FileNotFoundError(
                "FastSurfer atlas not found. Provide --fs-mri-path OR --fastsurfer-root + --subject "
                "with atlases stored as '<fastsurfer_root>/<subject>.nii' or "
                "'<fastsurfer_root>/<subject>.nii.gz'."
            )

    # Prepare outputs
    roi_masks: Dict[str, np.ndarray] = {}
    atlas_imgs: Dict[str, nib.Nifti1Image] = {}

    if chosen_mode == "mni":
        ROI_Q = ROI_QUERIES_OXFORD
        for roi_name, info in ROI_Q.items():
            print(f"[INFO] Using Harvard–Oxford atlas '{info['atlas']}' for ROI '{roi_name}'")
            atlas = datasets.fetch_atlas_harvard_oxford(info["atlas"])
            atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
            atlas_data = np.asarray(atlas_img.dataobj).astype(int)
            labels = list(atlas.labels)

            # case-insensitive substring match over label list
            matched_idx = [i for i, lab in enumerate(labels) if info["query"].lower() in str(lab).lower()]
            if not matched_idx:
                raise ValueError(f"No HO labels match query '{info['query']}'")

            combined_mask = np.isin(atlas_data, matched_idx).astype(np.uint8)

            # resample to TI grid
            if (atlas_img.shape != ti_img.shape) or (not np.allclose(atlas_img.affine, ti_img.affine, atol=1e-5)):
                resampled = resample_from_to(
                    nib.Nifti1Image(combined_mask, atlas_img.affine), ti_img, order=0
                )
                combined_mask = np.asarray(resampled.dataobj).astype(bool)
                atlas_img = resampled
            else:
                combined_mask = combined_mask.astype(bool)

            roi_masks[roi_name] = combined_mask
            atlas_imgs[roi_name] = atlas_img

        return roi_masks, atlas_imgs

    # ---- FastSurfer path (chosen_mode == "fastsurfer") ----
    print(f"[INFO] Using FastSurfer/FreeSurfer atlas segmentation for subject '{subject}'")
    assert fs_atlas is not None, "Internal: fs_atlas must be resolved here."

    atlas_img = load_custom_atlas(fs_atlas)
    atlas_data = np.asarray(atlas_img.dataobj).astype(int)

    requested_rois = roi_names or list(DEFAULT_FASTSURFER_ROI_NAMES)
    for requested_roi in requested_rois:
        resolved_roi = resolve_fastsurfer_roi_name(requested_roi)
        roi_name = resolved_roi.canonical_name
        roi_key = roi_name.lower()

        ids = list(resolve_fastsurfer_roi_label_ids(roi_name))
        if not ids:
            raise ValueError(f"No FastSurfer label matches '{roi_name}'.")
        component_names = fastsurfer_roi_component_names(roi_name)
        present_label_ids = {
            int(label_id)
            for label_id in np.unique(atlas_data[np.isin(atlas_data, ids)])
        }
        missing_label_ids = sorted(set(ids) - present_label_ids)
        if missing_label_ids:
            message = (
                f"FastSurfer ROI '{roi_name}' resolved from alias '{resolved_roi.matched_alias}' "
                f"to label id(s) {ids} ({', '.join(component_names)}), but label id(s) "
                f"{missing_label_ids} are absent from atlas '{fs_atlas}'."
            )
            if any(label_id >= 11000 for label_id in ids):
                message += (
                    " This is a Destrieux/a2009s-style label. Use an aparc.a2009s+aseg "
                    "atlas for direct ROI comparisons, or request an explicit *_dkt "
                    "fallback alias when intentionally using a DKT/coarse atlas."
                )
            raise ValueError(message)

        combined_mask = np.isin(atlas_data, ids).astype(np.uint8)
        source_voxels = int(np.count_nonzero(combined_mask))
        if source_voxels == 0:
            message = (
                f"FastSurfer ROI '{roi_name}' resolved from alias '{resolved_roi.matched_alias}' "
                f"to label id(s) {ids} ({', '.join(component_names)}), but those label id(s) "
                f"are absent from atlas '{fs_atlas}'."
            )
            if any(label_id >= 11000 for label_id in ids):
                message += (
                    " This is a Destrieux/a2009s-style label. Use an aparc.a2009s+aseg "
                    "atlas for direct ROI comparisons, or request an explicit *_dkt "
                    "fallback alias when intentionally using a DKT/coarse atlas."
                )
            raise ValueError(message)

        # resample to TI grid if needed
        cropped = try_fast_crop_to_target(atlas_img, ti_img, combined_mask.astype(bool))
        if cropped is not None:
            combined_mask = cropped
            resampled_img = nib.Nifti1Image(cropped.astype(np.uint8), ti_img.affine, ti_img.header)
        else:
            # fallback (what you already have)
            resampled = resample_from_to(nib.Nifti1Image(combined_mask, atlas_img.affine), ti_img, order=0)
            combined_mask = np.asarray(resampled.dataobj).astype(bool)
            resampled_img = resampled

        resampled_voxels = int(np.count_nonzero(combined_mask))
        if resampled_voxels == 0:
            raise ValueError(
                f"FastSurfer ROI '{roi_name}' has {source_voxels} voxel(s) in atlas '{fs_atlas}' "
                "but 0 voxel(s) after mapping to the TI grid. Check that the subject atlas, "
                "TI image, and affine/header belong to the same subject and coordinate space."
            )

        roi_masks[roi_name] = combined_mask
        atlas_imgs[roi_name] = resampled_img

    return roi_masks, atlas_imgs

import csv
def write_csv(out_path: str, ijk: np.ndarray, xyz: np.ndarray, vals: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['I', 'J', 'K', 'X_mm', 'Y_mm', 'Z_mm', 'Value'])
        for (i, j, k), (x, y, z), v in zip(ijk, xyz, vals):
            writer.writerow([i, j, k, f"{x:.2f}", f"{y:.2f}", f"{z:.2f}", f"{v:.6g}"])

# ------------------ Misc from your original file ------------------

def close_gmsh_windows():
    stop_flag = True
    while stop_flag:
        try:
            result = subprocess.run(['bash', '-c', 'xdotool search --name "Gmsh" windowkill'])
            if result.returncode == 1:
                stop_flag = False
        except Exception:
            stop_flag = False

def format_output_dir(directory_path: str) -> None:
    if not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        return
    for fname in os.listdir(directory_path):
        fpath = os.path.join(directory_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"Deleted {fpath}")

def generate_mesh_from_nii(output_path: str, T1_path: str, T2_path: str = None) -> str:
    try:
        subprocess.run(["charm", T1_path, T2_path, output_path])
        return True
    except Exception as e:
        print(f"Error creating volumetric mesh: {e}")
        return False

def atomic_replace(src_path: str, dst_path: str, force_int: bool = False, int_dtype: str = "uint16"):
    src = Path(src_path); dst = Path(dst_path); dst_dir = dst.parent; dst_dir.mkdir(parents=True, exist_ok=True)
    dst_suffix = "".join(dst.suffixes)
    fd, tmp_name = tempfile.mkstemp(dir=dst_dir, suffix=dst_suffix); os.close(fd)
    try:
        img = nib.load(str(src))
        if force_int:
            data = np.asarray(img.dataobj).astype(int_dtype, copy=False)
            out = nib.Nifti1Image(data, img.affine, img.header)
            out.header.set_data_dtype(int_dtype)
            nib.save(out, tmp_name)
        else:
            nib.save(img, tmp_name)
        with open(tmp_name, "rb") as f: os.fsync(f.fileno())
        try:
            dfd = os.open(str(dst_dir), os.O_DIRECTORY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except Exception:
            pass
        os.replace(tmp_name, str(dst))
    except Exception:
        try: os.unlink(tmp_name)
        except Exception: pass
        raise

def merge_segmentation_maps(*args, **kwargs):
    # unchanged vs your version, retained for compatibility
    from typing import Dict
    from nibabel.processing import resample_from_to as _rf2
    import numpy as _np

    manual_seg, charm_seg = args[:2]
    man_img = nib.load(manual_seg) if not isinstance(manual_seg, nib.Nifti1Image) else manual_seg
    cha_img = nib.load(charm_seg)  if not isinstance(charm_seg,  nib.Nifti1Image)  else charm_seg

    if (man_img.shape != cha_img.shape) or (not np.allclose(man_img.affine, cha_img.affine, atol=1e-5)):
        man_img = _rf2(man_img, cha_img, order=0)

    man = _np.asarray(man_img.get_fdata(), dtype=_np.int32)
    cha = _np.asarray(cha_img.get_fdata(), dtype=_np.int32)
    envelope = cha > 0

    manual_skin_id = kwargs.get("manual_skin_id", 5)
    background_label = kwargs.get("background_label", 0)
    man_scalp = (man == manual_skin_id)
    outside = man_scalp & (~envelope)
    removed_voxels = int(outside.sum()); total_scalp_voxels = int(man_scalp.sum())
    out_arr = man.copy(); out_arr[outside] = background_label
    out_img = nib.Nifti1Image(out_arr.astype(np.int16, copy=False), cha_img.affine, cha_img.header)

    output_path = kwargs.get("output_path")
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        nib.save(out_img, output_path)

    debug = {
        "removed_outside_voxels": removed_voxels,
        "manual_scalp_voxels": total_scalp_voxels,
        "dilate_envelope_voxels": int(kwargs.get("dilate_envelope_voxels", 0)),
        "background_label": int(background_label),
    }
    return out_img, debug
