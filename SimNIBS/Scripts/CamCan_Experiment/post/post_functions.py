import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any, List, Literal

import numpy as np
import pandas as pd
import nibabel as nib
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




NiftiLike = Union[str, os.PathLike, nib.Nifti1Image]
AtlasMode = Literal["auto", "mni", "fastsurfer"]

# ------------------ ROI dictionaries ------------------

# Harvard–Oxford (MNI) queries (substring matches over HO label list)
ROI_QUERIES_OXFORD = {
    "M1":          {"atlas": "cort-maxprob-thr25-2mm", "query": "precentral gyrus"},
    "Hippocampus": {"atlas": "sub-maxprob-thr25-2mm",  "query": "hippocampus"},
}

# FastSurfer DKT+aseg (exact or substring match over our map values)
ROI_QUERIES_FASTSURFER = {
    "M1":          {"atlas": "fastsurfer", "query": "ctx-lh-precentral"},   # left M1
    "Hippocampus": {"atlas": "fastsurfer", "query": "Left-Hippocampus"},
}

EFIELD_PERCENTILE   = 95
WRITE_PER_VOXEL_CSV = True

# ------------------ FastSurfer DKT labels ------------------

fastsurfer_dkt_labels = {
    # Subcortical
    0: "Unknown", 2: "Left-Cerebral-White-Matter", 3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle", 5: "Left-Inf-Lat-Vent", 7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex", 10: "Left-Thalamus-Proper", 11: "Left-Caudate",
    12: "Left-Putamen", 13: "Left-Pallidum", 14: "3rd-Ventricle", 15: "4th-Ventricle",
    16: "Brain-Stem", 17: "Left-Hippocampus", 18: "Left-Amygdala", 24: "CSF",
    26: "Left-Accumbens-area", 28: "Left-VentralDC", 30: "Left-vessel", 31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter", 42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle", 44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter", 47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus-Proper", 50: "Right-Caudate", 51: "Right-Putamen",
    52: "Right-Pallidum", 53: "Right-Hippocampus", 54: "Right-Amygdala",
    58: "Right-Accumbens-area", 60: "Right-VentralDC", 62: "Right-vessel", 63: "Right-choroid-plexus",

    # Left cortex (DKT)
    1000: "ctx-lh-bankssts", 1001: "ctx-lh-caudalanteriorcingulate",
    1002: "ctx-lh-caudalmiddlefrontal", 1003: "ctx-lh-cuneus",
    1004: "ctx-lh-entorhinal", 1005: "ctx-lh-fusiform", 1006: "ctx-lh-inferiorparietal",
    1007: "ctx-lh-inferiortemporal", 1008: "ctx-lh-isthmuscingulate",
    1009: "ctx-lh-lateraloccipital", 1010: "ctx-lh-lateralorbitofrontal",
    1011: "ctx-lh-lingual", 1012: "ctx-lh-medialorbitofrontal", 1013: "ctx-lh-middletemporal",
    1014: "ctx-lh-parahippocampal", 1015: "ctx-lh-paracentral", 1016: "ctx-lh-parsopercularis",
    1017: "ctx-lh-parsorbitalis", 1018: "ctx-lh-parstriangularis", 1019: "ctx-lh-pericalcarine",
    1020: "ctx-lh-postcentral", 1021: "ctx-lh-posteriorcingulate", 1022: "ctx-lh-precentral",
    1023: "ctx-lh-precuneus", 1024: "ctx-lh-rostralanteriorcingulate",
    1025: "ctx-lh-rostralmiddlefrontal", 1026: "ctx-lh-superiorfrontal",
    1027: "ctx-lh-superiorparietal", 1028: "ctx-lh-superiortemporal",
    1029: "ctx-lh-supramarginal", 1030: "ctx-lh-frontalpole",
    1031: "ctx-lh-temporalpole", 1032: "ctx-lh-transversetemporal", 1033: "ctx-lh-insula",

    # Right cortex (DKT)
    2000: "ctx-rh-bankssts", 2001: "ctx-rh-caudalanteriorcingulate",
    2002: "ctx-rh-caudalmiddlefrontal", 2003: "ctx-rh-cuneus",
    2004: "ctx-rh-entorhinal", 2005: "ctx-rh-fusiform", 2006: "ctx-rh-inferiorparietal",
    2007: "ctx-rh-inferiortemporal", 2008: "ctx-rh-isthmuscingulate",
    2009: "ctx-rh-lateraloccipital", 2010: "ctx-rh-lateralorbitofrontal",
    2011: "ctx-rh-lingual", 2012: "ctx-rh-medialorbitofrontal", 2013: "ctx-rh-middletemporal",
    2014: "ctx-rh-parahippocampal", 2015: "ctx-rh-paracentral", 2016: "ctx-rh-parsopercularis",
    2017: "ctx-rh-parsorbitalis", 2018: "ctx-rh-parstriangularis", 2019: "ctx-rh-pericalcarine",
    2020: "ctx-rh-postcentral", 2021: "ctx-rh-posteriorcingulate", 2022: "ctx-rh-precentral",
    2023: "ctx-rh-precuneus", 2024: "ctx-rh-rostralanteriorcingulate",
    2025: "ctx-rh-rostralmiddlefrontal", 2026: "ctx-rh-superiorfrontal",
    2027: "ctx-rh-superiorparietal", 2028: "ctx-rh-superiortemporal",
    2029: "ctx-rh-supramarginal", 2030: "ctx-rh-frontalpole",
    2031: "ctx-rh-temporalpole", 2032: "ctx-rh-transversetemporal", 2033: "ctx-rh-insula",
}

# ------------------ Overlay helpers ------------------

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
    cmap: str = "jet",
    dpi: int = 150,
    alpha: float = 0.85
) -> tuple[str, str, Optional[str]]:

    ti_arr = load_ti_as_scalar(ti_img)
    ti_scalar_img = nib.Nifti1Image(ti_arr, ti_img.affine, ti_img.header)

    t1_on_ti  = resample_to_img(t1_img, ti_scalar_img, interpolation="continuous")
    roi_on_ti = resample_to_img(roi_mask_img, ti_scalar_img, interpolation="nearest")

    arr = np.asarray(ti_arr, dtype=float)
    finite_pos = np.isfinite(arr) & (arr > 0)
    if not np.any(finite_pos):
        raise ValueError("TI has no positive finite voxels.")
    thr_percentile = float(np.percentile(arr[finite_pos], percentile))
    thr_fixed = float(hard_threshold)
    global_vmax = float(np.nanmax(arr[finite_pos]))

    roi_data = np.asarray(roi_on_ti.dataobj)
    roi_coords = np.argwhere(roi_data > 0)
    if roi_coords.size:
        center_ijk = roi_coords.mean(axis=0)
        center_xyz = nib.affines.apply_affine(roi_on_ti.affine, center_ijk)
        center_xyz = np.asarray(center_xyz, dtype=float)
        center_xyz[2] += float(z_offset_mm)
        cut_coords = tuple(float(x) for x in center_xyz)
    else:
        cut_coords = (0.0, 0.0, 0.0)

    def _plot_overlay(thr_value: Optional[float], label: str):
        if thr_value is None:
            overlay_data = arr
            subset = arr[finite_pos]
        else:
            overlay_data = np.where(arr >= thr_value, arr, 0.0)
            subset = overlay_data[overlay_data > 0]
        vmin = float(np.nanmin(subset)) if subset.size else 0.0
        vmax = global_vmax
        overlay_img = nib.Nifti1Image(overlay_data, ti_img.affine, ti_img.header)

        display = plot_anat(
            t1_on_ti, display_mode="ortho", dim=0, annotate=True,
            draw_cross=True, colorbar=False, black_bg=True, cut_coords=cut_coords,
            title=f"TI ≥ {thr_value:.3f} ({label})" if thr_value is not None else "TI (full field)",
        )
        display.add_overlay(
            overlay_img, colorbar=True, vmin=vmin, vmax=vmax, cmap="viridis"
        )
        display.add_contours(
            roi_on_ti, levels=[0.5], colors=[contour_color], linewidths=contour_linewidth
        )

        if subject:
            out_path = f"{out_prefix}_{subject}_{label}.png"
        else:
            out_path = f"{out_prefix}_{label}.png"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        display.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        display.close()
        return out_path

    png_full = _plot_overlay(None, "full") if include_full_field else None
    png_percentile = _plot_overlay(thr_percentile, f"top{int(percentile)}")
    png_fixed = _plot_overlay(thr_fixed, f"above{hard_threshold:.2f}")
    return png_percentile, png_fixed, png_full


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
) -> tuple[str, str, Optional[str]]:
    """
    Overlay TI on T1 with ROI contour using per-image vmin/vmax.
    Each image uses min/max of its own subset.
    """
    ti_arr = load_ti_as_scalar(ti_img)
    ti_scalar_img = nib.Nifti1Image(ti_arr, ti_img.affine, ti_img.header)

    t1_on_ti = resample_to_img(t1_img, ti_scalar_img, interpolation="continuous")
    roi_on_ti = resample_to_img(roi_mask_img, ti_scalar_img, interpolation="nearest")

    arr = np.asarray(ti_arr, dtype=float)
    finite_pos = np.isfinite(arr) & (arr > 0)
    if not np.any(finite_pos):
        raise ValueError("TI has no positive finite voxels.")
    thr_percentile = float(np.percentile(arr[finite_pos], percentile))
    thr_fixed = float(hard_threshold)

    roi_data = np.asarray(roi_on_ti.dataobj)
    roi_coords = np.argwhere(roi_data > 0)
    if roi_coords.size:
        center_ijk = roi_coords.mean(axis=0)
        center_xyz = nib.affines.apply_affine(roi_on_ti.affine, center_ijk)
        center_xyz = np.asarray(center_xyz, dtype=float)
        center_xyz[2] += float(z_offset_mm)
        cut_coords = tuple(float(x) for x in center_xyz)
    else:
        cut_coords = (0.0, 0.0, 0.0)

    def _plot_overlay(thr_value: Optional[float], label: str):
        if thr_value is None:
            overlay_data = arr
            subset = arr[finite_pos]
        else:
            overlay_data = np.where(arr >= thr_value, arr, 0.0)
            subset = overlay_data[overlay_data > 0]
        vmin = float(np.nanmin(subset)) if subset.size else 0.0
        vmax = float(np.nanmax(subset)) if subset.size else 1.0
        overlay_img = nib.Nifti1Image(overlay_data, ti_img.affine, ti_img.header)

        display = plot_anat(
            t1_on_ti, display_mode="ortho", dim=0, annotate=True,
            draw_cross=True, colorbar=False, black_bg=True, cut_coords=cut_coords,
            title=f"TI ≥ {thr_value:.3f} ({label})" if thr_value is not None else "TI (full field)",
        )
        display.add_overlay(
            overlay_img, colorbar=True, vmin=vmin, vmax=vmax, cmap=cmap
        )
        display.add_contours(
            roi_on_ti, levels=[0.5], colors=[contour_color], linewidths=contour_linewidth
        )

        if subject:
            out_path = f"{out_prefix}_{subject}_{label}.png"
        else:
            out_path = f"{out_prefix}_{label}.png"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        display.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        display.close()
        return out_path

    png_full = _plot_overlay(None, "full") if include_full_field else None
    png_percentile = _plot_overlay(thr_percentile, f"top{int(percentile)}")
    png_fixed = _plot_overlay(thr_fixed, f"above{hard_threshold:.2f}")
    return png_percentile, png_fixed, png_full

# ------------------ ROI extraction core ------------------

def load_custom_atlas(atlas_path: NiftiLike) -> nib.Nifti1Image:
    if isinstance(atlas_path, (str, os.PathLike)):
        return nib.load(str(atlas_path))
    if isinstance(atlas_path, nib.Nifti1Image):
        return atlas_path
    raise TypeError(f"Expected path or NIfTI image, got {type(atlas_path)}")

def _resolve_fastsurfer_atlas(subject: str, fastsurfer_root: Optional[str], explicit_path: Optional[str]) -> Optional[str]:
    """
    Try several canonical locations for aparc.DKTatlas+aseg.deep.nii.gz.
    Priority: explicit_path > {fastsurfer_root}/{subject}.nii.gz >
              {fastsurfer_root}/{subject}/mri/... > env SUBJECTS_DIR.
    """
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if fastsurfer_root:
        candidates.append(Path(fastsurfer_root) / f"{subject}.nii.gz")
    env_sd = os.environ.get("SUBJECTS_DIR")
    if env_sd:
        candidates.append(Path(env_sd) / subject / "mri" / "aparc.DKTatlas+aseg.deep.nii.gz")

    for c in candidates:
        if c.is_file():
            return str(c)
    return None

def roi_masks_on_ti_grid(
    ti_img: nib.Nifti1Image,
    *,
    atlas_mode: AtlasMode = "auto",
    subject: Optional[str] = None,
    fastsurfer_root: Optional[str] = None,
    fastsurfer_atlas_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, nib.Nifti1Image]]:
    """
    Build ROI masks on the TI grid using either Harvard–Oxford (MNI) or subject FastSurfer atlas.

    atlas_mode:
      - "auto": use FastSurfer if a subject atlas is found (subject != 'MNI152'), else Harvard–Oxford
      - "fastsurfer": require a FastSurfer atlas (error if missing)
      - "mni": always use Harvard–Oxford

    FastSurfer search order: fastsurfer_atlas_path (explicit) >
                             {fastsurfer_root}/{subject}/mri/aparc.DKTatlas+aseg.deep.nii.gz >
                             $SUBJECTS_DIR/{subject}/mri/aparc.DKTatlas+aseg.deep.nii.gz
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
                "or set $SUBJECTS_DIR. Expected 'aparc.DKTatlas+aseg.deep.nii.gz'."
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
    print(f"[INFO] Using FastSurfer segementation for subject '{subject}'")
    assert fs_atlas is not None, "Internal: fs_atlas must be resolved here."

    atlas_img = load_custom_atlas(fs_atlas)
    atlas_data = np.asarray(atlas_img.dataobj).astype(int)

    # Build reverse map name->id (case-insensitive)
    id_to_name = fastsurfer_dkt_labels
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}

    ROI_Q = ROI_QUERIES_FASTSURFER
    for roi_name, info in ROI_Q.items():
        q = info["query"].lower()

        # Try exact name match first, then substring over all names
        if q in name_to_id:
            ids = [name_to_id[q]]
        else:
            ids = [k for k, v in id_to_name.items() if q in v.lower()]
        if not ids:
            raise ValueError(f"No FastSurfer labels match query '{info['query']}'")

        combined_mask = np.isin(atlas_data, ids).astype(np.uint8)

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
