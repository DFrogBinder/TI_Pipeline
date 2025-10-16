import os
import shutil
import tempfile
import json
import numpy as np
import nibabel as nib
import subprocess

import csv, numpy as np, nibabel as nib

from pathlib import Path
from scipy.ndimage import binary_dilation  
from typing import Tuple, Union, Optional, Sequence
from typing import Optional, Sequence, Tuple, Dict, Any, Union

from nilearn import datasets, image as nli, plotting
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine
from nilearn.image import resample_to_img
from nilearn.plotting import plot_anat

NiftiLike = Union[str, os.PathLike, nib.Nifti1Image]

# Which ROIs to extract. The queries are matched case-insensitively against labels.
# We fetch the appropriate HO atlas (cortical OR subcortical) per-ROI under the hood.
ROI_QUERIES = {
    "M1":          {"atlas": "cort-maxprob-thr25-2mm", "query": "precentral gyrus"},
    "Hippocampus": {"atlas": "sub-maxprob-thr25-2mm",  "query": "hippocampus"},
}

EFIELD_PERCENTILE     = 95   # top percentile (e.g., 90 or 95)
WRITE_PER_VOXEL_CSV   = True # CSV per voxel; set False if files get too big
# ------------------------------------------------
#region post_funciton

def overlay_ti_thresholds_on_t1_with_roi(
    *,
    ti_img: nib.Nifti1Image,         # 3D scalar TI (or 4D vector -> handled)
    t1_img: nib.Nifti1Image,         # 3D T1
    roi_mask_img: nib.Nifti1Image,   # 0/1 mask for M1
    out_prefix: str,                 # base name (saves 2 PNGs)
    percentile: float = 95.0,
    hard_threshold: float = 200,
    contour_color: str = "lime",
    contour_linewidth: float = 2.5,
    cmap: str = "jet",
    dpi: int = 150,
    alpha: float = 0.85
) -> tuple[str, str]:
    """
    Save two PNG overlays:
      1. TI ≥ percentile
      2. TI ≥ hard_threshold
    Both show T1 as background and the ROI outlined.

    Returns (percentile_path, hard_threshold_path).
    """
    
    data = ti_img.get_fdata()
    finite = np.isfinite(data)
    if np.any(finite):
        vmin = np.percentile(data[finite], 2)
        vmax = np.percentile(data[finite], 98)
        if vmin >= vmax:  # fallback if data are weird
            vmin, vmax = np.nanmin(data[finite]), np.nanmax(data[finite])
    
    # Ensure scalar TI
    ti_arr = load_ti_as_scalar(ti_img)
    ti_scalar_img = nib.Nifti1Image(ti_arr, ti_img.affine, ti_img.header)

    # Resample to TI grid
    t1_on_ti  = resample_to_img(t1_img, ti_scalar_img, interpolation="continuous")
    roi_on_ti = resample_to_img(roi_mask_img, ti_scalar_img, interpolation="nearest")

    # Compute thresholds
    arr = np.asarray(ti_arr, dtype=float)
    finite_pos = np.isfinite(arr) & (arr > 0)
    if not np.any(finite_pos):
        raise ValueError("TI has no positive finite voxels.")
    thr_percentile = float(np.percentile(arr[finite_pos], percentile))
    thr_fixed = float(hard_threshold)

    def _plot_overlay(thr_value: float, label: str):
        masked = np.where(arr >= thr_value, arr, 0.0)
        shown = masked[masked > 0]
        if np.any(shown):
            vmin = np.percentile(shown, 2)
            vmax = np.percentile(shown, 98)
            if vmin >= vmax:  # fallback if data are weird
                vmin, vmax = np.nanmin(shown), np.nanmax(shown)

        display = plot_anat(
            t1_on_ti,
            display_mode="ortho",
            dim=0,
            annotate=True,
            draw_cross=False,
            colorbar=False,      # <- NO colorbar for anatom
            black_bg=True,
            title=f"TI ≥ {thr_value:.3f} ({label})",
            )
        display.add_overlay(
            ti_img,
            colorbar=True,       # <- YES colorbar for E-field
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",      # or any cmap you prefer
        )
        display.add_contours(
            roi_on_ti,
            levels=[0.5],
            colors=[contour_color],
            linewidths=contour_linewidth
        )

        out_path = f"{out_prefix}_{label}.png"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        display.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        display.close()
        return out_path

    # Generate both overlays
    png_percentile = _plot_overlay(thr_percentile, f"top{int(percentile)}")
    png_fixed      = _plot_overlay(thr_fixed, f"above{hard_threshold:.2f}")

    return png_percentile, png_fixed


def normalize_roi_name(name: str) -> str:
    """Make a filesystem-safe ROI name."""
    return "".join(c if c.isalnum() else "_" for c in name.strip().replace(" ", "_"))
def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
def vol_mm3(img: nib.Nifti1Image) -> float:
    """Compute voxel volume in mm^3."""
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms))
def load_ti_as_scalar(img: nib.Nifti1Image) -> np.ndarray:
    """Load TI data as a scalar array, converting from vector if needed."""
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 3:
        # Vector field: compute magnitude
        data = np.linalg.norm(data, axis=3)
    elif data.ndim != 3:
        raise ValueError(f"Unexpected TI data shape: {data.shape}")
    return data
def save_masked_nii(data: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    """Save a NIfTI file with data masked by mask, using ref_img's affine and header."""
    masked_data = np.where(mask, data, 0)
    out_img = nib.Nifti1Image(masked_data, ref_img.affine, ref_img.header)
    nib.save(out_img, out_path)
from nilearn import plotting
import numpy as np
import nibabel as nib

def make_overlay_png(out_png, overlay_img, bg_img=None, title=None, roi_mask_img=None, abs_colour=False):
    """
    overlay_img: NIfTI of the E-field scalar (TI). The colorbar will use its values.
    bg_img:      T1 (resampled to TI grid), shown without a colorbar.
    roi_mask_img: optional binary mask to be drawn as a contour.
    """
    # Choose sensible vmin/vmax from E-field data (ignore NaNs/Infs)
    data = overlay_img.get_fdata()
    finite = np.isfinite(data)
    if np.any(finite):
        if abs_colour == True:
            vmin = np.min(data[finite])
            vmax = np.max(data[finite])
        else:
            vmin = np.percentile(data[finite], 2)
            vmax = np.percentile(data[finite], 98)
        if vmin >= vmax:  # fallback if data are weird
            vmin, vmax = np.nanmin(data[finite]), np.nanmax(data[finite])
    else:
        vmin = vmax = None  # nilearn will handle, but values won't mean much

    if bg_img is not None:
        disp = plotting.plot_anat(
            bg_img,
            annotate=False,
            draw_cross=False,
            black_bg=True,
            colorbar=False,      # <- NO colorbar for anatomy
        )
        # Add the E-field as an overlay WITH colorbar
        disp.add_overlay(
            overlay_img,
            colorbar=True,       # <- YES colorbar for E-field
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",      # or any cmap you prefer
        )
    else:
        # Fall back to showing the E-field alone, colorbar included
        disp = plotting.plot_img(
            overlay_img,
            annotate=False,
            draw_cross=False,
            black_bg=True,
            colorbar=True,       # <- colorbar reflects overlay
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

    # Optional: add ROI contour on top
    if roi_mask_img is not None:
        disp.add_contours(roi_mask_img, levels=[0.5], linewidths=1.5, colors="yellow")

    if title:
        disp.title(title)

    disp.savefig(out_png, dpi=300)
    disp.close()

def basic_stats(data: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute basic statistics (min, max, mean, std) on data, optionally within a mask."""
    if mask is not None:
        data = data[mask]
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.min(finite_data)),
        "max": float(np.max(finite_data)),
        "mean": float(np.mean(finite_data)),
        "std": float(np.std(finite_data)),
    }
def roi_masks_on_ti_grid(ti_img: nib.Nifti1Image) -> Tuple[Dict[str, np.ndarray], Dict[str, nib.Nifti1Image]]:
    """Fetch ROI masks on the TI image grid.
    Returns a dict of ROI name -> boolean mask arrays, and a dict of ROI name -> nibabel images.
    """
    def as_niimg(maybe_img):
        """Return a nibabel image whether input is a path or already an image."""
        # Already a nibabel spatial image (covers Nifti1Image, etc.)
        if isinstance(maybe_img, nib.spatialimages.SpatialImage):
            return maybe_img
        # Path-like (string or Path)
        if isinstance(maybe_img, (str, Path)):
            return nib.load(str(maybe_img))
        # Anything with get_fdata() quacks like a niimg
        if hasattr(maybe_img, "get_fdata"):
            return maybe_img
        raise TypeError(f"Expected path or niimg, got {type(maybe_img)}")
    roi_masks = {}
    atlas_imgs = {}
    for roi_name, info in ROI_QUERIES.items():
        atlas = datasets.fetch_atlas_harvard_oxford(info["atlas"])
        atlas_img = as_niimg(atlas.maps)
        
        atlas_data = np.asarray(atlas_img.dataobj)
        labels = atlas.labels

        # Find label indices matching the query (case-insensitive substring match)
        matched_indices = [i for i, label in enumerate(labels) if info["query"].lower() in label.lower()]
        if not matched_indices:
            raise ValueError(f"No labels found matching query '{info['query']}' in atlas '{info['atlas']}'")

        # Build combined mask for all matched labels
        combined_mask = np.isin(atlas_data, matched_indices)

        # Resample to TI grid if needed
        if (atlas_img.shape != ti_img.shape) or (not np.allclose(atlas_img.affine, ti_img.affine, atol=1e-5)):
            resampled_img = resample_from_to(nib.Nifti1Image(combined_mask.astype(np.uint8), atlas_img.affine), ti_img, order=0)
            combined_mask = np.asarray(resampled_img.dataobj).astype(bool)
            atlas_img = resampled_img  # update to resampled image

        roi_masks[roi_name] = combined_mask
        atlas_imgs[roi_name] = atlas_img

    return roi_masks, atlas_imgs
def extract_table(mask: np.ndarray, ref_img: nib.Nifti1Image, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract voxel indices, coordinates, and data values within the mask."""
    ijk = np.argwhere(mask)
    xyz = apply_affine(ref_img.affine, ijk)
    vals = data[mask]
    return ijk, xyz, vals
def write_csv(out_path: str, ijk: np.ndarray, xyz: np.ndarray, vals: np.ndarray) -> None:
    """Write voxel indices, coordinates, and values to a
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['I', 'J', 'K', 'X_mm', 'Y_mm', 'Z_mm', 'Value'])
        for (i, j, k), (x, y, z), v in zip(ijk, xyz, vals):
            writer.writerow([i, j, k, f"{x:.2f}", f"{y:.2f}", f"{z:.2f}", f"{v:.6g}"]) CSV file."""
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['I', 'J', 'K', 'X_mm', 'Y_mm', 'Z_mm', 'Value'])
        for (i, j, k), (x, y, z), v in zip(ijk, xyz, vals):
            writer.writerow([i, j, k, f"{x:.2f}", f"{y:.2f}", f"{z:.2f}", f"{v:.6g}"])      



def close_gmsh_windows():
    """Kill any lingering Gmsh GUI windows."""
    stop_flag = True
    while stop_flag:
        try:
            result = subprocess.run(
                ['bash', '-c', 'xdotool search --name "Gmsh" windowkill']
            )
            if result.returncode == 1:
                stop_flag = False
        except Exception:
            stop_flag = False

#region 3d gen functions
def format_output_dir(directory_path: str) -> None:
    """Delete all files in a folder (leave subfolders intact)."""
    if not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        return
    for fname in os.listdir(directory_path):
        fpath = os.path.join(directory_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"Deleted {fpath}")
def generate_mesh_from_nii( output_path: str, T1_path: str, T2_path: str = None) -> str:
    try:
        subprocess.run(["charm", T1_path, T2_path, output_path])
        return True
    except Exception as e:
        print(f"Error creating volumetric mesh: {e}")
        return False

def atomic_replace(src_path: str, dst_path: str, force_int: bool = False, int_dtype: str = "uint16"):
    """
    Atomically replace dst_path with src_path's NIfTI contents.
    Handles .nii <-> .nii.gz re-encoding and preserves the source file.
    Optionally force integer dtype for segmentation maps.

    Args:
        src_path: path to the source NIfTI (.nii or .nii.gz)
        dst_path: path to the target NIfTI (.nii or .nii.gz)
        force_int: if True, cast data to integer (use for label maps)
        int_dtype: integer dtype to cast to when force_int=True (e.g., 'uint16' or 'int16')
    """
    src = Path(src_path)
    dst = Path(dst_path)
    dst_dir = dst.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp file in the destination directory so the final os.replace is atomic
    dst_suffix = "".join(dst.suffixes)  # handles '.nii.gz'
    fd, tmp_name = tempfile.mkstemp(dir=dst_dir, suffix=dst_suffix)
    os.close(fd)

    try:
        img = nib.load(str(src))
        if force_int:
            data = np.asarray(img.dataobj).astype(int_dtype, copy=False)
            out = nib.Nifti1Image(data, img.affine, img.header)
            out.header.set_data_dtype(int_dtype)
            nib.save(out, tmp_name)
        else:
            # Re-save in the destination's format (nibabel picks compression from extension)
            nib.save(img, tmp_name)

        # Ensure file contents are flushed to disk
        with open(tmp_name, "rb") as f:
            os.fsync(f.fileno())

        # Fsync the directory metadata (best-effort)
        try:
            dfd = os.open(str(dst_dir), os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass

        # Atomic swap
        os.replace(tmp_name, str(dst))
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise

def merge_segmentation_maps(
    manual_seg: NiftiLike,
    charm_seg: NiftiLike,
    *,
    output_path: Optional[str] = None,
    manual_skin_id: int = 5,
    background_label: int = 0,
    dilate_envelope_voxels: int = 0,
    # optional artifacts
    save_envelope_path: Optional[str] = None,
    save_outside_mask_path: Optional[str] = None,
    qc_dir: Optional[str] = None,
    make_summary_json: bool = False,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """
    Clip manual scalp to CHARM's head envelope (>0), as suggested:
      1) envelope = (CHARM > 0)
      2) manual scalp outside envelope -> set to background_label
      3) keep all other manual labels as-is
    If grids differ, manual is resampled to CHARM grid using nearest-neighbor.

    Parameters
    ----------
    manual_seg : path or NIfTI
        Manual/Jake segmentation (integer labels).
    charm_seg : path or NIfTI
        CHARM segmentation (integer labels). Defines target grid & envelope.
    output_path : str or None
        If given, save the clipped segmentation here (.nii or .nii.gz).
    manual_skin_id : int
        Label id used for scalp/skin in the manual segmentation.
    background_label : int
        Label id to assign where manual scalp lies outside the envelope.
    dilate_envelope_voxels : int
        Optional morphological dilation steps for the envelope (forgiving on tight areas).
    save_envelope_path : str or None
        If set, save the (possibly dilated) envelope as a NIfTI (uint8).
    save_outside_mask_path : str or None
        If set, save the mask of manual scalp voxels removed (uint8).
    qc_dir : str or None
        If set, save a few axial QC PNGs (requires matplotlib).
    make_summary_json : bool
        If True and qc_dir is set, write a summary.json there.

    Returns
    -------
    out_img : nib.Nifti1Image
        The clipped segmentation on CHARM's grid.
    debug : dict
        Stats and parameter echo.
    """
    def _resample_nn(src_img, tgt_img):
        from nibabel.processing import resample_from_to
        return resample_from_to(src_img, tgt_img, order=0)

    def _maybe_dilate(mask: np.ndarray, steps: int) -> np.ndarray:
        if steps <= 0:
            return mask
        try:
            from scipy.ndimage import binary_dilation
        except Exception:
            # SciPy not available -> skip dilation
            return mask
        out = mask.copy()
        for _ in range(int(steps)):
            out = binary_dilation(out)
        return out

    # --- Load ---
    man_img = nib.load(manual_seg) if not isinstance(manual_seg, nib.Nifti1Image) else manual_seg
    cha_img = nib.load(charm_seg)  if not isinstance(charm_seg,  nib.Nifti1Image)  else charm_seg

    # --- Align manual -> CHARM grid if needed (NN preserves labels) ---
    if (man_img.shape != cha_img.shape) or (not np.allclose(man_img.affine, cha_img.affine, atol=1e-5)):
        man_img = _resample_nn(man_img, cha_img)

    man = np.asarray(man_img.get_fdata(), dtype=np.int32)
    cha = np.asarray(cha_img.get_fdata(), dtype=np.int32)

    # --- Build envelope (>0), optional dilation ---
    envelope = cha > 0
    envelope = _maybe_dilate(envelope, dilate_envelope_voxels)

    # --- Manual scalp & outside mask ---
    man_scalp = (man == manual_skin_id)
    outside = man_scalp & (~envelope)

    removed_voxels = int(outside.sum())
    total_scalp_voxels = int(man_scalp.sum())

    # --- Apply clipping ---
    out_arr = man.copy()
    out_arr[outside] = background_label

    out_img = nib.Nifti1Image(out_arr.astype(np.int16, copy=False), cha_img.affine, cha_img.header)

    # --- Save main output (optional) ---
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        nib.save(out_img, output_path)

    # --- Save side artifacts (optional) ---
    if save_envelope_path:
        env_img = nib.Nifti1Image(envelope.astype(np.uint8, copy=False), cha_img.affine, cha_img.header)
        nib.save(env_img, save_envelope_path)

    if save_outside_mask_path:
        outside_img = nib.Nifti1Image(outside.astype(np.uint8, copy=False), cha_img.affine, cha_img.header)
        nib.save(outside_img, save_outside_mask_path)

    # --- QC PNGs (optional) ---
    if qc_dir:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(qc_dir, exist_ok=True)
            Z = out_arr.shape[2]
            zs = sorted(set([Z // 2, Z // 3, (2 * Z) // 3]))
            for i, z in enumerate(zs, 1):
                base = envelope[:, :, z].astype(float)
                pre_scalp = (man[:, :, z] == manual_skin_id)
                post_scalp = (out_arr[:, :, z] == manual_skin_id)
                removed = outside[:, :, z]

                fig = plt.figure(figsize=(6, 6), dpi=120)
                plt.imshow(base, vmin=0, vmax=1)  # grayscale envelope
                # simple scatter overlays (fast & readable)
                ys, xs = np.where(pre_scalp)
                plt.scatter(xs, ys, s=0.15, label="manual scalp (pre)")
                ys, xs = np.where(removed)
                if ys.size:
                    plt.scatter(xs, ys, s=0.15, label="removed (outside)")
                ys, xs = np.where(post_scalp)
                plt.scatter(xs, ys, s=0.15, label="manual scalp (post)")
                plt.title(f"QC: z={z}")
                plt.legend(markerscale=10, loc="lower right", fontsize=7)
                plt.axis("off")
                out_png = os.path.join(qc_dir, f"qc_axial_{i}_z{z}.png")
                plt.savefig(out_png, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass  # QC is optional; ignore plotting errors

        if make_summary_json:
            summary = {
                "manual_skin_id": int(manual_skin_id),
                "background_label": int(background_label),
                "dilate_envelope_voxels": int(dilate_envelope_voxels),
                "stats": {
                    "manual_scalp_voxels": total_scalp_voxels,
                    "removed_outside_voxels": removed_voxels,
                },
                "shapes": {
                    "output_shape": list(map(int, out_arr.shape)),
                },
                "paths": {
                    "output_path": output_path,
                    "envelope_path": save_envelope_path,
                    "outside_mask_path": save_outside_mask_path,
                },
            }
            with open(os.path.join(qc_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    debug = {
        "removed_outside_voxels": removed_voxels,
        "manual_scalp_voxels": total_scalp_voxels,
        "dilate_envelope_voxels": int(dilate_envelope_voxels),
        "background_label": int(background_label),
    }
    return out_img, debug

