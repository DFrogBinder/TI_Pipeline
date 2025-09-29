import os
import shutil
import tempfile
import json
import numpy as np
import nibabel as nib
import csv, numpy as np, nibabel as nib

from pathlib import Path
from scipy.ndimage import binary_dilation  
from typing import Tuple, Union, Optional, Sequence
from typing import Optional, Sequence, Tuple, Dict, Any, Union

from nilearn import datasets, image as nli, plotting
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine

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

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_ti_as_scalar(img: nib.Nifti1Image) -> np.ndarray:
    """Return scalar field; if 4D with last dim=3, convert vector -> magnitude."""
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 3:
        data = np.linalg.norm(data, axis=-1).astype(np.float32)
    return data

def vol_mm3(img: nib.Nifti1Image) -> float:
    z = img.header.get_zooms()[:3]
    return float(z[0] * z[1] * z[2])

def _labels_to_lower(labels):
    out = []
    for l in labels:
        if isinstance(l, bytes):
            out.append(l.decode("utf-8", errors="ignore").lower())
        else:
            out.append(str(l).lower())
    return out

def fetch_and_resample_ho(atlas_name: str, target_img: nib.Nifti1Image):
    """Fetch a Harvard–Oxford atlas and resample it to target_img grid (nearest-neighbor)."""
    ho = datasets.fetch_atlas_harvard_oxford(atlas_name)
    ho_img = nli.load_img(ho.maps)
    ho_res = resample_from_to(ho_img, target_img, order=0)  # order=0 => NN for labels
    labels = _labels_to_lower(ho.labels)
    return ho_res, labels

def find_label_indices(labels, substr):
    """Return sorted indices for labels containing substr (case-insensitive)."""
    ss = substr.lower()
    idx = [i for i, name in enumerate(labels) if ss in name]
    return sorted(idx)

def roi_masks_on_ti_grid(ti_img: nib.Nifti1Image):
    """
    Build ROI masks for each entry in ROI_QUERIES on the TI grid.
    Returns:
        roi_masks: dict[str, np.ndarray(bool)]
        atlas_imgs: dict[str, nib.Nifti1Image]  # resampled atlas per unique atlas_name
    """
    # group queries by atlas to avoid double fetching
    atlas_groups = {}
    for roi_name, conf in ROI_QUERIES.items():
        atlas_groups.setdefault(conf["atlas"], []).append((roi_name, conf["query"]))

    atlas_imgs = {}
    roi_masks  = {k: None for k in ROI_QUERIES.keys()}

    for atlas_name, roi_list in atlas_groups.items():
        resampled_img, labels = fetch_and_resample_ho(atlas_name, ti_img)
        atlas_imgs[atlas_name] = resampled_img
        atlas_data = resampled_img.get_fdata().astype(np.int32, copy=False)

        for roi_name, query in roi_list:
            ids = find_label_indices(labels, query)
            if not ids:
                raise RuntimeError(f"'{query}' not found in Harvard–Oxford labels for atlas '{atlas_name}'.")
            mask = np.isin(atlas_data, ids)
            roi_masks[roi_name] = mask

    return roi_masks, atlas_imgs

def extract_table(mask: np.ndarray, ti_img: nib.Nifti1Image, values: np.ndarray):
    """Return ijk, xyz(mm), and value arrays for voxels where mask==True."""
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0,3), dtype=int), np.empty((0,3), dtype=float), np.empty((0,), dtype=float)
    xyz = apply_affine(ti_img.affine, ijk)
    vals = values[mask]
    return ijk, xyz, vals

def write_csv(path, ijk, xyz, vals):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["i","j","k","x_mm","y_mm","z_mm","value"])
        for (i,j,k), (x,y,z), v in zip(ijk, xyz, vals):
            w.writerow([int(i), int(j), int(k), float(x), float(y), float(z), float(v)])

def save_masked_nii(path, mask, ti_img, values):
    """Save the scalar field zeroed outside mask."""
    arr = np.zeros_like(values, dtype=np.float32)
    arr[mask] = values[mask]
    nib.save(nib.Nifti1Image(arr, ti_img.affine, ti_img.header), path)

def basic_stats(arr: np.ndarray):
    if arr.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p95=np.nan, max=np.nan)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, p95=np.nan, max=np.nan)
    return dict(
        n=arr.size,
        mean=float(np.nanmean(arr)),
        median=float(np.nanmedian(arr)),
        p95=float(np.nanpercentile(arr, 95)),
        max=float(np.nanmax(arr)),
    )

def make_overlay_png(out_png, stat_img, roi_img, bg_img=None, thr=None, title=None):
    """Render TI as stat map with ROI contours. Uses bg_img if provided."""
    display = plotting.plot_stat_map(
        stat_img,
        bg_img=bg_img if bg_img is not None else None,
        display_mode='ortho',
        threshold=thr,
        colorbar=True,
        annotate=True,
        title=title or "",
    )
    display.add_contours(roi_img, levels=[0.5], linewidths=2)
    display.savefig(out_png, dpi=200)
    display.close()

def _normalize_roi_name(s: str) -> str:
    s = (s or '').strip().lower()
    if s in {'m1','motor','precentral','precentral gyrus','primary motor cortex'}:
        return 'M1'
    if s in {'hip','hipp','hippocampus'}:
        return 'Hippocampus'
    raise SystemExit(f"--plot-roi must be one of: M1, Hippocampus (got: {s})")

def merge_segmentation_maps(
    manual_seg: NiftiLike,
    charm_seg: NiftiLike,
    manual_skin_id: int = 5,
    charm_skin_ids: Tuple[int, ...] = (5,),
    # behaviour controls
    background_label: int = 0,           # what to write where "manual skin" lies outside CHARM skin
    charm_skin_dilate_voxels: int = 0,
    resample_manual: bool = False,
    # optional: dilate CHARM skin to be slightly more permissive
    # --- mask saving/return ---
    skin_mask_out: Optional[str] = None,
    return_mask: bool = False,
    mask_preserve_labels: bool = False,
    return_debug: bool = False,
) -> Union[
    nib.Nifti1Image,
    Tuple[nib.Nifti1Image, nib.Nifti1Image],
    Tuple[nib.Nifti1Image, dict],
    Tuple[nib.Nifti1Image, nib.Nifti1Image, dict],
]:
    """
    Replace the skin/scalp in the MANUAL segmentation with the skin/scalp from CHARM,
    *and zero any manual skin voxels that lie outside the CHARM skin envelope*.
    Output lives on CHARM's grid (higher resolution).

    Parameters
    ----------
    manual_seg : path or NIfTI
        Manually corrected full-head label map (coarser, source for non-skin tissues).
    charm_seg : path or NIfTI
        CHARM full-head label map (target grid & skin envelope).
    manual_skin_id : int
        Label to use for skin in the merged output (your manual scheme).
    charm_skin_ids : tuple[int,...]
        Label(s) in CHARM that represent outer head tissues (skin/scalp).
    background_label : int
        Label to assign where manual skin exists *outside* the CHARM skin mask (default 0).
    charm_skin_dilate_voxels : int
        If >0, dilate CHARM skin mask by this many voxel steps to avoid over-clipping tight areas.
    skin_mask_out : str or None
        If provided, save a NIfTI of the (binary or label-preserving) CHARM skin mask.
    return_mask : bool
        If True, also return the mask NIfTI.
    mask_preserve_labels : bool
        If False: mask is binary (0/1). If True: keep CHARM label IDs where skin, 0 elsewhere.
    return_debug : bool
        If True, also return a dict with inventories and replaced/cleared voxel counts.
    """
    # --- Load inputs ---
    man_img = nib.load(manual_seg) if not isinstance(manual_seg, nib.Nifti1Image) else manual_seg
    cha_img = nib.load(charm_seg)  if not isinstance(charm_seg,  nib.Nifti1Image)  else charm_seg

    if resample_manual:
        # --- Resample MANUAL -> CHARM grid (NN preserves labels) ---
        man_img = resample_from_to(man_img, (cha_img.shape, cha_img.affine), order=0)

    # --- Arrays on CHARM grid ---
    man_arr = np.asarray(man_img.get_fdata(), dtype=np.int32)
    cha_arr = np.asarray(cha_img.get_fdata(), dtype=np.int32)

    # --- Masks ---
    charm_skin_ids = np.asarray(charm_skin_ids, dtype=np.int32)
    charm_skin_mask = np.isin(cha_arr, charm_skin_ids)

    # Optional dilation to be a bit more tolerant
    if charm_skin_dilate_voxels > 0:
        mask = charm_skin_mask
        for _ in range(int(charm_skin_dilate_voxels)):
            mask = binary_dilation(mask)
        charm_skin_mask = mask

    manual_skin_mask = (man_arr == manual_skin_id)

    # --- Start from MANUAL (we want to keep its non-skin labels) ---
    merged = man_arr.copy()

    # 1) Zero any manual skin that lies OUTSIDE the CHARM skin mask
    #    (this removes coarse "spikes"/overreach from the manual software)
    outside_mask = manual_skin_mask & (~charm_skin_mask)
    merged[outside_mask] = background_label

    # 2) Within the CHARM skin envelope, set skin to manual_skin_id
    #    (ensures skin exists exactly where CHARM says it should)
    merged[charm_skin_mask] = manual_skin_id

    # --- Build outputs (on CHARM space) ---
    merged_img = nib.Nifti1Image(merged.astype(np.int16, copy=False), cha_img.affine, cha_img.header)
    merged_img.set_data_dtype(np.int16)

    # Construct the mask image (binary or label-preserving)
    if mask_preserve_labels:
        mask_arr = np.where(charm_skin_mask, cha_arr, 0).astype(np.int16, copy=False)
    else:
        mask_arr = charm_skin_mask.astype(np.uint8, copy=False)

    mask_img = nib.Nifti1Image(mask_arr, cha_img.affine, cha_img.header)
    mask_img.set_data_dtype(mask_arr.dtype)

    # Save mask if requested
    if skin_mask_out:
        nib.save(mask_img, skin_mask_out)

    # --- Prepare return values ---
    results: Tuple = (merged_img,)
    if return_mask:
        results += (mask_img,)

    if return_debug:
        man_u, man_c = np.unique(man_arr, return_counts=True)
        cha_u, cha_c = np.unique(cha_arr, return_counts=True)
        debug = {
            "manual_labels": dict(zip(man_u.tolist(), man_c.tolist())),
            "charm_labels": dict(zip(cha_u.tolist(), cha_c.tolist())),
            "cleared_manual_skin_outside_charm": int(outside_mask.sum()),
            "written_skin_inside_charm": int(charm_skin_mask.sum()),
            "charm_skin_ids": charm_skin_ids.tolist(),
            "background_label": int(background_label),
            "charm_skin_dilate_voxels": int(charm_skin_dilate_voxels),
        }
        results += (debug,)

    return results[0] if len(results) == 1 else results

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

def threshold_gap_fill(
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