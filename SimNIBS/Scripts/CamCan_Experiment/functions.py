import os
from typing import Tuple, Union, Optional, Sequence
import numpy as np
import nibabel as nib

import tempfile
import shutil
from pathlib import Path

import os, csv, numpy as np, nibabel as nib
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


def fix_csf_and_scalp_gaps_along_z(
    seg: NiftiLike,
    output_path: Optional[str] = None,
    *,
    csf_label: int,
    scalp_label: int,
    skull_label: Optional[int] = None,              # if None, gaps between scalp and CSF filled with scalp_label
    background_labels: Sequence[int] = (0,),       # labels considered “holes/air”
    max_csf_gap: int = 6,                          # max length (voxels) to fill as CSF pocket
    max_between_gap: int = 8,                      # max length to fill between scalp and CSF
    return_debug: bool = False,
) -> Union[nib.Nifti1Image, Tuple[nib.Nifti1Image, dict]]:
    """
    Sweep along +Z for each (x,y) column to fix small gaps:
      (A) CSF pockets: short non-CSF runs bounded by CSF above and below -> set to CSF.
      (B) Scalp–CSF gaps: short runs between nearest scalp (outer) and subsequent CSF -> fill with skull_label
          if provided, else scalp_label.

    Parameters
    ----------
    seg : path or NIfTI
        Integer label volume (already on target grid).
    output_path : str or None
        If provided, save the fixed NIfTI here (handles .nii.gz).
    csf_label : int
        Label ID for CSF.
    scalp_label : int
        Label ID for scalp/skin.
    skull_label : int or None
        Label to use when closing gaps between scalp and CSF. If None, uses scalp_label.
    background_labels : sequence[int]
        Labels treated as “empty/hole” when considering fills (typically [0]).
    max_csf_gap : int
        Max pocket length (in voxels) that will be filled as CSF.
    max_between_gap : int
        Max gap length (in voxels) between scalp and CSF that will be closed.
    return_debug : bool
        If True, also return a dict of counts.

    Returns
    -------
    fixed_img : NIfTI
    [debug] : dict with counts if return_debug=True
    """
    NiftiLike = Union[str, nib.Nifti1Image]
    
    img = nib.load(seg) if not isinstance(seg, nib.Nifti1Image) else seg
    arr = np.asarray(img.get_fdata(), dtype=np.int32)   # work in int32, labels
    assert arr.ndim == 3, "Only 3D label volumes are supported."

    X, Y, Z = arr.shape
    out = arr.copy()

    bg_set = set(int(b) for b in background_labels)
    fill_between_label = skull_label if skull_label is not None else scalp_label

    csf_filled_total = 0
    between_filled_total = 0

    # Helper to fill short gaps in a boolean mask along 1D signal
    def _fill_short_gaps(mask_true: np.ndarray, max_gap: int) -> Tuple[np.ndarray, int]:
        """
        Given a boolean array where True marks the target class (e.g., CSF),
        fill any contiguous False-runs that are strictly bounded by True on both
        sides and whose length <= max_gap.
        Returns new mask and count of voxels flipped True.
        """
        m = mask_true.copy()
        n = len(m)
        if n == 0:
            return m, 0

        i = 0
        filled = 0
        while i < n:
            if not m[i]:
                # start of a False-run
                j = i + 1
                while j < n and not m[j]:
                    j += 1
                # Now [i, j) is a False-run; it is bounded if i>0 and j<n and m[i-1]==True and m[j]==True
                if i > 0 and j < n and m[i-1] and m[j]:
                    run_len = j - i
                    if run_len <= max_gap:
                        m[i:j] = True
                        filled += run_len
                i = j
            else:
                i += 1
        return m, filled

    # Main sweep over columns (x,y) along +Z
    for x in range(X):
        for y in range(Y):
            col = out[x, y, :]  # shape (Z,)

            # --- A) CSF pockets ---
            csf_mask = (col == csf_label)
            csf_mask_filled, added = _fill_short_gaps(csf_mask, max_csf_gap)
            if added:
                col[csf_mask_filled & ~csf_mask] = csf_label
                csf_filled_total += int(added)

            # --- B) Gaps between scalp and CSF (only near the *outer* surface) ---
            # Find first scalp index going in +Z, then the next CSF index deeper.
            # If there is a short gap between them consisting only of backgrounds (or tiny noise), fill it.
            # (We do not try to correct long anatomical gaps or overwrite brain/other tissues.)
            # 1) first scalp voxel (outermost)
            scalp_idxs = np.flatnonzero(col == scalp_label)
            if scalp_idxs.size == 0:
                out[x, y, :] = col
                continue
            s0 = int(scalp_idxs[0])

            # 2) first CSF voxel deeper than s0
            csf_deeper = np.flatnonzero((col == csf_label) & (np.arange(Z) > s0))
            if csf_deeper.size == 0:
                out[x, y, :] = col
                continue
            c0 = int(csf_deeper[0])

            # 3) consider the gap (s0+1 .. c0-1)
            if c0 - s0 > 1:  # there is a gap
                gap_slice = slice(s0 + 1, c0)
                gap_vals = col[gap_slice]

                # Only fill if the gap is short and doesn't contain "forbidden" tissue.
                if (c0 - s0 - 1) <= max_between_gap:
                    # permit filling if gap is wholly backgrounds or tiny specks not equal to scalp/csf
                    # Basic rule: if every voxel in gap is background OR one of (background, scalp, csf),
                    # we treat it as a hole. To be stricter, demand mostly background:
                    if np.all(np.isin(gap_vals, list(bg_set))):
                        col[gap_slice] = fill_between_label
                        between_filled_total += (c0 - s0 - 1)

            out[x, y, :] = col  # write back

    fixed_img = nib.Nifti1Image(out.astype(np.int16, copy=False), img.affine, img.header)
    fixed_img.set_data_dtype(np.int16)

    if output_path:
        nib.save(fixed_img, output_path)

    if return_debug:
        return fixed_img, {
            "voxels_filled_as_csf_pockets": int(csf_filled_total),
            "voxels_filled_between_scalp_and_csf": int(between_filled_total),
            "params": {
                "csf_label": int(csf_label),
                "scalp_label": int(scalp_label),
                "skull_label": None if skull_label is None else int(skull_label),
                "background_labels": list(map(int, bg_set)),
                "max_csf_gap": int(max_csf_gap),
                "max_between_gap": int(max_between_gap),
            },
        }
    return fixed_img
