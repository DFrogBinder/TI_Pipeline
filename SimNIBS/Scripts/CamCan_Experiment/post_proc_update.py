#!/home/boyan/anaconda3/envs/simnibs_post/bin/python3
import os, sys, csv, argparse, numpy as np, nibabel as nib
from nilearn import datasets, image as nli, plotting
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine

# -------------------- CONFIG --------------------
output_root = '/home/boyan/sandbox/TI_Pipeline/SimNIBS/MNI152_TI_Output'
ti_path     = os.path.join(output_root, "ti_brain_only.nii.gz")   # FEM result (scalar or 3-vec)
t1_path     = "/home/boyan/sandbox/simnibs4_examples/m2m_MNI152/T1.nii.gz"

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

def main():

    # --- CLI args ---
    parser = argparse.ArgumentParser(description='Post-process TI maps with ROI overlays')
    parser.add_argument('--plot-roi', default='Hippocampus',
                        help='Which ROI to plot in figures: M1 or Hippocampus (case-insensitive). Default: M1')
    args = parser.parse_args()
    selected_roi = _normalize_roi_name(args.plot_roi)

    ensure_dir(output_root)

    # --- Load volumes ---
    ti_img = nib.load(ti_path)
    ti_data = load_ti_as_scalar(ti_img)

    # ROIs on TI grid
    roi_masks, atlas_imgs = roi_masks_on_ti_grid(ti_img)

    # Finite mask + percentile threshold over all finite voxels
    finite = np.isfinite(ti_data)
    if not np.any(finite):
        raise RuntimeError("No finite values in ti_data; check your masking.")

    thr = np.nanpercentile(ti_data[finite], EFIELD_PERCENTILE)
    topP_mask = finite & (ti_data >= thr)

    # --- Background for overlays: T1 if available, else TI itself ---
    bg_img = None
    if t1_path and os.path.exists(t1_path):
        try:
            bg_img = nib.load(t1_path)
            bg_img = resample_from_to(bg_img, ti_img, order=1)
        except Exception as e:
            print(f"Warning: could not load/resample T1: {e}")
            bg_img = None

    vox_vol = vol_mm3(ti_img)
    vol_ml = lambda mask: mask.sum()*vox_vol/1e3

    # --- Save global masks ---
    nib.save(nib.Nifti1Image(topP_mask.astype(np.uint8), ti_img.affine),
             os.path.join(output_root, f"efield_top{EFIELD_PERCENTILE}pct_mask.nii.gz"))
    save_masked_nii(os.path.join(output_root, f"TI_in_Top{EFIELD_PERCENTILE}.nii.gz"), topP_mask, ti_img, ti_data)

    # --- Per-ROI processing ---
    for roi_name, mask in roi_masks.items():
        if mask is None:
            continue

        overlap_mask = topP_mask & mask

        print(f"[{roi_name}] Threshold @ {EFIELD_PERCENTILE}th pct = {thr:.6g}")
        print(f"[{roi_name}] ROI voxels:       {mask.sum():,} ({vol_ml(mask):.3f} mL)")
        print(f"[{roi_name}] Overlap voxels:   {overlap_mask.sum():,} ({vol_ml(overlap_mask):.3f} mL)")
        print(f"[{roi_name}] Overlap within top-P: {overlap_mask.sum()/max(1, topP_mask.sum()):.2%}")
        print(f"[{roi_name}] Overlap within ROI:   {overlap_mask.sum()/max(1, mask.sum()):.2%}")

        # Extract per-voxel tables
        if WRITE_PER_VOXEL_CSV:
            roi_ijk,  roi_xyz,  roi_vals  = extract_table(mask,         ti_img, ti_data)
            top_ijk,  top_xyz,  top_vals  = extract_table(topP_mask,    ti_img, ti_data)
            ov_ijk,   ov_xyz,   ov_vals   = extract_table(overlap_mask, ti_img, ti_data)

            write_csv(os.path.join(output_root, f"{roi_name}_values.csv"),                   roi_ijk, roi_xyz, roi_vals)
            write_csv(os.path.join(output_root, f"Top{EFIELD_PERCENTILE}_values.csv"),      top_ijk, top_xyz, top_vals)
            write_csv(os.path.join(output_root, f"{roi_name}_Top{EFIELD_PERCENTILE}_overlap_values.csv"), ov_ijk, ov_xyz, ov_vals)

            print(f"Stats ({roi_name}):",  basic_stats(roi_vals))
            print(f"Stats (Top{EFIELD_PERCENTILE}):", basic_stats(top_vals))
            print(f"Stats (Overlap {roi_name}∩Top{EFIELD_PERCENTILE}):", basic_stats(ov_vals))

        # Save NIfTI masks and masked maps
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), ti_img.affine),
                 os.path.join(output_root, f"atlas_{roi_name}_mask.nii.gz"))
        nib.save(nib.Nifti1Image(overlap_mask.astype(np.uint8), ti_img.affine),
                 os.path.join(output_root, f"{roi_name}_overlap_top{EFIELD_PERCENTILE}pct_mask.nii.gz"))

        save_masked_nii(os.path.join(output_root, f"TI_in_{roi_name}.nii.gz"),                mask,         ti_img, ti_data)
        save_masked_nii(os.path.join(output_root, f"TI_in_{roi_name}_Top{EFIELD_PERCENTILE}.nii.gz"), overlap_mask, ti_img, ti_data)

        # Overlays only for the selected ROI
        if roi_name == selected_roi:
            make_overlay_png(
            os.path.join(output_root, f"overlay_TI_with_{roi_name}_contour.png"),
            stat_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            roi_img=nib.Nifti1Image(mask.astype(np.uint8), ti_img.affine),
            bg_img=bg_img,
            thr=None,
            title=f"TI field with {roi_name} contour (Selected ROI: {selected_roi})"
        )

            make_overlay_png(
                os.path.join(output_root, f"overlay_TI_top{EFIELD_PERCENTILE}pct_with_{roi_name}.png"),
            stat_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            roi_img=nib.Nifti1Image(mask.astype(np.uint8), ti_img.affine),
            bg_img=bg_img,
            thr=thr,
            title=f"TI field (≥ {EFIELD_PERCENTILE}th pct) + {roi_name} contour (Selected ROI: {selected_roi})"
        )

    # Optional: show the resampled atlas for the selected ROI (sanity check)
    for atlas_name, img in atlas_imgs.items():
        # only show the atlas used by the selected ROI
        if atlas_name != ROI_QUERIES[selected_roi]['atlas']:
            continue
        make_overlay_png(
            os.path.join(output_root, f"overlay_{atlas_name}_labels_on_BG.png"),
            stat_img=img,  # visualize label extents
            roi_img=nib.Nifti1Image(np.zeros_like(ti_data, dtype=np.uint8), ti_img.affine),
            bg_img=bg_img,
            thr=None,
            title=f"Harvard–Oxford (resampled for {selected_roi}): {atlas_name}"
        )

if __name__ == "__main__":
    main()
