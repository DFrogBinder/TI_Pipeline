#!/home/boyan/anaconda3/envs/simnibs_post/bin/python3
import os, sys, csv, argparse, numpy as np, nibabel as nib
from nilearn import datasets, image as nli, plotting
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine
from functions import *

# -------------------- CONFIG --------------------
output_root = '/home/boyan/sandbox/Jake_Data/camcan_test_run/sub-CC110062/anat/post'
ti_path     = os.path.join(os.path.dirname(output_root), 'SimNIBS',"ti_brain_only.nii.gz")   # FEM result (scalar or 3-vec)
t1_path     = "/home/boyan/sandbox/simnibs4_examples/m2m_MNI152/T1.nii.gz"

# Which ROIs to extract. The queries are matched case-insensitively against labels.
# We fetch the appropriate HO atlas (cortical OR subcortical) per-ROI under the hood.
ROI_QUERIES = {
    "M1":          {"atlas": "cort-maxprob-thr25-1mm", "query": "precentral gyrus"},
    "Hippocampus": {"atlas": "sub-maxprob-thr25-1mm",  "query": "hippocampus"},
}

EFIELD_PERCENTILE     = 95   # top percentile (e.g., 90 or 95)
WRITE_PER_VOXEL_CSV   = True # CSV per voxel; set False if files get too big
# ------------------------------------------------


def main():

    # --- CLI args ---
    parser = argparse.ArgumentParser(description='Post-process TI maps with ROI overlays')
    parser.add_argument('--plot-roi', default='Hippocampus',
                        help='Which ROI to plot in figures: M1 or Hippocampus (case-insensitive). Default: M1')
    args = parser.parse_args()
    selected_roi = normalize_roi_name(args.plot_roi)

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
    save_masked_nii(ti_data, topP_mask, ti_img, os.path.join(output_root, f"TI_in_Top{EFIELD_PERCENTILE}.nii.gz"))

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

        save_masked_nii(ti_data,mask,ti_img,os.path.join(output_root, f"TI_in_{roi_name}.nii.gz"))
        save_masked_nii(ti_data, overlap_mask, ti_img ,os.path.join(output_root, f"TI_in_{roi_name}_Top{EFIELD_PERCENTILE}.nii.gz"))

        # Overlays only for the selected ROI
        if roi_name == selected_roi:
            make_overlay_png(
                out_png =os.path.join(output_root, f"overlay_TI_with_{roi_name}_contour.png"),
                overlay_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                bg_img=bg_img,
                title=f"TI field with {roi_name} contour (Selected ROI: {selected_roi})"
        )

            make_overlay_png(
                out_png=os.path.join(output_root, f"overlay_TI_top{EFIELD_PERCENTILE}pct_with_{roi_name}.png"),
                overlay_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                bg_img=bg_img,
                title=f"TI field (≥ {EFIELD_PERCENTILE}th pct) + {roi_name} contour (Selected ROI: {selected_roi})"
        )

    
    # 1) Binary ROI mask
    overlay_roi_outline(bg_img, ti_data,
                        out_path="overlay_top5.png", title="Top 5% TI region", linewidths=2.5)


    
    
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
