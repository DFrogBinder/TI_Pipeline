#!/home/boyan/anaconda3/envs/simnibs_post/bin/python3
import os, sys, csv, argparse, numpy as np, nibabel as nib
from nilearn import datasets, image as nli, plotting
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine
from functions import *

# -------------------- CONFIG --------------------
subject = 'MNI152'
output_root = f'/home/boyan/sandbox/Jake_Data/camcan_test_run/{subject}/M1_New_Params/post'
# output_root = '/home/boyan/sandbox/Jake_Data/camcan_test_run/MNI152/post'

ti_path     = os.path.join(os.path.dirname(output_root), 'SimNIBS',"ti_brain_only.nii.gz")   # FEM result (scalar or 3-vec)
# ti_path     = os.path.join(os.path.dirname(output_root), 'anat', 'SimNIBS',"ti_brain_only.nii.gz")   # FEM result (scalar or 3-vec)


t1_path     = os.path.join("/home","boyan","sandbox","simnibs4_exmaples",'m2m_MNI152',"T1.nii.gz")

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
        raise RuntimeError("[ERROR]No finite values in ti_data; check your masking.")

    thr = np.nanpercentile(ti_data[finite], EFIELD_PERCENTILE)
    topP_mask = finite & (ti_data >= thr)

    # --- Background for overlays: T1 if available, else TI itself ---
    bg_img = None
    if t1_path and os.path.exists(t1_path):
        try:
            bg_img = nib.load(t1_path)
            bg_img = resample_from_to(bg_img, ti_img, order=1)
        except Exception as e:
            print(f"[WARNING] could not load/resample T1: {e}")
            bg_img = None
    else:
        print("[WARNING] no T1 provided/found; using TI as background.")
    

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

        print(f"[INFO][{roi_name}] Threshold @ {EFIELD_PERCENTILE}th pct = {thr:.6g}")
        print(f"[INFO][{roi_name}] ROI voxels:       {mask.sum():,} ({vol_ml(mask):.3f} mL)")
        print(f"[INFO][{roi_name}] Overlap voxels:   {overlap_mask.sum():,} ({vol_ml(overlap_mask):.3f} mL)")
        print(f"[INFO][{roi_name}] Overlap within top-P: {overlap_mask.sum()/max(1, topP_mask.sum()):.2%}")
        print(f"[INFO][{roi_name}] Overlap within ROI:   {overlap_mask.sum()/max(1, mask.sum()):.2%}")

        # Extract per-voxel tables
        if WRITE_PER_VOXEL_CSV:
            roi_ijk,  roi_xyz,  roi_vals  = extract_table(mask,         ti_img, ti_data)
            top_ijk,  top_xyz,  top_vals  = extract_table(topP_mask,    ti_img, ti_data)
            ov_ijk,   ov_xyz,   ov_vals   = extract_table(overlap_mask, ti_img, ti_data)

            write_csv(os.path.join(output_root, f"{roi_name}_values.csv"),                   roi_ijk, roi_xyz, roi_vals)
            write_csv(os.path.join(output_root, f"Top{EFIELD_PERCENTILE}_values.csv"),      top_ijk, top_xyz, top_vals)
            write_csv(os.path.join(output_root, f"{roi_name}_Top{EFIELD_PERCENTILE}_overlap_values.csv"), ov_ijk, ov_xyz, ov_vals)

            print(f"[INFO] Stats ({roi_name}):",  basic_stats(roi_vals))
            print(f"[INFO] Stats (Top{EFIELD_PERCENTILE}):", basic_stats(top_vals))
            print(f"[INFO] Stats (Overlap {roi_name}∩Top{EFIELD_PERCENTILE}):", basic_stats(ov_vals))

        # Save NIfTI masks and masked maps
        if mask is not None:
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), ti_img.affine),
                    os.path.join(output_root, f"atlas_{roi_name}_mask.nii.gz"))
        else:
            print(f"[WARNING]: no voxels for {roi_name}; skipping saving ROI mask.")
            
        if overlap_mask is not None:
            nib.save(nib.Nifti1Image(overlap_mask.astype(np.uint8), ti_img.affine),
                    os.path.join(output_root, f"{roi_name}_overlap_top{EFIELD_PERCENTILE}pct_mask.nii.gz"))
        else:
            print(f"[WARNING]: no overlap voxels for {roi_name}; skipping saving overlap mask.")
            
        save_masked_nii(ti_data,mask,ti_img,os.path.join(output_root, f"TI_in_{roi_name}.nii.gz"))
        save_masked_nii(ti_data, overlap_mask, ti_img ,os.path.join(output_root, f"TI_in_{roi_name}_Top{EFIELD_PERCENTILE}.nii.gz"))

        # # Overlays only for the selected ROI
        # if roi_name == selected_roi:
        #     make_overlay_png(
        #         out_png =os.path.join(output_root, f"overlay_TI_with_{roi_name}_contour.png"),
        #         overlay_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
        #         bg_img=bg_img,
        #         title=f"TI field with {roi_name} contour (Selected ROI: {selected_roi})"
        # )

        #     make_overlay_png(
        #         out_png=os.path.join(output_root, f"overlay_TI_top{EFIELD_PERCENTILE}pct_with_{roi_name}.png"),
        #         overlay_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
        #         bg_img=bg_img,
        #         title=f"TI field (≥ {EFIELD_PERCENTILE}th pct) + {roi_name} contour (Selected ROI: {selected_roi})"
        # )
    
        # After roi_masks, ti_img, ti_data, and bg_img/t1_path are set
        # Build a NIfTI mask for M1 on the TI grid
        m1_bool = roi_masks["M1"]        # boolean array on TI grid
        m1_mask_img = nib.Nifti1Image(m1_bool.astype(np.uint8), ti_img.affine, ti_img.header)
        
        hippocampus_bool = roi_masks["Hippocampus"]        # boolean array on TI grid
        hippocampus_mask_img = nib.Nifti1Image(hippocampus_bool.astype(np.uint8), ti_img.affine, ti_img.header)


        # Load original T1 (not resampled) for the helper (it resamples internally)
        t1_img_full = nib.load(t1_path) if t1_path and os.path.exists(t1_path) else None
        if t1_img_full is None:
            raise RuntimeError("T1 not found; required for the requested overlay.")

        m1_mask_img = nib.Nifti1Image(roi_masks["M1"].astype(np.uint8), ti_img.affine, ti_img.header)
        hippocampus_mask_img = nib.Nifti1Image(roi_masks["Hippocampus"].astype(np.uint8), ti_img.affine, ti_img.header)
        t1_img_full = nib.load(t1_path)

        m1_out_base = os.path.join(output_root, "M1_TI_overlay")
        hipp_out_base = os.path.join(output_root, "Hippocampus_TI_overlay")

        if roi_name == selected_roi:
            png_95, png_02 = overlay_ti_thresholds_on_t1_with_roi(
                ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                t1_img=t1_img_full,
                roi_mask_img=m1_mask_img,
                out_prefix=m1_out_base,
                percentile=95.0,
                hard_threshold=200,
                # contour_color="yellow"
            )
            
            png_95, png_02 = overlay_ti_thresholds_on_t1_with_roi(
                ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                t1_img=t1_img_full,
                roi_mask_img=hippocampus_mask_img,
                out_prefix=hipp_out_base,
                percentile=95.0,
                hard_threshold=200,
                # contour_color="yellow"
            )

            print(f"[INFO] Saved percentile overlay: {png_95}")
            print(f"[INFO] Saved hard-threshold overlay: {png_02}")
            print(f"[INFO] Tranfsorming TI values to MNI152 space...")
            # Transform TI to MNI152 space by applyting both a linear and non-linear transform
    
    # Optional: show the resampled atlas for the selected ROI (sanity check)
    for atlas_name, img in atlas_imgs.items():
        # only show the atlas used by the selected ROI
        if atlas_name.lower() != ROI_QUERIES[selected_roi]['query'].lower():
            continue
        make_overlay_png(
            out_png=os.path.join(output_root, f"overlay_{atlas_name}_labels_on_BG.png"),
            overlay_img=img,  # visualize label extents
            bg_img=bg_img,
            title=f"Harvard–Oxford (resampled for {selected_roi}): {atlas_name}"
        )

if __name__ == "__main__":
    main()
