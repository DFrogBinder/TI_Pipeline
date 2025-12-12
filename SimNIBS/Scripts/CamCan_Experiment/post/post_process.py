# post_process.py
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.post_functions import *
from utils.paths import post_root, ti_brain_path, t1_path, fastsurfer_atlas_path

@dataclass
class PostProcessConfig:
    # Required-ish
    root_dir: str
    subject: str = "MNI152"
    ti_path: Optional[str] = None
    t1_path: Optional[str] = None

    # Atlas selection
    atlas_mode: str = "mni"          # "auto" | "mni" | "fastsurfer"
    fastsurfer_root: Optional[str] = None
    fs_mri_path: Optional[str] = None # explicit path to aparc.DKTatlas+aseg.deep.nii.gz

    # Behavior
    out_dir: Optional[str] = None
    plot_roi: str = "Hippocampus"     # which ROI to highlight in overlays
    percentile: float = 95.0
    hard_threshold: float = 200.0
    write_region_table: bool = True
    region_percentile: float = 95.0
    offtarget_threshold: float = 0.2  # V/m threshold for focality checks

    # Debug/logging
    verbose: bool = True


def _infer_paths(cfg: PostProcessConfig) -> Tuple[str, str, Optional[str]]:
    subj = cfg.subject
    root = os.path.abspath(cfg.root_dir)
    out_root = cfg.out_dir or str(post_root(root, subj))

    ti_path = cfg.ti_path or str(ti_brain_path(root, subj))
    if cfg.t1_path:
        t1_path = cfg.t1_path
    else:
        if subj.upper() == "MNI152":
            t1_path = "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz"
        else:
            t1_path = str(t1_path(root, subj))

    return out_root, ti_path, t1_path


def run_post_process(cfg: PostProcessConfig) -> Dict[str, dict]:
    """
    Library entrypoint. Returns a dict with useful results and file paths.
    Set breakpoints inside to step through.
    """
    # ---- Paths & IO ----
    out_dir, ti_path, t1_path = _infer_paths(cfg)
    if cfg.verbose:
        print(f"[cfg] subject={cfg.subject} | atlas_mode={cfg.atlas_mode}")
        print(f"[cfg] ti_path={ti_path}")
        print(f"[cfg] t1_path={t1_path}")
        print(f"[cfg] out_dir={out_dir}")

    ensure_dir(out_dir)
    ti_img = nib.load(ti_path)
    ti_data = load_ti_as_scalar(ti_img)

    # ---- Atlas / ROI masks on TI grid ----
    roi_masks, atlas_imgs = roi_masks_on_ti_grid(
        ti_img,
        atlas_mode=cfg.atlas_mode,
        subject=cfg.subject,
        fastsurfer_root=cfg.fastsurfer_root,
        fastsurfer_atlas_path=cfg.fs_mri_path,
    )

    # Resolve full FastSurfer atlas if available (for full-region summary)
    fs_atlas_path = None
    if cfg.atlas_mode in ("auto", "fastsurfer"):
        try:
        fs_atlas_path = _resolve_fastsurfer_atlas(cfg.subject, cfg.fastsurfer_root, cfg.fs_mri_path)
        except Exception:
            fs_atlas_path = None
    
    mask = roi_masks.get("Hippocampus")
    print("[INFO]:MODE CHECK")
    print("[INFO]:subject:", cfg.subject, "| atlas_mode:", cfg.atlas_mode)
    print("[INFO]:fs path:", cfg.fs_mri_path or (cfg.fastsurfer_root and os.path.join(cfg.fastsurfer_root, cfg.subject, "mri", "aparc.DKTatlas+aseg.deep.nii.gz")))
    print("[INFO]:mask dtype/shape:", mask.dtype, mask.shape if mask is not None else None)
    print("[INFO]:mask voxels >0:", int(mask.sum()) if mask is not None else 0)


    # ---- Thresholds ----
    finite = np.isfinite(ti_data)
    if not np.any(finite):
        raise RuntimeError("[ERROR] No finite values in ti_data; check your masking.")

    thr = float(np.nanpercentile(ti_data[finite], cfg.percentile))
    topP_mask = finite & (ti_data >= thr)

    # Background for overlays (resampled to TI grid if available)
    t1_img_full = None
    if t1_path and os.path.exists(t1_path):
        try:
            t1_img_full = nib.load(t1_path)
        except Exception as e:
            if cfg.verbose:
                print(f"[WARN] Failed loading T1 ({t1_path}): {e}")

    # Save global masks
    vox_vol = vol_mm3(ti_img)
    topP_mask_path = os.path.join(out_dir, f"efield_top{int(cfg.percentile)}pct_mask.nii.gz")
    nib.save(nib.Nifti1Image(topP_mask.astype(np.uint8), ti_img.affine), topP_mask_path)

    topP_ti_path = os.path.join(out_dir, f"TI_in_Top{int(cfg.percentile)}.nii.gz")
    save_masked_nii(ti_data, topP_mask, ti_img, topP_ti_path)

    # ---- Per-ROI products ----
    per_roi = {}
    per_roi_metrics = {}
    for roi_name, mask in roi_masks.items():
        if mask is None:
            continue

        overlap_mask = topP_mask & mask
        roi_mask_img = nib.Nifti1Image(mask.astype(np.uint8), ti_img.affine, ti_img.header)
        overlap_mask_img = nib.Nifti1Image(overlap_mask.astype(np.uint8), ti_img.affine, ti_img.header)

        if cfg.verbose:
            print(f"[ROI:{roi_name}] thr@{cfg.percentile}th = {thr:.6g}")
            print(f"[ROI:{roi_name}] ROI voxels: {mask.sum():,} ({mask.sum()*vox_vol/1e3:.3f} mL)")
            print(f"[ROI:{roi_name}] Overlap voxels: {overlap_mask.sum():,} "
                  f"({overlap_mask.sum()*vox_vol/1e3:.3f} mL)")

        # Tables
        ijk_r, xyz_r, vals_r = extract_table(mask,         ti_img, ti_data)
        ijk_t, xyz_t, vals_t = extract_table(topP_mask,    ti_img, ti_data)
        ijk_o, xyz_o, vals_o = extract_table(overlap_mask, ti_img, ti_data)

        roi_csv         = os.path.join(out_dir, f"{roi_name}_values.csv")
        topp_csv        = os.path.join(out_dir, f"Top{int(cfg.percentile)}_values.csv")
        overlap_csv     = os.path.join(out_dir, f"{roi_name}_Top{int(cfg.percentile)}_overlap_values.csv")
        write_csv(roi_csv,     ijk_r, xyz_r, vals_r)
        write_csv(topp_csv,    ijk_t, xyz_t, vals_t)
        write_csv(overlap_csv, ijk_o, xyz_o, vals_o)

        # Save masks
        atlas_mask_path = os.path.join(out_dir, f"atlas_{roi_name}_mask.nii.gz")
        overlap_mask_path = os.path.join(out_dir, f"{roi_name}_overlap_top{int(cfg.percentile)}pct_mask.nii.gz")
        nib.save(roi_mask_img, atlas_mask_path)
        nib.save(overlap_mask_img, overlap_mask_path)

        # Save masked TI
        ti_in_roi_path        = os.path.join(out_dir, f"TI_in_{roi_name}.nii.gz")
        ti_in_roi_topP_path   = os.path.join(out_dir, f"TI_in_{roi_name}_Top{int(cfg.percentile)}.nii.gz")
        save_masked_nii(ti_data, mask,         ti_img, ti_in_roi_path)
        save_masked_nii(ti_data, overlap_mask, ti_img, ti_in_roi_topP_path)

        per_roi[roi_name] = dict(
            roi_csv=roi_csv,
            topp_csv=topp_csv,
            overlap_csv=overlap_csv,
            atlas_mask=atlas_mask_path,
            overlap_mask=overlap_mask_path,
            ti_in_roi=ti_in_roi_path,
            ti_in_roi_topP=ti_in_roi_topP_path,
        )

        per_roi_metrics[roi_name] = dict(
            roi_voxels=int(mask.sum()),
            overlap_top_voxels=int(overlap_mask.sum()),
            roi_volume_mm3=float(mask.sum() * vox_vol),
            overlap_volume_mm3=float(overlap_mask.sum() * vox_vol),
            overlap_fraction=float(overlap_mask.sum() / mask.sum()) if mask.sum() else 0.0,
        )
        
        hippo_outline_path = None
        hippo_label_path   = None
        hippo_boldmask_path = None

        if roi_name.lower() == "hippocampus":
            outline = make_outline(mask, iterations=1)  # increase to 2-3 if you want thicker edge
            hippo_outline_path = os.path.join(out_dir, "atlas_Hippocampus_outline_mask.nii.gz")
            nib.save(nib.Nifti1Image(outline, ti_img.affine), hippo_outline_path)

            # 2b) Make a "bold" (high intensity) filled mask for viewers that fade 0/1 overlays
            bold = (mask.astype(np.uint8) * 255)
            hippo_boldmask_path = os.path.join(out_dir, "atlas_Hippocampus_mask_255.nii.gz")
            nib.save(nib.Nifti1Image(bold, ti_img.affine), hippo_boldmask_path)

            # 2c) Export a label-ID volume on the TI grid (17 for L, 53 for R; 0 elsewhere)
            #     This is often the easiest to color distinctly in Freeview/FSLEyes.
            #     We reconstruct it from the original atlas image on the TI grid:
            ids = [17, 53]  # bilateral; change to [17] or [53] if you want unilateral
            # Start from zeros, paint the mask with ID 17/53 depending on which side each voxel belongs to.
            # If you don't need per-side split, you can just use e.g. 17 for all.
            # Here’s a simple approach: put 77 for left, 88 for right (or keep 17/53 if preferred).
            label_vol = np.zeros(mask.shape, dtype=np.uint16)
            # Heuristic split by x in world space (left/right). Feel free to skip and just set 17/53 both.
            ijk = np.argwhere(mask)
            xyz = nib.affines.apply_affine(ti_img.affine, ijk)
            # Left hemisphere has x<0 in RAS (typical); adjust if your orientation differs.
            left_idx  = (xyz[:, 0] < 0)
            right_idx = ~left_idx
            if ijk.size > 0:
                if left_idx.any():  label_vol[tuple(ijk[left_idx].T)]  = 17
                if right_idx.any(): label_vol[tuple(ijk[right_idx].T)] = 53

            hippo_label_path = os.path.join(out_dir, "atlas_Hippocampus_labels_on_TI.nii.gz")
            nib.save(nib.Nifti1Image(label_vol, ti_img.affine), hippo_label_path)

            # Track in the return dict
            per_roi[roi_name].update(dict(
                hippo_outline=hippo_outline_path,
                hippo_mask_255=hippo_boldmask_path,
                hippo_label_ids=hippo_label_path,
            ))

    # ---- Full atlas region summaries (FastSurfer only) ----
    region_table_path = None
    region_df = None
    if cfg.write_region_table and fs_atlas_path:
        try:
            fs_atlas_img = resample_atlas_to_ti_grid(nib.load(fs_atlas_path), ti_img)
            region_df = summarize_atlas_regions(
                nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
                fs_atlas_img,
                fastsurfer_dkt_labels,
                percentile=cfg.region_percentile,
            )
            region_table_path = os.path.join(out_dir, "region_stats_fastsurfer.csv")
            region_df.to_csv(region_table_path, index=False)
            if cfg.verbose:
                print(f"[INFO] Saved region stats to {region_table_path}")
        except Exception as e:
            if cfg.verbose:
                print(f"[WARN] Skipped region table: {e}")


    # ---- Pretty overlays for selected ROI (optional) ----
    overlay_paths = {}
    sel = cfg.plot_roi
    sel_norm = normalize_roi_name(sel)
    if t1_img_full is not None and sel in roi_masks:
        roi_mask_img = nib.Nifti1Image(roi_masks[sel].astype(np.uint8), ti_img.affine, ti_img.header)
        out_base = os.path.join(out_dir, f"{sel_norm}_TI_overlay")
        png_95, png_02 = overlay_ti_thresholds_on_t1_with_roi(
            ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            t1_img=t1_img_full,
            roi_mask_img=roi_mask_img,
            out_prefix=out_base,
            percentile=cfg.percentile,
            hard_threshold=cfg.hard_threshold,
        )
        overlay_paths[sel] = [png_95, png_02]
    elif t1_img_full is None:
        if cfg.verbose:
            print("[INFO] Skipping overlays (T1 not found).")

    # ---- Subject-level robustness metrics ----
    subject_metrics = dict(
        subject=cfg.subject,
        percentile=cfg.percentile,
        percentile_value=float(thr),
        voxel_volume_mm3=vox_vol,
        top_percentile_voxels=int(topP_mask.sum()),
        rois=per_roi_metrics,
    )

    metrics_path = os.path.join(out_dir, "subject_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(subject_metrics, f, indent=2)

    # ---- Return everything useful for tests / notebooks ----
    return dict(
        config=cfg,
        ti_path=ti_path,
        t1_path=t1_path,
        out_dir=out_dir,
        percentile_threshold=thr,
        topP_mask_path=topP_mask_path,
        topP_ti_path=topP_ti_path,
        per_roi=per_roi,
        overlays=overlay_paths,
        region_table=region_table_path,
        region_df=region_df,
        metrics_path=metrics_path,
    )

cfg = PostProcessConfig(
    root_dir="CHANGE_ME",
    subject="MNI152",
    atlas_mode="auto",
    fastsurfer_root=None,
    fs_mri_path=None,
    plot_roi="Hippocampus",
    percentile=95.0,
    hard_threshold=200.0,
    verbose=True,
)

if __name__ == "__main__":
    raise SystemExit(
        "Import run_post_process and supply a PostProcessConfig; "
        "this module no longer runs with hard-coded paths."
    )
