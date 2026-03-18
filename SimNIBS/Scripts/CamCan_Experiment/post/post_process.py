# post_process.py
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import json
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from post.post_functions import (
    _resolve_fastsurfer_atlas,
    fastsurfer_dkt_labels,
    make_outline,
    overlay_ti_thresholds_on_t1_with_roi,
    overlay_ti_thresholds_on_t1_with_roi_individual_scale,
    roi_masks_on_ti_grid,
    write_csv,
)
from utils.paths import post_root, ti_brain_path, t1_path
from utils.ti_utils import (
    ensure_dir,
    extract_table,
    load_ti_as_scalar,
    normalize_roi_name,
    resample_atlas_to_ti_grid,
    save_masked_nii,
    summarize_atlas_regions,
    vol_mm3,
)

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
    fs_mri_path: Optional[str] = None # explicit path to the subject atlas NIfTI

    # Behavior
    out_dir: Optional[str] = None
    plot_roi: str = "ctx-lh-precentral"     # which ROI to highlight in overlays
    percentile: float = 95.0
    hard_threshold: float = 200.0
    overlay_z_offset_mm: float = 0.0
    overlay_full_field: bool = False
    write_region_table: bool = True
    region_percentile: float = 95.0
    offtarget_threshold: float = 0.2  # V/m threshold for focality checks

    # Debug/logging
    verbose: bool = True


def _infer_paths(cfg: PostProcessConfig) -> Tuple[str, str, Optional[str], str]:
    subj = cfg.subject
    root = os.path.abspath(cfg.root_dir)
    out_root = cfg.out_dir or str(post_root(root, subj))

    ti_path = cfg.ti_path or str(ti_brain_path(root, subj))
    if cfg.t1_path:
        t1_file = str(Path(cfg.t1_path).expanduser())
        t1_source = "cfg.t1_path"
    else:
        if subj.upper() == "MNI152":
            t1_file = "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz"
            t1_source = "built-in MNI152 template fallback"
        else:
            t1_file = str(t1_path(root, subj))
            t1_source = f"derived from utils.paths.t1_path(root={root!r}, subject={subj!r})"

    return out_root, ti_path, t1_file, t1_source


def _nearby_t1_candidates(t1_candidate: Path) -> Tuple[Path, ...]:
    candidates = []
    name = t1_candidate.name

    if name.endswith(".nii.gz"):
        candidates.append(t1_candidate.with_name(name[:-3]))
    elif t1_candidate.suffix == ".nii":
        candidates.append(t1_candidate.with_name(f"{name}.gz"))

    return tuple(candidates)


def _log_t1_lookup_details(
    cfg: PostProcessConfig,
    t1_candidate: Optional[str],
    t1_source: str,
    *,
    error: Optional[Exception] = None,
) -> None:
    if not cfg.verbose:
        return

    print(f"[INFO] Overlay T1 source: {t1_source}")
    if not t1_candidate:
        print("[WARN] Overlay T1 path is empty; overlays that need T1 will be skipped.")
        return

    expanded = Path(t1_candidate).expanduser()
    resolved = expanded.resolve()
    print(f"[INFO] Overlay T1 candidate: {t1_candidate}")
    print(f"[INFO] Overlay T1 resolved path: {resolved}")
    print(f"[INFO] Overlay T1 exists={resolved.exists()} is_file={resolved.is_file()}")

    nearby_existing = [str(path) for path in _nearby_t1_candidates(resolved) if path.is_file()]
    if nearby_existing:
        print(f"[INFO] Nearby existing T1 candidate(s): {', '.join(nearby_existing)}")

    if error is not None:
        print(f"[WARN] Failed loading T1 for overlays: {type(error).__name__}: {error}")
    elif not resolved.is_file():
        print("[WARN] T1 file for overlays was not found at the resolved path above.")


def run_post_process(cfg: PostProcessConfig) -> Dict[str, dict]:
    """
    Library entrypoint. Returns a dict with useful results and file paths.
    Set breakpoints inside to step through.
    """
    # ---- Paths & IO ----
    out_dir, ti_path, t1_path, t1_path_source = _infer_paths(cfg)
    if cfg.verbose:
        print(f"[cfg] subject={cfg.subject} | atlas_mode={cfg.atlas_mode}")
        print(f"[cfg] ti_path={ti_path}")
        print(f"[cfg] t1_path={t1_path}")
        print(f"[cfg] t1_path_source={t1_path_source}")
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
        roi_names=[cfg.plot_roi] if cfg.plot_roi else None,
    )

    selected_plot_roi = cfg.plot_roi
    if selected_plot_roi not in roi_masks and len(roi_masks) == 1:
        selected_plot_roi = next(iter(roi_masks))

    # Resolve full FastSurfer atlas if available (for full-region summary)
    fs_atlas_path = None
    if cfg.atlas_mode in ("auto", "fastsurfer"):
        # try:
        fs_atlas_path = _resolve_fastsurfer_atlas(cfg.subject, cfg.fastsurfer_root, cfg.fs_mri_path)
        # except Exception:
        #     fs_atlas_path = None
    
    mask = roi_masks.get(selected_plot_roi)
    print("[INFO]:MODE CHECK")
    print("[INFO]:subject:", cfg.subject, "| atlas_mode:", cfg.atlas_mode)
    print("[INFO]:fs path:", cfg.fs_mri_path or (cfg.fastsurfer_root and os.path.join(cfg.fastsurfer_root, f"{cfg.subject}.nii.gz")))
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
    if t1_path:
        try:
            t1_resolved = Path(t1_path).expanduser()
            if t1_resolved.is_file():
                t1_img_full = nib.load(str(t1_resolved))
            else:
                _log_t1_lookup_details(cfg, t1_path, t1_path_source)
        except Exception as e:
            _log_t1_lookup_details(cfg, t1_path, t1_path_source, error=e)

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

        if "hippocampus" in roi_name.lower():
            roi_stub = normalize_roi_name(roi_name)
            outline = make_outline(mask, iterations=1)  # increase to 2-3 if you want thicker edge
            hippo_outline_path = os.path.join(out_dir, f"atlas_{roi_stub}_outline_mask.nii.gz")
            nib.save(nib.Nifti1Image(outline, ti_img.affine), hippo_outline_path)

            # 2b) Make a "bold" (high intensity) filled mask for viewers that fade 0/1 overlays
            bold = (mask.astype(np.uint8) * 255)
            hippo_boldmask_path = os.path.join(out_dir, f"atlas_{roi_stub}_mask_255.nii.gz")
            nib.save(nib.Nifti1Image(bold, ti_img.affine), hippo_boldmask_path)

            # 2c) Export a label-ID volume on the TI grid (17 for L, 53 for R; 0 elsewhere)
            #     This is often the easiest to color distinctly in Freeview/FSLEyes.
            #     We reconstruct it from the original atlas image on the TI grid:
            label_vol = np.zeros(mask.shape, dtype=np.uint16)
            if roi_name.lower().startswith("left-hippocampus"):
                label_vol[mask] = 17
            elif roi_name.lower().startswith("right-hippocampus"):
                label_vol[mask] = 53
            else:
                ijk = np.argwhere(mask)
                xyz = nib.affines.apply_affine(ti_img.affine, ijk)
                left_idx = xyz[:, 0] < 0
                right_idx = ~left_idx
                if ijk.size > 0:
                    if left_idx.any():
                        label_vol[tuple(ijk[left_idx].T)] = 17
                    if right_idx.any():
                        label_vol[tuple(ijk[right_idx].T)] = 53

            hippo_label_path = os.path.join(out_dir, f"atlas_{roi_stub}_labels_on_TI.nii.gz")
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
    sel = selected_plot_roi
    sel_norm = normalize_roi_name(sel)
    if t1_img_full is not None and sel in roi_masks:
        roi_mask_img = nib.Nifti1Image(roi_masks[sel].astype(np.uint8), ti_img.affine, ti_img.header)
        out_base = os.path.join(out_dir, f"{sel_norm}_TI_overlay")
        png_95, png_02, png_full = overlay_ti_thresholds_on_t1_with_roi(
            ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            t1_img=t1_img_full,
            roi_mask_img=roi_mask_img,
            out_prefix=out_base,
            subject=cfg.subject,
            z_offset_mm=cfg.overlay_z_offset_mm,
            include_full_field=cfg.overlay_full_field,
            percentile=cfg.percentile,
            hard_threshold=cfg.hard_threshold,
        )
        dyn_base = f"{out_base}_dynamic"
        dyn_95, dyn_02, dyn_full = overlay_ti_thresholds_on_t1_with_roi_individual_scale(
            ti_img=nib.Nifti1Image(ti_data, ti_img.affine, ti_img.header),
            t1_img=t1_img_full,
            roi_mask_img=roi_mask_img,
            out_prefix=dyn_base,
            subject=cfg.subject,
            z_offset_mm=cfg.overlay_z_offset_mm,
            include_full_field=cfg.overlay_full_field,
            percentile=cfg.percentile,
            hard_threshold=cfg.hard_threshold,
        )
        overlay_paths[sel] = [
            p for p in (png_95, png_02, png_full, dyn_95, dyn_02, dyn_full) if p is not None
        ]
    elif t1_img_full is None:
        if cfg.verbose:
            print("[INFO] Skipping overlays because no T1 background image could be loaded.")

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

if __name__ == "__main__":
    raise SystemExit(
        "Use post/run_post_processing.py to run batch post-processing. "
        "This module is intended to be imported and called as a library."
    )
