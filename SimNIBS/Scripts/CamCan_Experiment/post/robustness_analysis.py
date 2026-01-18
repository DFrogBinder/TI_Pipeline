#!/usr/bin/env python3
"""
Standalone robustness analysis (per-subject + population).

Computes:
  - ROI peak/mean and deltas vs MNI152 baseline (CSV input)
  - Whole-brain focality voxels > threshold and delta vs MNI152
  - Neighboring-region mean/peak (atlas-adjacent to ROI)
  - Distances from ROI centroid to CSF/skull (if labels provided)
  - Distances from ROI centroid to electrode centers (CSV input)
  - Population IQR/CV and worst-case (lowest peak) subjects
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_dilation, distance_transform_edt

from post.post_functions import (
    _resolve_fastsurfer_atlas,
    fastsurfer_dkt_labels,
    roi_masks_on_ti_grid,
)
from utils.paths import ti_brain_path
from utils.ti_utils import load_ti_as_scalar, resample_atlas_to_ti_grid, vol_mm3


@dataclass
class RobustnessConfig:
    root: str
    subjects: Optional[List[str]] = None
    roi_name: str = "Hippocampus"
    atlas_mode: str = "fastsurfer"  # adjacency requires label atlas
    fastsurfer_root: Optional[str] = None
    fs_mri_path_template: Optional[str] = None  # e.g. "/path/{subject}.nii.gz"
    mni_baseline_csv: Optional[str] = None
    focality_threshold: float = 0.2
    out_dir: Optional[str] = None

    # Tissue distances (label IDs on the atlas)
    csf_labels: Optional[List[int]] = None  # default [24] if None
    skull_labels: Optional[List[int]] = None  # if None, skull distance not computed

    # Electrode centers (world mm): CSV with columns subject,electrode,x,y,z
    electrode_csv: Optional[str] = None
    # If electrode_csv is not provided, look up names in EEG positions files.
    electrode_names: Optional[List[str]] = None
    eeg_positions_path_template: Optional[str] = None  # e.g. "{root}/{subject}/anat/m2m_{subject}/eeg_positions.csv"

    # Neighbor extraction
    neighbor_dilation_iter: int = 1

    verbose: bool = True


def discover_subjects(root: Path, subjects: Optional[Iterable[str]]) -> List[str]:
    if subjects:
        return list(subjects)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _load_mni_baseline_csv(path: Path, roi_name: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    if "roi" in df.columns:
        df = df[df["roi"].str.lower() == roi_name.lower()]
    if "metric" in df.columns and "value" in df.columns:
        return {str(k): float(v) for k, v in zip(df["metric"], df["value"])}
    raise ValueError("Baseline CSV must include columns 'metric' and 'value' (and optional 'roi').")


def _load_electrode_centers(path: Path) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    df = pd.read_csv(path)
    required = {"subject", "electrode", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"Electrode CSV missing required columns: {sorted(required)}")
    by_subject: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for _, row in df.iterrows():
        subj = str(row["subject"])
        coord = np.array([row["x"], row["y"], row["z"]], dtype=float)
        by_subject.setdefault(subj, []).append((str(row["electrode"]), coord))
    return by_subject


def _read_eeg_positions(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if {"name", "x", "y", "z"}.issubset(cols):
            out = {}
            for _, row in df.iterrows():
                name = str(row[cols["name"]])
                out[name] = np.array(
                    [row[cols["x"]], row[cols["y"]], row[cols["z"]]], dtype=float
                )
            return out
    except Exception:
        pass

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            name = parts[0]
            try:
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
            except ValueError:
                continue
            out[name] = xyz
    return out


def _electrode_centers_from_names(
    root: Path,
    subject: str,
    names: List[str],
    template: Optional[str],
) -> List[Tuple[str, np.ndarray]]:
    if template:
        eeg_path = Path(template.format(root=str(root), subject=subject))
    else:
        eeg_path = root / subject / "anat" / f"m2m_{subject}" / "eeg_positions.csv"
    positions = _read_eeg_positions(eeg_path)
    centers = []
    for name in names:
        if name in positions:
            centers.append((name, positions[name]))
    return centers


def _roi_centroid_world(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return None
    center_ijk = ijk.mean(axis=0)
    xyz = nib.affines.apply_affine(affine, center_ijk)
    return np.asarray(xyz, dtype=float)


def _roi_centroid_ijk(mask: np.ndarray) -> Optional[np.ndarray]:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return None
    return ijk.mean(axis=0)


def _distance_to_tissue(
    tissue_mask: np.ndarray,
    roi_centroid_ijk: np.ndarray,
    zooms: Tuple[float, float, float],
) -> float:
    inv = ~tissue_mask
    dist_map = distance_transform_edt(inv, sampling=zooms)
    idx = np.round(roi_centroid_ijk).astype(int)
    idx = np.clip(idx, [0, 0, 0], np.array(dist_map.shape) - 1)
    return float(dist_map[tuple(idx)])


def _neighbor_labels(
    roi_mask: np.ndarray,
    atlas_data: np.ndarray,
    dilation_iter: int,
) -> List[int]:
    if roi_mask.sum() == 0:
        return []
    roi_ids = set(np.unique(atlas_data[roi_mask]))
    roi_ids.discard(0)
    dilated = binary_dilation(roi_mask, iterations=dilation_iter)
    border = dilated & (~roi_mask)
    neighbor_ids = set(np.unique(atlas_data[border]))
    neighbor_ids.discard(0)
    neighbor_ids -= roi_ids
    return sorted(int(x) for x in neighbor_ids)


def _neighbor_stats(
    neighbor_ids: List[int],
    atlas_data: np.ndarray,
    ti_data: np.ndarray,
    finite: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for lab_id in neighbor_ids:
        mask = atlas_data == lab_id
        vals = ti_data[mask & finite]
        if vals.size == 0:
            continue
        rows.append(
            {
                "label_id": int(lab_id),
                "label_name": fastsurfer_dkt_labels.get(int(lab_id), f"Label-{lab_id}"),
                "voxels": int(mask.sum()),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows)


def run_robustness(cfg: RobustnessConfig) -> Path:
    root = Path(cfg.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    subjects = discover_subjects(root, cfg.subjects)
    if not subjects:
        raise SystemExit("No subjects found.")

    out_dir = Path(cfg.out_dir or (root / "robustness_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_out = out_dir / "subjects"
    subj_out.mkdir(parents=True, exist_ok=True)

    baseline = {}
    if cfg.mni_baseline_csv:
        baseline = _load_mni_baseline_csv(Path(cfg.mni_baseline_csv), cfg.roi_name)

    electrode_centers = {}
    if cfg.electrode_csv:
        electrode_centers = _load_electrode_centers(Path(cfg.electrode_csv))

    csf_labels = cfg.csf_labels or [24]

    per_subject_rows = []
    worst_case = []

    for subj in subjects:
        ti_path = ti_brain_path(str(root), subj)
        if not ti_path.is_file():
            if cfg.verbose:
                print(f"[WARN] Missing TI for {subj}: {ti_path}")
            continue

        ti_img = nib.load(str(ti_path))
        ti_data = load_ti_as_scalar(ti_img)
        finite = np.isfinite(ti_data)

        roi_masks, _ = roi_masks_on_ti_grid(
            ti_img,
            atlas_mode=cfg.atlas_mode,
            subject=subj,
            fastsurfer_root=cfg.fastsurfer_root,
            fastsurfer_atlas_path=cfg.fs_mri_path_template.format(subject=subj)
            if cfg.fs_mri_path_template
            else None,
        )
        if cfg.roi_name not in roi_masks:
            if cfg.verbose:
                print(f"[WARN] ROI '{cfg.roi_name}' not found for {subj}")
            continue
        roi_mask = roi_masks[cfg.roi_name]

        roi_vals = ti_data[roi_mask & finite]
        roi_peak = float(np.max(roi_vals)) if roi_vals.size else np.nan
        roi_mean = float(np.mean(roi_vals)) if roi_vals.size else np.nan

        focality_voxels = int(np.sum(ti_data[finite] > cfg.focality_threshold))

        roi_peak_delta = roi_peak - baseline.get("roi_peak", np.nan)
        focality_delta = focality_voxels - baseline.get("focality_voxels", np.nan)

        # Neighbor stats
        fs_path = _resolve_fastsurfer_atlas(
            subj,
            cfg.fastsurfer_root,
            cfg.fs_mri_path_template.format(subject=subj) if cfg.fs_mri_path_template else None,
        )
        neighbor_df = pd.DataFrame()
        if fs_path:
            atlas_img = resample_atlas_to_ti_grid(nib.load(fs_path), ti_img)
            atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
            neighbor_ids = _neighbor_labels(roi_mask, atlas_data, cfg.neighbor_dilation_iter)
            neighbor_df = _neighbor_stats(neighbor_ids, atlas_data, ti_data, finite)
            neighbor_csv = subj_out / f"{subj}_neighbors.csv"
            neighbor_df.to_csv(neighbor_csv, index=False)

        # Distances to CSF/skull
        roi_centroid_ijk = _roi_centroid_ijk(roi_mask)
        roi_centroid_xyz = _roi_centroid_world(roi_mask, ti_img.affine)
        csf_dist = np.nan
        skull_dist = np.nan
        if fs_path and roi_centroid_ijk is not None:
            atlas_img = resample_atlas_to_ti_grid(nib.load(fs_path), ti_img)
            atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
            zooms = ti_img.header.get_zooms()[:3]
            csf_mask = np.isin(atlas_data, csf_labels)
            if np.any(csf_mask):
                csf_dist = _distance_to_tissue(csf_mask, roi_centroid_ijk, zooms)
            if cfg.skull_labels:
                skull_mask = np.isin(atlas_data, cfg.skull_labels)
                if np.any(skull_mask):
                    skull_dist = _distance_to_tissue(skull_mask, roi_centroid_ijk, zooms)

        # Electrode distances
        electrode_rows = []
        if roi_centroid_xyz is not None and subj in electrode_centers:
            for name, coord in electrode_centers[subj]:
                dist = float(np.linalg.norm(coord - roi_centroid_xyz))
                electrode_rows.append({"electrode": name, "distance_mm": dist})
            pd.DataFrame(electrode_rows).to_csv(subj_out / f"{subj}_electrode_distances.csv", index=False)
        elif roi_centroid_xyz is not None and cfg.electrode_names:
            centers = _electrode_centers_from_names(
                root,
                subj,
                cfg.electrode_names,
                cfg.eeg_positions_path_template,
            )
            for name, coord in centers:
                dist = float(np.linalg.norm(coord - roi_centroid_xyz))
                electrode_rows.append({"electrode": name, "distance_mm": dist})
            if electrode_rows:
                pd.DataFrame(electrode_rows).to_csv(
                    subj_out / f"{subj}_electrode_distances.csv", index=False
                )

        per_subject_rows.append(
            {
                "subject": subj,
                "roi": cfg.roi_name,
                "roi_peak": roi_peak,
                "roi_mean": roi_mean,
                "roi_peak_delta_mni": roi_peak_delta,
                "focality_voxels": focality_voxels,
                "focality_delta_mni": focality_delta,
                "roi_centroid_x": float(roi_centroid_xyz[0]) if roi_centroid_xyz is not None else np.nan,
                "roi_centroid_y": float(roi_centroid_xyz[1]) if roi_centroid_xyz is not None else np.nan,
                "roi_centroid_z": float(roi_centroid_xyz[2]) if roi_centroid_xyz is not None else np.nan,
                "csf_distance_mm": csf_dist,
                "skull_distance_mm": skull_dist,
            }
        )
        worst_case.append((subj, roi_peak))

    per_subject_df = pd.DataFrame(per_subject_rows)
    per_subject_df.to_csv(out_dir / "per_subject_metrics.csv", index=False)

    # Population summaries
    def _iqr(series: pd.Series) -> float:
        return float(series.quantile(0.75) - series.quantile(0.25))

    pop_rows = []
    for metric in ["roi_peak", "roi_mean", "focality_voxels"]:
        if metric not in per_subject_df.columns or per_subject_df.empty:
            continue
        vals = per_subject_df[metric].dropna()
        if vals.empty:
            continue
        pop_rows.append(
            {
                "metric": metric,
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "iqr": _iqr(vals),
                "cv": float(vals.std(ddof=0) / vals.mean()) if vals.mean() else np.nan,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    pd.DataFrame(pop_rows).to_csv(out_dir / "population_summary.csv", index=False)

    worst_case_sorted = sorted(
        [(s, p) for s, p in worst_case if np.isfinite(p)],
        key=lambda x: x[1],
    )[:10]
    pd.DataFrame(worst_case_sorted, columns=["subject", "roi_peak"]).to_csv(
        out_dir / "worst_case_subjects.csv", index=False
    )

    if cfg.verbose:
        print(f"[INFO] Robustness outputs in: {out_dir}")

    return out_dir


if __name__ == "__main__":
    cfg = RobustnessConfig(
        root="/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/Sample Data",
        subjects=['sub-CC110056'],
        roi_name="Hippocampus",
        atlas_mode="fastsurfer",
        fastsurfer_root=None,
        fs_mri_path_template='/home/boyan/sandbox/Jake_Data/atlases/sub-CC110056.nii.gz',  # e.g. "/path/to/atlases/{subject}.nii.gz"
        mni_baseline_csv="/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/post/configs/mni_baseline_placeholder.csv",
        focality_threshold=0.2,
        out_dir=None,
        csf_labels=[24],
        skull_labels=None,
        electrode_csv="/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/CamCan_Experiment/post/configs/electrode_centers_placeholder.csv",
        electrode_names=["Fp2", "P8", "T7", "P7"],
        eeg_positions_path_template=None,
        neighbor_dilation_iter=1,
        verbose=True,
    )
    run_robustness(cfg)
