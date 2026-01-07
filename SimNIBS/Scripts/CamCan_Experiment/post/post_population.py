"""
Population-level post-processing for TI simulations.

Aggregates per-subject outputs produced by post_process.py (region_stats_fastsurfer.csv
and subject_metrics.json) to derive variability, robustness, and hotspot summaries.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def load_subject_region_table(subj: str, path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df.insert(0, "subject", subj)
    return df


def load_subject_metrics(subj: str, path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    data["subject"] = subj
    return data


def discover_subjects(root: Path, subjects: Optional[Iterable[str]], region_filename: str) -> List[str]:
    if subjects:
        return list(subjects)
    found: List[str] = []
    for subj_dir in root.iterdir():
        if not subj_dir.is_dir():
            continue
        region_path = subj_dir / "anat" / "post" / region_filename
        if region_path.is_file():
            found.append(subj_dir.name)
    return sorted(found)


def aggregate_regions(df: pd.DataFrame, peak_threshold: float) -> pd.DataFrame:
    grouped = df.groupby(["label_id", "label_name"])
    rows = []
    for (lab_id, lab_name), g in grouped:
        peak = g["max"]
        mean_vals = g["mean"]
        vol = g["volume_mm3"]
        rows.append(
            {
                "label_id": lab_id,
                "label_name": lab_name,
                "subjects": g["subject"].nunique(),
                "mean_of_mean": float(mean_vals.mean()),
                "median_of_mean": float(mean_vals.median()),
                "iqr_mean": iqr(mean_vals),
                "cv_mean": float(mean_vals.std(ddof=0) / mean_vals.mean()) if mean_vals.mean() else np.nan,
                "mean_peak": float(peak.mean()),
                "median_peak": float(peak.median()),
                "iqr_peak": iqr(peak),
                "cv_peak": float(peak.std(ddof=0) / peak.mean()) if peak.mean() else np.nan,
                "min_peak": float(peak.min()),
                "max_peak": float(peak.max()),
                "frac_peak_gt_thr": float((peak > peak_threshold).sum() / len(peak)) if len(peak) else np.nan,
                "mean_volume_mm3": float(vol.mean()),
                "median_volume_mm3": float(vol.median()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["label_id"], inplace=True)
    return out


def correlation_volume_intensity(df: pd.DataFrame) -> pd.DataFrame:
    corr_rows = []
    for metric in ("mean", "max"):
        c = df[["volume_mm3", metric]].corr().iloc[0, 1]
        corr_rows.append({"metric": metric, "pearson_r": float(c)})
    return pd.DataFrame(corr_rows)


def subject_target_table(
    all_regions: pd.DataFrame,
    subject_metrics: List[dict],
    target_roi: str,
    template_peak: Optional[float],
) -> pd.DataFrame:
    target = all_regions[all_regions["label_name"].str.lower() == target_roi.lower()].copy()
    if template_peak is not None and not target.empty:
        target["drop_vs_template"] = (template_peak - target["max"]) / template_peak

    # Merge in overlap fractions if available
    overlap_rows = []
    for sm in subject_metrics:
        roi = sm.get("rois", {}).get(target_roi, {})
        overlap_rows.append(
            {
                "subject": sm.get("subject"),
                "roi_overlap_fraction": roi.get("overlap_fraction"),
                "top_percentile_voxels": sm.get("top_percentile_voxels"),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    if not overlap_df.empty:
        target = target.merge(overlap_df, on="subject", how="left")
    return target


def load_template_peak(template_csv: Optional[Path], target_roi: str) -> Optional[float]:
    if not template_csv or not template_csv.is_file():
        return None
    df = pd.read_csv(template_csv)
    match = df[df["label_name"].str.lower() == target_roi.lower()]
    if match.empty:
        return None
    return float(match["max"].iloc[0])


def run_population(
    *,
    root: Path,
    subjects: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
    region_filename: str = "region_stats_fastsurfer.csv",
    metrics_filename: str = "subject_metrics.json",
    peak_threshold: float = 0.2,
    target_roi: str = "Hippocampus",
    template_region_csv: Optional[Path] = None,
) -> Path:
    out_dir = Path(out_dir or (root / "population_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(root, subjects, region_filename)
    if not subjects:
        raise SystemExit("No subjects found with region stats.")

    region_tables = []
    subj_metrics = []
    for subj in subjects:
        subj_root = root / subj / "anat" / "post"
        reg_path = subj_root / region_filename
        met_path = subj_root / metrics_filename

        df = load_subject_region_table(subj, reg_path)
        if df is not None:
            region_tables.append(df)

        sm = load_subject_metrics(subj, met_path)
        if sm is not None:
            subj_metrics.append(sm)

    if not region_tables:
        raise SystemExit("No per-subject region tables were loaded.")

    all_regions = pd.concat(region_tables, ignore_index=True)
    all_regions.to_csv(out_dir / "all_region_values.csv", index=False)

    summary = aggregate_regions(all_regions, peak_threshold)
    summary.to_csv(out_dir / "population_region_summary.csv", index=False)

    corr = correlation_volume_intensity(all_regions)
    corr.to_csv(out_dir / "volume_intensity_correlation.csv", index=False)

    template_peak = load_template_peak(template_region_csv, target_roi)
    subj_df = subject_target_table(all_regions, subj_metrics, target_roi, template_peak)
    subj_df.to_csv(out_dir / "subject_robustness.csv", index=False)

    print(f"[INFO] Aggregated {len(subjects)} subject(s). Outputs in: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Aggregate TI post-processing across subjects.")
    parser.add_argument("--root", required=True, help="Root directory containing subject folders.")
    parser.add_argument("--subjects", nargs="*", help="Explicit list of subject IDs to include.")
    parser.add_argument("--out-dir", default=None, help="Output directory for population summaries.")
    parser.add_argument("--region-filename", default="region_stats_fastsurfer.csv", help="Per-subject region stats filename.")
    parser.add_argument("--metrics-filename", default="subject_metrics.json", help="Per-subject metrics filename.")
    parser.add_argument("--peak-threshold", type=float, default=0.2, help="Threshold (V/m) for 'stimulated' peak fraction.")
    parser.add_argument("--target-roi", default="Hippocampus", help="ROI name for robustness/target-drop reporting.")
    parser.add_argument("--template-region-csv", default=None, help="Template (MNI) region stats CSV for baseline peaks.")

    args = parser.parse_args()

    run_population(
        root=Path(args.root).expanduser(),
        subjects=args.subjects,
        out_dir=Path(args.out_dir).expanduser() if args.out_dir else None,
        region_filename=args.region_filename,
        metrics_filename=args.metrics_filename,
        peak_threshold=args.peak_threshold,
        target_roi=args.target_roi,
        template_region_csv=Path(args.template_region_csv).expanduser()
        if args.template_region_csv
        else None,
    )


if __name__ == "__main__":
    main()
