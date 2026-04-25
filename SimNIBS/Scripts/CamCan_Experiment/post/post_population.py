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

from post.metric_extensions import flatten_subject_metric_payload


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


def flatten_subject_metrics(subject_metrics: List[dict], target_roi: str) -> pd.DataFrame:
    rows = []
    for payload in subject_metrics:
        try:
            row = flatten_subject_metric_payload(payload, target_roi)
        except Exception:
            continue
        row["subject"] = payload.get("subject")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_subject_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    excluded = {"subject", "target_roi"}
    rows = []
    for column in df.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "subjects": int(values.shape[0]),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "iqr": iqr(values),
                "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                "cv": float(values.std(ddof=0) / values.mean()) if values.mean() else np.nan,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("metric").reset_index(drop=True)


def collect_neighbor_metrics(subject_metrics: List[dict]) -> pd.DataFrame:
    rows = []
    for payload in subject_metrics:
        subject = payload.get("subject")
        extended = payload.get("extended_metrics", {})
        if not isinstance(extended, dict):
            continue
        for row in extended.get("neighbors", []):
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "subject": subject,
                    "label_id": row.get("label_id"),
                    "label_name": row.get("label_name"),
                    "voxels": row.get("voxels"),
                    "volume_mm3": row.get("volume_mm3"),
                    "mean": row.get("mean"),
                    "max": row.get("max"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_neighbor_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["label_id", "label_name"], dropna=False)
    rows = []
    for (label_id, label_name), group in grouped:
        rows.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "subjects": int(group["subject"].nunique()),
                "mean_of_mean": float(pd.to_numeric(group["mean"], errors="coerce").mean()),
                "median_of_mean": float(pd.to_numeric(group["mean"], errors="coerce").median()),
                "iqr_mean": iqr(pd.to_numeric(group["mean"], errors="coerce").dropna()),
                "mean_peak": float(pd.to_numeric(group["max"], errors="coerce").mean()),
                "median_peak": float(pd.to_numeric(group["max"], errors="coerce").median()),
                "iqr_peak": iqr(pd.to_numeric(group["max"], errors="coerce").dropna()),
                "mean_volume_mm3": float(pd.to_numeric(group["volume_mm3"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["label_name", "label_id"]).reset_index(drop=True)


def subject_anatomy_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    performance_metrics = [
        "roi_peak",
        "roi_mean",
        "roi_peak_abs_delta_mni",
        "roi_mean_abs_delta_mni",
        "focality_voxels_gt_threshold",
        "focality_volume_mm3_gt_threshold",
    ]
    anatomy_metrics = [
        "csf_distance_mm",
        "skull_distance_mm",
        "electrode_distance_mean_mm",
        "electrode_distance_min_mm",
        "electrode_distance_max_mm",
    ]

    rows = []
    for performance_metric in performance_metrics:
        if performance_metric not in df.columns:
            continue
        for anatomy_metric in anatomy_metrics:
            if anatomy_metric not in df.columns:
                continue
            pair = df[[performance_metric, anatomy_metric]].apply(pd.to_numeric, errors="coerce").dropna()
            if pair.shape[0] < 3:
                continue
            corr = pair.corr(method="pearson").iloc[0, 1]
            rows.append(
                {
                    "performance_metric": performance_metric,
                    "anatomy_metric": anatomy_metric,
                    "subjects": int(pair.shape[0]),
                    "pearson_r": float(corr),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["performance_metric", "anatomy_metric"]
    ).reset_index(drop=True)


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

    flat_metrics = flatten_subject_metrics(subject_metrics, target_roi)
    if not flat_metrics.empty:
        rename_map = {
            "overlap_fraction": "roi_overlap_fraction",
            "focality_voxels_gt_threshold": "focality_voxels_gt_threshold",
            "focality_volume_mm3_gt_threshold": "focality_volume_mm3_gt_threshold",
            "roi_peak_abs_delta_mni": "roi_peak_abs_delta_mni",
            "roi_mean_abs_delta_mni": "roi_mean_abs_delta_mni",
            "csf_distance_mm": "csf_distance_mm",
            "skull_distance_mm": "skull_distance_mm",
            "electrode_distance_mean_mm": "electrode_distance_mean_mm",
            "electrode_distance_min_mm": "electrode_distance_min_mm",
            "electrode_distance_max_mm": "electrode_distance_max_mm",
            "neighbor_mean_of_means": "neighbor_mean_of_means",
            "neighbor_max_of_max": "neighbor_max_of_max",
        }
        keep_cols = ["subject"] + [column for column in rename_map if column in flat_metrics.columns]
        overlap_df = flat_metrics.loc[:, keep_cols].rename(columns=rename_map)
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

    flat_subject_metrics = flatten_subject_metrics(subj_metrics, target_roi)
    if not flat_subject_metrics.empty:
        flat_subject_metrics.to_csv(out_dir / "subject_metric_values.csv", index=False)
        summarize_subject_metrics(flat_subject_metrics).to_csv(
            out_dir / "population_subject_metric_summary.csv", index=False
        )
        if "roi_peak" in flat_subject_metrics.columns:
            (
                flat_subject_metrics.loc[:, ["subject", "roi_peak"]]
                .dropna()
                .sort_values("roi_peak", ascending=True)
                .head(10)
                .to_csv(out_dir / "worst_case_subjects.csv", index=False)
            )

        anatomy_corr = subject_anatomy_correlations(flat_subject_metrics)
        if not anatomy_corr.empty:
            anatomy_corr.to_csv(out_dir / "population_anatomy_correlations.csv", index=False)

    neighbor_metrics = collect_neighbor_metrics(subj_metrics)
    if not neighbor_metrics.empty:
        neighbor_metrics.to_csv(out_dir / "subject_neighbor_metrics.csv", index=False)
        summarize_neighbor_metrics(neighbor_metrics).to_csv(
            out_dir / "population_neighbor_summary.csv", index=False
        )

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
