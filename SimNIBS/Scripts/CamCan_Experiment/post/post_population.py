"""
Population-level post-processing for TI simulations.

Aggregates per-subject outputs produced by post_process.py (region_stats_fastsurfer.csv
and subject_metrics.json) to derive variability, robustness, and hotspot summaries.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from post.metric_extensions import flatten_subject_metric_payload
from post.pipeline_layers import (
    load_subject_metrics_payload,
    subject_metrics_payload_analysis_complete,
)
from utils.roi_registry import resolve_fastsurfer_roi_label_ids


COHORT_MANIFEST_COLUMNS = [
    "subject",
    "has_region_table",
    "has_complete_subject_metrics",
    "included",
]


@dataclass
class PopulationData:
    subjects: List[str]
    subject_metrics: List[dict]
    all_regions: pd.DataFrame
    flat_subject_metrics: pd.DataFrame
    neighbor_metrics: pd.DataFrame
    target_roi: str
    cohort_manifest: pd.DataFrame


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def numeric_summary(values: pd.Series) -> dict:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "std": np.nan,
            "cv": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    mean = float(numeric.mean())
    return {
        "count": int(numeric.shape[0]),
        "mean": mean,
        "median": float(numeric.median()),
        "iqr": iqr(numeric),
        "std": float(numeric.std(ddof=1)) if numeric.shape[0] > 1 else 0.0,
        "cv": float(numeric.std(ddof=0) / mean) if mean else np.nan,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def load_subject_region_table(subj: str, path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df.insert(0, "subject", subj)
    return df


def load_subject_metrics(subj: str, path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    data = load_subject_metrics_payload(path)
    if data is None or not subject_metrics_payload_analysis_complete(data):
        return None
    data["subject"] = subj
    return data


def subject_has_complete_outputs(post_root: Path, region_filename: str, metrics_filename: str) -> bool:
    region_path = post_root / region_filename
    metrics_path = post_root / metrics_filename
    return region_path.is_file() and subject_metrics_payload_analysis_complete(
        load_subject_metrics_payload(metrics_path)
    )


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
    output_columns = [
        "metric",
        "subjects",
        "mean",
        "median",
        "iqr",
        "std",
        "cv",
        "min",
        "max",
    ]
    if df.empty:
        return pd.DataFrame(columns=output_columns)

    excluded = {"subject", "target_roi"}
    rows = []
    for column in df.columns:
        if column in excluded:
            continue
        stats = numeric_summary(df[column])
        if stats["count"] == 0:
            continue
        rows.append(
            {
                "metric": column,
                "subjects": stats["count"],
                "mean": stats["mean"],
                "median": stats["median"],
                "iqr": stats["iqr"],
                "std": stats["std"],
                "cv": stats["cv"],
                "min": stats["min"],
                "max": stats["max"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows, columns=output_columns).sort_values("metric").reset_index(drop=True)


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
        mean_stats = numeric_summary(group["mean"])
        peak_stats = numeric_summary(group["max"])
        volume_stats = numeric_summary(group["volume_mm3"])
        rows.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "subjects": int(group["subject"].nunique()),
                "mean_of_mean": mean_stats["mean"],
                "median_of_mean": mean_stats["median"],
                "iqr_mean": mean_stats["iqr"],
                "std_mean": mean_stats["std"],
                "cv_mean": mean_stats["cv"],
                "min_mean": mean_stats["min"],
                "max_mean": mean_stats["max"],
                "mean_peak": peak_stats["mean"],
                "median_peak": peak_stats["median"],
                "iqr_peak": peak_stats["iqr"],
                "std_peak": peak_stats["std"],
                "cv_peak": peak_stats["cv"],
                "min_peak": peak_stats["min"],
                "max_peak": peak_stats["max"],
                "mean_volume_mm3": volume_stats["mean"],
            }
        )
    return pd.DataFrame(rows).sort_values(["label_name", "label_id"]).reset_index(drop=True)


def subject_anatomy_correlations(df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "performance_metric",
        "anatomy_metric",
        "subjects",
        "pearson_r",
    ]
    if df.empty:
        return pd.DataFrame(columns=output_columns)

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
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows, columns=output_columns).sort_values(
        ["performance_metric", "anatomy_metric"]
    ).reset_index(drop=True)


def discover_subjects(
    root: Path,
    subjects: Optional[Iterable[str]],
    region_filename: str,
    metrics_filename: str,
) -> List[str]:
    if subjects:
        return list(subjects)
    found: List[str] = []
    for subj_dir in root.iterdir():
        if not subj_dir.is_dir():
            continue
        post_root = subj_dir / "anat" / "post"
        if subject_has_complete_outputs(post_root, region_filename, metrics_filename):
            found.append(subj_dir.name)
    return sorted(found)


def load_population_data(
    *,
    root: Path,
    subjects: Optional[Iterable[str]],
    region_filename: str,
    metrics_filename: str,
    target_roi: str,
) -> PopulationData:
    discovered_subjects = discover_subjects(root, subjects, region_filename, metrics_filename)

    included_subjects: List[str] = []
    subject_metrics: List[dict] = []
    region_tables: List[pd.DataFrame] = []
    manifest_rows = []

    for subj in discovered_subjects:
        subj_root = root / subj / "anat" / "post"
        region_path = subj_root / region_filename
        metrics_path = subj_root / metrics_filename

        has_region_table = region_path.is_file()
        metrics_payload = load_subject_metrics(subj, metrics_path)
        has_complete_subject_metrics = metrics_payload is not None
        included = bool(has_region_table and has_complete_subject_metrics)

        if not has_complete_subject_metrics:
            continue

        manifest_rows.append(
            {
                "subject": subj,
                "has_region_table": has_region_table,
                "has_complete_subject_metrics": has_complete_subject_metrics,
                "included": included,
            }
        )

        if not included:
            continue

        region_table = load_subject_region_table(subj, region_path)
        if region_table is None:
            continue

        included_subjects.append(subj)
        subject_metrics.append(metrics_payload)
        region_tables.append(region_table)

    all_regions = (
        pd.concat(region_tables, ignore_index=True)
        if region_tables
        else pd.DataFrame()
    )
    flat_subject_metrics = flatten_subject_metrics(subject_metrics, target_roi)
    neighbor_metrics = collect_neighbor_metrics(subject_metrics)
    cohort_manifest = pd.DataFrame(manifest_rows, columns=COHORT_MANIFEST_COLUMNS)

    return PopulationData(
        subjects=included_subjects,
        subject_metrics=subject_metrics,
        all_regions=all_regions,
        flat_subject_metrics=flat_subject_metrics,
        neighbor_metrics=neighbor_metrics,
        target_roi=target_roi,
        cohort_manifest=cohort_manifest,
    )


def aggregate_regions(df: pd.DataFrame, peak_threshold: float) -> pd.DataFrame:
    grouped = df.groupby(["label_id", "label_name"])
    rows = []
    for (lab_id, lab_name), g in grouped:
        peak = pd.to_numeric(g["max"], errors="coerce")
        peak_stats = numeric_summary(g["max"])
        mean_stats = numeric_summary(g["mean"])
        volume_stats = numeric_summary(g["volume_mm3"])
        rows.append(
            {
                "label_id": lab_id,
                "label_name": lab_name,
                "subjects": g["subject"].nunique(),
                "mean_of_mean": mean_stats["mean"],
                "median_of_mean": mean_stats["median"],
                "iqr_mean": mean_stats["iqr"],
                "cv_mean": mean_stats["cv"],
                "mean_peak": peak_stats["mean"],
                "median_peak": peak_stats["median"],
                "iqr_peak": peak_stats["iqr"],
                "cv_peak": peak_stats["cv"],
                "min_peak": peak_stats["min"],
                "max_peak": peak_stats["max"],
                "frac_peak_gt_thr": float((peak > peak_threshold).sum() / len(peak)) if len(peak) else np.nan,
                "mean_volume_mm3": volume_stats["mean"],
                "median_volume_mm3": volume_stats["median"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["label_id"], inplace=True)
    return out


def correlation_volume_intensity(df: pd.DataFrame) -> pd.DataFrame:
    corr_rows = []
    for metric in ("mean", "max"):
        pair = df[["volume_mm3", metric]].apply(pd.to_numeric, errors="coerce").dropna()
        c = pair.corr().iloc[0, 1] if pair.shape[0] >= 2 else np.nan
        corr_rows.append({"metric": metric, "pearson_r": float(c)})
    return pd.DataFrame(corr_rows)


def regional_volume_intensity_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr_rows = []
    grouped = df.groupby(["label_id", "label_name"])
    for (label_id, label_name), group in grouped:
        for metric in ("mean", "max"):
            pair = group.loc[:, ["subject", "volume_mm3", metric]].copy()
            pair["volume_mm3"] = pd.to_numeric(pair["volume_mm3"], errors="coerce")
            pair[metric] = pd.to_numeric(pair[metric], errors="coerce")
            pair = pair.dropna(subset=["volume_mm3", metric])
            subject_count = int(pair["subject"].nunique())
            if subject_count < 3:
                continue
            corr = pair[["volume_mm3", metric]].corr(method="pearson").iloc[0, 1]
            corr_rows.append(
                {
                    "label_id": label_id,
                    "label_name": label_name,
                    "subjects": subject_count,
                    "metric": metric,
                    "pearson_r": float(corr),
                }
            )
    return pd.DataFrame(
        corr_rows,
        columns=["label_id", "label_name", "subjects", "metric", "pearson_r"],
    ).sort_values(["label_id", "metric"]).reset_index(drop=True)


def subject_target_table(
    all_regions: pd.DataFrame,
    subject_metrics: List[dict],
    target_roi: str,
    template_peak: Optional[float],
) -> pd.DataFrame:
    target = all_regions[all_regions["label_name"].str.lower() == target_roi.lower()].copy()
    flat_metrics = flatten_subject_metrics(subject_metrics, target_roi)
    if target.empty and not flat_metrics.empty:
        target = flat_metrics.loc[:, ["subject"]].copy()
        try:
            label_ids = ",".join(str(label_id) for label_id in resolve_fastsurfer_roi_label_ids(target_roi))
        except ValueError:
            label_ids = ""
        target["label_id"] = label_ids
        target["label_name"] = target_roi
        metric_map = {
            "roi_voxels": "voxels",
            "roi_volume_mm3": "volume_mm3",
            "roi_mean": "mean",
            "roi_peak": "max",
        }
        for source, destination in metric_map.items():
            if source in flat_metrics.columns:
                target[destination] = flat_metrics[source]

    if template_peak is not None and not target.empty and "max" in target.columns:
        target["drop_vs_template"] = (template_peak - target["max"]) / template_peak

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
    if match.empty and "label_id" in df.columns:
        try:
            label_ids = set(resolve_fastsurfer_roi_label_ids(target_roi))
        except ValueError:
            label_ids = set()
        if label_ids:
            match = df[pd.to_numeric(df["label_id"], errors="coerce").isin(label_ids)]
    if match.empty:
        return None
    return float(pd.to_numeric(match["max"], errors="coerce").max())


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

    data = load_population_data(
        root=root,
        subjects=subjects,
        region_filename=region_filename,
        metrics_filename=metrics_filename,
        target_roi=target_roi,
    )
    data.cohort_manifest.to_csv(out_dir / "population_cohort_manifest.csv", index=False)

    if not data.subjects:
        raise SystemExit("No subjects found with region stats.")
    if data.all_regions.empty:
        raise SystemExit("No per-subject region tables were loaded.")

    all_regions = data.all_regions
    all_regions.to_csv(out_dir / "all_region_values.csv", index=False)

    summary = aggregate_regions(all_regions, peak_threshold)
    summary.to_csv(out_dir / "population_region_summary.csv", index=False)

    corr = correlation_volume_intensity(all_regions)
    corr.to_csv(out_dir / "volume_intensity_correlation.csv", index=False)
    regional_corr = regional_volume_intensity_correlation(all_regions)
    regional_corr.to_csv(out_dir / "regional_volume_intensity_correlation.csv", index=False)

    template_peak = load_template_peak(template_region_csv, target_roi)
    subj_df = subject_target_table(all_regions, data.subject_metrics, target_roi, template_peak)
    subj_df.to_csv(out_dir / "subject_robustness.csv", index=False)

    flat_subject_metrics = data.flat_subject_metrics
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

    neighbor_metrics = data.neighbor_metrics
    if not neighbor_metrics.empty:
        neighbor_metrics.to_csv(out_dir / "subject_neighbor_metrics.csv", index=False)
        summarize_neighbor_metrics(neighbor_metrics).to_csv(
            out_dir / "population_neighbor_summary.csv", index=False
        )

    print(f"[INFO] Aggregated {len(data.subjects)} subject(s). Outputs in: {out_dir}")
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
