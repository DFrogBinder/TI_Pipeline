#!/usr/bin/env python3
"""Build the 0.20 V/m companion figures for the CamCan final-132 analysis.

The validated schema-2 manuscript package contains threshold metrics at
0.18 and 0.15 V/m.  The earlier validated post-processing archive contains
the corresponding per-repeat target and whole-brain suprathreshold voxel
counts at 0.20 V/m.  This script:

1. matches all 5,280 legacy records to the schema-2 subject/ROI/repeat keys;
2. verifies that the anatomical ROI and whole-brain voxel denominators agree;
3. reconstructs the schema-2 target, off-target, whole-brain, and localization
   percentages at 0.20 V/m;
4. arithmetic-mean aggregates the metrics across ten repeats;
5. recomputes the corrected MNI152 comparator from its TI NIfTIs and the exact
   archived Destrieux atlas, while verifying that the recomputed 0.18/0.15
   values reproduce the downloaded schema-2 MNI rows; and
6. writes publication figures, compact audit tables, captions, and a manifest.

No voxelwise values are interpolated and no threshold value is estimated from
the 0.18/0.15 summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import percentileofscore, spearmanr

from build_camcan_publication_outputs import (
    FIGURE_BLUE,
    FIGURE_GRAY,
    FIGURE_GREEN,
    FIGURE_ORANGE,
    FIGURE_RED,
    ROI_LABELS,
    ROI_ORDER,
    save_figure,
    setup_matplotlib,
)
from camcan_manuscript_analysis import (
    _compute_mni_record,
    _threshold_slug,
    _write_effectiveness_spread_figure,
    manuscript_metric_names,
)


EXPECTED_SUBJECTS = 132
EXPECTED_REPEATS = 10
EXPECTED_RECORDS = 5280
DEFAULT_THRESHOLD = 0.20
DEFAULT_MNI_ATLAS_MEMBER = "atlases_a2009s/sub-mni152-cw256.nii.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_file_slug(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "p")


def safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    if (denominator <= 0).any():
        raise RuntimeError("A percentage denominator is not positive.")
    return numerator.astype(float) / denominator.astype(float) * 100.0


def find_runs_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "runs" and path.is_dir():
        return path
    candidate = path / "runs"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Could not find the extracted post-processing runs directory below {path}."
    )


def parse_legacy_record(path: Path, expected_threshold: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    roi_parent = next(
        (part for part in path.parts if part.endswith("_Runs")), None
    )
    dataset_parent = next(
        (part for part in path.parts if "_Data_" in part), None
    )
    if roi_parent is None or dataset_parent is None:
        raise RuntimeError(f"Could not infer ROI/repeat from {path}.")
    roi = roi_parent.removesuffix("_Runs")
    repeat = dataset_parent.rsplit("_", 1)[-1].zfill(2)
    target_name = payload["target_roi"]
    target_roi = payload["rois"][target_name]
    target_qc = payload["threshold_qc"]["rois"][target_name][
        "metric_threshold"
    ]
    whole_qc = payload["threshold_qc"]["whole_brain"]["metric_threshold"]
    if not math.isclose(
        float(target_qc["threshold"]),
        expected_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Unexpected legacy threshold in {path}: {target_qc['threshold']}"
        )
    if target_qc["comparator"] != ">=" or whole_qc["comparator"] != ">=":
        raise RuntimeError(f"Unexpected threshold comparator in {path}.")
    return {
        "subject": payload["subject"],
        "roi": roi,
        "repeat": repeat,
        "legacy_roi_voxels": int(target_roi["roi_voxels"]),
        "legacy_whole_brain_voxels": int(payload["whole_brain_voxels"]),
        "target_count": int(target_qc["voxels"]),
        "whole_count": int(whole_qc["voxels"]),
    }


def load_legacy_counts(
    runs_root: Path,
    expected_threshold: float,
) -> pd.DataFrame:
    paths = sorted(runs_root.rglob("subject_metrics.json"))
    if len(paths) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS:,} subject_metrics.json files; "
            f"found {len(paths):,}."
        )
    frame = pd.DataFrame(
        parse_legacy_record(path, expected_threshold) for path in paths
    )
    if frame.duplicated(["subject", "roi", "repeat"]).any():
        raise RuntimeError("Legacy 0.20 V/m records contain duplicate keys.")
    if set(frame["roi"]) != set(ROI_ORDER):
        raise RuntimeError("Legacy records do not contain the four planned ROIs.")
    if (frame["target_count"] > frame["whole_count"]).any():
        raise RuntimeError("A target suprathreshold count exceeds its whole-brain count.")
    if (frame["target_count"] < 0).any() or (frame["whole_count"] < 0).any():
        raise RuntimeError("A legacy suprathreshold count is negative.")
    return frame


def reconstruct_repeat_metrics(
    repeat_frame: pd.DataFrame,
    legacy_frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    repeat_frame = repeat_frame.copy()
    repeat_frame["repeat"] = repeat_frame["repeat"].astype(str).str.zfill(2)
    if len(repeat_frame) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS:,} schema-2 repeat records."
        )
    if repeat_frame.duplicated(["subject", "roi", "repeat"]).any():
        raise RuntimeError("Schema-2 repeat metrics contain duplicate keys.")
    merged = repeat_frame.merge(
        legacy_frame,
        on=["subject", "roi", "repeat"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise RuntimeError(f"Legacy/schema-2 record-key mismatch: {counts}")
    roi_mismatch = int(
        (merged["roi_voxels"] != merged["legacy_roi_voxels"]).sum()
    )
    whole_mismatch = int(
        (
            merged["whole_brain_voxels"]
            != merged["legacy_whole_brain_voxels"]
        ).sum()
    )
    if roi_mismatch or whole_mismatch:
        raise RuntimeError(
            "Legacy/schema-2 voxel denominators differ: "
            f"ROI={roi_mismatch}, whole brain={whole_mismatch}."
        )

    metric_slug = _threshold_slug(threshold)
    target = merged["target_count"].astype(int)
    whole = merged["whole_count"].astype(int)
    off_target = whole - target
    if (off_target > merged["off_target_voxels"]).any():
        raise RuntimeError("An off-target suprathreshold count exceeds its denominator.")

    voxel_volume = merged["voxel_volume_mm3"].astype(float)
    merged[f"target_coverage_voxels_ge_{metric_slug}"] = target
    merged[f"target_coverage_volume_mm3_ge_{metric_slug}"] = (
        target * voxel_volume
    )
    merged[f"target_coverage_percent_ge_{metric_slug}"] = safe_percent(
        target, merged["roi_voxels"]
    )
    merged[f"whole_brain_coverage_voxels_ge_{metric_slug}"] = whole
    merged[f"whole_brain_coverage_volume_mm3_ge_{metric_slug}"] = (
        whole * voxel_volume
    )
    merged[f"whole_brain_coverage_percent_ge_{metric_slug}"] = safe_percent(
        whole, merged["whole_brain_voxels"]
    )
    merged[f"off_target_coverage_voxels_ge_{metric_slug}"] = off_target
    merged[f"off_target_coverage_volume_mm3_ge_{metric_slug}"] = (
        off_target * voxel_volume
    )
    merged[f"off_target_coverage_percent_ge_{metric_slug}"] = safe_percent(
        off_target, merged["off_target_voxels"]
    )
    target_array = target.to_numpy(dtype=float)
    whole_array = whole.to_numpy(dtype=float)
    localization = np.zeros_like(whole_array, dtype=float)
    np.divide(
        target_array * 100.0,
        whole_array,
        out=localization,
        where=whole_array > 0,
    )
    merged[
        f"threshold_localization_percent_in_roi_ge_{metric_slug}"
    ] = localization

    audit = {
        "record_keys_matched": int(len(merged)),
        "roi_voxel_denominator_mismatches": roi_mismatch,
        "whole_brain_voxel_denominator_mismatches": whole_mismatch,
        "zero_whole_brain_suprathreshold_records": int((whole == 0).sum()),
    }
    drop_columns = [
        "_merge",
        "legacy_roi_voxels",
        "legacy_whole_brain_voxels",
        "target_count",
        "whole_count",
    ]
    return merged.drop(columns=drop_columns), audit


def aggregate_subject_metrics(
    repeat_frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    metric_slug = _threshold_slug(threshold)
    metrics = [
        name
        for name in manuscript_metric_names((threshold,))
        if name.endswith(f"_ge_{metric_slug}")
    ]
    groups = ["subject", "roi", "canonical_roi"]
    means = (
        repeat_frame.groupby(groups, sort=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    sds = (
        repeat_frame.groupby(groups, sort=False)[metrics]
        .std(ddof=1, numeric_only=True)
        .add_suffix("__repeat_sd")
        .reset_index()
    )
    counts = (
        repeat_frame.groupby(groups, sort=False)
        .size()
        .rename("repeat_count")
        .reset_index()
    )
    result = means.merge(sds, on=groups).merge(counts, on=groups)
    if len(result) != EXPECTED_SUBJECTS * len(ROI_ORDER):
        raise RuntimeError("Unexpected subject-level 0.20 V/m row count.")
    if not (result["repeat_count"] == EXPECTED_REPEATS).all():
        raise RuntimeError("At least one subject/ROI lacks exactly ten repeats.")
    return result


def recompute_mni_metrics(
    *,
    baseline_parent: Path,
    atlas_path: Path,
    existing_mni: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, float]:
    thresholds = (threshold, 0.18, 0.15)
    rows = [
        _compute_mni_record(
            roi=roi,
            baseline_parent=baseline_parent,
            mni_atlas_path=atlas_path,
            thresholds=thresholds,
            top_percentile=95.0,
            robust_max_percentile=99.9,
            upper_tail_fraction=0.01,
        )
        for roi in ROI_ORDER
    ]
    computed = pd.DataFrame(rows).set_index("roi")
    existing = existing_mni.set_index("roi")
    validation_columns = manuscript_metric_names((0.18, 0.15))
    differences = (
        computed[validation_columns].astype(float)
        - existing[validation_columns].astype(float)
    ).abs()
    maximum_difference = float(differences.to_numpy().max())
    if maximum_difference > 1e-10:
        raise RuntimeError(
            "Recomputed corrected MNI152 values do not reproduce the "
            "downloaded 0.18/0.15 schema-2 rows; maximum absolute "
            f"difference={maximum_difference:.12g}."
        )
    return computed.reset_index(), maximum_difference


def merge_subject_threshold(
    existing_subject: pd.DataFrame,
    subject_threshold: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["subject", "roi", "canonical_roi"]
    threshold_columns = [
        column for column in subject_threshold.columns if column not in keys
    ]
    if "repeat_count" in threshold_columns:
        threshold_columns.remove("repeat_count")
    merged = existing_subject.merge(
        subject_threshold[keys + threshold_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_SUBJECTS * len(ROI_ORDER):
        raise RuntimeError("Subject-level threshold merge is incomplete.")
    return merged


def plot_effectiveness_spread(
    subject_frame: pd.DataFrame,
    mni_by_roi: pd.DataFrame,
    figures_dir: Path,
    threshold: float,
) -> None:
    metric_slug = _threshold_slug(threshold)
    file_slug = threshold_file_slug(threshold)
    x_metric = f"target_coverage_percent_ge_{metric_slug}"
    y_metric = f"off_target_coverage_percent_ge_{metric_slug}"
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.4))
    for axis, roi in zip(axes.flat, ROI_ORDER):
        rows = subject_frame.loc[subject_frame["roi"] == roi]
        x = rows[x_metric].astype(float)
        y = rows[y_metric].astype(float)
        rho = float(spearmanr(x, y).statistic)
        axis.scatter(
            x,
            y,
            s=20,
            alpha=0.52,
            color=FIGURE_BLUE,
            edgecolors="none",
        )
        if np.ptp(x) > 0:
            coefficients = np.polyfit(x, y, 1)
            grid = np.linspace(float(x.min()), float(x.max()), 100)
            axis.plot(
                grid,
                coefficients[0] * grid + coefficients[1],
                color=FIGURE_GRAY,
                linewidth=1.2,
                linestyle="--",
            )
        axis.scatter(
            [mni_by_roi.loc[roi, x_metric]],
            [mni_by_roi.loc[roi, y_metric]],
            s=70,
            marker="D",
            color=FIGURE_ORANGE,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        axis.text(
            0.04,
            0.95,
            rf"Spearman $\rho$ = {rho:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set_title(ROI_LABELS[roi])
        axis.set_xlabel(f"Target coverage ≥{threshold:.2f} V/m (%)")
        axis.set_ylabel(f"Off-target coverage ≥{threshold:.2f} V/m (%)")
        axis.grid(True, color="#E5E7EB", linewidth=0.6)
        axis.set_xlim(left=min(0.0, float(x.min()) - 2))
        axis.set_ylim(bottom=min(0.0, float(y.min()) - 1))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FIGURE_BLUE,
            alpha=0.65,
            markersize=6,
            label="CamCan subject (ten-repeat mean)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=FIGURE_ORANGE,
            markersize=7,
            label="MNI152 reference",
        ),
    ]
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.84,
        wspace=0.26,
        hspace=0.35,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"Within-ROI effectiveness–spread relationship at {threshold:.2f} V/m",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    save_figure(
        fig,
        figures_dir
        / f"figure_effectiveness_off_target_relationship_ge_{file_slug}",
    )


def plot_target_coverage_ecdf(
    subject_frame: pd.DataFrame,
    mni_by_roi: pd.DataFrame,
    figures_dir: Path,
    threshold: float,
) -> None:
    metric_slug = _threshold_slug(threshold)
    file_slug = threshold_file_slug(threshold)
    metric = f"target_coverage_percent_ge_{metric_slug}"
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), constrained_layout=True)
    for axis, roi in zip(axes.flat, ROI_ORDER):
        values = np.sort(
            subject_frame.loc[
                subject_frame["roi"] == roi, metric
            ].to_numpy(dtype=float)
        )
        cumulative = np.arange(1, len(values) + 1) / len(values)
        mni_value = float(mni_by_roi.loc[roi, metric])
        axis.step(
            values,
            cumulative,
            where="post",
            color=FIGURE_BLUE,
            linewidth=2,
        )
        axis.axvline(
            mni_value,
            color=FIGURE_ORANGE,
            linestyle="--",
            linewidth=1.6,
            label=f"MNI152 ({mni_value:.1f}%)",
        )
        for cutoff in [25, 50, 75]:
            axis.axvline(
                cutoff, color="#D1D5DB", linewidth=0.7, zorder=0
            )
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 1.02)
        axis.set_title(ROI_LABELS[roi])
        axis.set_xlabel(f"Target coverage ≥{threshold:.2f} V/m (%)")
        axis.set_ylabel("Cumulative proportion of subjects")
        axis.legend(frameon=False, loc="lower right")
        axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.6)
    fig.suptitle(
        f"Inter-individual distribution of target coverage at {threshold:.2f} V/m",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(
        fig,
        figures_dir / f"figure_target_coverage_ecdf_ge_{file_slug}",
    )


def plot_threshold_sensitivity(
    subject_frame: pd.DataFrame,
    mni_by_roi: pd.DataFrame,
    figures_dir: Path,
    threshold: float,
) -> None:
    thresholds = [threshold, 0.18, 0.15]
    x = np.arange(len(thresholds))
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
    for axis, roi in zip(axes.flat, ROI_ORDER):
        rows = subject_frame.loc[subject_frame["roi"] == roi]
        for label, stem, color in [
            ("Target coverage", "target_coverage_percent", FIGURE_BLUE),
            ("Off-target coverage", "off_target_coverage_percent", FIGURE_RED),
        ]:
            metrics = [
                f"{stem}_ge_{_threshold_slug(value)}"
                for value in thresholds
            ]
            means = np.array([rows[metric].mean() for metric in metrics])
            sds = np.array([rows[metric].std(ddof=1) for metric in metrics])
            mni = np.array(
                [mni_by_roi.loc[roi, metric] for metric in metrics]
            )
            axis.errorbar(
                x - 0.035,
                means,
                yerr=sds,
                marker="o",
                color=color,
                linewidth=1.8,
                capsize=3,
                markersize=5,
                label=f"CamCan {label}",
            )
            axis.plot(
                x + 0.035,
                mni,
                marker="D",
                color=color,
                linewidth=1.2,
                linestyle=":",
                markersize=5,
                alpha=0.9,
                label=f"MNI152 {label}",
            )
        axis.set_xticks(x, [f"{value:.2f}" for value in thresholds])
        axis.set_xlabel("TI-field threshold (V/m)")
        axis.set_ylabel("Coverage (%)")
        axis.set_ylim(0, 105)
        axis.set_title(ROI_LABELS[roi])
        axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.80,
        wspace=0.26,
        hspace=0.36,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Threshold sensitivity of target effectiveness and off-target spread",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    save_figure(
        fig, figures_dir / "figure_threshold_sensitivity_with_0p20"
    )


def mni_context_rows(
    subject_frame: pd.DataFrame,
    mni_by_roi: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    metric_slug = _threshold_slug(threshold)
    metrics = [
        (
            "Field magnitude",
            "ROI median field",
            "roi_median_v_per_m",
        ),
        (
            "Field magnitude",
            "ROI P99.9",
            "roi_robust_max_p99_9_v_per_m",
        ),
        (
            f"Threshold performance ({threshold:.2f} V/m)",
            f"Target coverage ≥{threshold:.2f}",
            f"target_coverage_percent_ge_{metric_slug}",
        ),
        (
            f"Threshold performance ({threshold:.2f} V/m)",
            f"Off-target coverage ≥{threshold:.2f}",
            f"off_target_coverage_percent_ge_{metric_slug}",
        ),
        (
            f"Threshold performance ({threshold:.2f} V/m)",
            f"Whole-brain coverage ≥{threshold:.2f}",
            f"whole_brain_coverage_percent_ge_{metric_slug}",
        ),
        (
            f"Threshold performance ({threshold:.2f} V/m)",
            f"Localization ≥{threshold:.2f}",
            f"threshold_localization_percent_in_roi_ge_{metric_slug}",
        ),
        (
            "Rank-based sensitivity",
            "Top-5% target coverage",
            "top_5_percent_target_coverage_percent",
        ),
        (
            "Rank-based sensitivity",
            "Top-5% localization",
            "top_5_percent_localization_percent_in_roi",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for roi in ROI_ORDER:
        roi_subjects = subject_frame.loc[subject_frame["roi"] == roi]
        for order, (section, label, metric) in enumerate(metrics):
            values = roi_subjects[metric].astype(float)
            mni_value = float(mni_by_roi.loc[roi, metric])
            rows.append(
                {
                    "roi": roi,
                    "ROI": ROI_LABELS[roi],
                    "section": section,
                    "label": label,
                    "metric": metric,
                    "order": order,
                    "camcan_n": int(len(values)),
                    "camcan_mean": float(values.mean()),
                    "camcan_sd": float(values.std(ddof=1)),
                    "camcan_median": float(values.median()),
                    "mni152": mni_value,
                    "mni152_percentile_within_camcan": float(
                        percentileofscore(values, mni_value, kind="mean")
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_mni_context(
    context: pd.DataFrame,
    figures_dir: Path,
    threshold: float,
) -> None:
    threshold_section = f"Threshold performance ({threshold:.2f} V/m)"
    category_colors = {
        "Field magnitude": FIGURE_BLUE,
        threshold_section: FIGURE_RED,
        "Rank-based sensitivity": FIGURE_GREEN,
    }
    file_slug = threshold_file_slug(threshold)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    for axis, roi in zip(axes.flat, ROI_ORDER):
        rows = context.loc[context["roi"] == roi].sort_values(
            "order", ascending=False
        )
        y = np.arange(len(rows))
        colors = [category_colors[value] for value in rows["section"]]
        axis.axvspan(25, 75, color="#F3F4F6", zorder=0)
        axis.axvline(50, color=FIGURE_GRAY, linewidth=1, linestyle="--")
        axis.scatter(
            rows["mni152_percentile_within_camcan"],
            y,
            color=colors,
            s=42,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.set_yticks(y, rows["label"])
        axis.set_xlim(0, 100)
        axis.set_xlabel("MNI152 percentile within CamCan (%)")
        axis.set_title(ROI_LABELS[roi])
        axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.6)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markersize=7,
            label=label,
        )
        for label, color in category_colors.items()
    ]
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        bottom=0.08,
        top=0.82,
        wspace=0.50,
        hspace=0.32,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Position of the MNI152 reference within each CamCan distribution",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    save_figure(
        fig,
        figures_dir / f"figure_mni152_percentile_context_ge_{file_slug}",
    )


def build_association_table(
    subject_frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    metric_slug = _threshold_slug(threshold)
    x_metric = f"target_coverage_percent_ge_{metric_slug}"
    rows = []
    for roi in ROI_ORDER:
        selected = subject_frame.loc[subject_frame["roi"] == roi]
        x = selected[x_metric].astype(float)
        for label, y_metric in [
            (
                "Effectiveness–spread coupling",
                f"off_target_coverage_percent_ge_{metric_slug}",
            ),
            (
                "Effectiveness–localization relationship",
                f"threshold_localization_percent_in_roi_ge_{metric_slug}",
            ),
        ]:
            correlation = spearmanr(x, selected[y_metric].astype(float))
            rows.append(
                {
                    "ROI": ROI_LABELS[roi],
                    "Threshold (V/m)": threshold,
                    "Relationship": label,
                    "n": int(len(x)),
                    "Spearman rho": float(correlation.statistic),
                    "Spearman p (descriptive)": float(correlation.pvalue),
                }
            )
    return pd.DataFrame(rows)


def build_summary_table(
    subject_frame: pd.DataFrame,
    mni_by_roi: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    metric_slug = _threshold_slug(threshold)
    metrics = [
        (
            "Target coverage",
            f"target_coverage_percent_ge_{metric_slug}",
        ),
        (
            "Off-target coverage",
            f"off_target_coverage_percent_ge_{metric_slug}",
        ),
        (
            "Whole-brain coverage",
            f"whole_brain_coverage_percent_ge_{metric_slug}",
        ),
        (
            "Suprathreshold localization in target",
            f"threshold_localization_percent_in_roi_ge_{metric_slug}",
        ),
    ]
    rows = []
    for roi in ROI_ORDER:
        selected = subject_frame.loc[subject_frame["roi"] == roi]
        for label, metric in metrics:
            values = selected[metric].astype(float)
            rows.append(
                {
                    "ROI": ROI_LABELS[roi],
                    "Threshold (V/m)": threshold,
                    "Outcome": label,
                    "CamCan n": int(len(values)),
                    "CamCan mean (%)": float(values.mean()),
                    "CamCan SD (%)": float(values.std(ddof=1)),
                    "CamCan median (%)": float(values.median()),
                    "CamCan Q1 (%)": float(values.quantile(0.25)),
                    "CamCan Q3 (%)": float(values.quantile(0.75)),
                    "MNI152 (%)": float(mni_by_roi.loc[roi, metric]),
                }
            )
    return pd.DataFrame(rows)


def build_captions(
    association: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    file_slug = threshold_file_slug(threshold)
    spread = association.loc[
        association["Relationship"] == "Effectiveness–spread coupling"
    ].set_index("ROI")
    rho_text = ", ".join(
        f"{ROI_LABELS[roi]} rho={spread.loc[ROI_LABELS[roi], 'Spearman rho']:.2f}"
        for roi in ROI_ORDER
    )
    records = [
        {
            "Figure file stem": (
                f"figure_effectiveness_off_target_relationship_ge_{file_slug}"
            ),
            "Short title": (
                f"Target effectiveness and off-target spread at {threshold:.2f} V/m"
            ),
            "Caption": (
                "Relationship between target effectiveness and off-target "
                "spread during temporal-interference (TI) stimulation in 132 "
                "CamCan participants. Separate panels show fixed electrode "
                "montages targeting the left hippocampus, left primary motor "
                "cortex (M1), right dorsolateral prefrontal cortex (DLPFC), "
                f"and right thalamus. Target coverage is the percentage of "
                "voxels in the anatomical target region whose TI field was "
                f"at least {threshold:.2f} V/m; off-target coverage is the "
                "percentage of finite whole-brain voxels outside that target "
                "meeting the same threshold. Each blue point is one "
                "participant after metrics were calculated separately for "
                "ten independently remeshed models and arithmetic-mean "
                "aggregated. The orange diamond is the corrected MNI152 "
                "reference simulated with the same target-specific montage "
                "and currents. Gray dashed lines are descriptive least-"
                "squares fits. Panel Spearman correlations are descriptive "
                f"({rho_text}); no cross-ROI inference was performed."
            ),
        },
        {
            "Figure file stem": (
                f"figure_target_coverage_ecdf_ge_{file_slug}"
            ),
            "Short title": (
                f"Distribution of target coverage at {threshold:.2f} V/m"
            ),
            "Caption": (
                "Inter-individual distribution of target coverage during "
                "temporal-interference (TI) stimulation in 132 CamCan "
                "participants. Separate panels show fixed electrode montages "
                "targeting the left hippocampus, left primary motor cortex "
                "(M1), right dorsolateral prefrontal cortex (DLPFC), and "
                "right thalamus. Target coverage is the percentage of "
                "anatomical target voxels whose TI field was at least "
                f"{threshold:.2f} V/m. The blue empirical cumulative "
                "distribution gives the proportion of participants at or "
                "below each coverage value; curves farther right indicate "
                "greater target coverage. Each participant value is the "
                "arithmetic mean of metrics calculated independently for "
                "ten remeshed models. The orange dashed line is the corrected "
                "MNI152 reference. Gray lines at 25%, 50%, and 75% are "
                "coverage guides, not cohort quartiles."
            ),
        },
        {
            "Figure file stem": "figure_threshold_sensitivity_with_0p20",
            "Short title": "Sensitivity across 0.20, 0.18, and 0.15 V/m",
            "Caption": (
                "Sensitivity of target and off-target coverage to the "
                "temporal-interference (TI) field threshold in 132 CamCan "
                "participants. Panels correspond to montages targeting the "
                "left hippocampus, left primary motor cortex (M1), right "
                "dorsolateral prefrontal cortex (DLPFC), and right thalamus. "
                "Blue denotes target coverage and red denotes off-target "
                "coverage. Solid lines and circles show CamCan means; error "
                "bars are one between-participant standard deviation. Each "
                "participant value is a ten-repeat arithmetic mean. Dotted "
                "lines and diamonds show the single corrected MNI152 "
                "reference and therefore have no error bars. Lowering the "
                "criterion classifies more tissue as suprathreshold; it does "
                "not alter the simulated field."
            ),
        },
        {
            "Figure file stem": (
                f"figure_mni152_percentile_context_ge_{file_slug}"
            ),
            "Short title": (
                f"MNI152 context using {threshold:.2f}-V/m threshold metrics"
            ),
            "Caption": (
                "Position of the corrected MNI152 reference within outcome "
                "distributions from 132 CamCan participants undergoing "
                "simulated temporal-interference stimulation. Each point is "
                "the percentile rank of the single MNI152 value within the "
                "corresponding distribution of participant ten-repeat means; "
                "0%, 50%, and 100% denote the bottom, median, and top of that "
                "distribution. Blue points are target-region field magnitude "
                "metrics, red points are target, off-target, whole-brain, and "
                f"localization metrics evaluated at {threshold:.2f} V/m, and "
                "green points are whole-brain top-5% rank-based metrics. The "
                "gray band spans the 25th–75th percentiles and the dashed "
                "line marks the cohort median. Percentile direction depends "
                "on the metric: high target coverage and high off-target "
                "coverage do not have the same interpretation. MNI152 is a "
                "single descriptive reference, not a comparison group."
            ),
        },
    ]
    return pd.DataFrame(records)


def write_captions(
    captions: pd.DataFrame,
    out_dir: Path,
) -> None:
    captions.to_csv(out_dir / "figure_captions_0p20.csv", index=False)
    lines = ["# Self-contained captions for the 0.20 V/m figure set", ""]
    captions_dir = out_dir / "captions"
    captions_dir.mkdir(exist_ok=True)
    for row in captions.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row['Short title']}",
                "",
                f"Files: `{row['Figure file stem']}.png` and "
                f"`{row['Figure file stem']}.pdf`",
                "",
                row["Caption"],
                "",
            ]
        )
        (captions_dir / f"{row['Figure file stem']}_caption.txt").write_text(
            row["Caption"] + "\n", encoding="utf-8"
        )
    (out_dir / "figure_captions_0p20.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest(
    *,
    out_dir: Path,
    input_dir: Path,
    legacy_runs_root: Path,
    baseline_parent: Path,
    atlas_zip: Path,
    atlas_member: str,
    threshold: float,
    audit: dict[str, Any],
    mni_maximum_difference: float,
    outputs: list[Path],
) -> Path:
    manifest_path = out_dir / "threshold_0p20_figure_manifest.json"
    payload = {
        "status": "complete",
        "threshold_v_per_m": threshold,
        "subjects": EXPECTED_SUBJECTS,
        "rois": ROI_ORDER,
        "repeats_per_subject_roi": EXPECTED_REPEATS,
        "repeat_level_records": EXPECTED_RECORDS,
        "aggregation": (
            "Each threshold metric was calculated independently per repeat, "
            "then arithmetic-mean aggregated across ten repeats."
        ),
        "camcan_0p20_source": (
            "Validated per-repeat >=0.20 V/m voxel counts in the earlier "
            "subject_metrics.json post-processing archive."
        ),
        "mni_source": (
            "Corrected ROI-specific MNI152 TI NIfTIs and the exact archived "
            "Destrieux/a2009s atlas."
        ),
        "input_dir": str(input_dir),
        "legacy_runs_root": str(legacy_runs_root),
        "mni_baseline_parent": str(baseline_parent),
        "mni_atlas_zip": str(atlas_zip),
        "mni_atlas_member": atlas_member,
        "source_sha256": {
            "analysis_manifest.json": sha256_file(
                input_dir / "analysis_manifest.json"
            ),
            "repeat_level_metrics.csv": sha256_file(
                input_dir / "repeat_level_metrics.csv"
            ),
            "subject_level_repeat_mean_metrics.csv": sha256_file(
                input_dir / "subject_level_repeat_mean_metrics.csv"
            ),
            "mni152_baseline_metrics.csv": sha256_file(
                input_dir / "mni152_baseline_metrics.csv"
            ),
            "mni_atlas_zip": sha256_file(atlas_zip),
        },
        "validation": {
            **audit,
            "mni_existing_0p18_0p15_max_abs_difference": (
                mni_maximum_difference
            ),
            "mni_validation_tolerance": 1e-10,
        },
        "outputs": {
            str(path.relative_to(out_dir)): sha256_file(path)
            for path in sorted(outputs)
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--legacy-post-root", type=Path, required=True)
    parser.add_argument("--mni-baseline-parent", type=Path, required=True)
    parser.add_argument("--mni-atlas-zip", type=Path, required=True)
    parser.add_argument(
        "--mni-atlas-member",
        default=DEFAULT_MNI_ATLAS_MEMBER,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    legacy_runs_root = find_runs_root(args.legacy_post_root)
    baseline_parent = args.mni_baseline_parent.expanduser().resolve()
    atlas_zip = args.mni_atlas_zip.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    threshold = float(args.threshold)
    if not math.isclose(threshold, DEFAULT_THRESHOLD, abs_tol=1e-12):
        raise ValueError(
            "This audited bridge is specifically for the legacy 0.20 V/m counts."
        )

    required = [
        input_dir / "analysis_manifest.json",
        input_dir / "repeat_level_metrics.csv",
        input_dir / "subject_level_repeat_mean_metrics.csv",
        input_dir / "mni152_baseline_metrics.csv",
        atlas_zip,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))
    manifest = json.loads(
        (input_dir / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("analysis_schema_version") != 2
        or manifest.get("status") != "complete"
        or manifest.get("repeat_level_records") != EXPECTED_RECORDS
    ):
        raise RuntimeError("Input is not the validated final-132 schema-2 package.")

    repeat_frame = pd.read_csv(
        input_dir / "repeat_level_metrics.csv",
        dtype={"repeat": str},
    )
    existing_subject = pd.read_csv(
        input_dir / "subject_level_repeat_mean_metrics.csv"
    )
    existing_mni = pd.read_csv(input_dir / "mni152_baseline_metrics.csv")
    legacy_frame = load_legacy_counts(legacy_runs_root, threshold)
    repeat_extended, audit = reconstruct_repeat_metrics(
        repeat_frame, legacy_frame, threshold
    )
    subject_threshold = aggregate_subject_metrics(repeat_extended, threshold)
    subject_extended = merge_subject_threshold(
        existing_subject, subject_threshold
    )

    with tempfile.TemporaryDirectory(prefix="camcan_mni_atlas_") as temporary:
        temporary_dir = Path(temporary)
        with zipfile.ZipFile(atlas_zip) as archive:
            if args.mni_atlas_member not in archive.namelist():
                raise FileNotFoundError(
                    f"Atlas member not found in archive: {args.mni_atlas_member}"
                )
            archive.extract(args.mni_atlas_member, temporary_dir)
        atlas_path = temporary_dir / args.mni_atlas_member
        mni_extended, mni_maximum_difference = recompute_mni_metrics(
            baseline_parent=baseline_parent,
            atlas_path=atlas_path,
            existing_mni=existing_mni,
            threshold=threshold,
        )

    mni_by_roi = mni_extended.set_index("roi")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    metric_slug = _threshold_slug(threshold)
    threshold_metrics = [
        name
        for name in manuscript_metric_names((threshold,))
        if name.endswith(f"_ge_{metric_slug}")
    ]
    repeat_export = repeat_extended[
        ["subject", "roi", "repeat", "canonical_roi"] + threshold_metrics
    ].copy()
    subject_export_columns = [
        "subject",
        "roi",
        "canonical_roi",
        *threshold_metrics,
        *[f"{metric}__repeat_sd" for metric in threshold_metrics],
        "repeat_count",
    ]
    subject_export = subject_threshold[subject_export_columns].copy()
    mni_export = mni_extended[
        ["subject", "roi"] + threshold_metrics
    ].copy()
    summary = build_summary_table(
        subject_extended, mni_by_roi, threshold
    )
    association = build_association_table(subject_extended, threshold)
    context = mni_context_rows(subject_extended, mni_by_roi, threshold)
    tables = {
        "repeat_level_metrics_ge_0p20.csv": repeat_export,
        "subject_level_repeat_mean_metrics_ge_0p20.csv": subject_export,
        "mni152_baseline_metrics_ge_0p20.csv": mni_export,
        "table_threshold_0p20_descriptive_statistics.csv": summary,
        "table_threshold_0p20_within_roi_associations.csv": association,
        "table_mni152_percentile_context_ge_0p20.csv": context,
    }
    for filename, frame in tables.items():
        frame.to_csv(tables_dir / filename, index=False)

    setup_matplotlib()
    plot_effectiveness_spread(
        subject_extended, mni_by_roi, figures_dir, threshold
    )
    plot_target_coverage_ecdf(
        subject_extended, mni_by_roi, figures_dir, threshold
    )
    plot_threshold_sensitivity(
        subject_extended, mni_by_roi, figures_dir, threshold
    )
    plot_mni_context(context, figures_dir, threshold)
    file_slug = threshold_file_slug(threshold)
    _write_effectiveness_spread_figure(
        subject_frame=subject_extended,
        mni_frame=mni_extended,
        threshold=threshold,
        spread_scope="off_target",
        output_base=(
            figures_dir
            / f"effectiveness_vs_off_target_coverage_ge_{file_slug}"
        ),
    )
    _write_effectiveness_spread_figure(
        subject_frame=subject_extended,
        mni_frame=mni_extended,
        threshold=threshold,
        spread_scope="whole_brain",
        output_base=(
            figures_dir
            / f"effectiveness_vs_whole_brain_coverage_ge_{file_slug}"
        ),
    )

    captions = build_captions(association, threshold)
    write_captions(captions, out_dir)
    generated = [
        *figures_dir.glob("*0p20*"),
        *tables_dir.glob("*0p20*"),
        out_dir / "figure_captions_0p20.csv",
        out_dir / "figure_captions_0p20.md",
        *list((out_dir / "captions").glob("*0p20*")),
    ]
    generated = sorted({path for path in generated if path.is_file()})
    manifest_path = write_manifest(
        out_dir=out_dir,
        input_dir=input_dir,
        legacy_runs_root=legacy_runs_root,
        baseline_parent=baseline_parent,
        atlas_zip=atlas_zip,
        atlas_member=args.mni_atlas_member,
        threshold=threshold,
        audit=audit,
        mni_maximum_difference=mni_maximum_difference,
        outputs=generated,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "threshold_v_per_m": threshold,
                "repeat_level_records": len(repeat_export),
                "subject_level_records": len(subject_export),
                "mni_baselines": len(mni_export),
                "figures_written": len(list(figures_dir.glob("*0p20*"))),
                "tables_written": len(tables),
                "manifest": str(manifest_path),
                "validation": {
                    **audit,
                    "mni_existing_0p18_0p15_max_abs_difference": (
                        mni_maximum_difference
                    ),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
