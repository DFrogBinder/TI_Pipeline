#!/usr/bin/env python3
"""Assess whether the age-balanced 14-subject cohort spans generic-montage fields.

The analysis compares the MNI152-derived generic montage simulated on each
participant head with the corresponding MNI152 baseline. It is descriptive:
it assesses field-distribution coverage after the cohort has been selected,
but does not redefine the cohort or claim statistical equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "optimized_vs_unoptimized_age_cohort"
SOURCE_PACKAGE = (
    Path.home()
    / "sandbox"
    / "Jake_Data"
    / "CamCan-PostData"
    / "manuscript-ready-supervisor-revision-0p20-v4"
    / "source_tables"
    / "cohort"
)
DEFAULT_COHORT_TABLE = SOURCE_PACKAGE / "subject_level_repeat_mean_metrics.csv"
DEFAULT_MNI_TABLE = SOURCE_PACKAGE / "mni152_baseline_metrics.csv"
DEFAULT_ANALYSIS_MANIFEST = SOURCE_PACKAGE / "analysis_manifest.json"
DEFAULT_SELECTION_MANIFEST = (
    REPO_ROOT
    / "CamCan_Experiment"
    / "cohort_pipeline"
    / "cohorts"
    / "optimized_vs_unoptimized_age_balanced_14"
    / "selection_manifest.csv"
)
DEFAULT_SELECTION_AUDIT = (
    REPO_ROOT
    / "output"
    / "spreadsheet"
    / "final_132_optimized_vs_unoptimized_age_balanced_14"
    / "eligible_132_selection_audit.csv"
)
DEFAULT_OPTIMIZED_SEVEN = (
    REPO_ROOT
    / "CamCan_Experiment"
    / "cohort_pipeline"
    / "cohorts"
    / "optimized_best_worst_7"
    / "subjects.txt"
)
DEFAULT_OUTPUT_DIR = TASK_DIR / "downloads" / "Results" / "representativeness"

EXPECTED_SUBJECTS = 132
EXPECTED_SELECTED = 14
EXPECTED_OPTIMIZED = 7
EXPECTED_REPEATS = 10
TARGET_ORDER = (
    "Left_Hippocampus",
    "Left_M1",
    "Right_DLPC",
    "Right_Thalamus",
)
TARGET_LABELS = {
    "Left_Hippocampus": "Left hippocampus",
    "Left_M1": "Left M1",
    "Right_DLPC": "Right DLPFC",
    "Right_Thalamus": "Right thalamus",
}
SEX_ORDER = ("FEMALE", "MALE")
SEX_LABELS = {"FEMALE": "Female", "MALE": "Male"}
SEX_COLORS = {"FEMALE": "#D55E00", "MALE": "#0072B2"}
SEX_MARKERS = {"FEMALE": "o", "MALE": "s"}


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    relative_label: str
    transform: str


METRICS = (
    MetricSpec(
        "roi_mean_v_per_m",
        "Target-ROI mean field",
        "Difference from MNI152 (%)",
        "percent",
    ),
    MetricSpec(
        "roi_median_v_per_m",
        "Target-ROI median field",
        "Difference from MNI152 (%)",
        "percent",
    ),
    MetricSpec(
        "target_coverage_percent_ge_0p2",
        "Target coverage at 0.20 V/m",
        "Difference from MNI152 (percentage points)",
        "percentage_points",
    ),
    MetricSpec(
        "off_target_coverage_percent_ge_0p2",
        "Off-target coverage at 0.20 V/m",
        "Difference from MNI152 (percentage points)",
        "percentage_points",
    ),
)
PRIMARY_METRIC = METRICS[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path, expected: int) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != expected or len(values) != len(set(values)):
        raise ValueError(
            f"{path} must contain {expected} unique subject IDs; found {len(values)} rows."
        )
    return values


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def empirical_percentile(reference: np.ndarray, value: float) -> float:
    """Return the mid-rank empirical percentile of value in reference."""

    reference = np.asarray(reference, dtype=float)
    if reference.size == 0 or not np.isfinite(reference).all() or not math.isfinite(value):
        raise ValueError("Empirical percentiles require finite, non-empty values.")
    below = np.count_nonzero(reference < value)
    equal = np.count_nonzero(reference == value)
    return 100.0 * (below + 0.5 * equal) / reference.size


def design_matrix(frame: pd.DataFrame, age_center: float) -> np.ndarray:
    age_per_10 = (frame["age_years_precise"].to_numpy(dtype=float) - age_center) / 10.0
    male = frame["recorded_sex"].eq("MALE").to_numpy(dtype=float)
    return np.column_stack(
        [np.ones(len(frame)), age_per_10, male, age_per_10 * male]
    )


def fit_adjusted_diagnostics(
    reference: pd.DataFrame,
    selected: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fit on the unselected reference and return honest residual diagnostics.

    The reference distribution uses leave-one-out (PRESS) residuals. Selected
    residuals are predictions from the model fitted only to the reference set.
    Both are standardized to the reference PRESS-residual distribution.
    """

    age_center = float(reference["age_years_precise"].mean())
    x_ref = design_matrix(reference, age_center)
    y_ref = reference[value_column].to_numpy(dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(x_ref, y_ref, rcond=None)
    if rank != x_ref.shape[1]:
        raise ValueError(f"Age/sex design matrix is rank deficient for {value_column}.")

    fitted = x_ref @ beta
    residual = y_ref - fitted
    hat = np.sum(x_ref * (x_ref @ np.linalg.inv(x_ref.T @ x_ref)), axis=1)
    if np.any(hat >= 1.0):
        raise ValueError("Invalid leverage encountered while calculating PRESS residuals.")
    press = residual / (1.0 - hat)
    press_mean = float(np.mean(press))
    press_sd = float(np.std(press, ddof=1))
    if not math.isfinite(press_sd) or press_sd <= 0:
        raise ValueError(f"Reference residual scale is not positive for {value_column}.")

    ref_result = reference.copy()
    ref_result["adjusted_residual"] = press
    ref_result["adjusted_residual_z"] = (press - press_mean) / press_sd

    x_selected = design_matrix(selected, age_center)
    selected_result = selected.copy()
    selected_result["adjusted_residual"] = (
        selected[value_column].to_numpy(dtype=float) - x_selected @ beta
    )
    selected_result["adjusted_residual_z"] = (
        selected_result["adjusted_residual"] - press_mean
    ) / press_sd

    z_ref = ref_result["adjusted_residual_z"].to_numpy(dtype=float)
    z_selected = selected_result["adjusted_residual_z"].to_numpy(dtype=float)
    selected_result["adjusted_reference_percentile"] = [
        empirical_percentile(z_ref, value) for value in z_selected
    ]
    ks_result = stats.ks_2samp(z_selected, z_ref, alternative="two-sided", method="auto")
    diagnostics = {
        "age_center_years": age_center,
        "adjusted_smd": float(np.mean(z_selected) - np.mean(z_ref)),
        "adjusted_variance_ratio": float(np.var(z_selected, ddof=1) / np.var(z_ref, ddof=1)),
        "adjusted_ks_statistic": float(ks_result.statistic),
        "adjusted_ks_p_value": float(ks_result.pvalue),
        "selected_span_reference_percent": float(
            100.0
            * np.mean((z_ref >= np.min(z_selected)) & (z_ref <= np.max(z_selected)))
        ),
        "selected_max_abs_adjusted_z": float(np.max(np.abs(z_selected))),
        "selected_abs_adjusted_z_gt_2_count": int(np.count_nonzero(np.abs(z_selected) > 2.0)),
    }
    return ref_result, selected_result, diagnostics


def transform_from_mni(values: pd.Series, baseline: float, transform: str) -> pd.Series:
    if not math.isfinite(baseline):
        raise ValueError("MNI152 baseline is non-finite.")
    if transform == "percent":
        if baseline == 0:
            raise ValueError("Cannot calculate percentage difference from a zero baseline.")
        return 100.0 * (values.astype(float) - baseline) / baseline
    if transform == "percentage_points":
        return values.astype(float) - baseline
    raise ValueError(f"Unknown transform: {transform}")


def load_and_validate(
    cohort_table: Path,
    mni_table: Path,
    analysis_manifest: Path,
    selection_manifest: Path,
    selection_audit: Path,
    optimized_seven: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str], dict]:
    for path in (
        cohort_table,
        mni_table,
        analysis_manifest,
        selection_manifest,
        selection_audit,
        optimized_seven,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    manifest = json.loads(analysis_manifest.read_text(encoding="utf-8"))
    manifest_expectations = {
        "analysis_schema_version": 4,
        "status": "complete",
        "subjects": EXPECTED_SUBJECTS,
        "repeats_per_subject_roi": EXPECTED_REPEATS,
        "subject_level_records": EXPECTED_SUBJECTS * len(TARGET_ORDER),
        "mni_baselines": len(TARGET_ORDER),
    }
    failures = {
        key: (manifest.get(key), expected)
        for key, expected in manifest_expectations.items()
        if manifest.get(key) != expected
    }
    if failures:
        raise ValueError(f"Cohort analysis manifest failed validation: {failures}")
    if tuple(manifest.get("rois", [])) != TARGET_ORDER:
        raise ValueError("Cohort manifest ROI order/scope does not match the four study targets.")

    cohort = pd.read_csv(cohort_table)
    baseline = pd.read_csv(mni_table)
    selected_manifest = pd.read_csv(selection_manifest)
    audit = pd.read_csv(selection_audit)
    optimized_ids = set(read_ids(optimized_seven, EXPECTED_OPTIMIZED))

    metric_columns = [metric.column for metric in METRICS]
    require_columns(
        cohort,
        ["subject", "roi", "repeat_count", *metric_columns],
        "cohort metric table",
    )
    require_columns(baseline, ["subject", "roi", *metric_columns], "MNI152 baseline table")
    require_columns(
        selected_manifest,
        ["subject", "age_years_precise", "recorded_sex", "age_band_index", "age_band_years"],
        "selection manifest",
    )
    require_columns(
        audit,
        [
            "subject",
            "age_years_precise",
            "recorded_sex",
            "held_out_current_optimized_7",
            "selected_new_14",
            "age_band_index",
            "age_band_years",
        ],
        "selection audit",
    )

    if len(cohort) != EXPECTED_SUBJECTS * len(TARGET_ORDER):
        raise ValueError(f"Expected 528 cohort rows, found {len(cohort)}.")
    if cohort["subject"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError("Cohort metric table does not contain exactly 132 unique subjects.")
    if set(cohort["roi"]) != set(TARGET_ORDER):
        raise ValueError("Cohort metric table does not contain the four expected ROIs.")
    if not cohort["repeat_count"].eq(EXPECTED_REPEATS).all():
        raise ValueError("Every cohort subject/ROI row must aggregate exactly 10 repeats.")
    if cohort.duplicated(["subject", "roi"]).any():
        raise ValueError("Cohort metric table contains duplicate subject/ROI rows.")
    if cohort[metric_columns].isna().any().any():
        raise ValueError("Cohort metric table contains missing analysis values.")

    if len(baseline) != len(TARGET_ORDER) or baseline["roi"].nunique() != len(TARGET_ORDER):
        raise ValueError("MNI152 table must contain one row for each of four targets.")
    if set(baseline["subject"]) != {"MNI152"} or set(baseline["roi"]) != set(TARGET_ORDER):
        raise ValueError("MNI152 table has an unexpected subject or ROI scope.")
    if baseline[metric_columns].isna().any().any():
        raise ValueError("MNI152 baseline table contains missing analysis values.")

    if len(selected_manifest) != EXPECTED_SELECTED or selected_manifest["subject"].nunique() != EXPECTED_SELECTED:
        raise ValueError("Selection manifest must contain exactly 14 unique subjects.")
    if len(audit) != EXPECTED_SUBJECTS or audit["subject"].nunique() != EXPECTED_SUBJECTS:
        raise ValueError("Selection audit must contain exactly 132 unique subjects.")
    if set(audit["recorded_sex"]) != set(SEX_ORDER):
        raise ValueError("Selection audit has unexpected recorded-sex values.")
    if set(cohort["subject"]) != set(audit["subject"]):
        raise ValueError("Cohort metrics and selection audit do not contain the same 132 subjects.")
    selected_ids = set(selected_manifest["subject"])
    if selected_ids & optimized_ids:
        raise ValueError("Selected 14 overlap the held-out optimized seven.")
    if set(audit.loc[audit["selected_new_14"].eq("YES"), "subject"]) != selected_ids:
        raise ValueError("Selection manifest and selection audit disagree on selected subjects.")
    if set(audit.loc[audit["held_out_current_optimized_7"].eq("YES"), "subject"]) != optimized_ids:
        raise ValueError("Selection audit and optimized-seven list disagree.")
    demographic_check = selected_manifest.merge(
        audit[["subject", "age_years_precise", "recorded_sex"]],
        on="subject",
        suffixes=("_selection", "_audit"),
        validate="one_to_one",
    )
    if not np.allclose(
        demographic_check["age_years_precise_selection"],
        demographic_check["age_years_precise_audit"],
    ) or not demographic_check["recorded_sex_selection"].eq(
        demographic_check["recorded_sex_audit"]
    ).all():
        raise ValueError("Selection manifest and audit demographics disagree.")

    return cohort, baseline, audit, optimized_ids, manifest


def build_analysis_tables(
    cohort: pd.DataFrame,
    baseline: pd.DataFrame,
    audit: pd.DataFrame,
    optimized_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demographics = audit[
        ["subject", "age_years_precise", "recorded_sex", "age_band_index", "age_band_years"]
    ].copy()
    demographics["group"] = "eligible_unselected"
    demographics.loc[demographics["subject"].isin(optimized_ids), "group"] = "prior_optimized_7"
    demographics.loc[audit["selected_new_14"].eq("YES"), "group"] = "selected_14"
    merged = cohort.merge(demographics, on="subject", validate="many_to_one")

    selected_profile_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    residual_frames: list[pd.DataFrame] = []

    for metric in METRICS:
        baseline_values = baseline.set_index("roi")[metric.column].to_dict()
        relative_column = f"{metric.column}_mni_relative"
        merged[relative_column] = np.nan
        for target in TARGET_ORDER:
            target_mask = merged["roi"].eq(target)
            merged.loc[target_mask, relative_column] = transform_from_mni(
                merged.loc[target_mask, metric.column],
                float(baseline_values[target]),
                metric.transform,
            )

            reference = merged.loc[
                target_mask & merged["group"].eq("eligible_unselected")
            ].copy()
            selected = merged.loc[target_mask & merged["group"].eq("selected_14")].copy()
            if len(reference) != EXPECTED_SUBJECTS - EXPECTED_SELECTED - EXPECTED_OPTIMIZED:
                raise ValueError(f"Unexpected reference count for {target}: {len(reference)}")
            if len(selected) != EXPECTED_SELECTED:
                raise ValueError(f"Unexpected selected count for {target}: {len(selected)}")

            reference_adjusted, selected_adjusted, diagnostics = fit_adjusted_diagnostics(
                reference, selected, relative_column
            )
            reference_adjusted["metric"] = metric.column
            reference_adjusted["target"] = target
            reference_adjusted["residual_group"] = "eligible_unselected"
            selected_adjusted["metric"] = metric.column
            selected_adjusted["target"] = target
            selected_adjusted["residual_group"] = "selected_14"
            residual_frames.extend([reference_adjusted, selected_adjusted])

            eligible_strata = merged.loc[
                target_mask & ~merged["group"].eq("prior_optimized_7")
            ]
            stratum_sizes: list[int] = []
            stratum_percentiles: list[float] = []
            for _, row in selected_adjusted.iterrows():
                stratum = eligible_strata.loc[
                    eligible_strata["age_band_index"].eq(row["age_band_index"])
                    & eligible_strata["recorded_sex"].eq(row["recorded_sex"]),
                    relative_column,
                ].to_numpy(dtype=float)
                stratum_sizes.append(len(stratum))
                stratum_percentiles.append(
                    empirical_percentile(stratum, float(row[relative_column]))
                )
            selected_adjusted["selection_stratum_n"] = stratum_sizes
            selected_adjusted["selection_stratum_percentile"] = stratum_percentiles
            selected_adjusted["metric_label"] = metric.label
            selected_adjusted["relative_value_label"] = metric.relative_label
            selected_adjusted["mni_baseline"] = float(baseline_values[target])
            selected_adjusted["mni_relative_value"] = selected_adjusted[relative_column]
            selected_profile_frames.append(
                selected_adjusted[
                    [
                        "subject",
                        "age_years_precise",
                        "recorded_sex",
                        "age_band_index",
                        "age_band_years",
                        "roi",
                        "metric",
                        "metric_label",
                        metric.column,
                        "mni_baseline",
                        "mni_relative_value",
                        "relative_value_label",
                        "selection_stratum_n",
                        "selection_stratum_percentile",
                        "adjusted_residual",
                        "adjusted_residual_z",
                        "adjusted_reference_percentile",
                    ]
                ].rename(columns={metric.column: "participant_value"})
            )

            selected_values = selected_adjusted[relative_column].to_numpy(dtype=float)
            reference_values = reference_adjusted[relative_column].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "relative_value_label": metric.relative_label,
                    "selected_n": len(selected_values),
                    "reference_n": len(reference_values),
                    "selected_mean_mni_relative": float(np.mean(selected_values)),
                    "selected_median_mni_relative": float(np.median(selected_values)),
                    "selected_min_mni_relative": float(np.min(selected_values)),
                    "selected_max_mni_relative": float(np.max(selected_values)),
                    "reference_mean_mni_relative": float(np.mean(reference_values)),
                    "reference_median_mni_relative": float(np.median(reference_values)),
                    "reference_min_mni_relative": float(np.min(reference_values)),
                    "reference_max_mni_relative": float(np.max(reference_values)),
                    **diagnostics,
                    "selected_stratum_percentile_median": float(np.median(stratum_percentiles)),
                    "selected_stratum_percentile_min": float(np.min(stratum_percentiles)),
                    "selected_stratum_percentile_max": float(np.max(stratum_percentiles)),
                }
            )

    selected_profiles = pd.concat(selected_profile_frames, ignore_index=True)
    summaries = pd.DataFrame(summary_rows)
    residuals = pd.concat(residual_frames, ignore_index=True)
    multivariate = build_multivariate_profile(residuals)
    return merged, selected_profiles, summaries, multivariate


def build_multivariate_profile(residuals: pd.DataFrame) -> pd.DataFrame:
    primary = residuals.loc[residuals["metric"].eq(PRIMARY_METRIC.column)].copy()
    pivot = primary.pivot(
        index=["subject", "residual_group"],
        columns="target",
        values="adjusted_residual_z",
    ).reset_index()
    missing_targets = sorted(set(TARGET_ORDER) - set(pivot.columns))
    if missing_targets or pivot[list(TARGET_ORDER)].isna().any().any():
        raise ValueError(f"Incomplete multivariate field profiles: {missing_targets}")
    reference = pivot.loc[pivot["residual_group"].eq("eligible_unselected")]
    reference_matrix = reference[list(TARGET_ORDER)].to_numpy(dtype=float)
    center = reference_matrix.mean(axis=0)
    covariance_inverse = np.linalg.pinv(np.cov(reference_matrix, rowvar=False, ddof=1))

    def distances(frame: pd.DataFrame) -> np.ndarray:
        delta = frame[list(TARGET_ORDER)].to_numpy(dtype=float) - center
        return np.sqrt(np.einsum("ij,jk,ik->i", delta, covariance_inverse, delta))

    pivot["mahalanobis_distance"] = distances(pivot)
    reference_distances = pivot.loc[
        pivot["residual_group"].eq("eligible_unselected"), "mahalanobis_distance"
    ].to_numpy(dtype=float)
    pivot["reference_distance_percentile"] = [
        empirical_percentile(reference_distances, value)
        for value in pivot["mahalanobis_distance"].to_numpy(dtype=float)
    ]
    return pivot.sort_values(["residual_group", "reference_distance_percentile"]).reset_index(
        drop=True
    )


def plot_primary_representativeness(
    merged: pd.DataFrame,
    summaries: pd.DataFrame,
    output_dir: Path,
) -> None:
    relative_column = f"{PRIMARY_METRIC.column}_mni_relative"
    primary_summary = summaries.loc[summaries["metric"].eq(PRIMARY_METRIC.column)].set_index(
        "target"
    )
    values = merged[relative_column].to_numpy(dtype=float)
    lower = math.floor((np.min(values) - 5.0) / 10.0) * 10.0
    upper = math.ceil((np.max(values) + 5.0) / 10.0) * 10.0
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), sharex=True, sharey=True)
    for panel_index, (target, ax) in enumerate(zip(TARGET_ORDER, axes.flat)):
        target_data = merged.loc[merged["roi"].eq(target)].copy()
        ax.scatter(
            target_data["age_years_precise"],
            target_data[relative_column],
            s=17,
            color="#B8BDC5",
            alpha=0.58,
            edgecolors="none",
            zorder=1,
        )
        prior = target_data.loc[target_data["group"].eq("prior_optimized_7")]
        ax.scatter(
            prior["age_years_precise"],
            prior[relative_column],
            s=42,
            marker="x",
            color="#222222",
            linewidths=1.2,
            zorder=3,
        )
        for sex in SEX_ORDER:
            selected = target_data.loc[
                target_data["group"].eq("selected_14")
                & target_data["recorded_sex"].eq(sex)
            ]
            ax.scatter(
                selected["age_years_precise"],
                selected[relative_column],
                s=58,
                marker=SEX_MARKERS[sex],
                facecolor=SEX_COLORS[sex],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        ax.axhline(0, color="#4C566A", linewidth=0.9, linestyle="--", zorder=0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, zorder=0)
        ax.set_title(TARGET_LABELS[target], fontweight="bold")
        ax.set_ylim(lower, upper)
        ax.set_xlim(16, 88)
        row = primary_summary.loc[target]
        ax.text(
            0.03,
            0.96,
            f"Adjusted SMD = {row['adjusted_smd']:+.2f}\n"
            f"Selected span: {row['selected_span_reference_percent']:.0f}% of reference",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.88, "edgecolor": "#D1D5DB"},
        )
        ax.text(
            0.00,
            1.045,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Age (years)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean target-ROI field difference from MNI152 (%)")

    legend_handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5, color="#B8BDC5", label="Final-132"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SEX_COLORS["FEMALE"], markeredgecolor="white", label="Selected female"),
        plt.Line2D([], [], linestyle="none", marker="s", markersize=7, markerfacecolor=SEX_COLORS["MALE"], markeredgecolor="white", label="Selected male"),
        plt.Line2D([], [], linestyle="none", marker="x", markersize=7, color="#222222", label="Prior optimized 7"),
        plt.Line2D([], [], linestyle="--", linewidth=1.0, color="#4C566A", label="MNI152 baseline"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Generic-montage target fields in the selected 14 and final-132 cohort",
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "Participant values are ten-repeat means using optimizer-matched ROIs. Selection did not use field outcomes",
        ha="center",
        va="top",
        fontsize=10,
        color="#374151",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.105, hspace=0.25, wspace=0.12)
    fig.savefig(output_dir / "mni_field_representativeness.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir / "mni_field_representativeness.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_threshold_coverage_representativeness(
    merged: pd.DataFrame,
    summaries: pd.DataFrame,
    profiles: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot target coverage at 0.20 V/m as a separate supporting figure."""

    metric = next(
        item for item in METRICS if item.column == "target_coverage_percent_ge_0p2"
    )
    relative_column = f"{metric.column}_mni_relative"
    metric_summary = summaries.loc[summaries["metric"].eq(metric.column)].set_index(
        "target"
    )
    baseline_by_target = (
        profiles.loc[profiles["metric"].eq(metric.column)]
        .groupby("roi", observed=True)["mni_baseline"]
        .first()
        .to_dict()
    )
    if set(baseline_by_target) != set(TARGET_ORDER):
        raise ValueError("Could not recover one target-coverage MNI152 baseline per target.")

    values = merged[relative_column].to_numpy(dtype=float)
    lower = math.floor((np.min(values) - 5.0) / 10.0) * 10.0
    upper = math.ceil((np.max(values) + 5.0) / 10.0) * 10.0
    annotation_positions = {
        "Left_Hippocampus": (0.03, 0.96, "left", "top"),
        "Left_M1": (0.97, 0.96, "right", "top"),
        "Right_DLPC": (0.97, 0.96, "right", "top"),
        "Right_Thalamus": (0.97, 0.04, "right", "bottom"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), sharex=True, sharey=True)
    for panel_index, (target, ax) in enumerate(zip(TARGET_ORDER, axes.flat)):
        target_data = merged.loc[merged["roi"].eq(target)].copy()
        ax.scatter(
            target_data["age_years_precise"],
            target_data[relative_column],
            s=17,
            color="#B8BDC5",
            alpha=0.58,
            edgecolors="none",
            zorder=1,
        )
        prior = target_data.loc[target_data["group"].eq("prior_optimized_7")]
        ax.scatter(
            prior["age_years_precise"],
            prior[relative_column],
            s=42,
            marker="x",
            color="#222222",
            linewidths=1.2,
            zorder=3,
        )
        for sex in SEX_ORDER:
            selected = target_data.loc[
                target_data["group"].eq("selected_14")
                & target_data["recorded_sex"].eq(sex)
            ]
            ax.scatter(
                selected["age_years_precise"],
                selected[relative_column],
                s=58,
                marker=SEX_MARKERS[sex],
                facecolor=SEX_COLORS[sex],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        ax.axhline(0, color="#4C566A", linewidth=0.9, linestyle="--", zorder=0)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, zorder=0)
        ax.set_title(TARGET_LABELS[target], fontweight="bold")
        ax.set_ylim(lower, upper)
        ax.set_xlim(16, 88)
        row = metric_summary.loc[target]
        annotation_x, annotation_y, annotation_ha, annotation_va = annotation_positions[
            target
        ]
        ax.text(
            annotation_x,
            annotation_y,
            f"MNI152 coverage = {baseline_by_target[target]:.1f}%\n"
            f"Adjusted SMD = {row['adjusted_smd']:+.2f}\n"
            f"Selected span: {row['selected_span_reference_percent']:.0f}% of reference",
            transform=ax.transAxes,
            ha=annotation_ha,
            va=annotation_va,
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "#D1D5DB",
            },
        )
        ax.text(
            0.00,
            1.045,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Age (years)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Target coverage minus MNI152\n(percentage points)")

    legend_handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5, color="#B8BDC5", label="Final-132"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SEX_COLORS["FEMALE"], markeredgecolor="white", label="Selected female"),
        plt.Line2D([], [], linestyle="none", marker="s", markersize=7, markerfacecolor=SEX_COLORS["MALE"], markeredgecolor="white", label="Selected male"),
        plt.Line2D([], [], linestyle="none", marker="x", markersize=7, color="#222222", label="Prior optimized 7"),
        plt.Line2D([], [], linestyle="--", linewidth=1.0, color="#4C566A", label="MNI152 baseline"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Target coverage at 0.20 V/m in the selected 14 and final-132 cohort",
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "Percentage of the optimizer-matched target ROI at or above 0.20 V/m, shown relative to the target-specific MNI152 value",
        ha="center",
        va="top",
        fontsize=10,
        color="#374151",
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.88,
        bottom=0.105,
        hspace=0.25,
        wspace=0.12,
    )
    fig.savefig(
        output_dir / "mni_target_coverage_representativeness.png",
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "mni_target_coverage_representativeness.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def fmt_range(values: pd.Series, digits: int = 2) -> str:
    return f"{values.min():.{digits}f} to {values.max():.{digits}f}"


def write_report(
    output_dir: Path,
    summaries: pd.DataFrame,
    profiles: pd.DataFrame,
    multivariate: pd.DataFrame,
) -> None:
    primary = summaries.loc[summaries["metric"].eq(PRIMARY_METRIC.column)].copy()
    supporting = summaries.loc[~summaries["metric"].eq(PRIMARY_METRIC.column)].copy()
    selected_multi = multivariate.loc[multivariate["residual_group"].eq("selected_14")]
    adjusted_smd_range = fmt_range(primary["adjusted_smd"])
    coverage_range = fmt_range(primary["selected_span_reference_percent"], 1)
    max_abs_z = float(primary["selected_max_abs_adjusted_z"].max())
    count_abs_z_gt_2 = int(primary["selected_abs_adjusted_z_gt_2_count"].sum())
    multivariate_range = fmt_range(selected_multi["reference_distance_percentile"], 1)
    supporting_abs_smd_max_index = supporting["adjusted_smd"].abs().idxmax()
    supporting_abs_smd_max = supporting.loc[supporting_abs_smd_max_index]
    supporting_coverage_range = fmt_range(
        supporting["selected_span_reference_percent"], 1
    )

    target_lines = []
    for target in TARGET_ORDER:
        row = primary.loc[primary["target"].eq(target)].iloc[0]
        target_lines.append(
            f"- {TARGET_LABELS[target]}: selected MNI-relative mean-field range "
            f"{row['selected_min_mni_relative']:.1f}% to {row['selected_max_mni_relative']:.1f}%, "
            f"age/sex-adjusted SMD {row['adjusted_smd']:+.2f}, and selected residual span "
            f"covering {row['selected_span_reference_percent']:.1f}% of the 111-person reference distribution."
        )

    mean_field_caption = """**Generic-montage target fields in the selected 14 and final-132 cohort.** Each
point shows the ten-repeat mean target-ROI field for the MNI152-derived generic
montage on one participant head, expressed as a percentage difference from the
corresponding MNI152-head result. The selected age-balanced participants are
highlighted by recorded sex, and the seven previously optimized participants
are marked separately because they were selected using field outcomes. SMD is
the difference in mean age/recorded-sex-adjusted residual between the selected
14 and the 111-person reference group, divided by the reference residual
standard deviation. Selected span is the percentage of reference residuals
falling between the minimum and maximum selected residuals. The horizontal
dashed line denotes the MNI152 value."""

    threshold_caption = """**Target coverage at 0.20 V/m in the selected 14 and final-132 cohort.** Each
point shows the percentage of the optimizer-matched target ROI with a simulated
field at or above 0.20 V/m, averaged across ten remeshing repeats and expressed
as a percentage-point difference from the corresponding MNI152-head result.
The selected age-balanced participants are highlighted by recorded sex, and
the seven previously optimized participants are marked separately. SMD and
selected span use the same age/recorded-sex-adjusted 111-person reference as
the mean-field figure. The dashed line denotes the MNI152 coverage. The 0.20
V/m cutoff is an analysis convention, not a validated therapeutic threshold."""

    report = f"""# MNI-based field representativeness of the selected 14

## Bottom line

The 14 participants were selected using age and recorded sex, without using an
electric-field outcome. Their existing MNI152-derived generic-montage results
were then compared with both the corresponding MNI152 template value and the
field distribution in the final-132 cohort.

Across the four targets, the age/recorded-sex-adjusted standardized mean
differences for the primary target-ROI mean field ranged from {adjusted_smd_range}.
The observed ranges of the selected participants covered {coverage_range}% of
the adjusted residual distributions in the 111 eligible participants who were
neither selected nor part of the outcome-selected optimized seven. Across all
56 selected participant-target records, {count_abs_z_gt_2} had an absolute
adjusted residual above 2 reference standard deviations, and the maximum was
{max_abs_z:.2f}. In the four-target multivariate field profile, selected-subject
distance percentiles ranged from {multivariate_range}% relative to the same
reference group.

For the supporting median-field and 0.20 V/m coverage outcomes, selected
residual ranges covered {supporting_coverage_range}% of the reference
distributions. The largest supporting standardized mean difference was
{supporting_abs_smd_max['adjusted_smd']:+.2f} for
{supporting_abs_smd_max['metric_label'].lower()} in the
{TARGET_LABELS[supporting_abs_smd_max['target']].lower()}. This threshold-based
outcome was more heterogeneous than the continuous mean-field outcome and is
therefore retained explicitly in the supporting tables rather than being used
to redefine the primary cohort.

These diagnostics characterize how the selected participants sit within the
available generic-montage field variation. They do not establish formal
equivalence, and the final-132 cohort itself is a QC-selected modelling cohort
rather than a population-random sample.

## Primary target-field results

{chr(10).join(target_lines)}

## Methods

- Source: complete schema-4 optimizer-matched analysis of 132 participants,
  four targets, and ten independently remeshed simulations per participant and
  target.
- Generic configuration: the MNI152-derived montage was simulated on every
  participant head. The reference value was the same montage simulated on the
  MNI152 head model.
- Primary quantity: percentage difference in the ten-repeat mean target-ROI
  field from the target-specific MNI152 value.
- Supporting quantities: target-ROI median field, target coverage at or above
  0.20 V/m, and off-target coverage at or above 0.20 V/m. Coverage differences
  are expressed in percentage points.
- ROI boundary: the same parcel-clipped spherical construction logic was used
  for participant and MNI152 analyses. The voxelized realizations are aligned
  definitions, not identical masks.
- Reference group: 111 final-132 participants after excluding the selected 14
  and the seven previously optimized outcome-selected participants.
- Adjustment: for each target and metric, the reference data were modelled
  using continuous age, recorded sex, and their interaction. Reference
  prediction errors were leave-one-out PRESS residuals. Selected participants
  were predicted from the model fitted only to the 111-person reference group.
- The standardized mean difference, residual range coverage, two-sample
  Kolmogorov-Smirnov statistic, within-selection-stratum percentile, and
  four-target Mahalanobis distance are descriptive diagnostics. No arbitrary
  pass/fail threshold was imposed.

## Figure captions

### Mean-field figure

{mean_field_caption}

### Target-coverage figure

{threshold_caption}

## Suggested concise email statement

Before proceeding, I assessed the 14 participants against the complete
final-132 generic-montage distribution using the optimizer-matched target ROIs.
Their mean target fields were close to the eligible reference distribution
after accounting for age and recorded sex, and the selected ranges covered
75-95% of its residual variation across the four targets (Figure attached).
Threshold-based coverage was more heterogeneous, particularly for right DLPFC,
so I would retain the outcome-independent cohort and include baseline field
position as a prespecified covariate or sensitivity analysis rather than
reselecting participants on the outcome.
"""
    (output_dir / "REPRESENTATIVENESS_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "figure_caption.txt").write_text(
        mean_field_caption.replace("**", "").strip() + "\n",
        encoding="utf-8",
    )
    (output_dir / "target_coverage_figure_caption.txt").write_text(
        threshold_caption.replace("**", "").strip() + "\n",
        encoding="utf-8",
    )


def write_manifest(
    output_dir: Path,
    inputs: dict[str, Path],
    source_manifest: dict,
    summaries: pd.DataFrame,
    multivariate: pd.DataFrame,
) -> None:
    primary = summaries.loc[summaries["metric"].eq(PRIMARY_METRIC.column)]
    selected_multi = multivariate.loc[multivariate["residual_group"].eq("selected_14")]
    manifest = {
        "schema_version": 1,
        "analysis": "mni_field_representativeness_of_age_balanced_14",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "interpretation": "descriptive_representativeness_not_equivalence",
        "source_analysis": {
            "analysis_schema_version": source_manifest["analysis_schema_version"],
            "subjects": source_manifest["subjects"],
            "rois": source_manifest["rois"],
            "repeats_per_subject_roi": source_manifest["repeats_per_subject_roi"],
            "primary_roi_definition": source_manifest["primary_roi_definition"],
        },
        "validated_counts": {
            "final_132_subjects": EXPECTED_SUBJECTS,
            "selected_subjects": EXPECTED_SELECTED,
            "prior_optimized_subjects": EXPECTED_OPTIMIZED,
            "eligible_unselected_reference_subjects": EXPECTED_SUBJECTS
            - EXPECTED_SELECTED
            - EXPECTED_OPTIMIZED,
            "targets": len(TARGET_ORDER),
            "repeats_per_subject_target": EXPECTED_REPEATS,
        },
        "analysis_choices": {
            "primary_metric": PRIMARY_METRIC.column,
            "supporting_metrics": [metric.column for metric in METRICS[1:]],
            "field_transform": "100 * (participant - MNI152) / MNI152",
            "coverage_transform": "participant percentage - MNI152 percentage",
            "adjustment_model": "continuous age + recorded sex + age by recorded sex",
            "reference_residuals": "leave-one-out PRESS residuals",
            "selected_residuals": "predicted by model fitted to 111-person reference only",
            "decision_rule": "descriptive diagnostics; no pass/fail or equivalence threshold",
        },
        "primary_summary": {
            "adjusted_smd_min": float(primary["adjusted_smd"].min()),
            "adjusted_smd_max": float(primary["adjusted_smd"].max()),
            "selected_span_reference_percent_min": float(
                primary["selected_span_reference_percent"].min()
            ),
            "selected_span_reference_percent_max": float(
                primary["selected_span_reference_percent"].max()
            ),
            "selected_max_abs_adjusted_z": float(
                primary["selected_max_abs_adjusted_z"].max()
            ),
            "selected_multivariate_distance_percentile_min": float(
                selected_multi["reference_distance_percentile"].min()
            ),
            "selected_multivariate_distance_percentile_max": float(
                selected_multi["reference_distance_percentile"].max()
            ),
        },
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in inputs.items()
        },
        "outputs": [
            "mni_field_representativeness.png",
            "mni_field_representativeness.pdf",
            "mni_target_coverage_representativeness.png",
            "mni_target_coverage_representativeness.pdf",
            "representativeness_summary.csv",
            "selected_14_mni_field_profiles.csv",
            "multivariate_field_profiles.csv",
            "REPRESENTATIVENESS_REPORT.md",
            "figure_caption.txt",
            "target_coverage_figure_caption.txt",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def write_checksums(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files)
    (output_dir / "SHA256SUMS").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess whether the selected age/recorded-sex-balanced 14-subject "
            "cohort spans the final-132 MNI-relative generic-montage field distribution."
        )
    )
    parser.add_argument("--cohort-table", type=Path, default=DEFAULT_COHORT_TABLE)
    parser.add_argument("--mni-table", type=Path, default=DEFAULT_MNI_TABLE)
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS_MANIFEST)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--selection-audit", type=Path, default=DEFAULT_SELECTION_AUDIT)
    parser.add_argument("--optimized-seven", type=Path, default=DEFAULT_OPTIMIZED_SEVEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory after all inputs validate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "cohort_table": args.cohort_table,
        "mni_table": args.mni_table,
        "analysis_manifest": args.analysis_manifest,
        "selection_manifest": args.selection_manifest,
        "selection_audit": args.selection_audit,
        "optimized_seven": args.optimized_seven,
    }
    cohort, baseline, audit, optimized_ids, source_manifest = load_and_validate(
        args.cohort_table,
        args.mni_table,
        args.analysis_manifest,
        args.selection_manifest,
        args.selection_audit,
        args.optimized_seven,
    )
    merged, profiles, summaries, multivariate = build_analysis_tables(
        cohort, baseline, audit, optimized_ids
    )

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. Use --force to replace it."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    summaries.to_csv(args.output_dir / "representativeness_summary.csv", index=False)
    profiles.to_csv(args.output_dir / "selected_14_mni_field_profiles.csv", index=False)
    multivariate.to_csv(args.output_dir / "multivariate_field_profiles.csv", index=False)
    plot_primary_representativeness(merged, summaries, args.output_dir)
    plot_threshold_coverage_representativeness(
        merged, summaries, profiles, args.output_dir
    )
    write_report(args.output_dir, summaries, profiles, multivariate)
    write_manifest(args.output_dir, inputs, source_manifest, summaries, multivariate)
    write_checksums(args.output_dir)

    primary = summaries.loc[summaries["metric"].eq(PRIMARY_METRIC.column)]
    print(f"Wrote representativeness package: {args.output_dir}")
    print(
        "Primary adjusted SMD range: "
        f"{primary['adjusted_smd'].min():+.3f} to {primary['adjusted_smd'].max():+.3f}"
    )
    print(
        "Selected residual-span coverage: "
        f"{primary['selected_span_reference_percent'].min():.1f}% to "
        f"{primary['selected_span_reference_percent'].max():.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
