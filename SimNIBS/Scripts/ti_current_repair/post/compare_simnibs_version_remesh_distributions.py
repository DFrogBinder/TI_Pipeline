#!/usr/bin/env python3
"""Compare SimNIBS v3.2.6 and v4 remesh-repeat ROI-field distributions.

This is an official, read-only post-processing stage. It consumes the four
canonical schema-v2 optimizer-ROI metric tables, selects the 40-run remesh
condition, locates each v3 value in its matched v4 distribution, and performs
participant-level paired inference. Technical repeats are summarized within
participant and are never treated as independent inferential observations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = {
    "left_hippocampus": {
        "label": "Left hippocampus",
        "roi": "Left_Hippocampus",
        "requested_volume_mm3": 200.0,
    },
    "right_m1": {
        "label": "Right M1",
        "roi": "Right_M1",
        "requested_volume_mm3": 100.0,
    },
}

METRICS = (
    "roi_median_v_per_m",
    "roi_mean_v_per_m",
    "roi_p95_v_per_m",
)
PRIMARY_METRIC = "roi_median_v_per_m"
EXPECTED_TAGS = {f"repeat_{index:03d}" for index in range(1, 41)}
REQUIRED_COLUMNS = {
    "schema_version",
    "subject",
    "condition",
    "repeat_tag",
    "roi",
    "roi_voxels",
    "requested_roi_volume_mm3",
    "achieved_roi_volume_mm3",
    "roi_radius_mm",
    "finite_roi_voxels",
    "nonfinite_roi_voxels",
    "finite_roi_fraction",
    *METRICS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_midrank_percentile(value: float, reference: np.ndarray) -> float:
    """Return an empirical percentile with ties assigned half weight."""

    reference = np.asarray(reference, dtype=float)
    if reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("reference must contain finite values")
    if not math.isfinite(float(value)):
        raise ValueError("value must be finite")
    less = int(np.count_nonzero(reference < value))
    equal = int(np.count_nonzero(reference == value))
    return 100.0 * (less + 0.5 * equal) / reference.size


def common_language_probability(v3: np.ndarray, v4: np.ndarray) -> float:
    """Return P(v3 > v4), assigning half weight to exact ties."""

    v3 = np.asarray(v3, dtype=float)
    v4 = np.asarray(v4, dtype=float)
    if v3.size == 0 or v4.size == 0:
        raise ValueError("both distributions must be non-empty")
    if not np.isfinite(v3).all() or not np.isfinite(v4).all():
        raise ValueError("both distributions must contain finite values")
    pairwise = v3[:, None] - v4[None, :]
    return float(np.mean(pairwise > 0) + 0.5 * np.mean(pairwise == 0))


def exact_paired_sign_flip(differences: np.ndarray) -> dict[str, float | int]:
    """Two-sided exact sign-flip test of the participant mean difference."""

    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array")
    if not np.isfinite(differences).all():
        raise ValueError("differences must be finite")
    n = int(differences.size)
    observed = abs(float(differences.mean()))
    permuted = np.fromiter(
        (
            abs(float(np.mean(differences * np.asarray(signs, dtype=float))))
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        ),
        dtype=float,
        count=2**n,
    )
    p_value = float(np.count_nonzero(permuted >= observed - 1e-15) / len(permuted))
    return {
        "participants": n,
        "permutations": int(len(permuted)),
        "observed_mean_difference_v_per_m": float(differences.mean()),
        "two_sided_p": p_value,
    }


def _validate_table(frame: pd.DataFrame, target: str, version: str) -> None:
    label = f"{version} {target}"
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{label}: missing columns {sorted(missing)}")
    if len(frame) != 800:
        raise RuntimeError(f"{label}: expected 800 rows, observed {len(frame)}")
    if set(frame["schema_version"].astype(int)) != {2}:
        raise RuntimeError(f"{label}: expected schema version 2")
    meta = TARGETS[target]
    if set(frame["roi"].astype(str)) != {meta["roi"]}:
        raise RuntimeError(f"{label}: unexpected ROI values")
    if set(frame["condition"].astype(str)) != {"remesh", "fixed_mesh"}:
        raise RuntimeError(f"{label}: unexpected conditions")
    if frame["subject"].nunique() != 10:
        raise RuntimeError(f"{label}: expected ten subjects")
    requested = frame["requested_roi_volume_mm3"].to_numpy(dtype=float)
    if not np.allclose(requested, meta["requested_volume_mm3"]):
        raise RuntimeError(f"{label}: unexpected requested ROI volume")
    achieved = frame["achieved_roi_volume_mm3"].to_numpy(dtype=float)
    if not np.isfinite(achieved).all() or np.any(achieved + 1e-9 < requested):
        raise RuntimeError(f"{label}: invalid achieved ROI volume")
    keys = frame[["subject", "condition", "repeat_tag"]]
    if keys.duplicated().any():
        raise RuntimeError(f"{label}: duplicate subject-condition-repeat keys")
    counts = frame.groupby(["subject", "condition"], sort=False).size()
    if len(counts) != 20 or set(counts.astype(int)) != {40}:
        raise RuntimeError(f"{label}: each subject-condition must contain 40 runs")
    tags = frame.groupby(["subject", "condition"], sort=False)["repeat_tag"].agg(set)
    if not all(observed == EXPECTED_TAGS for observed in tags):
        raise RuntimeError(f"{label}: repeat tags do not cover repeat_001..repeat_040")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: non-finite scalar ROI metrics")
    if np.any(values < 0):
        raise RuntimeError(f"{label}: negative scalar ROI metrics")
    roi_voxels = frame["roi_voxels"].to_numpy(dtype=int)
    finite = frame["finite_roi_voxels"].to_numpy(dtype=int)
    nonfinite = frame["nonfinite_roi_voxels"].to_numpy(dtype=int)
    if np.any(roi_voxels <= 0) or np.any(finite <= 0) or np.any(nonfinite < 0):
        raise RuntimeError(f"{label}: invalid finite-support counts")
    if not np.array_equal(finite + nonfinite, roi_voxels):
        raise RuntimeError(f"{label}: inconsistent finite-support counts")
    if not np.allclose(
        frame["finite_roi_fraction"].to_numpy(dtype=float),
        finite / roi_voxels,
        atol=5e-4,
    ):
        raise RuntimeError(f"{label}: inconsistent finite-support fractions")
    support = frame.groupby("subject", sort=False).agg(
        roi_voxels=("roi_voxels", "nunique"),
        achieved_volumes=("achieved_roi_volume_mm3", "nunique"),
    )
    if not (support == 1).all().all():
        raise RuntimeError(f"{label}: ROI support changes within a subject")


def load_inputs(
    v3_left_hippocampus_csv: Path,
    v3_right_m1_csv: Path,
    v4_left_hippocampus_csv: Path,
    v4_right_m1_csv: Path,
) -> pd.DataFrame:
    """Load and validate the four immutable canonical input tables."""

    paths = {
        ("v3", "left_hippocampus"): Path(v3_left_hippocampus_csv),
        ("v3", "right_m1"): Path(v3_right_m1_csv),
        ("v4", "left_hippocampus"): Path(v4_left_hippocampus_csv),
        ("v4", "right_m1"): Path(v4_right_m1_csv),
    }
    frames: list[pd.DataFrame] = []
    subject_sets: dict[tuple[str, str], set[str]] = {}
    for (version, target), path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{version} {target}: input is missing or empty: {path}")
        frame = pd.read_csv(path)
        _validate_table(frame, target, version)
        subject_sets[(version, target)] = set(frame["subject"].astype(str))
        frame.insert(0, "target", target)
        frame.insert(0, "version", version)
        frames.append(frame)

    for target in TARGETS:
        if subject_sets[("v3", target)] != subject_sets[("v4", target)]:
            raise RuntimeError(f"{target}: v3 and v4 subject sets differ")
    if subject_sets[("v3", "left_hippocampus")] != subject_sets[("v3", "right_m1")]:
        raise RuntimeError("v3 target subject sets differ")
    if subject_sets[("v4", "left_hippocampus")] != subject_sets[("v4", "right_m1")]:
        raise RuntimeError("v4 target subject sets differ")

    combined = pd.concat(frames, ignore_index=True)
    remesh_keys: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for version in ("v3", "v4"):
        for target in TARGETS:
            selected = combined.query(
                "version == @version and target == @target and condition == 'remesh'"
            )
            remesh_keys[(version, target)] = set(
                zip(selected["subject"].astype(str), selected["repeat_tag"].astype(str))
            )
    for target in TARGETS:
        if remesh_keys[("v3", target)] != remesh_keys[("v4", target)]:
            raise RuntimeError(f"{target}: v3 and v4 remesh repeat keys differ")
    return combined


def _distribution_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_v_per_m": float(np.mean(values)),
        f"{prefix}_median_v_per_m": float(np.median(values)),
        f"{prefix}_sd_v_per_m": float(np.std(values, ddof=1)),
        f"{prefix}_q1_v_per_m": float(np.percentile(values, 25)),
        f"{prefix}_q3_v_per_m": float(np.percentile(values, 75)),
        f"{prefix}_min_v_per_m": float(np.min(values)),
        f"{prefix}_max_v_per_m": float(np.max(values)),
    }


def _location(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "below"
    if value > upper:
        return "above"
    return "within"


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust a finite vector of p-values while preserving row order."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p-values must be a finite one-dimensional array")
    order = np.argsort(p_values, kind="stable")
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def build_comparison_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build run-location, participant-summary, and inference tables."""

    remesh = frame.loc[frame["condition"] == "remesh"].copy()
    run_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        target_frame = remesh.loc[remesh["target"] == target]
        for subject in sorted(target_frame["subject"].astype(str).unique()):
            subject_frame = target_frame.loc[target_frame["subject"] == subject]
            for metric in METRICS:
                v3_frame = subject_frame.loc[subject_frame["version"] == "v3"].sort_values(
                    "repeat_tag"
                )
                v4_frame = subject_frame.loc[subject_frame["version"] == "v4"].sort_values(
                    "repeat_tag"
                )
                v3 = v3_frame[metric].to_numpy(dtype=float)
                v4 = v4_frame[metric].to_numpy(dtype=float)
                if len(v3) != 40 or len(v4) != 40:
                    raise RuntimeError(
                        f"{target} {subject} {metric}: expected 40 remesh runs per version"
                    )
                percentiles = np.asarray(
                    [empirical_midrank_percentile(value, v4) for value in v3],
                    dtype=float,
                )
                for repeat_tag, value, percentile in zip(
                    v3_frame["repeat_tag"].astype(str), v3, percentiles
                ):
                    run_rows.append(
                        {
                            "target": target,
                            "subject": subject,
                            "metric": metric,
                            "repeat_tag": repeat_tag,
                            "v3_value_v_per_m": float(value),
                            "percentile_in_v4": float(percentile),
                            "v4_reference_runs": 40,
                        }
                    )

                v4_lower, v4_upper = np.percentile(v4, [2.5, 97.5])
                v3_median = float(np.median(v3))
                row: dict[str, Any] = {
                    "target": target,
                    "subject": subject,
                    "metric": metric,
                    "runs_per_version": 40,
                    **_distribution_stats(v3, "v3"),
                    **_distribution_stats(v4, "v4"),
                    "v3_minus_v4_mean_v_per_m": float(np.mean(v3) - np.mean(v4)),
                    "v3_minus_v4_median_v_per_m": float(v3_median - np.median(v4)),
                    "v3_to_v4_median_ratio": (
                        float(v3_median / np.median(v4))
                        if float(np.median(v4)) != 0.0
                        else math.nan
                    ),
                    "probability_v3_greater_than_v4": common_language_probability(v3, v4),
                    "v3_median_percentile_in_v4": empirical_midrank_percentile(v3_median, v4),
                    "v3_run_percentile_median": float(np.median(percentiles)),
                    "v3_run_percentile_q1": float(np.percentile(percentiles, 25)),
                    "v3_run_percentile_q3": float(np.percentile(percentiles, 75)),
                    "v3_run_percentile_min": float(np.min(percentiles)),
                    "v3_run_percentile_max": float(np.max(percentiles)),
                    "v4_central_95_lower_v_per_m": float(v4_lower),
                    "v4_central_95_upper_v_per_m": float(v4_upper),
                    "v3_median_location_in_v4_observed_range": _location(
                        v3_median, float(np.min(v4)), float(np.max(v4))
                    ),
                    "v3_median_location_in_v4_central_95": _location(
                        v3_median, float(v4_lower), float(v4_upper)
                    ),
                    "v3_minimum_finite_roi_fraction": float(
                        v3_frame["finite_roi_fraction"].min()
                    ),
                    "v3_median_finite_roi_fraction": float(
                        v3_frame["finite_roi_fraction"].median()
                    ),
                    "v4_minimum_finite_roi_fraction": float(
                        v4_frame["finite_roi_fraction"].min()
                    ),
                    "v4_median_finite_roi_fraction": float(
                        v4_frame["finite_roi_fraction"].median()
                    ),
                }
                participant_rows.append(row)

    run_percentiles = pd.DataFrame(run_rows)
    participant_metrics = pd.DataFrame(participant_rows)
    primary = participant_metrics.loc[
        participant_metrics["metric"] == PRIMARY_METRIC
    ].reset_index(drop=True)

    inference_rows: list[dict[str, Any]] = []
    for (target, metric), group in participant_metrics.groupby(
        ["target", "metric"], sort=False
    ):
        differences = group["v3_minus_v4_median_v_per_m"].to_numpy(dtype=float)
        test = exact_paired_sign_flip(differences)
        sd = float(np.std(differences, ddof=1))
        inference_rows.append(
            {
                "target": target,
                "metric": metric,
                "participants": int(test["participants"]),
                "technical_repeats_per_participant_version": 40,
                "participant_summary": "median_across_remesh_repeats",
                "mean_paired_difference_v3_minus_v4_v_per_m": float(
                    np.mean(differences)
                ),
                "median_paired_difference_v3_minus_v4_v_per_m": float(
                    np.median(differences)
                ),
                "sd_paired_difference_v_per_m": sd,
                "cohen_dz": float(np.mean(differences) / sd) if sd > 0 else math.nan,
                "exact_two_sided_sign_flip_p": float(test["two_sided_p"]),
                "exact_sign_flip_permutations": int(test["permutations"]),
            }
        )
    participant_inference = pd.DataFrame(inference_rows)
    participant_inference["holm_adjusted_p_across_six_tests"] = _holm_adjust(
        participant_inference["exact_two_sided_sign_flip_p"].to_numpy(dtype=float)
    )
    return {
        "run_percentiles": run_percentiles,
        "participant_metrics": participant_metrics,
        "primary_summary": primary,
        "participant_inference": participant_inference,
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dee9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _plot_distribution_overlay(
    remesh: pd.DataFrame, target: str, out_dir: Path
) -> None:
    selected = remesh.loc[remesh["target"] == target]
    v4 = selected.loc[selected["version"] == "v4"]
    order = (
        v4.groupby("subject")[PRIMARY_METRIC]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    positions = np.arange(1, len(order) + 1, dtype=float)
    v4_values = [
        v4.loc[v4["subject"] == subject, PRIMARY_METRIC].to_numpy(dtype=float)
        for subject in order
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    box = ax.boxplot(
        v4_values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        tick_labels=order,
        medianprops={"color": "#0b132b", "linewidth": 1.5},
        boxprops={"facecolor": "#78a6d1", "edgecolor": "#315b7d", "alpha": 0.7},
        whiskerprops={"color": "#315b7d"},
        capprops={"color": "#315b7d"},
    )
    del box
    for position, subject in zip(positions, order):
        values = selected.loc[
            (selected["version"] == "v3") & (selected["subject"] == subject),
            PRIMARY_METRIC,
        ].to_numpy(dtype=float)
        jitter = np.random.default_rng(20261021 + int(position)).uniform(
            -0.16, 0.16, len(values)
        )
        ax.scatter(
            position + jitter,
            values,
            s=14,
            color="#d1495b",
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            position,
            np.median(values),
            s=42,
            marker="D",
            color="#8f1d2c",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
    _style_axis(ax)
    ax.set_title(f"{TARGETS[target]['label']}: v3 location within the v4 remesh distribution")
    ax.set_ylabel("Optimizer-ROI median |E| (V/m)")
    ax.set_xlabel("Participant (ordered by v4 median)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.plot([], [], color="#315b7d", linewidth=8, alpha=0.7, label="v4: 40-run box plot")
    ax.scatter([], [], color="#d1495b", s=20, label="v3: individual remesh run")
    ax.scatter([], [], color="#8f1d2c", marker="D", s=42, label="v3: 40-run median")
    ax.legend(frameon=False, fontsize=8, loc="best")
    stem = out_dir / f"{target}_v4_distribution_with_v3_overlay"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _plot_percentile_location(
    primary: pd.DataFrame, target: str, out_dir: Path
) -> None:
    selected = primary.loc[primary["target"] == target].sort_values(
        "v3_run_percentile_median"
    )
    y = np.arange(len(selected), dtype=float)
    medians = selected["v3_run_percentile_median"].to_numpy(dtype=float)
    q1 = selected["v3_run_percentile_q1"].to_numpy(dtype=float)
    q3 = selected["v3_run_percentile_q3"].to_numpy(dtype=float)
    minima = selected["v3_run_percentile_min"].to_numpy(dtype=float)
    maxima = selected["v3_run_percentile_max"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax.hlines(y, minima, maxima, color="#aab4c0", linewidth=1.2, label="range")
    ax.hlines(y, q1, q3, color="#315b7d", linewidth=5, label="interquartile range")
    ax.scatter(medians, y, color="#d1495b", s=40, zorder=3, label="median percentile")
    ax.axvline(50, color="#0b132b", linestyle="--", linewidth=1, label="v4 median")
    ax.set_yticks(y, selected["subject"].astype(str))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile of each v3 run in the matched 40-run v4 distribution")
    ax.set_ylabel("Participant")
    ax.set_title(f"{TARGETS[target]['label']}: empirical v3 percentile location")
    ax.grid(axis="x", color="#d8dee9", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    stem = out_dir / f"{target}_v3_percentile_location"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "NA" if not math.isfinite(float(value)) else f"{float(value):.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format_value(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *rows])


def _write_text_outputs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    inference = tables["participant_inference"].copy()
    report_columns = [
        "target",
        "metric",
        "participants",
        "mean_paired_difference_v3_minus_v4_v_per_m",
        "cohen_dz",
        "exact_two_sided_sign_flip_p",
        "holm_adjusted_p_across_six_tests",
    ]
    report = f"""# SimNIBS v3.2.6 versus v4 remesh-distribution comparison

## Scope and estimand

This read-only official post-processing analysis compares the 40 completed
`remesh` repetitions from SimNIBS v3.2.6 with the corresponding 40 repetitions
from SimNIBS v4 for ten participants and two targets. The primary outcome is
`{PRIMARY_METRIC}`; ROI mean and ROI p95 are prespecified sensitivity outcomes.

The 40 runs are technical replicates, not inferential units. Each participant's
version-specific 40-run median is the unit entering the paired analysis. The
reported two-sided p-values are exact participant-level sign-flip tests over
all 2^10 sign assignments; no run-level p-values are calculated. The median
ROI field is prespecified as primary, while the mean and p95 are sensitivity
outcomes. Holm-adjusted p-values across all six target-outcome tests are also
reported to make the multiplicity boundary explicit.

## Distribution-location summaries

For each v3 run, its empirical midrank percentile is calculated against the 40
matched v4 values. Exact ties receive half weight. `P(v3 > v4)` is the fraction
of all 40 x 40 cross-version pairs in which v3 is larger, again with half weight
for ties. These are descriptive distribution-location measures, not additional
independent tests.

## Participant-level paired inference

{_markdown_table(inference, report_columns)}

## ROI and finite-support boundary

Both versions use aligned optimizer-matched spherical ROI constructions with
the same anatomical target and requested physical volume. Version-specific
mesh or voxel realizations are not assumed to be identical masks. The tables
therefore retain version-specific finite-support summaries, and the analysis
fails if scalar metrics are non-finite or if support accounting is inconsistent.

This analysis uses continuous field values and does not treat 0.2 V/m as a
therapeutic or biological threshold.
"""
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")

    captions = """# Stand-alone figure captions

## SimNIBS-version remesh distributions

For each participant, the box plot shows the distribution of the primary
optimizer-ROI outcome (median electric-field magnitude, V/m) across 40 SimNIBS
v4 remesh repetitions. Red points show the 40 SimNIBS v3.2.6 remesh values and
the dark-red diamond shows their median. Participants are ordered by their v4
median. Repetitions are technical replicates; the participant is the
inferential unit. Left hippocampus and right M1 are shown separately.

## Empirical percentile location of v3 within v4

For each participant, every SimNIBS v3.2.6 remesh value was assigned an
empirical midrank percentile in the matched 40-value SimNIBS v4 distribution.
Points show the median of the 40 percentiles, thick lines the interquartile
range, and thin lines the full range. The dashed line marks the v4 median
(50th percentile). Exact ties receive half weight. Values are descriptive
within-participant distribution locations; repetitions are not independent
inferential observations.
"""
    (out_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "v3_left_hippocampus_csv": Path(args.v3_left_hippocampus_csv),
        "v3_right_m1_csv": Path(args.v3_right_m1_csv),
        "v4_left_hippocampus_csv": Path(args.v4_left_hippocampus_csv),
        "v4_right_m1_csv": Path(args.v4_right_m1_csv),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all inputs and return scope without writing any output."""

    frame = load_inputs(**_input_paths(args))
    remesh = frame.loc[frame["condition"] == "remesh"]
    return {
        "status": "ready",
        "input_rows": int(len(frame)),
        "selected_remesh_rows": int(len(remesh)),
        "versions": int(frame["version"].nunique()),
        "participants": int(frame["subject"].nunique()),
        "targets": int(frame["target"].nunique()),
        "remesh_repeats_per_participant_target_version": 40,
        "metrics": list(METRICS),
        "primary_metric": PRIMARY_METRIC,
        "source_policy": "read-only",
        "meshing_or_fem_tasks": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the validated comparison and write an auditable result package."""

    paths = _input_paths(args)
    frame = load_inputs(**paths)
    tables = build_comparison_tables(frame)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    table_files = {
        "run_percentiles": "v3_run_percentiles_in_v4.csv",
        "participant_metrics": "participant_metric_summary.csv",
        "primary_summary": "participant_target_primary_summary.csv",
        "participant_inference": "participant_level_inference.csv",
    }
    for key, filename in table_files.items():
        tables[key].to_csv(out_dir / filename, index=False)

    remesh = frame.loc[frame["condition"] == "remesh"]
    for target in TARGETS:
        _plot_distribution_overlay(remesh, target, out_dir)
        _plot_percentile_location(tables["primary_summary"], target, out_dir)
    _write_text_outputs(tables, out_dir)

    checksum_candidates = sorted(
        path for path in out_dir.iterdir() if path.name not in {"SHA256SUMS", "analysis_manifest.json"}
    )
    checksum_text = "".join(
        f"{_sha256_file(path)}  {path.name}\n" for path in checksum_candidates
    )
    (out_dir / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")

    artifact_paths = sorted(
        path for path in out_dir.iterdir() if path.name != "analysis_manifest.json"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "simnibs_version_remesh_distribution_comparison",
        "status": "complete",
        "input_rows": int(len(frame)),
        "selected_remesh_rows": int(len(remesh)),
        "run_percentile_rows": int(len(tables["run_percentiles"])),
        "participant_metric_rows": int(len(tables["participant_metrics"])),
        "participant_target_primary_rows": int(len(tables["primary_summary"])),
        "participant_inference_rows": int(len(tables["participant_inference"])),
        "participants": int(frame["subject"].nunique()),
        "targets": list(TARGETS),
        "versions": ["v3", "v4"],
        "condition": "remesh",
        "technical_repeats_per_participant_target_version": 40,
        "metrics": list(METRICS),
        "primary_metric": PRIMARY_METRIC,
        "inferential_unit": "participant",
        "technical_repeats_are_inferential_units": False,
        "test": "exact_two_sided_paired_sign_flip_of_participant_medians",
        "multiplicity_adjustment": "holm_across_all_six_target_metric_tests",
        "empirical_percentile_tie_weight": 0.5,
        "common_language_probability_tie_weight": 0.5,
        "source_table_schema_version": 2,
        "source_policy": "read-only",
        "meshing_or_fem_tasks": 0,
        "target_definitions": TARGETS,
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": {
            key: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for key, path in paths.items()
        },
        "outputs": table_files,
        "artifacts": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--v3-left-hippocampus-csv", type=Path, required=True)
    parser.add_argument("--v3-right-m1-csv", type=Path, required=True)
    parser.add_argument("--v4-left-hippocampus-csv", type=Path, required=True)
    parser.add_argument("--v4-right-m1-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser(
        "preflight", help="validate inputs and print scope without writing"
    )
    _add_common_arguments(preflight_parser)
    run_parser = subparsers.add_parser("run", help="write the complete comparison package")
    _add_common_arguments(run_parser)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = preflight(args) if args.command == "preflight" else run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
