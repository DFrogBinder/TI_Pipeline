"""Participant-level summaries and exact paired inference.

The 40 simulations in each participant/version/target/condition cell are
technical repeats.  This module first reduces those repeats to participant
cell summaries and only then performs paired inference across participants.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .settings import CONDITIONS, METRICS, SUBJECTS, TARGETS, VERSIONS


CELL_KEYS = ("version", "target", "condition", "subject")
REQUIRED_RUN_COLUMNS = {
    *CELL_KEYS,
    "repeat_tag",
    "roi_voxels",
    "achieved_roi_volume_mm3",
    "finite_roi_voxels",
    "nonfinite_roi_voxels",
    "finite_roi_fraction",
    *METRICS,
}
OUTCOME_PREFIX = {"cell_median": "central", "cell_sd": "variability"}


class StatisticsError(RuntimeError):
    """Raised when the run-level or participant-level matrix is incomplete."""


def _ordered(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    current = frame.copy()
    orders = {
        "subject": {value: index for index, value in enumerate(SUBJECTS)},
        "version": {value: index for index, value in enumerate(VERSIONS)},
        "target": {value: index for index, value in enumerate(TARGETS)},
        "condition": {value: index for index, value in enumerate(CONDITIONS)},
        "metric": {value: index for index, value in enumerate(METRICS)},
    }
    sort_columns: list[str] = []
    for column in columns:
        if column in orders:
            helper = f"_{column}_order"
            current[helper] = current[column].map(orders[column])
            sort_columns.append(helper)
        else:
            sort_columns.append(column)
    helpers = [column for column in current if column.startswith("_") and column.endswith("_order")]
    return (
        current.sort_values(sort_columns, kind="stable")
        .drop(columns=helpers)
        .reset_index(drop=True)
    )


def _validate_run_frame(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_RUN_COLUMNS.difference(frame.columns))
    if missing:
        raise StatisticsError(f"Run-level frame is missing columns: {missing}")
    expected_rows = len(SUBJECTS) * len(VERSIONS) * len(TARGETS) * len(CONDITIONS) * 40
    if len(frame) != expected_rows:
        raise StatisticsError(
            f"Expected {expected_rows} complete run rows, observed {len(frame)}"
        )
    dimensions = (
        ("subject", SUBJECTS),
        ("version", VERSIONS),
        ("target", TARGETS),
        ("condition", CONDITIONS),
    )
    for column, expected in dimensions:
        if set(frame[column].astype(str)) != set(expected):
            raise StatisticsError(f"Unexpected {column} values")
    if frame[[*CELL_KEYS, "repeat_tag"]].duplicated().any():
        raise StatisticsError("Duplicate participant-cell-repeat key")
    expected_tags = {f"repeat_{index:03d}" for index in range(1, 41)}
    groups = frame.groupby(list(CELL_KEYS), sort=False, observed=True)
    if len(groups) != 80:
        raise StatisticsError(f"Expected 80 participant cells, observed {len(groups)}")
    for key, group in groups:
        if len(group) != 40 or set(group["repeat_tag"].astype(str)) != expected_tags:
            raise StatisticsError(f"Incomplete repeat coverage for {key}")
    numeric = frame[[*METRICS, "finite_roi_fraction"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise StatisticsError("Run-level values must be finite")


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce 3,200 runs to 80 participant cells for each scalar metric."""

    _validate_run_frame(frame)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(CELL_KEYS), sort=False, observed=True):
        geometry = group[["roi_voxels", "achieved_roi_volume_mm3"]].nunique()
        if (geometry != 1).any():
            raise StatisticsError(f"ROI geometry changes within cell {key}")
        support = group["nonfinite_roi_voxels"].to_numpy(dtype=int)
        common = {
            **dict(zip(CELL_KEYS, key)),
            "run_count": int(len(group)),
            "roi_voxels": int(group["roi_voxels"].iloc[0]),
            "achieved_roi_volume_mm3": float(
                group["achieved_roi_volume_mm3"].iloc[0]
            ),
            "minimum_finite_roi_voxels": int(
                group["finite_roi_voxels"].min()
            ),
            "minimum_finite_fraction": float(
                group["finite_roi_fraction"].min()
            ),
            "affected_run_count": int(np.count_nonzero(support > 0)),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "cell_mean": mean,
                    "cell_median": float(np.median(values)),
                    "cell_sd": sd,
                    "cell_cv": sd / mean if mean != 0 else math.nan,
                    "cell_iqr": float(
                        np.percentile(values, 75) - np.percentile(values, 25)
                    ),
                    "cell_range": float(np.ptp(values)),
                }
            )
    result = pd.DataFrame(rows)
    return _ordered(
        result, ("version", "target", "condition", "subject", "metric")
    )


def exact_sign_flip(values: np.ndarray | list[float]) -> dict[str, float | int]:
    """Return exact two-sided paired sign-flip inference for one contrast."""

    observed_values = np.asarray(values, dtype=float)
    if observed_values.ndim != 1 or not len(observed_values):
        raise ValueError("Sign-flip values must be a non-empty vector")
    if not np.isfinite(observed_values).all():
        raise ValueError("Sign-flip values must be finite")
    n = len(observed_values)
    observed = abs(float(np.mean(observed_values)))
    assignments = 2**n
    extreme = 0
    tolerance = np.finfo(float).eps * max(1.0, observed) * 16
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        statistic = abs(float(np.mean(observed_values * np.asarray(signs))))
        if statistic + tolerance >= observed:
            extreme += 1
    sd = float(np.std(observed_values, ddof=1)) if n > 1 else math.nan
    cohen_dz = (
        float(np.mean(observed_values)) / sd
        if math.isfinite(sd) and sd > 0
        else math.nan
    )
    return {
        "n": n,
        "assignments": assignments,
        "mean_difference": float(np.mean(observed_values)),
        "median_difference": float(np.median(observed_values)),
        "sd_difference": sd,
        "cohen_dz": cohen_dz,
        "exact_p_value": extreme / assignments,
    }


def holm_adjust(values: np.ndarray | list[float]) -> np.ndarray:
    """Holm-adjust a finite vector while preserving its original order."""

    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("Holm p-values must be a finite vector")
    if np.any((raw < 0) | (raw > 1)):
        raise ValueError("Holm p-values must be between zero and one")
    if not len(raw):
        return raw.copy()
    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    scaled = (len(raw) - np.arange(len(raw))) * ranked
    adjusted_ranked = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def _validate_cell_summary(cell_summary: pd.DataFrame, outcome: str) -> None:
    if outcome not in OUTCOME_PREFIX:
        raise ValueError(f"Unsupported participant-cell outcome: {outcome}")
    required = {*CELL_KEYS, "metric", outcome}
    missing = sorted(required.difference(cell_summary.columns))
    if missing:
        raise StatisticsError(f"Cell summary is missing columns: {missing}")
    if len(cell_summary) != 80 * len(METRICS):
        raise StatisticsError("Expected 80 participant cells per metric")
    if cell_summary[[*CELL_KEYS, "metric"]].duplicated().any():
        raise StatisticsError("Duplicate participant-cell metric row")


def _contrast_row(
    *,
    family: str,
    contrast_type: str,
    label: str,
    values: pd.Series,
    metadata: dict[str, str],
) -> dict[str, Any]:
    ordered = values.reindex(SUBJECTS)
    if ordered.isna().any():
        raise StatisticsError(f"Incomplete participant pairing for {label}")
    raw = ordered.to_numpy(dtype=float)
    inference = exact_sign_flip(raw)
    participant_values = {
        subject: float(value) for subject, value in zip(SUBJECTS, raw)
    }
    return {
        "family": family,
        "contrast_type": contrast_type,
        "contrast_label": label,
        **metadata,
        **inference,
        "participant_contrasts_json": json.dumps(
            participant_values, separators=(",", ":")
        ),
    }


def _finish_family(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p_value"] = holm_adjust(
        frame["exact_p_value"].to_numpy(dtype=float)
    )
    return frame


def build_contrasts(
    cell_summary: pd.DataFrame, outcome: str
) -> dict[str, pd.DataFrame]:
    """Build version, condition, and difference-in-differences contrasts."""

    _validate_cell_summary(cell_summary, outcome)
    prefix = OUTCOME_PREFIX[outcome]
    version_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        for condition in CONDITIONS:
            for metric in METRICS:
                subset = cell_summary.loc[
                    (cell_summary["target"] == target)
                    & (cell_summary["condition"] == condition)
                    & (cell_summary["metric"] == metric)
                ]
                pivot = subset.pivot(index="subject", columns="version", values=outcome)
                values = pivot["simnibs326"] - pivot["simnibs4"]
                version_rows.append(
                    _contrast_row(
                        family=f"{prefix}_version",
                        contrast_type="version",
                        label="simnibs326_minus_simnibs4",
                        values=values,
                        metadata={
                            "target": target,
                            "condition": condition,
                            "version": "",
                            "metric": metric,
                            "outcome": outcome,
                        },
                    )
                )

        for version in VERSIONS:
            for metric in METRICS:
                subset = cell_summary.loc[
                    (cell_summary["target"] == target)
                    & (cell_summary["version"] == version)
                    & (cell_summary["metric"] == metric)
                ]
                pivot = subset.pivot(index="subject", columns="condition", values=outcome)
                values = pivot["fixed_mesh"] - pivot["remesh"]
                condition_rows.append(
                    _contrast_row(
                        family=f"{prefix}_condition",
                        contrast_type="condition",
                        label="fixed_mesh_minus_remesh",
                        values=values,
                        metadata={
                            "target": target,
                            "condition": "",
                            "version": version,
                            "metric": metric,
                            "outcome": outcome,
                        },
                    )
                )

        for metric in METRICS:
            subset = cell_summary.loc[
                (cell_summary["target"] == target)
                & (cell_summary["metric"] == metric)
            ].copy()
            subset["cell"] = subset["version"] + "__" + subset["condition"]
            pivot = subset.pivot(index="subject", columns="cell", values=outcome)
            values = (
                pivot["simnibs326__fixed_mesh"]
                - pivot["simnibs4__fixed_mesh"]
                - pivot["simnibs326__remesh"]
                + pivot["simnibs4__remesh"]
            )
            interaction_rows.append(
                _contrast_row(
                    family=f"{prefix}_interaction",
                    contrast_type="interaction",
                    label=(
                        "(simnibs326_minus_simnibs4)_fixed_minus_"
                        "(simnibs326_minus_simnibs4)_remesh"
                    ),
                    values=values,
                    metadata={
                        "target": target,
                        "condition": "fixed_minus_remesh",
                        "version": "simnibs326_minus_simnibs4",
                        "metric": metric,
                        "outcome": outcome,
                    },
                )
            )

    return {
        "version": _finish_family(version_rows),
        "condition": _finish_family(condition_rows),
        "interaction": _finish_family(interaction_rows),
    }


def build_support_audit(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize finite ROI support without assigning causal meaning."""

    _validate_run_frame(frame)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(CELL_KEYS), sort=False, observed=True):
        nonfinite = group["nonfinite_roi_voxels"].to_numpy(dtype=int)
        rows.append(
            {
                **dict(zip(CELL_KEYS, key)),
                "runs": int(len(group)),
                "roi_voxels": int(group["roi_voxels"].iloc[0]),
                "achieved_roi_volume_mm3": float(
                    group["achieved_roi_volume_mm3"].iloc[0]
                ),
                "minimum_finite_fraction": float(
                    group["finite_roi_fraction"].min()
                ),
                "median_finite_fraction": float(
                    group["finite_roi_fraction"].median()
                ),
                "runs_with_incomplete_support": int(
                    np.count_nonzero(nonfinite > 0)
                ),
                "runs_with_complete_support": int(
                    np.count_nonzero(nonfinite == 0)
                ),
                "maximum_nonfinite_roi_voxels": int(nonfinite.max()),
            }
        )
    audit = _ordered(
        pd.DataFrame(rows), ("version", "target", "condition", "subject")
    )
    affected = audit["runs_with_incomplete_support"] > 0
    summary = {
        "status": "complete",
        "groups": int(len(audit)),
        "groups_with_incomplete_support": int(affected.sum()),
        "groups_with_complete_support_only": int((~affected).sum()),
        "runs": int(audit["runs"].sum()),
        "runs_with_incomplete_support": int(
            audit["runs_with_incomplete_support"].sum()
        ),
        "minimum_finite_fraction": float(
            audit["minimum_finite_fraction"].min()
        ),
        "interpretation": (
            "Finite-support differences are a sensitivity and provenance "
            "boundary. Any association with field values is descriptive and "
            "does not establish causation."
        ),
    }
    return audit, summary
