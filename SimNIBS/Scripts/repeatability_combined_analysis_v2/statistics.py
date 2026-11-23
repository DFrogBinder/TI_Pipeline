"""Participant-level summaries for nested technical repeatability runs."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from .settings import (
    CONDITIONS,
    METRICS,
    OUTCOMES,
    REPEAT_COUNT,
    SUBJECTS,
    TARGETS,
    VERSIONS,
    expected_participant_cells,
    expected_run_rows,
)


CELL_KEYS = ("subject", "version", "target", "condition", "outcome_mode")
REQUIRED_COLUMNS = {
    *CELL_KEYS,
    "repeat",
    "repeat_tag",
    *METRICS,
    "finite_volume_fraction",
    "source_msh_path",
    "source_msh_sha256",
    "source_field_name",
    "geometry_sha256",
    "roi_provenance_sha256",
    "roi_provenance_json",
    "extraction_implementation",
    "extraction_version",
    "simnibs_version",
    "analysis_dtype",
    "stored_field_dtype",
    "preliminary",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _provenance_digest(value: object) -> str:
    parsed = json.loads(str(value))
    canonical = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_final_matrix(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Final matrix is missing columns: {missing}")
    if len(frame) != expected_run_rows():
        raise ValueError(
            f"Expected {expected_run_rows()} final run rows, observed {len(frame)}"
        )
    for column, expected in (
        ("subject", SUBJECTS),
        ("version", VERSIONS),
        ("target", TARGETS),
        ("condition", CONDITIONS),
        ("outcome_mode", OUTCOMES),
    ):
        if set(frame[column].astype(str)) != set(expected):
            if column == "version":
                raise ValueError(
                    "Final matrix must contain only SimNIBS 3.2.6 and 4.6; "
                    "4.0.1/preliminary rows are excluded"
                )
            raise ValueError(f"Final matrix has unexpected {column} values")
    preliminary = frame["preliminary"].astype(bool)
    if preliminary.any():
        raise ValueError("Preliminary rows cannot enter the final analysis matrix")
    if not frame["source_msh_path"].astype(str).str.lower().str.endswith(".msh").all():
        raise ValueError("Every final metric must come from a native .msh source")
    if not frame["analysis_dtype"].eq("float64").all() or not frame[
        "stored_field_dtype"
    ].eq("float64").all():
        raise ValueError("Every final analytical and stored field dtype must be float64")
    digest_columns = (
        "source_msh_sha256",
        "geometry_sha256",
        "roi_provenance_sha256",
    )
    if any(
        not frame[column].astype(str).str.fullmatch(SHA256_PATTERN).all()
        for column in digest_columns
    ):
        raise ValueError("Every final row must contain complete SHA-256 provenance")
    try:
        observed_roi_digests = frame["roi_provenance_json"].map(_provenance_digest)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Every final row must contain valid ROI provenance JSON") from error
    if not observed_roi_digests.eq(
        frame["roi_provenance_sha256"].astype(str)
    ).all():
        raise ValueError("ROI provenance JSON and SHA-256 provenance disagree")
    if not frame["extraction_implementation"].eq(
        "repeatability_v2.native_mesh_extraction"
    ).all() or not frame["extraction_version"].astype(str).eq("2").all():
        raise ValueError("Final rows must use the approved native extraction implementation")
    expected_simnibs = frame["version"].map(
        {"simnibs326": "3.2.6", "simnibs46": "4.6.0"}
    )
    if not frame["simnibs_version"].astype(str).eq(expected_simnibs).all():
        raise ValueError("Metric SimNIBS provenance disagrees with final version labels")
    expected_fields = frame["outcome_mode"].map(
        {
            "ti_target": "TImax",
            "tdcs_pair1_matched_target": "E",
            "tdcs_pair1_under_anode": "E",
        }
    )
    if not frame["source_field_name"].astype(str).eq(expected_fields).all():
        raise ValueError("Metric source-field provenance disagrees with outcome mode")
    key_columns = [*CELL_KEYS, "repeat"]
    if frame[key_columns].duplicated().any():
        raise ValueError("Duplicate final run key")
    numeric = frame[[*METRICS, "finite_volume_fraction"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Final analytical values must be finite")
    if not numeric["finite_volume_fraction"].between(0.0, 1.0, inclusive="right").all():
        raise ValueError("Final finite-volume fractions must be in (0, 1]")
    expected_tags = {f"repeat_{value:03d}" for value in range(1, REPEAT_COUNT + 1)}
    groups = frame.groupby(list(CELL_KEYS), sort=False, observed=True)
    if len(groups) != expected_participant_cells():
        raise ValueError("Final matrix does not contain 240 participant cells")
    for key, group in groups:
        if len(group) != REPEAT_COUNT or set(group["repeat_tag"].astype(str)) != expected_tags:
            raise ValueError(f"Incomplete 40-repeat participant cell: {key}")
    fixed = frame.loc[frame["condition"] == "fixed_mesh"]
    fixed_groups = fixed.groupby(
        ["subject", "version", "target"], sort=False, observed=True
    )["geometry_sha256"].nunique()
    if not fixed_groups.eq(1).all():
        raise ValueError("Fixed geometry changed across repeats or outcome rows")


def summarize_participant_cells(frame: pd.DataFrame) -> pd.DataFrame:
    validate_final_matrix(frame)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(CELL_KEYS), sort=False, observed=True):
        common = dict(zip(CELL_KEYS, key))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            mean = float(np.mean(values, dtype=np.float64))
            sd = float(np.std(values, ddof=1, dtype=np.float64))
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "run_count": REPEAT_COUNT,
                    "cell_mean": mean,
                    "cell_median": float(np.median(values)),
                    "cell_sd": sd,
                    "cell_cv": sd / mean if mean != 0 else math.nan,
                    "cell_iqr": float(
                        np.percentile(values, 75) - np.percentile(values, 25)
                    ),
                    "minimum_finite_volume_fraction": float(
                        group["finite_volume_fraction"].min()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["version", "target", "condition", "outcome_mode", "subject", "metric"],
        kind="stable",
    ).reset_index(drop=True)


def _exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values, dtype=np.float64)))
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 16
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
        )
        extreme += int(statistic + tolerance >= observed)
    return extreme / (2 ** len(values))


def build_matched_decomposition(cell_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    matched = cell_summary.loc[
        cell_summary["outcome_mode"].isin(
            ("ti_target", "tdcs_pair1_matched_target")
        )
    ]
    for version in VERSIONS:
        for target in TARGETS:
            for condition in CONDITIONS:
                for metric in METRICS:
                    subset = matched.loc[
                        (matched["version"] == version)
                        & (matched["target"] == target)
                        & (matched["condition"] == condition)
                        & (matched["metric"] == metric)
                    ]
                    for repeatability, value_column in (
                        ("absolute_sd", "cell_sd"),
                        ("relative_cv", "cell_cv"),
                    ):
                        pivot = subset.pivot(
                            index="subject",
                            columns="outcome_mode",
                            values=value_column,
                        ).reindex(SUBJECTS)
                        if pivot.isna().any().any():
                            raise ValueError("Matched decomposition lacks participant pairing")
                        differences = (
                            pivot["tdcs_pair1_matched_target"]
                            - pivot["ti_target"]
                        ).to_numpy(dtype=np.float64)
                        rows.append(
                            {
                                "version": version,
                                "target": target,
                                "condition": condition,
                                "metric": metric,
                                "repeatability_measure": repeatability,
                                "comparison": "tdcs_pair1_matched_target_minus_ti_target",
                                "participant_count": len(SUBJECTS),
                                "mean_paired_difference": float(
                                    np.mean(differences, dtype=np.float64)
                                ),
                                "median_paired_difference": float(
                                    np.median(differences)
                                ),
                                "positive_difference_count": int(
                                    np.count_nonzero(differences > 0)
                                ),
                                "negative_difference_count": int(
                                    np.count_nonzero(differences < 0)
                                ),
                                "exact_sign_flip_p_value": _exact_sign_flip(
                                    differences
                                ),
                                "interpretation": "matched_roi_ti_component_variability_decomposition",
                                "assumes_ti_doubles_variability": False,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_under_anode(cell_summary: pd.DataFrame) -> pd.DataFrame:
    subset = cell_summary.loc[
        cell_summary["outcome_mode"] == "tdcs_pair1_under_anode"
    ]
    rows: list[dict[str, object]] = []
    for key, group in subset.groupby(
        ["version", "target", "condition", "metric"], sort=False, observed=True
    ):
        for repeatability, value_column in (
            ("absolute_sd", "cell_sd"),
            ("relative_cv", "cell_cv"),
        ):
            values = group.set_index("subject")[value_column].reindex(SUBJECTS)
            if values.isna().any():
                raise ValueError("Under-anode summary lacks participant coverage")
            rows.append(
                {
                    **dict(zip(("version", "target", "condition", "metric"), key)),
                    "outcome_mode": "tdcs_pair1_under_anode",
                    "repeatability_measure": repeatability,
                    "participant_count": len(SUBJECTS),
                    "participant_mean": float(values.mean()),
                    "participant_median": float(values.median()),
                    "interpretation": "separate_application_facing_tdcs",
                    "location_matched_to_ti": False,
                }
            )
    return pd.DataFrame(rows)
