"""Round-trip-safe assembly of schema-compatible native metric tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ti_current_repair.repeatability_v2.precision import read_metric_table

from .settings import CONDITIONS, METRICS, OUTCOMES, TARGETS, VERSIONS
from .statistics import validate_final_matrix


@dataclass(frozen=True)
class MetricSource:
    version: str
    target: str
    condition: str
    outcome_mode: str
    path: Path
    preliminary: bool


def assemble_sources(
    sources: Iterable[MetricSource], *, final: bool
) -> pd.DataFrame:
    materialized = tuple(sources)
    if not materialized:
        raise ValueError("No native metric sources were supplied")
    if final:
        preliminary = [
            source
            for source in materialized
            if source.preliminary or source.version == "simnibs401"
        ]
        if preliminary:
            raise ValueError("SimNIBS 4.0.1/preliminary sources cannot enter final analysis")
        expected_keys = {
            (version, target, condition, outcome)
            for version in VERSIONS
            for target in TARGETS
            for condition in CONDITIONS
            for outcome in OUTCOMES
        }
        observed_keys = {
            (
                source.version,
                source.target,
                source.condition,
                source.outcome_mode,
            )
            for source in materialized
        }
        if observed_keys != expected_keys or len(materialized) != len(expected_keys):
            raise ValueError("Final analysis sources do not cover the exact 24 tables")
    frames: list[pd.DataFrame] = []
    for source in materialized:
        rows = read_metric_table(
            source.path,
            float_fields=(*METRICS, "finite_volume_fraction"),
            float_precision="round_trip",
        )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError(f"Native metric source is empty: {source.path}")
        for column, expected in (
            ("target", source.target),
            ("condition", source.condition),
            ("outcome_mode", source.outcome_mode),
        ):
            if column in frame and set(frame[column].astype(str)) != {expected}:
                raise ValueError(f"{source.path} has unexpected {column} values")
            frame[column] = expected
        if "version" in frame and set(frame["version"].astype(str)) != {
            source.version
        }:
            raise ValueError(f"{source.path} has unexpected version values")
        frame["version"] = source.version
        frame["preliminary"] = bool(source.preliminary)
        frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
        for field in METRICS:
            frame[field] = pd.to_numeric(frame[field], errors="raise").astype(
                "float64"
            )
        frame["finite_volume_fraction"] = pd.to_numeric(
            frame["finite_volume_fraction"], errors="raise"
        ).astype("float64")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if "electrode_name" in combined:
        combined["electrode_name"] = combined["electrode_name"].fillna("")
    if final:
        validate_final_matrix(combined)
    return combined

