"""Fail-closed assembly of the four canonical 9+1 metric tables."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .settings import (
    CONDITIONS,
    EXCLUDED_SUBJECT,
    METRICS,
    REPLACEMENT_SUBJECT,
    SUBJECTS,
    AnalysisConfig,
)


class AssemblyError(RuntimeError):
    """Raised when a metric source cannot prove the approved analysis scope."""


TARGETS = {
    "left_hippocampus": {"roi": "Left_Hippocampus", "volume": 200.0},
    "right_m1": {"roi": "Right_M1", "volume": 100.0},
}
REQUIRED_COLUMNS = {
    "schema_version",
    "subject",
    "condition",
    "repeat_tag",
    "roi",
    "requested_roi_volume_mm3",
    "achieved_roi_volume_mm3",
    "roi_radius_mm",
    "roi_voxels",
    "finite_roi_voxels",
    "nonfinite_roi_voxels",
    "finite_roi_fraction",
    *METRICS,
}


@dataclass(frozen=True)
class SourceSelection:
    name: str
    path: Path
    size_bytes: int
    sha256: str
    observed_rows: int
    selected_rows: int
    rejected_rows: int
    conditions: tuple[str, ...]
    subjects: tuple[str, ...]
    selected_sha256: str


@dataclass(frozen=True)
class CanonicalTable:
    key: str
    fieldnames: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.fieldnames)


@dataclass(frozen=True)
class AssemblyPlan:
    config: AnalysisConfig
    selections: tuple[SourceSelection, ...]
    tables: tuple[CanonicalTable, ...]

    @property
    def total_rows(self) -> int:
        return sum(table.row_count for table in self.tables)


@dataclass(frozen=True)
class _SourceSpec:
    name: str
    path: Path
    target: str
    observed_subjects: tuple[str, ...]
    observed_conditions: tuple[str, ...]
    selected_subjects: tuple[str, ...]
    selected_conditions: tuple[str, ...]
    table_key: str

    @property
    def observed_rows(self) -> int:
        return (
            len(self.observed_subjects) * len(self.observed_conditions) * 40
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _subject_order(subject: str) -> int:
    try:
        return SUBJECTS.index(subject)
    except ValueError:
        return len(SUBJECTS)


def _stable_sort(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_subject_order"] = ordered["subject"].map(_subject_order)
    ordered["_condition_order"] = ordered["condition"].map(
        {condition: index for index, condition in enumerate(CONDITIONS)}
    )
    ordered["_repeat_order"] = ordered["repeat_tag"].str.removeprefix(
        "repeat_"
    ).astype(int)
    return (
        ordered.sort_values(
            ["_subject_order", "_condition_order", "_repeat_order"],
            kind="stable",
        )
        .drop(columns=["_subject_order", "_condition_order", "_repeat_order"])
        .reset_index(drop=True)
    )


def _missing_replacement_message(path: Path) -> str:
    return (
        f"Replacement v4 schema-v2 optimizer metrics are unavailable at {path}. "
        "A separate read-only refresh of 160 existing NIfTI metrics is required; "
        "this analysis will not extract them implicitly."
    )


def validate_metric_table(
    frame: pd.DataFrame,
    *,
    target: str,
    allowed_conditions: tuple[str, ...],
    expected_subjects: tuple[str, ...],
    expected_rows: int,
    label: str,
) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise AssemblyError(f"{label}: missing columns {missing}")
    if set(pd.to_numeric(frame["schema_version"], errors="coerce")) != {2}:
        raise AssemblyError(f"{label}: expected schema version 2")
    if set(frame["subject"].astype(str)) != set(expected_subjects):
        raise AssemblyError(f"{label}: unexpected subject set")
    if set(frame["condition"].astype(str)) != set(allowed_conditions):
        raise AssemblyError(f"{label}: unexpected conditions")

    metadata = TARGETS[target]
    if set(frame["roi"].astype(str)) != {metadata["roi"]}:
        raise AssemblyError(f"{label}: unexpected ROI")
    requested = pd.to_numeric(
        frame["requested_roi_volume_mm3"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(requested).all() or not np.allclose(
        requested, metadata["volume"]
    ):
        raise AssemblyError(f"{label}: unexpected requested ROI volume")
    achieved = pd.to_numeric(
        frame["achieved_roi_volume_mm3"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(achieved).all() or np.any(achieved + 1e-9 < requested):
        raise AssemblyError(f"{label}: invalid achieved ROI volume")

    metric_values = frame[list(METRICS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise AssemblyError(f"{label}: non-finite scalar metric")
    if np.any(metric_values < 0):
        raise AssemblyError(f"{label}: negative scalar metric")

    roi_voxels = pd.to_numeric(frame["roi_voxels"], errors="coerce")
    finite = pd.to_numeric(frame["finite_roi_voxels"], errors="coerce")
    nonfinite = pd.to_numeric(frame["nonfinite_roi_voxels"], errors="coerce")
    fractions = pd.to_numeric(frame["finite_roi_fraction"], errors="coerce")
    if (
        roi_voxels.isna().any()
        or finite.isna().any()
        or nonfinite.isna().any()
        or fractions.isna().any()
        or (roi_voxels <= 0).any()
        or (finite <= 0).any()
        or (nonfinite < 0).any()
        or not (finite + nonfinite).equals(roi_voxels)
        or not np.allclose(
            fractions.to_numpy(dtype=float),
            finite.to_numpy(dtype=float) / roi_voxels.to_numpy(dtype=float),
            atol=5e-4,
        )
    ):
        raise AssemblyError(f"{label}: inconsistent finite ROI support")

    if frame[["subject", "condition", "repeat_tag"]].duplicated().any():
        raise AssemblyError(f"{label}: duplicate subject-condition-repeat key")
    expected_tags = {f"repeat_{index:03d}" for index in range(1, 41)}
    groups = frame.groupby(["subject", "condition"], sort=False)
    if len(groups) != len(expected_subjects) * len(allowed_conditions):
        raise AssemblyError(f"{label}: incomplete subject-condition cells")
    for key, group in groups:
        if len(group) != 40 or set(group["repeat_tag"].astype(str)) != expected_tags:
            raise AssemblyError(f"{label}: incomplete repeat coverage for {key}")
    if len(frame) != expected_rows:
        raise AssemblyError(
            f"{label}: expected {expected_rows} rows, observed {len(frame)}"
        )

    geometry = frame.groupby("subject", sort=False).agg(
        roi_voxels=("roi_voxels", "nunique"),
        achieved=("achieved_roi_volume_mm3", "nunique"),
        radius=("roi_radius_mm", "nunique"),
    )
    if not (geometry == 1).all().all():
        raise AssemblyError(f"{label}: ROI geometry changes within a subject")


def _source_specs(config: AnalysisConfig) -> tuple[_SourceSpec, ...]:
    base9 = tuple(subject for subject in config.subjects if subject != REPLACEMENT_SUBJECT)
    original10 = (*base9, EXCLUDED_SUBJECT)
    replacement = (REPLACEMENT_SUBJECT,)
    both = ("remesh", "fixed_mesh")
    sources = config.sources
    return (
        _SourceSpec("v3_base9_left", sources.v3_base9_left_csv, "left_hippocampus", base9, both, base9, both, "simnibs326_left_hippocampus"),
        _SourceSpec("v3_base9_right", sources.v3_base9_right_csv, "right_m1", base9, both, base9, both, "simnibs326_right_m1"),
        _SourceSpec("v3_replacement_left", sources.v3_replacement_left_csv, "left_hippocampus", replacement, both, replacement, both, "simnibs326_left_hippocampus"),
        _SourceSpec("v3_replacement_right", sources.v3_replacement_right_csv, "right_m1", replacement, both, replacement, both, "simnibs326_right_m1"),
        _SourceSpec("v4_historical_left", sources.v4_historical_left_csv, "left_hippocampus", original10, both, base9, ("remesh",), "simnibs4_left_hippocampus"),
        _SourceSpec("v4_historical_right", sources.v4_historical_right_csv, "right_m1", original10, both, base9, ("remesh",), "simnibs4_right_m1"),
        _SourceSpec("v4_corrected_fixed_left", sources.v4_corrected_fixed_left_csv, "left_hippocampus", original10, ("fixed_mesh",), base9, ("fixed_mesh",), "simnibs4_left_hippocampus"),
        _SourceSpec("v4_corrected_fixed_right", sources.v4_corrected_fixed_right_csv, "right_m1", original10, ("fixed_mesh",), base9, ("fixed_mesh",), "simnibs4_right_m1"),
        _SourceSpec("v4_replacement_left", sources.v4_replacement_left_csv, "left_hippocampus", replacement, both, replacement, both, "simnibs4_left_hippocampus"),
        _SourceSpec("v4_replacement_right", sources.v4_replacement_right_csv, "right_m1", replacement, both, replacement, both, "simnibs4_right_m1"),
    )


def build_assembly_plan(config: AnalysisConfig) -> AssemblyPlan:
    selections: list[SourceSelection] = []
    selected_by_table: dict[str, list[pd.DataFrame]] = {}
    for spec in _source_specs(config):
        if not spec.path.is_file() or spec.path.stat().st_size == 0:
            if spec.name.startswith("v4_replacement"):
                raise AssemblyError(_missing_replacement_message(spec.path))
            raise AssemblyError(f"Required metric source is missing or empty: {spec.path}")
        frame = pd.read_csv(spec.path)
        if spec.name.startswith("v4_replacement") and not REQUIRED_COLUMNS.issubset(
            frame.columns
        ):
            raise AssemblyError(_missing_replacement_message(spec.path))
        validate_metric_table(
            frame,
            target=spec.target,
            allowed_conditions=spec.observed_conditions,
            expected_subjects=spec.observed_subjects,
            expected_rows=spec.observed_rows,
            label=spec.name,
        )
        selected = frame[
            frame["subject"].isin(spec.selected_subjects)
            & frame["condition"].isin(spec.selected_conditions)
        ].copy()
        selected = _stable_sort(selected)
        expected_selected = (
            len(spec.selected_subjects) * len(spec.selected_conditions) * 40
        )
        if len(selected) != expected_selected:
            raise AssemblyError(
                f"{spec.name}: expected {expected_selected} selected rows, "
                f"observed {len(selected)}"
            )
        selections.append(
            SourceSelection(
                name=spec.name,
                path=spec.path,
                size_bytes=spec.path.stat().st_size,
                sha256=_sha256_file(spec.path),
                observed_rows=len(frame),
                selected_rows=len(selected),
                rejected_rows=len(frame) - len(selected),
                conditions=spec.selected_conditions,
                subjects=tuple(
                    subject for subject in SUBJECTS if subject in set(selected["subject"])
                ),
                selected_sha256=_frame_sha256(selected),
            )
        )
        selected_by_table.setdefault(spec.table_key, []).append(selected)

    tables: list[CanonicalTable] = []
    for key in (
        "simnibs326_left_hippocampus",
        "simnibs326_right_m1",
        "simnibs4_left_hippocampus",
        "simnibs4_right_m1",
    ):
        parts = selected_by_table[key]
        fieldnames: list[str] = []
        for part in parts:
            fieldnames.extend(column for column in part.columns if column not in fieldnames)
        combined = pd.concat(
            [part.reindex(columns=fieldnames) for part in parts], ignore_index=True
        )
        combined = _stable_sort(combined)
        target = "left_hippocampus" if key.endswith("left_hippocampus") else "right_m1"
        validate_metric_table(
            combined,
            target=target,
            allowed_conditions=CONDITIONS,
            expected_subjects=config.subjects,
            expected_rows=config.expected_rows_per_table,
            label=key,
        )
        if EXCLUDED_SUBJECT in set(combined["subject"]):
            raise AssemblyError(f"{key}: excluded subject survived assembly")
        tables.append(
            CanonicalTable(
                key=key,
                fieldnames=tuple(fieldnames),
                rows=tuple(
                    tuple(row)
                    for row in combined.itertuples(index=False, name=None)
                ),
            )
        )
    plan = AssemblyPlan(
        config=config,
        selections=tuple(selections),
        tables=tuple(tables),
    )
    if plan.total_rows != config.expected_rows:
        raise AssemblyError(
            f"Expected {config.expected_rows} total rows, observed {plan.total_rows}"
        )
    return plan


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_inputs(plan: AssemblyPlan, destination: Path) -> dict[str, Path]:
    destination = destination.resolve()
    for selection in plan.selections:
        if not selection.path.is_file() or _sha256_file(selection.path) != selection.sha256:
            raise AssemblyError(f"Source changed after planning: {selection.path}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    output_manifest: dict[str, dict[str, object]] = {}
    for table in plan.tables:
        path = destination / f"{table.key}.csv"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        table.to_frame().to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(path)
        outputs[table.key] = path
        output_manifest[table.key] = {
            "path": path.name,
            "rows": table.row_count,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "source_outputs_modified": False,
        "subjects": list(plan.config.subjects),
        "excluded_subject": EXCLUDED_SUBJECT,
        "replacement_subject": REPLACEMENT_SUBJECT,
        "total_rows": plan.total_rows,
        "sources": [
            {**asdict(selection), "path": str(selection.path)}
            for selection in plan.selections
        ],
        "outputs": output_manifest,
    }
    _write_json_atomic(destination / "assembly_manifest.json", payload)
    return outputs
