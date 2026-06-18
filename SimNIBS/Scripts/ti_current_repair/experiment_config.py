#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared configuration helpers for paired repeatability experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


VALID_MESH_MODES = {"remesh", "fixed_mesh"}
VALID_COMPARE_METRICS = {"median_roi", "mean_roi", "peak_roi"}


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    mesh_mode: str
    repeat_count: int
    description: str = ""


@dataclass(frozen=True)
class AnalysisConfig:
    roi_preset: str | None = None
    roi_name: str | None = None
    roi_labels: list[int] | None = None
    atlas_dir: str | None = None
    compare_cohort_root: str | None = None
    cohort_region_name: str | None = None
    cohort_region_label: int | None = None
    compare_metric: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    source_root: Path
    experiment_root: Path
    subjects: list[str]
    conditions: list[ConditionConfig]
    analysis: AnalysisConfig


@dataclass(frozen=True)
class ExperimentTask:
    subject: str
    condition_name: str
    mesh_mode: str
    repeat_index: int
    repeat_tag: str


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty string for '{field_name}'.")
    return value.strip()


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name=field_name)


def _require_positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"Expected '{field_name}' to be an integer >= 1.")
    return value


def _optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Expected '{field_name}' to be an integer when provided.")
    return value


def _parse_condition(raw: object, *, index: int) -> ConditionConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"Condition #{index + 1} must be a JSON object.")
    name = _require_string(raw.get("name"), field_name=f"conditions[{index}].name")
    mesh_mode = _require_string(
        raw.get("mesh_mode"),
        field_name=f"conditions[{index}].mesh_mode",
    )
    if mesh_mode not in VALID_MESH_MODES:
        raise ValueError(
            f"Unsupported mesh_mode '{mesh_mode}' for condition '{name}'. "
            f"Valid values: {', '.join(sorted(VALID_MESH_MODES))}."
        )
    repeat_count = _require_positive_int(
        raw.get("repeat_count"),
        field_name=f"conditions[{index}].repeat_count",
    )
    description = _optional_string(
        raw.get("description"),
        field_name=f"conditions[{index}].description",
    ) or ""
    return ConditionConfig(
        name=name,
        mesh_mode=mesh_mode,
        repeat_count=repeat_count,
        description=description,
    )


def _parse_analysis(raw: object) -> AnalysisConfig:
    if raw is None:
        return AnalysisConfig()
    if not isinstance(raw, dict):
        raise ValueError("The 'analysis' section must be a JSON object when provided.")
    roi_labels = raw.get("roi_labels")
    if roi_labels is not None:
        if not isinstance(roi_labels, list) or not roi_labels:
            raise ValueError("'analysis.roi_labels' must be a non-empty list of integers.")
        parsed_labels: list[int] = []
        for idx, value in enumerate(roi_labels):
            if not isinstance(value, int):
                raise ValueError(
                    f"'analysis.roi_labels[{idx}]' must be an integer; got {value!r}."
                )
            parsed_labels.append(value)
        roi_labels = parsed_labels
    compare_metric = _optional_string(
        raw.get("compare_metric"),
        field_name="analysis.compare_metric",
    )
    if compare_metric is not None and compare_metric not in VALID_COMPARE_METRICS:
        raise ValueError(
            f"Unsupported analysis.compare_metric '{compare_metric}'. "
            f"Valid values: {', '.join(sorted(VALID_COMPARE_METRICS))}."
        )
    return AnalysisConfig(
        roi_preset=_optional_string(raw.get("roi_preset"), field_name="analysis.roi_preset"),
        roi_name=_optional_string(raw.get("roi_name"), field_name="analysis.roi_name"),
        roi_labels=roi_labels,
        atlas_dir=_optional_string(raw.get("atlas_dir"), field_name="analysis.atlas_dir"),
        compare_cohort_root=_optional_string(
            raw.get("compare_cohort_root"),
            field_name="analysis.compare_cohort_root",
        ),
        cohort_region_name=_optional_string(
            raw.get("cohort_region_name"),
            field_name="analysis.cohort_region_name",
        ),
        cohort_region_label=_optional_int(
            raw.get("cohort_region_label"),
            field_name="analysis.cohort_region_label",
        ),
        compare_metric=compare_metric,
    )


def load_experiment_config(path: str | Path, *, validate_paths: bool = True) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a top-level JSON object.")

    source_root = Path(
        _require_string(raw.get("source_root"), field_name="source_root")
    ).expanduser().resolve()
    experiment_root = Path(
        _require_string(raw.get("experiment_root"), field_name="experiment_root")
    ).expanduser().resolve()

    raw_subjects = raw.get("subjects")
    if not isinstance(raw_subjects, list) or not raw_subjects:
        raise ValueError("'subjects' must be a non-empty JSON list.")
    subjects = [_require_string(value, field_name=f"subjects[{idx}]") for idx, value in enumerate(raw_subjects)]
    if len(set(subjects)) != len(subjects):
        raise ValueError("'subjects' contains duplicates. Each subject must appear only once.")

    raw_conditions = raw.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("'conditions' must be a non-empty JSON list.")
    conditions = [_parse_condition(value, index=idx) for idx, value in enumerate(raw_conditions)]
    condition_names = [condition.name for condition in conditions]
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("'conditions' contains duplicate names. Each condition name must be unique.")

    if validate_paths:
        if not source_root.is_dir():
            raise FileNotFoundError(f"Configured source_root does not exist: {source_root}")
        experiment_root.mkdir(parents=True, exist_ok=True)

    return ExperimentConfig(
        config_path=config_path,
        source_root=source_root,
        experiment_root=experiment_root,
        subjects=subjects,
        conditions=conditions,
        analysis=_parse_analysis(raw.get("analysis")),
    )


def condition_by_name(config: ExperimentConfig, name: str) -> ConditionConfig:
    for condition in config.conditions:
        if condition.name == name:
            return condition
    raise KeyError(f"Condition '{name}' not present in config: {config.config_path}")


def repeat_tag(index: int, width: int = 3) -> str:
    return f"repeat_{index:0{width}d}"


def subject_repeatability_root(config: ExperimentConfig, subject: str) -> Path:
    return config.experiment_root / f"{subject}_repeatability"


def subject_condition_root(config: ExperimentConfig, subject: str, condition_name: str) -> Path:
    return subject_repeatability_root(config, subject) / condition_name


def subject_condition_repeats_root(config: ExperimentConfig, subject: str, condition_name: str) -> Path:
    return subject_condition_root(config, subject, condition_name) / "repeats"


def subject_condition_mesh_cache_root(config: ExperimentConfig, subject: str, condition_name: str) -> Path:
    return subject_condition_root(config, subject, condition_name) / "mesh_cache" / subject / "anat"


def subject_analysis_root(config: ExperimentConfig, subject: str) -> Path:
    return config.experiment_root / "_analysis" / subject


def condition_manifest_path(config: ExperimentConfig, subject: str, condition_name: str) -> Path:
    return subject_condition_root(config, subject, condition_name) / "condition_manifest.json"


def iter_experiment_tasks(
    config: ExperimentConfig,
    *,
    subjects: list[str] | None = None,
    condition_names: list[str] | None = None,
) -> list[ExperimentTask]:
    selected_subjects = subjects or config.subjects
    selected_conditions = (
        [condition_by_name(config, name) for name in condition_names]
        if condition_names
        else config.conditions
    )

    tasks: list[ExperimentTask] = []
    for subject in selected_subjects:
        if subject not in config.subjects:
            raise KeyError(f"Subject '{subject}' is not present in config: {config.config_path}")
        for condition in selected_conditions:
            for repeat_index in range(1, condition.repeat_count + 1):
                tasks.append(
                    ExperimentTask(
                        subject=subject,
                        condition_name=condition.name,
                        mesh_mode=condition.mesh_mode,
                        repeat_index=repeat_index,
                        repeat_tag=repeat_tag(repeat_index),
                    )
                )
    return tasks


def template_config_dict() -> dict[str, object]:
    return {
        "source_root": "/mnt/parscratch/users/cop23bi/repeatability-ti-dataset",
        "experiment_root": "/mnt/parscratch/users/cop23bi/repeatability-ti-experiment",
        "subjects": [
            "sub-CC110056",
            "sub-CC120120",
            "sub-CC210124",
        ],
        "conditions": [
            {
                "name": "remesh",
                "mesh_mode": "remesh",
                "repeat_count": 40,
                "description": "Generate a fresh mesh for every repeat to capture remeshing + FEM variation.",
            },
            {
                "name": "fixed_mesh",
                "mesh_mode": "fixed_mesh",
                "repeat_count": 40,
                "description": "Generate one mesh once per subject, then reuse it across repeats to isolate FEM variation.",
            },
        ],
        "analysis": {
            "roi_preset": "left-hippocampus",
            "atlas_dir": "/mnt/parscratch/cop23bi/ZIPs/atlases",
            "compare_cohort_root": "/media/boyan/main/PhD/Left_Hippocampus_Data",
            "cohort_region_name": "Left-Hippocampus",
            "cohort_region_label": 17,
            "compare_metric": "median_roi",
        },
    }


def write_template_config(path: str | Path) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(template_config_dict(), fh, indent=2)
        fh.write("\n")
    return out_path
