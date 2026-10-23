"""Immutable identity, paths, and numerical scope for one replacement subject."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

from simnibs_326_corrected_label_repeatability.settings import (
    BASE9_SUBJECTS,
    campaign_paths_from_environment,
)


SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FINAL132_SUBJECTS = (
    SCRIPTS_ROOT
    / "CamCan_Experiment"
    / "cohort_pipeline"
    / "cohorts"
    / "final_132"
    / "subjects.txt"
)
DEFAULT_CORRECTED_V4_SCAFFOLD = Path(
    "/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
)
DEFAULT_ATLAS_ROOT = Path("/mnt/parscratch/users/cop23bi/ZIPs/atlases")
DEFAULT_TARGETS_CSV = SCRIPTS_ROOT / "utils" / "targets.csv"
DEFAULT_WORK_ROOT = Path("/mnt/parscratch/users/cop23bi/simnibs_subject_replacements")
SUBJECT_PATTERN = re.compile(r"^sub-CC[0-9]{6}$")
REPEAT_COUNT = 40


@dataclass(frozen=True)
class ReplacementConfig:
    subject: str
    replaces: str
    legacy_v3_root: Path
    final132_subjects_file: Path = DEFAULT_FINAL132_SUBJECTS
    corrected_v4_scaffold_root: Path = DEFAULT_CORRECTED_V4_SCAFFOLD
    atlas_root: Path = DEFAULT_ATLAS_ROOT
    targets_csv: Path = DEFAULT_TARGETS_CSV
    work_root: Path = DEFAULT_WORK_ROOT


@dataclass(frozen=True)
class ReplacementPaths:
    root: Path
    pipeline_root: Path
    manifest: Path
    singleton_subjects_file: Path
    v4_corrected_root: Path
    v3_headmodel_root: Path
    v3_left_root: Path
    v3_right_root: Path
    v4_left_root: Path
    v4_right_root: Path


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _overlap(left: Path, right: Path) -> bool:
    left = _resolved(left)
    right = _resolved(right)
    return left == right or left in right.parents or right in left.parents


def _canonical(value: str, label: str) -> str:
    if not SUBJECT_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a canonical sub-CC###### subject ID")
    return value


def validate_candidate(config: ReplacementConfig) -> tuple[str, ...]:
    """Validate one runtime candidate against the authoritative cohort list."""
    subject = _canonical(config.subject, "subject")
    replaces = _canonical(config.replaces, "replaces")
    if subject == replaces:
        raise ValueError("Replacement subject must differ from the replaced subject")
    if subject in BASE9_SUBJECTS:
        raise ValueError(f"Candidate {subject} is already a retained base-nine subject")
    try:
        cohort = tuple(
            line.strip()
            for line in config.final132_subjects_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise ValueError(
            f"Authoritative final-132 subject list is unreadable: "
            f"{config.final132_subjects_file}"
        ) from exc
    invalid = [value for value in cohort if not SUBJECT_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(f"Authoritative subject list contains non-canonical IDs: {invalid}")
    duplicates = sorted(value for value, count in Counter(cohort).items() if count > 1)
    if duplicates:
        raise ValueError(f"Authoritative subject list contains duplicate IDs: {duplicates}")
    if cohort.count(subject) != 1:
        raise ValueError(f"Candidate {subject} must occur exactly once in final-132")
    return cohort


def replacement_paths(config: ReplacementConfig) -> ReplacementPaths:
    subject = _canonical(config.subject, "subject")
    root = _resolved(config.work_root) / subject
    pipeline_root = root / "_pipeline"
    return ReplacementPaths(
        root=root,
        pipeline_root=pipeline_root,
        manifest=pipeline_root / "replacement_manifest.json",
        singleton_subjects_file=pipeline_root / "subjects.txt",
        v4_corrected_root=root / "inputs" / "corrected_v4",
        v3_headmodel_root=root / "simnibs326" / "headmodels",
        v3_left_root=root / "simnibs326" / "left_hippocampus",
        v3_right_root=root / "simnibs326" / "right_m1",
        v4_left_root=root / "simnibs4" / "left_hippocampus",
        v4_right_root=root / "simnibs4" / "right_m1",
    )


def assert_replacement_root_isolation(
    config: ReplacementConfig,
    paths: ReplacementPaths | None = None,
) -> dict[str, dict[str, str]]:
    resolved = paths or replacement_paths(config)
    writable = {
        "v4_corrected_root": resolved.v4_corrected_root,
        "v3_headmodel_root": resolved.v3_headmodel_root,
        "v3_left_root": resolved.v3_left_root,
        "v3_right_root": resolved.v3_right_root,
        "v4_left_root": resolved.v4_left_root,
        "v4_right_root": resolved.v4_right_root,
    }
    production = campaign_paths_from_environment({})
    protected = {
        "final132_subjects_file": config.final132_subjects_file,
        "corrected_v4_scaffold_root": config.corrected_v4_scaffold_root,
        "legacy_v3_root": config.legacy_v3_root,
        "atlas_root": config.atlas_root,
        "targets_csv": config.targets_csv,
        **{
            name: getattr(production, name)
            for name in (
                "headmodel_root",
                "left_root",
                "right_root",
                "completed_v3_left_root",
                "completed_v3_right_root",
                "completed_v4_left_root",
                "completed_v4_right_root",
            )
        },
    }
    items = list(writable.items())
    for index, (left_name, left_path) in enumerate(items):
        for right_name, right_path in items[index + 1 :]:
            if _overlap(left_path, right_path):
                raise ValueError(
                    f"Replacement writable roots overlap: {left_name}={left_path}; "
                    f"{right_name}={right_path}"
                )
    for writable_name, writable_path in writable.items():
        for protected_name, protected_path in protected.items():
            if _overlap(writable_path, protected_path):
                raise ValueError(
                    f"Replacement writable/protected roots overlap: "
                    f"{writable_name}={writable_path}; "
                    f"{protected_name}={protected_path}"
                )
    return {
        "writable_roots": {
            name: str(_resolved(path)) for name, path in writable.items()
        },
        "protected_read_only_roots": {
            name: str(_resolved(path)) for name, path in protected.items()
        },
    }


def replacement_scope(config: ReplacementConfig) -> dict[str, object]:
    validate_candidate(config)
    paths = replacement_paths(config)
    return {
        "subject": config.subject,
        "replaces": config.replaces,
        "targets": ["left-hippocampus", "right-m1"],
        "conditions": ["remesh", "fixed_mesh"],
        "repeats_per_condition": REPEAT_COUNT,
        "v3_reference_tasks": 1,
        "v3_headmodel_tasks": 1,
        "per_version": {
            "simulation_tasks": 2 * 2 * REPEAT_COUNT,
            "component_fem_solves": 2 * 2 * REPEAT_COUNT * 2,
        },
        "both_versions": {
            "simulation_tasks": 2 * 2 * 2 * REPEAT_COUNT,
            "component_fem_solves": 2 * 2 * 2 * REPEAT_COUNT * 2,
        },
        "arrays": {
            "reference": "0-0%1",
            "headmodel": "0-0%1",
            "simulation": "0-39%50",
            "roi": "0-0%1",
        },
        "root_isolation": assert_replacement_root_isolation(config, paths),
    }
