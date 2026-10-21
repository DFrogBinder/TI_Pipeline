"""Pinned scope and paths for the completed balanced-10 audit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
REPEAT_COUNT = 40
CONDITIONS = ("remesh", "fixed_mesh")
SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC520083",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)


@dataclass(frozen=True)
class TargetSpec:
    key: str
    roi: str
    atlas_labels: tuple[int, ...]
    v3_root: Path
    v4_root: Path
    remesh_job_id: str
    fixed_job_id: str


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def target_specs() -> dict[str, TargetSpec]:
    return {
        "left-hippocampus": TargetSpec(
            key="left-hippocampus",
            roi="Left_Hippocampus",
            atlas_labels=(17,),
            v3_root=_path(
                "FINITE_SUPPORT_V3_LEFT_ROOT",
                "/mnt/parscratch/users/cop23bi/"
                "final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1",
            ),
            v4_root=_path(
                "FINITE_SUPPORT_V4_LEFT_ROOT",
                "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10",
            ),
            remesh_job_id=os.environ.get("FINITE_SUPPORT_V3_LEFT_REMESH_JOB", "11789553"),
            fixed_job_id=os.environ.get("FINITE_SUPPORT_V3_LEFT_FIXED_JOB", "11797124"),
        ),
        "right-m1": TargetSpec(
            key="right-m1",
            roi="Right_M1",
            atlas_labels=(12129,),
            v3_root=_path(
                "FINITE_SUPPORT_V3_RIGHT_ROOT",
                "/mnt/parscratch/users/cop23bi/"
                "final_132_repeatability_balanced_10_simnibs326_right_m1_v1",
            ),
            v4_root=_path(
                "FINITE_SUPPORT_V4_RIGHT_ROOT",
                "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1",
            ),
            remesh_job_id=os.environ.get("FINITE_SUPPORT_V3_RIGHT_REMESH_JOB", "11789558"),
            fixed_job_id=os.environ.get("FINITE_SUPPORT_V3_RIGHT_FIXED_JOB", "11797126"),
        ),
    }


def atlas_root() -> Path:
    return _path("FINITE_SUPPORT_ATLAS_ROOT", "/mnt/parscratch/users/cop23bi/ZIPs/atlases")


def default_output_root() -> Path:
    return _path(
        "FINITE_SUPPORT_AUDIT_ROOT",
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_headreco/"
        "_simnibs326/finite_support_forensics_v1",
    )


def cell_from_index(index: int) -> tuple[TargetSpec, int, str]:
    targets = tuple(target_specs().values())
    cell_count = len(targets) * len(SUBJECTS)
    if index < 0 or index >= cell_count:
        raise IndexError(f"cell index {index} outside 0..{cell_count - 1}")
    target = targets[index // len(SUBJECTS)]
    subject_index = index % len(SUBJECTS)
    return target, subject_index, SUBJECTS[subject_index]


def repeat_root(root: Path, subject: str, condition: str, repeat_number: int) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if repeat_number < 1 or repeat_number > REPEAT_COUNT:
        raise ValueError(f"repeat number outside 1..{REPEAT_COUNT}: {repeat_number}")
    return (
        root
        / f"{subject}_repeatability"
        / condition
        / "repeats"
        / f"repeat_{repeat_number:03d}"
    )


def production_task_id(subject_index: int, repeat_number: int) -> int:
    if subject_index < 0 or subject_index >= len(SUBJECTS):
        raise IndexError(subject_index)
    if repeat_number < 1 or repeat_number > REPEAT_COUNT:
        raise ValueError(repeat_number)
    return subject_index * REPEAT_COUNT + repeat_number - 1
