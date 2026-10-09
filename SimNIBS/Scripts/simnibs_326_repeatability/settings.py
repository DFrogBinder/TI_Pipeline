#!/usr/bin/env python3
"""Pinned settings for the SimNIBS 3.2.6 repeatability campaign."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent

SIMNIBS_MODULE = "SimNIBS/3.2.6-foss-2023a"
MATLAB_MODULE = os.environ.get("SIMNIBS326_MATLAB_MODULE", "MATLAB/2023b")
PARTITION = "sheffield"
CPUS_PER_TASK = 8
MEMORY = "32G"
TIME_LIMIT = "08:00:00"
MAX_CONCURRENT = 50
MAX_RETRIES = 1
REPEAT_COUNT = 40

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


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


SOURCE_ROOT = _path_from_env(
    "SIMNIBS326_SOURCE_ROOT",
    "/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected",
)
SCAFFOLD_ROOT = _path_from_env(
    "SIMNIBS326_SCAFFOLD_ROOT",
    "/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_simnibs326_headreco",
)
ATLAS_DIR = _path_from_env(
    "SIMNIBS326_ATLAS_DIR",
    "/mnt/parscratch/users/cop23bi/ZIPs/atlases",
)
TARGETS_CSV = _path_from_env(
    "SIMNIBS326_TARGETS_CSV",
    SCRIPTS_ROOT / "utils" / "targets.csv",
)

PROTECTED_CHARM_ROOTS = {
    "corrected_v4_scaffolds": Path(
        "/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
    ).resolve(),
    "left_hippocampus_experiment": Path(
        "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10"
    ).resolve(),
    "right_m1_experiment": Path(
        "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1"
    ).resolve(),
}


@dataclass(frozen=True)
class TargetSettings:
    key: str
    label: str
    experiment_root: Path


TARGETS = {
    "left-hippocampus": TargetSettings(
        key="left-hippocampus",
        label="left hippocampus",
        experiment_root=_path_from_env(
            "SIMNIBS326_LEFT_ROOT",
            "/mnt/parscratch/users/cop23bi/"
            "final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1",
        ),
    ),
    "right-m1": TargetSettings(
        key="right-m1",
        label="right M1",
        experiment_root=_path_from_env(
            "SIMNIBS326_RIGHT_ROOT",
            "/mnt/parscratch/users/cop23bi/"
            "final_132_repeatability_balanced_10_simnibs326_right_m1_v1",
        ),
    ),
}


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def assert_output_root_isolation() -> dict[str, dict[str, str]]:
    """Fail before writes if a v3 output could touch protected input/v4 roots."""
    writable = {
        "v3_scaffolds": SCAFFOLD_ROOT,
        **{
            f"v3_{key.replace('-', '_')}": value.experiment_root
            for key, value in TARGETS.items()
        },
    }
    protected = {
        "staged_source": SOURCE_ROOT,
        "atlas_source": ATLAS_DIR,
        **PROTECTED_CHARM_ROOTS,
    }

    writable_items = list(writable.items())
    for index, (left_name, left_path) in enumerate(writable_items):
        for right_name, right_path in writable_items[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise RuntimeError(
                    "SimNIBS 3.2.6 writable roots overlap: "
                    f"{left_name}={left_path} and {right_name}={right_path}"
                )

    for writable_name, writable_path in writable.items():
        for protected_name, protected_path in protected.items():
            if _paths_overlap(writable_path, protected_path):
                raise RuntimeError(
                    "Refusing SimNIBS 3.2.6 writes because a writable root "
                    "overlaps a protected source/CHARM root: "
                    f"{writable_name}={writable_path}; "
                    f"{protected_name}={protected_path}"
                )

    return {
        "writable_v3_roots": {key: str(value) for key, value in writable.items()},
        "protected_read_only_roots": {
            key: str(value) for key, value in protected.items()
        },
    }


HEAD_MODEL_STRATEGY = "headreco-all-scaffold-then-fresh-volumemesh"
SEGMENTATION_PROVENANCE = (
    "SimNIBS 3.2.6 headreco regenerated from the same T1/T2 inputs. "
    "The corrected SimNIBS 4 CHARM label image is retained as an input-provenance "
    "file but is not consumed by headreco and the v3/v4 head models are not identical."
)


def target_settings(value: str) -> TargetSettings:
    try:
        return TARGETS[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported target {value!r}; expected one of {sorted(TARGETS)}"
        ) from exc


def scaffold_subject_root(subject: str) -> Path:
    return SCAFFOLD_ROOT / subject / "anat"


def source_anat_root(subject: str) -> Path:
    return SOURCE_ROOT / subject / "anat"


def source_paths(subject: str) -> tuple[Path, Path, Path]:
    anat = source_anat_root(subject)
    return (
        anat / f"{subject}_T1w.nii",
        anat / f"{subject}_T2w.nii",
        anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
    )
