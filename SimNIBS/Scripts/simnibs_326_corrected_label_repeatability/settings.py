"""Pinned configuration and path isolation for the harmonized v3 campaign."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ENV_PREFIX = "SIMNIBS326H_"
PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent
TARGETS_CSV = Path(
    os.environ.get(
        f"{ENV_PREFIX}TARGETS_CSV",
        str(SCRIPTS_ROOT / "utils" / "targets.csv"),
    )
).expanduser().resolve(strict=False)
SIMNIBS_MODULE = "SimNIBS/3.2.6-foss-2023a"
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


@dataclass(frozen=True)
class CampaignPaths:
    corrected_root: Path
    legacy_reference_root: Path
    atlas_root: Path
    headmodel_root: Path
    left_root: Path
    right_root: Path
    completed_v3_left_root: Path
    completed_v3_right_root: Path
    completed_v4_left_root: Path
    completed_v4_right_root: Path


@dataclass(frozen=True)
class SubjectPaths:
    subject: str
    corrected_label: Path
    corrected_t1: Path
    legacy_anat: Path
    legacy_m2m: Path
    legacy_ready_marker: Path
    staged_anat: Path
    staged_m2m: Path
    harmonized_label: Path
    harmonization_qc: Path
    harmonization_figure: Path
    head_mesh: Path
    compatibility_mesh: Path
    reference_receipt: Path
    headmodel_receipt: Path


@dataclass(frozen=True)
class TargetSettings:
    key: str
    label: str
    experiment_root: Path


_DEFAULTS = {
    "CORRECTED_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_corrected"
    ),
    "LEGACY_REFERENCE_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_headreco"
    ),
    "ATLAS_ROOT": "/mnt/parscratch/users/cop23bi/ZIPs/atlases",
    "HEADMODEL_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2"
    ),
    "LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "left_hippocampus_v2"
    ),
    "RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_right_m1_v2"
    ),
    "COMPLETED_V3_LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1"
    ),
    "COMPLETED_V3_RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_right_m1_v1"
    ),
    "COMPLETED_V4_LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10"
    ),
    "COMPLETED_V4_RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_right_m1"
    ),
}


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def campaign_paths_from_environment(
    environ: Mapping[str, str] | None = None,
) -> CampaignPaths:
    """Resolve every campaign root from one namespaced environment contract."""
    source = os.environ if environ is None else environ

    def get(key: str) -> Path:
        return _resolved(source.get(f"{ENV_PREFIX}{key}", _DEFAULTS[key]))

    return CampaignPaths(
        corrected_root=get("CORRECTED_ROOT"),
        legacy_reference_root=get("LEGACY_REFERENCE_ROOT"),
        atlas_root=get("ATLAS_ROOT"),
        headmodel_root=get("HEADMODEL_ROOT"),
        left_root=get("LEFT_ROOT"),
        right_root=get("RIGHT_ROOT"),
        completed_v3_left_root=get("COMPLETED_V3_LEFT_ROOT"),
        completed_v3_right_root=get("COMPLETED_V3_RIGHT_ROOT"),
        completed_v4_left_root=get("COMPLETED_V4_LEFT_ROOT"),
        completed_v4_right_root=get("COMPLETED_V4_RIGHT_ROOT"),
    )


def subject_paths(paths: CampaignPaths, subject: str) -> SubjectPaths:
    if subject not in SUBJECTS:
        raise ValueError(f"Unsupported subject {subject!r}; expected one of {SUBJECTS}")
    corrected_anat = paths.corrected_root / subject / "anat"
    legacy_anat = paths.legacy_reference_root / subject / "anat"
    staged_anat = paths.headmodel_root / subject / "anat"
    receipt_root = paths.headmodel_root / "_pipeline" / "subjects" / subject
    return SubjectPaths(
        subject=subject,
        corrected_label=(
            corrected_anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"
        ),
        corrected_t1=corrected_anat / f"{subject}_T1w.nii",
        legacy_anat=legacy_anat,
        legacy_m2m=legacy_anat / f"m2m_{subject}",
        legacy_ready_marker=legacy_anat / ".simnibs326_scaffold_ready.json",
        staged_anat=staged_anat,
        staged_m2m=staged_anat / f"m2m_{subject}",
        harmonized_label=(
            paths.headmodel_root
            / "harmonized_labels"
            / f"{subject}_v3_harmonized_t1fs_grid.nii.gz"
        ),
        harmonization_qc=(
            paths.headmodel_root / "harmonized_labels" / "qc" / f"{subject}.json"
        ),
        harmonization_figure=(
            paths.headmodel_root / "harmonized_labels" / "figures" / f"{subject}.png"
        ),
        head_mesh=staged_anat / f"{subject}.msh",
        compatibility_mesh=staged_anat / f"m2m_{subject}" / f"{subject}.msh",
        reference_receipt=receipt_root / "reference_stage.json",
        headmodel_receipt=receipt_root / "headmodel_complete.json",
    )


def target_settings(value: str, paths: CampaignPaths | None = None) -> TargetSettings:
    resolved = paths or campaign_paths_from_environment()
    targets = {
        "left-hippocampus": TargetSettings(
            "left-hippocampus", "left hippocampus", resolved.left_root
        ),
        "right-m1": TargetSettings("right-m1", "right M1", resolved.right_root),
    }
    try:
        return targets[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported target {value!r}; expected one of {sorted(targets)}"
        ) from exc


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def assert_root_isolation(paths: CampaignPaths) -> dict[str, dict[str, str]]:
    """Fail before writes if a new root overlaps any writable/protected root."""
    writable = {
        "headmodel_root": paths.headmodel_root,
        "left_root": paths.left_root,
        "right_root": paths.right_root,
    }
    protected = {
        "corrected_root": paths.corrected_root,
        "legacy_reference_root": paths.legacy_reference_root,
        "atlas_root": paths.atlas_root,
        "completed_v3_left_root": paths.completed_v3_left_root,
        "completed_v3_right_root": paths.completed_v3_right_root,
        "completed_v4_left_root": paths.completed_v4_left_root,
        "completed_v4_right_root": paths.completed_v4_right_root,
    }
    writable_items = list(writable.items())
    for index, (left_name, left_path) in enumerate(writable_items):
        for right_name, right_path in writable_items[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise RuntimeError(
                    "Writable roots overlap: "
                    f"{left_name}={left_path}; {right_name}={right_path}"
                )
    for writable_name, writable_path in writable.items():
        for protected_name, protected_path in protected.items():
            if _paths_overlap(writable_path, protected_path):
                raise RuntimeError(
                    "Writable and protected roots overlap: "
                    f"{writable_name}={writable_path}; "
                    f"{protected_name}={protected_path}"
                )
    return {
        "writable_roots": {name: str(path) for name, path in writable.items()},
        "protected_read_only_roots": {
            name: str(path) for name, path in protected.items()
        },
    }


def campaign_scope(paths: CampaignPaths | None = None) -> dict[str, object]:
    resolved = paths or campaign_paths_from_environment()
    per_target_condition = len(SUBJECTS) * REPEAT_COUNT
    return {
        "subjects": list(SUBJECTS),
        "subject_count": len(SUBJECTS),
        "targets": ["left-hippocampus", "right-m1"],
        "conditions": ["remesh", "fixed_mesh"],
        "repeats_per_condition": REPEAT_COUNT,
        "total_simulation_tasks": per_target_condition * 2 * 2,
        "component_fem_solves": per_target_condition * 2 * 2 * 2,
        "arrays": {
            "left_hippocampus_remesh": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
            "left_hippocampus_fixed": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
            "right_m1_remesh": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
            "right_m1_fixed": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
        },
        "resources": {
            "simnibs_module": SIMNIBS_MODULE,
            "partition": PARTITION,
            "cpus_per_task": CPUS_PER_TASK,
            "memory": MEMORY,
            "time_limit": TIME_LIMIT,
            "max_concurrent": MAX_CONCURRENT,
            "max_retries": MAX_RETRIES,
        },
        "root_isolation": assert_root_isolation(resolved),
    }
