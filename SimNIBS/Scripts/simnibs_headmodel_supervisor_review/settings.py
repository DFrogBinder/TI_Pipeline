"""Pinned roots and cohort for the supervisor-review package."""

from __future__ import annotations

from dataclasses import dataclass, fields
import os
from pathlib import Path
from typing import Mapping

from simnibs_326_corrected_label_repeatability.settings import BASE9_SUBJECTS


ENV_PREFIX = "SIMNIBS_REVIEW_"
SIMNIBS326_MODULE = "SimNIBS/3.2.6-foss-2023a"
PARTITION = "sheffield"
CPUS_PER_TASK = 8
MEMORY = "32G"
TIME_LIMIT = "08:00:00"
EXPECTED_ARTIFACTS = 8

_DEFAULTS = {
    "ORIGINAL_CHARM_ROOT": "/mnt/parscratch/users/cop23bi/charm_segmentations",
    "CORRECTED_V4_SCAFFOLD_ROOT": (
        "/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
    ),
    "LEGACY_V3_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_headreco"
    ),
    "HARMONIZED_V3_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2"
    ),
    "ATLAS_ROOT": "/mnt/parscratch/users/cop23bi/ZIPs/atlases",
    "COMPLETED_V4_LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10"
    ),
    "OUTPUT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "simnibs_headmodel_supervisor_review_package_v1"
    ),
}


@dataclass(frozen=True)
class ReviewPaths:
    original_charm_root: Path
    corrected_v4_scaffold_root: Path
    legacy_v3_root: Path
    harmonized_v3_root: Path
    atlas_root: Path
    completed_v4_left_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        for field in fields(self):
            value = Path(getattr(self, field.name)).expanduser().resolve(strict=False)
            object.__setattr__(self, field.name, value)

    @property
    def protected_roots(self) -> tuple[Path, ...]:
        return (
            self.original_charm_root,
            self.corrected_v4_scaffold_root,
            self.legacy_v3_root,
            self.harmonized_v3_root,
            self.atlas_root,
            self.completed_v4_left_root,
        )

    @property
    def selection_receipt(self) -> Path:
        return self.output_root / "selection.json"

    @property
    def package_dir(self) -> Path:
        return self.output_root / "package"

    @property
    def archive_path(self) -> Path:
        return self.output_root / "simnibs_headmodel_supervisor_review.tar.gz"

    @property
    def completion_receipt(self) -> Path:
        return self.output_root / "complete.json"


def paths_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ReviewPaths:
    source = os.environ if environ is None else environ

    def value(key: str) -> Path:
        return Path(source.get(f"{ENV_PREFIX}{key}", _DEFAULTS[key]))

    return ReviewPaths(
        original_charm_root=value("ORIGINAL_CHARM_ROOT"),
        corrected_v4_scaffold_root=value("CORRECTED_V4_SCAFFOLD_ROOT"),
        legacy_v3_root=value("LEGACY_V3_ROOT"),
        harmonized_v3_root=value("HARMONIZED_V3_ROOT"),
        atlas_root=value("ATLAS_ROOT"),
        completed_v4_left_root=value("COMPLETED_V4_LEFT_ROOT"),
        output_root=value("OUTPUT_ROOT"),
    )


__all__ = [
    "BASE9_SUBJECTS",
    "CPUS_PER_TASK",
    "EXPECTED_ARTIFACTS",
    "MEMORY",
    "PARTITION",
    "ReviewPaths",
    "SIMNIBS326_MODULE",
    "TIME_LIMIT",
    "paths_from_environment",
]
