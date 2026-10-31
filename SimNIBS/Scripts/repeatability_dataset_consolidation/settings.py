"""Code-owned production scope for the combined repeatability dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC510415",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)
REPLACEMENT_SUBJECT = "sub-CC510415"
EXCLUDED_SUBJECT = "sub-CC520083"
VERSIONS = ("simnibs326", "simnibs4")
TARGETS = ("left_hippocampus", "right_m1")
CONDITIONS = ("remesh", "fixed_mesh")


@dataclass(frozen=True)
class SourceRoots:
    """Every source root named by the approved consolidation design."""

    v4_left_remesh: Path
    v4_right_remesh: Path
    v4_left_fixed: Path
    v4_right_fixed: Path
    v3_left: Path
    v3_right: Path
    v3_headmodels: Path
    v4_scaffolds: Path
    replacement_subject: Path
    paper_analysis: Path
    nested_analysis: Path
    legacy_v3_left: Path
    legacy_v3_right: Path
    legacy_v3_headreco: Path
    replacement_v3_scaffolds: Path


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset, archive, cohort, and receipt configuration."""

    output_root: Path
    shared_archive_root: Path
    shared_receipt_root: Path
    deletion_log_root: Path
    sources: SourceRoots
    subjects: tuple[str, ...] = SUBJECTS
    replacement_subject: str = REPLACEMENT_SUBJECT
    excluded_subject: str = EXCLUDED_SUBJECT
    repeat_count: int = 40

    @property
    def expected_cells(self) -> int:
        return (
            len(self.subjects)
            * len(VERSIONS)
            * len(TARGETS)
            * len(CONDITIONS)
        )

    @property
    def expected_simulations(self) -> int:
        return self.expected_cells * self.repeat_count

    @property
    def archive_sources(self) -> dict[str, Path]:
        return {
            "legacy_v3_left": self.sources.legacy_v3_left,
            "legacy_v3_right": self.sources.legacy_v3_right,
            "legacy_v3_headreco": self.sources.legacy_v3_headreco,
            "replacement_v3_scaffolds": self.sources.replacement_v3_scaffolds,
        }


def _under(root: Path, name: str) -> Path:
    return root / name


def production_config() -> DatasetConfig:
    """Return the sole production path profile exposed by the CLI."""
    parscratch = Path("/mnt/parscratch/users/cop23bi")
    shared = Path("/shared/boyan/Shared")
    sources = SourceRoots(
        v4_left_remesh=_under(parscratch, "final_132_repeatability_balanced_10"),
        v4_right_remesh=_under(
            parscratch, "final_132_repeatability_balanced_10_right_m1"
        ),
        v4_left_fixed=_under(
            parscratch, "final_132_repeatability_balanced_10_spherical_fixed_v1"
        ),
        v4_right_fixed=_under(
            parscratch,
            "final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1",
        ),
        v3_left=_under(
            parscratch,
            "final_132_repeatability_balanced_10_simnibs326_harmonized_"
            "base9_left_hippocampus_v2",
        ),
        v3_right=_under(
            parscratch,
            "final_132_repeatability_balanced_10_simnibs326_harmonized_"
            "base9_right_m1_v2",
        ),
        v3_headmodels=_under(
            parscratch, "ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2"
        ),
        v4_scaffolds=_under(parscratch, "CamCan_Corrected_v4_Scaffolds"),
        replacement_subject=_under(
            parscratch, "simnibs_subject_replacements/sub-CC510415"
        ),
        paper_analysis=_under(parscratch, "final_132_repeatability_paper_update_v1"),
        nested_analysis=_under(parscratch, "final_132_repeatability_nested_40x40_v1"),
        legacy_v3_left=_under(
            parscratch,
            "final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1",
        ),
        legacy_v3_right=_under(
            parscratch, "final_132_repeatability_balanced_10_simnibs326_right_m1_v1"
        ),
        legacy_v3_headreco=_under(
            parscratch, "ti_dataset_final_132_balanced_10_simnibs326_headreco"
        ),
        replacement_v3_scaffolds=_under(parscratch, "simnibs326_subject_scaffolds_v1"),
    )
    return DatasetConfig(
        output_root=_under(parscratch, "final_132_repeatability_combined_dataset_v1"),
        shared_archive_root=_under(shared, "retired_repeatability_v3_sources_v1"),
        shared_receipt_root=_under(shared, ".repeatability_retirement_receipts"),
        deletion_log_root=_under(
            parscratch, "deletion_job_logs/repeatability_dataset_v1"
        ),
        sources=sources,
    )


def retirement_allowlist(config: DatasetConfig) -> dict[str, Path]:
    """Return exact code-owned retirement keys and canonical source paths."""
    sources = config.sources
    return {
        "v4_left_remesh": sources.v4_left_remesh,
        "v4_right_remesh": sources.v4_right_remesh,
        "v4_left_fixed": sources.v4_left_fixed,
        "v4_right_fixed": sources.v4_right_fixed,
        "v3_left": sources.v3_left,
        "v3_right": sources.v3_right,
        "v3_headmodels": sources.v3_headmodels,
        "replacement_subject": sources.replacement_subject,
        "paper_analysis": sources.paper_analysis,
        "nested_analysis": sources.nested_analysis,
        **config.archive_sources,
    }
