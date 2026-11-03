"""Code-owned production scope for the final 9+1 repeatability analysis."""

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
METRICS = (
    "roi_median_v_per_m",
    "roi_mean_v_per_m",
    "roi_p95_v_per_m",
)
PRIMARY_METRIC = "roi_median_v_per_m"


@dataclass(frozen=True)
class SourcePaths:
    """Every schema-v2 metric input and terminal receipt in production."""

    v3_base9_left_csv: Path
    v3_base9_right_csv: Path
    v3_replacement_left_csv: Path
    v3_replacement_right_csv: Path
    v4_historical_left_csv: Path
    v4_historical_right_csv: Path
    v4_corrected_fixed_left_csv: Path
    v4_corrected_fixed_right_csv: Path
    v4_replacement_left_csv: Path
    v4_replacement_right_csv: Path
    receipts: tuple[Path, ...]

    @property
    def metric_tables(self) -> dict[str, Path]:
        return {
            "v3_base9_left": self.v3_base9_left_csv,
            "v3_base9_right": self.v3_base9_right_csv,
            "v3_replacement_left": self.v3_replacement_left_csv,
            "v3_replacement_right": self.v3_replacement_right_csv,
            "v4_historical_left": self.v4_historical_left_csv,
            "v4_historical_right": self.v4_historical_right_csv,
            "v4_corrected_fixed_left": self.v4_corrected_fixed_left_csv,
            "v4_corrected_fixed_right": self.v4_corrected_fixed_right_csv,
            "v4_replacement_left": self.v4_replacement_left_csv,
            "v4_replacement_right": self.v4_replacement_right_csv,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable analysis destination, inputs, cohort, and repeat count."""

    output_root: Path
    sources: SourcePaths
    subjects: tuple[str, ...] = SUBJECTS
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
    def expected_rows_per_table(self) -> int:
        return len(self.subjects) * len(CONDITIONS) * self.repeat_count

    @property
    def expected_rows(self) -> int:
        return self.expected_cells * self.repeat_count


def _under(root: Path, name: str) -> Path:
    return root / name


def _metrics(root: Path, directory: str) -> Path:
    return root / "_post_processing" / directory / "optimizer_roi_metrics.csv"


def _receipt(root: Path) -> Path:
    return root / "_pipeline" / "workflow" / "complete.json"


def production_config() -> AnalysisConfig:
    """Return the sole production profile exposed by the HPC entry point."""

    parscratch = Path("/mnt/parscratch/users/cop23bi")
    replacement = _under(parscratch, "simnibs_subject_replacements") / REPLACEMENT_SUBJECT

    v3_left = _under(
        parscratch,
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_left_hippocampus_v2",
    )
    v3_right = _under(
        parscratch,
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_right_m1_v2",
    )
    v3_replacement_left = replacement / "simnibs326" / "left_hippocampus"
    v3_replacement_right = replacement / "simnibs326" / "right_m1"

    v4_left = _under(parscratch, "final_132_repeatability_balanced_10")
    v4_right = _under(parscratch, "final_132_repeatability_balanced_10_right_m1")
    v4_fixed_left = _under(
        parscratch, "final_132_repeatability_balanced_10_spherical_fixed_v1"
    )
    v4_fixed_right = _under(
        parscratch,
        "final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1",
    )
    v4_replacement_left = replacement / "simnibs4" / "left_hippocampus"
    v4_replacement_right = replacement / "simnibs4" / "right_m1"

    v3_metrics = "repeatability_optimizer_roi_metrics_simnibs326h_paired_v2"
    v4_metrics = "repeatability_optimizer_roi_metrics_v1"
    sources = SourcePaths(
        v3_base9_left_csv=_metrics(v3_left, v3_metrics),
        v3_base9_right_csv=_metrics(v3_right, v3_metrics),
        v3_replacement_left_csv=_metrics(v3_replacement_left, v3_metrics),
        v3_replacement_right_csv=_metrics(v3_replacement_right, v3_metrics),
        v4_historical_left_csv=_metrics(v4_left, v4_metrics),
        v4_historical_right_csv=_metrics(v4_right, v4_metrics),
        v4_corrected_fixed_left_csv=_metrics(
            v4_fixed_left, "optimizer_roi_metrics_v1"
        ),
        v4_corrected_fixed_right_csv=_metrics(
            v4_fixed_right, "optimizer_roi_metrics_v1"
        ),
        v4_replacement_left_csv=_metrics(v4_replacement_left, v4_metrics),
        v4_replacement_right_csv=_metrics(v4_replacement_right, v4_metrics),
        receipts=(
            _receipt(v3_left),
            _receipt(v3_right),
            _receipt(v3_replacement_left),
            _receipt(v3_replacement_right),
            _receipt(v4_left),
            _receipt(v4_right),
            _receipt(v4_fixed_left),
            _receipt(v4_fixed_right),
            _receipt(v4_replacement_left),
            _receipt(v4_replacement_right),
        ),
    )
    return AnalysisConfig(
        output_root=_under(
            parscratch, "final_132_repeatability_combined_analysis_v1"
        ),
        sources=sources,
    )
