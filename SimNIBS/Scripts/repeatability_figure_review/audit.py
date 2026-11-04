"""Validate and inventory the downloaded repeatability-analysis figures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


TARGETS = ("left_hippocampus", "right_m1")
COMPREHENSIVE_TYPES = (
    "central_values",
    "paired_version_contrasts",
    "technical_sd",
    "version_condition_interaction",
)
ESTABLISHED_TYPES = (
    "primary_median_roi_repeat_distributions",
    "single_repeat_subject_ranking_uncertainty",
)

EXPECTED_ANALYSIS_FIGURES = tuple(
    f"comprehensive/figures/{target}/{figure_type}.png"
    for target in TARGETS
    for figure_type in COMPREHENSIVE_TYPES
) + tuple(
    f"established/cross_version_remesh/{target}_{figure_type}.png"
    for target in TARGETS
    for figure_type in (
        "v3_percentile_location",
        "v4_distribution_with_v3_overlay",
    )
) + tuple(
    f"established/{version}/figures/{target}/{index:02d}_{figure_type}.png"
    for version in ("within_simnibs326", "within_simnibs4")
    for target in TARGETS
    for index, figure_type in enumerate(ESTABLISHED_TYPES, start=1)
)

REQUIRED_ANALYSIS_SOURCES = (
    "analysis_manifest.json",
    "inputs/simnibs326_left_hippocampus.csv",
    "inputs/simnibs326_right_m1.csv",
    "inputs/simnibs4_left_hippocampus.csv",
    "inputs/simnibs4_right_m1.csv",
)

PAPER_EQUIVALENCE_SPECS = (
    (
        "figure_target_field_repeatability.png",
        "updated_current",
        "paper_equivalents/figure_target_field_repeatability.png",
        "Rebuilt from the updated 9+1 SimNIBS 4 run-level established outputs.",
    ),
    (
        "figure_single_run_ranking_uncertainty.png",
        "updated_current",
        "paper_equivalents/figure_single_run_ranking_uncertainty.png",
        "Rebuilt from the updated 9+1 SimNIBS 4 ranking outputs.",
    ),
    (
        "figure_bootstrap_precision.png",
        "regenerate_current",
        "paper_equivalents/figure_bootstrap_precision.png",
        "Regenerated from the current 800-row-per-target SimNIBS 4 input tables.",
    ),
    (
        "figure_mesh_element_count_variability.png",
        "legacy_data_gap",
        "legacy_paper_figures/figure_mesh_element_count_variability.png",
        "The downloaded package contains no mesh-element metrics; this is legacy evidence, not an updated 9+1 equivalent.",
    ),
    (
        "figure_tissue_volume_deviations.png",
        "legacy_retained_subjects",
        "legacy_paper_figures/figure_tissue_volume_deviations.png",
        "The selected illustrative participants are retained, but the figure remains a legacy v4-only structural illustration.",
    ),
)


class SourceAuditError(RuntimeError):
    """Raised when the source package cannot support a trustworthy review."""


@dataclass(frozen=True)
class FigureRecord:
    relative_path: str
    family: str
    target: str
    figure_type: str
    disposition: str
    width: int
    height: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PaperEquivalence:
    paper_figure: str
    status: str
    companion_path: str
    rationale: str
    source_sha256: str


@dataclass(frozen=True)
class AuditBundle:
    source_root: Path
    manuscript_root: Path
    figures: tuple[FigureRecord, ...]
    paper_equivalences: tuple[PaperEquivalence, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SourceAuditError(f"Required source is missing or empty: {path}")


def _parse_figure(relative_path: str) -> tuple[str, str, str, str]:
    parts = Path(relative_path).parts
    stem = Path(relative_path).stem
    if parts[0] == "comprehensive":
        family = "comprehensive"
        target = parts[2]
        figure_type = stem
    elif parts[1] == "cross_version_remesh":
        family = "cross_version_remesh"
        target = next(target for target in TARGETS if stem.startswith(target))
        figure_type = stem.removeprefix(f"{target}_")
    else:
        family = parts[1]
        target = parts[3]
        figure_type = stem.split("_", 1)[1]

    dispositions = {
        ("comprehensive", "central_values"): "combine_and_clarify",
        ("comprehensive", "paired_version_contrasts"): "repair_unidentified_lines",
        ("comprehensive", "technical_sd"): "repair_dynamic_range",
        ("comprehensive", "version_condition_interaction"): "repair_participant_layout",
        ("cross_version_remesh", "v3_percentile_location"): "repair_boundary_clipping",
        ("cross_version_remesh", "v4_distribution_with_v3_overlay"): "combine_and_reduce_overplotting",
        ("within_simnibs326", "primary_median_roi_repeat_distributions"): "retain_with_determinism_annotation",
        ("within_simnibs326", "single_repeat_subject_ranking_uncertainty"): "replace_deterministic_blank_panel",
        ("within_simnibs4", "primary_median_roi_repeat_distributions"): "combine_as_paper_equivalent",
        ("within_simnibs4", "single_repeat_subject_ranking_uncertainty"): "combine_as_paper_equivalent",
    }
    return family, target, figure_type, dispositions[(family, figure_type)]


def audit_sources(source_root: Path, manuscript_root: Path) -> AuditBundle:
    """Return a fail-closed inventory of source and manuscript figures."""

    source_root = Path(source_root).expanduser().resolve()
    manuscript_root = Path(manuscript_root).expanduser().resolve()
    for relative in REQUIRED_ANALYSIS_SOURCES:
        _require_nonempty(source_root / relative)

    records: list[FigureRecord] = []
    for relative in EXPECTED_ANALYSIS_FIGURES:
        path = source_root / relative
        if not path.is_file():
            raise SourceAuditError(f"Required analysis figure is missing: {path}")
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        family, target, figure_type, disposition = _parse_figure(relative)
        records.append(
            FigureRecord(
                relative_path=relative,
                family=family,
                target=target,
                figure_type=figure_type,
                disposition=disposition,
                width=int(width),
                height=int(height),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )

    paper_dir = manuscript_root / "figures" / "combined"
    equivalences: list[PaperEquivalence] = []
    for paper_figure, status, companion_path, rationale in PAPER_EQUIVALENCE_SPECS:
        source = paper_dir / paper_figure
        _require_nonempty(source)
        equivalences.append(
            PaperEquivalence(
                paper_figure=paper_figure,
                status=status,
                companion_path=companion_path,
                rationale=rationale,
                source_sha256=sha256_file(source),
            )
        )

    return AuditBundle(
        source_root=source_root,
        manuscript_root=manuscript_root,
        figures=tuple(records),
        paper_equivalences=tuple(equivalences),
    )

