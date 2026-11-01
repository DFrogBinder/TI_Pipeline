"""Deterministic, immutable copy planning for the combined dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat

from .settings import CONDITIONS, TARGETS, DatasetConfig


class PlanningError(RuntimeError):
    """Raised when the approved source scope cannot form a safe copy plan."""


@dataclass(frozen=True)
class CopyItem:
    key: str
    source: Path
    destination: Path
    role: str
    required: bool = True
    materialize_links: bool = True
    subject: str | None = None
    version: str | None = None
    target: str | None = None
    condition: str | None = None
    source_device: int = 0
    source_inode: int = 0
    source_type: str = ""
    source_size: int = 0
    source_mtime_ns: int = 0
    source_ctime_ns: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source": str(self.source),
            "destination": self.destination.as_posix(),
            "role": self.role,
            "required": self.required,
            "materialize_links": self.materialize_links,
            "subject": self.subject,
            "version": self.version,
            "target": self.target,
            "condition": self.condition,
            "source_device": self.source_device,
            "source_inode": self.source_inode,
            "source_type": self.source_type,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "source_ctime_ns": self.source_ctime_ns,
        }


@dataclass(frozen=True)
class ConsolidationPlan:
    schema_version: int
    output_root: Path
    subjects: tuple[str, ...]
    excluded_subject: str
    replacement_subject: str
    repeat_count: int
    expected_cells: int
    expected_simulations: int
    items: tuple[CopyItem, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "output_root": str(self.output_root),
            "subjects": list(self.subjects),
            "excluded_subject": self.excluded_subject,
            "replacement_subject": self.replacement_subject,
            "repeat_count": self.repeat_count,
            "expected_cells": self.expected_cells,
            "expected_simulations": self.expected_simulations,
            "items": [item.as_dict() for item in self.items],
        }


def _inside(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _overlap(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


def _source_type(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _approved_roots(config: DatasetConfig) -> tuple[Path, ...]:
    return tuple(
        value.resolve(strict=False)
        for value in vars(config.sources).values()
        if isinstance(value, Path)
    )


def _validate_links(source: Path, approved_roots: tuple[Path, ...]) -> None:
    active_directories: set[tuple[int, int]] = set()
    validated_directories: set[tuple[int, int]] = set()

    def visit(candidate: Path) -> None:
        target = candidate
        if candidate.is_symlink():
            try:
                target = candidate.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise PlanningError(
                    f"Broken symlink in planned source: {candidate}"
                ) from exc
            if not any(_inside(target, root) for root in approved_roots):
                raise PlanningError(
                    f"Symlink target is outside approved sources: {candidate} -> {target}"
                )
        try:
            info = target.stat()
        except OSError as exc:
            raise PlanningError(f"Cannot inspect planned source: {target}") from exc
        if not stat.S_ISDIR(info.st_mode):
            return
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in active_directories:
            raise PlanningError(
                f"Symlink materialization cycle is not allowed: {candidate} -> {target}"
            )
        if identity in validated_directories:
            return
        active_directories.add(identity)
        try:
            for child in sorted(target.iterdir(), key=lambda path: path.name):
                visit(child)
        except OSError as exc:
            raise PlanningError(f"Cannot inspect planned source directory: {target}") from exc
        finally:
            active_directories.remove(identity)
        validated_directories.add(identity)

    visit(source)


def _safe_destination(destination: Path) -> Path:
    if destination.is_absolute() or not destination.parts:
        raise PlanningError(f"Unsafe destination: {destination}")
    if any(part in {"", ".", ".."} for part in destination.parts):
        raise PlanningError(f"Unsafe destination: {destination}")
    return destination


def _copy_item(
    *,
    key: str,
    source: Path,
    destination: Path,
    role: str,
    approved_roots: tuple[Path, ...],
    required: bool = True,
    materialize_links: bool = True,
    subject: str | None = None,
    version: str | None = None,
    target: str | None = None,
    condition: str | None = None,
) -> CopyItem:
    destination = _safe_destination(destination)
    if not source.exists() and not source.is_symlink():
        if required:
            raise PlanningError(f"Required source is missing: {source}")
        return CopyItem(
            key=key,
            source=source,
            destination=destination,
            role=role,
            required=False,
            materialize_links=materialize_links,
            subject=subject,
            version=version,
            target=target,
            condition=condition,
        )
    _validate_links(source, approved_roots)
    info = source.lstat()
    return CopyItem(
        key=key,
        source=source.resolve(strict=False),
        destination=destination,
        role=role,
        required=required,
        materialize_links=materialize_links,
        subject=subject,
        version=version,
        target=target,
        condition=condition,
        source_device=int(info.st_dev),
        source_inode=int(info.st_ino),
        source_type=_source_type(info.st_mode),
        source_size=int(info.st_size),
        source_mtime_ns=int(info.st_mtime_ns),
        source_ctime_ns=int(info.st_ctime_ns),
    )


def _scientific_destination(
    version: str,
    target: str,
    subject: str,
    condition: str,
) -> Path:
    return Path("data") / version / target / f"{subject}_repeatability" / condition


def _base_scientific_root(config: DatasetConfig, version: str, target: str, condition: str) -> Path:
    sources = config.sources
    if version == "simnibs326":
        return sources.v3_left if target == "left_hippocampus" else sources.v3_right
    mapping = {
        ("left_hippocampus", "remesh"): sources.v4_left_remesh,
        ("left_hippocampus", "fixed_mesh"): sources.v4_left_fixed,
        ("right_m1", "remesh"): sources.v4_right_remesh,
        ("right_m1", "fixed_mesh"): sources.v4_right_fixed,
    }
    return mapping[(target, condition)]


def _scientific_source(
    config: DatasetConfig,
    version: str,
    target: str,
    subject: str,
    condition: str,
) -> Path:
    if subject == config.replacement_subject:
        return (
            config.sources.replacement_subject
            / version
            / target
            / f"{subject}_repeatability"
            / condition
        )
    return (
        _base_scientific_root(config, version, target, condition)
        / f"{subject}_repeatability"
        / condition
    )


def _metadata_roots(config: DatasetConfig) -> dict[str, Path]:
    sources = config.sources
    return {
        "v4_left_remesh": sources.v4_left_remesh,
        "v4_right_remesh": sources.v4_right_remesh,
        "v4_left_fixed": sources.v4_left_fixed,
        "v4_right_fixed": sources.v4_right_fixed,
        "v3_left": sources.v3_left,
        "v3_right": sources.v3_right,
        "replacement_v3_left": sources.replacement_subject / "simnibs326/left_hippocampus",
        "replacement_v3_right": sources.replacement_subject / "simnibs326/right_m1",
        "replacement_v4_left": sources.replacement_subject / "simnibs4/left_hippocampus",
        "replacement_v4_right": sources.replacement_subject / "simnibs4/right_m1",
    }


def _append_source_metadata(
    items: list[CopyItem],
    config: DatasetConfig,
    approved_roots: tuple[Path, ...],
) -> None:
    for source_key, root in _metadata_roots(config).items():
        if not root.is_dir():
            raise PlanningError(f"Required experiment root is missing: {root}")
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if child.name.endswith("_repeatability"):
                continue
            items.append(
                _copy_item(
                    key=f"metadata:{source_key}:{child.name}",
                    source=child,
                    destination=Path("provenance/source_roots") / source_key / child.name,
                    role="source_metadata",
                    approved_roots=approved_roots,
                )
            )
    for child in sorted(config.sources.replacement_subject.iterdir(), key=lambda path: path.name):
        if child.name in {"simnibs326", "simnibs4"}:
            continue
        items.append(
            _copy_item(
                key=f"replacement_metadata:{child.name}",
                source=child,
                destination=Path("provenance/source_roots/replacement_subject_root")
                / child.name,
                role="replacement_metadata",
                approved_roots=approved_roots,
            )
        )


def _append_historical_configs(
    items: list[CopyItem],
    config: DatasetConfig,
    approved_roots: tuple[Path, ...],
) -> None:
    for source_key, root in _metadata_roots(config).items():
        config_root = root / "_pipeline/configs"
        if not config_root.is_dir() or config_root.is_symlink():
            continue
        for source in sorted(config_root.rglob("*"), key=lambda path: path.as_posix()):
            if not source.is_file() and not source.is_symlink():
                continue
            relative = source.relative_to(config_root)
            items.append(
                _copy_item(
                    key=f"historical_config:{source_key}:{relative.as_posix()}",
                    source=source,
                    destination=Path("provenance/configs/historical")
                    / source_key
                    / relative,
                    role="historical_config",
                    approved_roots=approved_roots,
                )
            )


def _append_v3_headmodels(
    items: list[CopyItem],
    config: DatasetConfig,
    approved_roots: tuple[Path, ...],
) -> None:
    for subject in config.subjects:
        replacement = subject == config.replacement_subject
        root = (
            config.sources.replacement_subject / "simnibs326/headmodels"
            if replacement
            else config.sources.v3_headmodels
        )
        prefix = Path("headmodels/simnibs326") / subject
        provenance_prefix = Path("provenance/headmodels/simnibs326") / subject
        records = (
            ("subject", root / subject, prefix, "v3_headmodel_subject"),
            (
                "label",
                root / "harmonized_labels" / f"{subject}_v3_harmonized_t1fs_grid.nii.gz",
                provenance_prefix / "harmonized_label.nii.gz",
                "v3_headmodel_label",
            ),
            (
                "qc",
                root / "harmonized_labels/qc" / f"{subject}.json",
                provenance_prefix / "harmonization_qc.json",
                "v3_headmodel_qc",
            ),
            (
                "figure",
                root / "harmonized_labels/figures" / f"{subject}.png",
                provenance_prefix / "harmonization_qc.png",
                "v3_headmodel_qc",
            ),
            (
                "receipts",
                root / "_pipeline/subjects" / subject,
                provenance_prefix / "receipts",
                "v3_headmodel_receipts",
            ),
        )
        for suffix, source, destination, role in records:
            items.append(
                _copy_item(
                    key=f"v3_headmodel:{subject}:{suffix}",
                    source=source,
                    destination=destination,
                    role=role,
                    approved_roots=approved_roots,
                    subject=subject,
                    version="simnibs326",
                )
            )

    cohort_roots = (
        ("base9", config.sources.v3_headmodels, "base9"),
        (
            "replacement",
            config.sources.replacement_subject / "simnibs326/headmodels",
            "external1",
        ),
    )
    for key, root, cohort in cohort_roots:
        for suffix, source in (
            ("headmodels", root / "_pipeline/headmodels" / cohort),
            ("reference_stage_collection.json", root / "_pipeline/reference_stage_collection.json"),
        ):
            items.append(
                _copy_item(
                    key=f"v3_headmodel_cohort:{key}:{suffix}",
                    source=source,
                    destination=Path("provenance/source_roots")
                    / f"v3_headmodels_{key}"
                    / suffix,
                    role="v3_headmodel_cohort_metadata",
                    approved_roots=approved_roots,
                )
            )
    replacement_analysis = (
        config.sources.replacement_subject
        / "simnibs326/headmodels/_pipeline/analysis/"
        "repeatability_optimizer_roi_paired_external1_v2"
    )
    items.append(
        _copy_item(
            key="analysis:replacement_v3_final",
            source=replacement_analysis,
            destination=Path(
                "analyses/replacement_subject/simnibs326_repeatability_analysis"
            ),
            role="replacement_v3_analysis",
            approved_roots=approved_roots,
            subject=config.replacement_subject,
            version="simnibs326",
        )
    )


def _append_v4_scaffolds(
    items: list[CopyItem],
    config: DatasetConfig,
    approved_roots: tuple[Path, ...],
) -> None:
    for subject in config.subjects:
        items.append(
            _copy_item(
                key=f"v4_scaffold:{subject}:subject",
                source=config.sources.v4_scaffolds / "subjects" / subject,
                destination=Path("headmodels/simnibs4") / subject / "subject_tree",
                role="v4_scaffold_subject",
                approved_roots=approved_roots,
                subject=subject,
                version="simnibs4",
            )
        )
        items.append(
            _copy_item(
                key=f"v4_scaffold:{subject}:provenance",
                source=config.sources.v4_scaffolds / "results" / f"{subject}.json",
                destination=Path("headmodels/simnibs4") / subject / "provenance.json",
                role="v4_scaffold_provenance",
                approved_roots=approved_roots,
                subject=subject,
                version="simnibs4",
            )
        )


def _validate_destinations(items: list[CopyItem]) -> None:
    seen: dict[Path, str] = {}
    for item in items:
        previous = seen.get(item.destination)
        if previous is not None:
            raise PlanningError(
                f"Duplicate destination {item.destination}: {previous}, {item.key}"
            )
        seen[item.destination] = item.key


def build_copy_plan(config: DatasetConfig) -> ConsolidationPlan:
    """Resolve the complete approved source-to-destination mapping."""
    approved_roots = _approved_roots(config)
    output = config.output_root.resolve(strict=False)
    for source in approved_roots:
        if _overlap(output, source):
            raise PlanningError(f"Dataset output overlaps source: {output} and {source}")
    if config.excluded_subject in config.subjects:
        raise PlanningError("Excluded subject appears in the canonical cohort")
    if config.replacement_subject not in config.subjects:
        raise PlanningError("Replacement subject is absent from the canonical cohort")

    items: list[CopyItem] = []
    for version in ("simnibs326", "simnibs4"):
        for target in TARGETS:
            for condition in CONDITIONS:
                for subject in config.subjects:
                    items.append(
                        _copy_item(
                            key=f"cell:{version}:{target}:{condition}:{subject}",
                            source=_scientific_source(
                                config, version, target, subject, condition
                            ),
                            destination=_scientific_destination(
                                version, target, subject, condition
                            ),
                            role="scientific_condition",
                            approved_roots=approved_roots,
                            subject=subject,
                            version=version,
                            target=target,
                            condition=condition,
                        )
                    )
    _append_source_metadata(items, config, approved_roots)
    _append_historical_configs(items, config, approved_roots)
    _append_v3_headmodels(items, config, approved_roots)
    _append_v4_scaffolds(items, config, approved_roots)
    items.extend(
        (
            _copy_item(
                key="analysis:paper_update_v1",
                source=config.sources.paper_analysis,
                destination=Path("analyses/repeatability_paper_update_v1"),
                role="paper_analysis",
                approved_roots=approved_roots,
            ),
            _copy_item(
                key="supplemental:nested_40x40_v1",
                source=config.sources.nested_analysis,
                destination=Path("supplemental/nested_40x40_v1"),
                role="nested_dataset",
                approved_roots=approved_roots,
            ),
        )
    )
    _validate_destinations(items)
    ordered = tuple(sorted(items, key=lambda item: item.key))
    return ConsolidationPlan(
        schema_version=1,
        output_root=output,
        subjects=config.subjects,
        excluded_subject=config.excluded_subject,
        replacement_subject=config.replacement_subject,
        repeat_count=config.repeat_count,
        expected_cells=config.expected_cells,
        expected_simulations=config.expected_simulations,
        items=ordered,
    )


def plan_sha256(plan: ConsolidationPlan) -> str:
    payload = json.dumps(
        plan.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
