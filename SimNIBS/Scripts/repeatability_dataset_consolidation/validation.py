"""Read-only scientific and source-identity validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .inventory import sha256_file
from .planning import ConsolidationPlan, CopyItem
from .settings import CONDITIONS, TARGETS, VERSIONS


class ScopeValidationError(RuntimeError):
    """Raised when planned scientific data are incomplete or no longer bound."""


@dataclass(frozen=True)
class ScopeValidation:
    subject_count: int
    cell_count: int
    simulation_count: int
    v3_headmodel_count: int
    v4_scaffold_count: int
    terminal_receipts: tuple[str, ...]

    def as_dict(self) -> dict[str, int]:
        return {
            "subjects": self.subject_count,
            "cells": self.cell_count,
            "simulations": self.simulation_count,
            "v3_headmodels": self.v3_headmodel_count,
            "v4_scaffolds": self.v4_scaffold_count,
        }


def validate_item_identity(item: CopyItem) -> None:
    source = item.source
    if not source.exists() and not source.is_symlink():
        if item.required:
            raise ScopeValidationError(f"Required planned source is missing: {source}")
        return
    info = source.lstat()
    if item.source_device and int(info.st_dev) != item.source_device:
        raise ScopeValidationError(f"Planned source device changed: {source}")
    if item.source_inode and int(info.st_ino) != item.source_inode:
        raise ScopeValidationError(f"Planned source inode changed: {source}")
    if item.source_size and int(info.st_size) != item.source_size:
        raise ScopeValidationError(f"Planned source size changed: {source}")
    if item.source_mtime_ns and int(info.st_mtime_ns) != item.source_mtime_ns:
        raise ScopeValidationError(f"Planned source modification time changed: {source}")
    if item.source_ctime_ns and int(info.st_ctime_ns) != item.source_ctime_ns:
        raise ScopeValidationError(f"Planned source status time changed: {source}")


def _experiment_root(item: CopyItem) -> Path:
    try:
        return item.source.parents[1]
    except IndexError as exc:
        raise ScopeValidationError(f"Scientific source has no experiment root: {item.source}") from exc


def _normalized_target(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def _receipt_integer(value: object, label: str, receipt: Path) -> int:
    if isinstance(value, bool):
        raise ScopeValidationError(f"Terminal source receipt has invalid {label}: {receipt}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ScopeValidationError(
            f"Terminal source receipt has invalid {label}: {receipt}"
        ) from exc
    if result < 0:
        raise ScopeValidationError(f"Terminal source receipt has invalid {label}: {receipt}")
    return result


def _terminal_receipt(
    experiment_root: Path,
    *,
    target: str,
    subjects: set[str],
    conditions: set[str],
    repeat_count: int,
) -> Path:
    receipt = experiment_root / "_pipeline/workflow/complete.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ScopeValidationError(f"Terminal source receipt is missing: {receipt}")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeValidationError(f"Terminal source receipt is unreadable: {receipt}") from exc
    if payload.get("status") != "complete":
        raise ScopeValidationError(f"Terminal source receipt is not terminal: {receipt}")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ScopeValidationError(f"Terminal source receipt has no scope: {receipt}")
    for stage in (payload.get("stage"), scope.get("stage")):
        if stage is not None and str(stage) != "final":
            raise ScopeValidationError(
                f"Terminal source receipt is not final: {receipt}"
            )

    observed_target = scope.get("target", scope.get("roi"))
    if _normalized_target(observed_target) != _normalized_target(target):
        raise ScopeValidationError(
            f"Terminal source receipt target differs from planned data: {receipt}"
        )
    subject_count = _receipt_integer(
        scope.get("subject_count"), "subject count", receipt
    )
    if subject_count < len(subjects):
        raise ScopeValidationError(
            f"Terminal source receipt subject scope is incomplete: {receipt}"
        )
    receipt_subjects = scope.get("subjects")
    if receipt_subjects is not None:
        if (
            not isinstance(receipt_subjects, list)
            or len(receipt_subjects) != subject_count
            or len(set(map(str, receipt_subjects))) != subject_count
            or not subjects.issubset(set(map(str, receipt_subjects)))
        ):
            raise ScopeValidationError(
                f"Terminal source receipt subject scope is incomplete: {receipt}"
            )
    observed_repeats = _receipt_integer(
        scope.get("repeats_per_condition"), "repeat count", receipt
    )
    if observed_repeats != repeat_count:
        raise ScopeValidationError(
            f"Terminal source receipt repeat count differs from planned data: {receipt}"
        )

    declared_conditions = scope.get("conditions")
    if declared_conditions is not None and not isinstance(declared_conditions, dict):
        raise ScopeValidationError(
            f"Terminal source receipt has invalid condition scope: {receipt}"
        )
    task_counts: dict[str, int] = {}
    condition_names = set(declared_conditions or {})
    condition_names.update(
        condition
        for condition in CONDITIONS
        if f"{condition}_tasks" in scope
    )
    if not conditions.issubset(condition_names):
        raise ScopeValidationError(
            f"Terminal source receipt condition scope is incomplete: {receipt}"
        )
    expected_per_condition = subject_count * repeat_count
    for condition in condition_names:
        task_key = f"{condition}_tasks"
        if task_key in scope:
            count = _receipt_integer(scope[task_key], task_key, receipt)
            if count != expected_per_condition:
                raise ScopeValidationError(
                    f"Terminal source receipt condition task count differs: {receipt}"
                )
            task_counts[condition] = count
            continue
        count = _receipt_integer(
            declared_conditions[condition], f"{condition} count", receipt
        )
        if count == repeat_count:
            task_counts[condition] = expected_per_condition
        elif count == expected_per_condition:
            task_counts[condition] = count
        else:
            raise ScopeValidationError(
                f"Terminal source receipt condition task count differs: {receipt}"
            )

    expected_total = sum(task_counts.values())
    total_fields = (
        "expected_ti_msh",
        "expected_ti_nifti",
        "total_simulation_tasks",
    )
    observed_totals = [
        _receipt_integer(scope[field], field, receipt)
        for field in total_fields
        if field in scope
    ]
    if not observed_totals:
        raise ScopeValidationError(
            f"Terminal source receipt has no scientific task count: {receipt}"
        )
    if any(total != expected_total for total in observed_totals):
        raise ScopeValidationError(
            f"Terminal source receipt task count differs from its scope: {receipt}"
        )
    return receipt.resolve()


def _validate_repeats(item: CopyItem, repeat_count: int) -> int:
    repeats = item.source / "repeats"
    if not repeats.is_dir() or repeats.is_symlink():
        raise ScopeValidationError(f"Repeat directory is missing or unsafe: {repeats}")
    observed = {path.name for path in repeats.iterdir() if path.is_dir() and not path.is_symlink()}
    expected = {f"repeat_{index:03d}" for index in range(1, repeat_count + 1)}
    if observed != expected:
        raise ScopeValidationError(
            f"Unexpected repeat tags for {item.key}: expected {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )
    for tag in sorted(expected):
        meshes = [
            path
            for path in (repeats / tag).rglob("TI.msh")
            if path.is_file() and not path.is_symlink()
        ]
        if len(meshes) != 1:
            raise ScopeValidationError(
                f"{item.key}/{tag}: expected exactly one TI.msh, observed {len(meshes)}"
            )
    return len(expected)


def _validate_replacement_analysis(plan: ConsolidationPlan) -> None:
    analyses = [
        item for item in plan.items if item.role == "replacement_v3_analysis"
    ]
    if len(analyses) != 1:
        raise ScopeValidationError(
            "Replacement v3 analysis scope must contain exactly one item"
        )
    root = analyses[0].source
    manifest = root / "results/analysis_manifest.json"
    archive = root / "simnibs326h_repeatability_analysis.tar.gz"
    checksum = root / "simnibs326h_repeatability_analysis.tar.gz.sha256"
    for path in (manifest, archive, checksum):
        if not path.is_file() or path.is_symlink():
            raise ScopeValidationError(
                f"Replacement v3 analysis artifact is missing or unsafe: {path}"
            )
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeValidationError(
            f"Replacement v3 analysis manifest is unreadable: {manifest}"
        ) from exc
    if manifest_payload.get("status") != "complete":
        raise ScopeValidationError("Replacement v3 analysis manifest is not complete")
    try:
        fields = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)
    except OSError as exc:
        raise ScopeValidationError(
            f"Replacement v3 analysis checksum is unreadable: {checksum}"
        ) from exc
    recorded_name = fields[1].lstrip("*") if len(fields) == 2 else ""
    recorded_path = Path(recorded_name)
    recorded_archive = (
        recorded_path
        if recorded_path.is_absolute()
        else checksum.parent / recorded_path
    ).resolve(strict=False)
    if (
        len(fields) != 2
        or len(fields[0]) != 64
        or any(character not in "0123456789abcdef" for character in fields[0])
        or recorded_archive != archive.resolve()
    ):
        raise ScopeValidationError(
            "Replacement v3 analysis checksum receipt is invalid"
        )
    if sha256_file(archive) != fields[0]:
        raise ScopeValidationError(
            "Replacement v3 analysis archive checksum does not match"
        )


def validate_scientific_scope(plan: ConsolidationPlan) -> ScopeValidation:
    """Validate exact cohort cells, repeats, receipts, and supporting bundles."""
    for item in plan.items:
        validate_item_identity(item)

    scientific = [item for item in plan.items if item.role == "scientific_condition"]
    for item in scientific:
        if item.subject == plan.excluded_subject:
            raise ScopeValidationError(
                f"Scientific item names excluded subject: {item.key}"
            )
        if item.subject not in plan.subjects:
            raise ScopeValidationError(f"Scientific item has unknown subject: {item.key}")

    observed_cells = {
        (item.version, item.target, item.condition, item.subject) for item in scientific
    }
    expected_cells = {
        (version, target, condition, subject)
        for version in VERSIONS
        for target in TARGETS
        for condition in CONDITIONS
        for subject in plan.subjects
    }
    if len(scientific) != len(observed_cells) or observed_cells != expected_cells:
        raise ScopeValidationError(
            f"Scientific cell scope differs: expected {len(expected_cells)}, "
            f"observed {len(scientific)} items/{len(observed_cells)} unique"
        )
    if len(expected_cells) != plan.expected_cells:
        raise ScopeValidationError("Plan expected-cell count is inconsistent")

    receipt_groups: dict[Path, list[CopyItem]] = {}
    for item in scientific:
        receipt_groups.setdefault(_experiment_root(item), []).append(item)
    receipts: list[Path] = []
    for experiment_root, items in sorted(
        receipt_groups.items(), key=lambda pair: str(pair[0])
    ):
        targets = {item.target for item in items}
        if len(targets) != 1 or None in targets:
            raise ScopeValidationError(
                f"Scientific source root mixes targets: {experiment_root}"
            )
        receipts.append(
            _terminal_receipt(
                experiment_root,
                target=str(next(iter(targets))),
                subjects={str(item.subject) for item in items},
                conditions={str(item.condition) for item in items},
                repeat_count=plan.repeat_count,
            )
        )
    simulations = sum(_validate_repeats(item, plan.repeat_count) for item in scientific)
    if simulations != plan.expected_simulations:
        raise ScopeValidationError(
            f"Simulation count differs: expected {plan.expected_simulations}, "
            f"observed {simulations}"
        )

    v3_subjects = {
        item.subject for item in plan.items if item.role == "v3_headmodel_subject"
    }
    v4_subjects = {
        item.subject for item in plan.items if item.role == "v4_scaffold_subject"
    }
    expected_subjects = set(plan.subjects)
    if v3_subjects != expected_subjects:
        raise ScopeValidationError("Selected v3 head-model bundle scope is incomplete")
    if v4_subjects != expected_subjects:
        raise ScopeValidationError("Selected v4 scaffold bundle scope is incomplete")
    _validate_replacement_analysis(plan)

    return ScopeValidation(
        subject_count=len(plan.subjects),
        cell_count=len(expected_cells),
        simulation_count=simulations,
        v3_headmodel_count=len(v3_subjects),
        v4_scaffold_count=len(v4_subjects),
        terminal_receipts=tuple(str(path) for path in sorted(set(receipts))),
    )
