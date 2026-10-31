"""Read-only scientific and source-identity validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

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


def _terminal_receipt(experiment_root: Path) -> Path:
    receipt = experiment_root / "_pipeline/workflow/complete.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ScopeValidationError(f"Terminal source receipt is missing: {receipt}")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeValidationError(f"Terminal source receipt is unreadable: {receipt}") from exc
    if payload.get("status") != "complete":
        raise ScopeValidationError(f"Terminal source receipt is not terminal: {receipt}")
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

    receipts = sorted({_terminal_receipt(_experiment_root(item)) for item in scientific})
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

    return ScopeValidation(
        subject_count=len(plan.subjects),
        cell_count=len(expected_cells),
        simulation_count=simulations,
        v3_headmodel_count=len(v3_subjects),
        v4_scaffold_count=len(v4_subjects),
        terminal_receipts=tuple(str(path) for path in receipts),
    )
