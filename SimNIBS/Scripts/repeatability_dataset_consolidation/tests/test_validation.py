from dataclasses import replace
import json

import pytest

from repeatability_dataset_consolidation.planning import CopyItem, build_copy_plan
from repeatability_dataset_consolidation.validation import (
    ScopeValidationError,
    validate_scientific_scope,
)
from repeatability_dataset_consolidation.workflow import preflight


def _first_cell(plan):
    return next(item for item in plan.items if item.role == "scientific_condition")


def test_preflight_validates_all_cells_repeats_receipts_and_headmodels(
    miniature_config,
) -> None:
    result = preflight(miniature_config)

    assert result["status"] == "ready"
    assert result["scope"] == {
        "subjects": 2,
        "cells": 16,
        "simulations": 32,
        "v3_headmodels": 2,
        "v4_scaffolds": 2,
    }
    assert result["source_outputs_modified"] is False
    assert result["free_bytes"] > 0
    assert not miniature_config.output_root.exists()


@pytest.mark.parametrize("observed", [1, 3])
def test_scope_rejects_wrong_repeat_count(miniature_config, observed: int) -> None:
    plan = build_copy_plan(miniature_config)
    cell = _first_cell(plan)
    repeats = cell.source / "repeats"
    if observed == 1:
        repeat = repeats / "repeat_002"
        for path in sorted(repeat.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        repeat.rmdir()
    else:
        mesh = (
            repeats
            / "repeat_003"
            / cell.subject
            / "anat/SimNIBS/Output"
            / cell.subject
            / "TI.msh"
        )
        mesh.parent.mkdir(parents=True)
        mesh.write_text("extra\n", encoding="utf-8")

    with pytest.raises(ScopeValidationError, match="repeat tags"):
        validate_scientific_scope(plan)


def test_scope_rejects_missing_field_mesh_and_incomplete_receipt(
    miniature_config,
) -> None:
    plan = build_copy_plan(miniature_config)
    cell = _first_cell(plan)
    mesh = next((cell.source / "repeats/repeat_001").rglob("TI.msh"))
    mesh.unlink()
    with pytest.raises(ScopeValidationError, match="exactly one TI.msh"):
        validate_scientific_scope(plan)

    mesh.write_text("restored\n", encoding="utf-8")
    experiment_root = cell.source.parents[1]
    receipt = experiment_root / "_pipeline/workflow/complete.json"
    receipt.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(ScopeValidationError, match="not terminal"):
        validate_scientific_scope(plan)


def test_scope_rejects_changed_headmodel_and_excluded_subject_item(
    miniature_config,
) -> None:
    plan = build_copy_plan(miniature_config)
    headmodel = next(item for item in plan.items if item.role == "v3_headmodel_subject")
    moved = headmodel.source.with_name(headmodel.source.name + "-moved")
    headmodel.source.rename(moved)
    with pytest.raises(ScopeValidationError, match="planned source is missing"):
        validate_scientific_scope(plan)
    moved.rename(headmodel.source)
    plan = build_copy_plan(miniature_config)

    excluded = CopyItem(
        key="cell:excluded",
        source=_first_cell(plan).source,
        destination=_first_cell(plan).destination,
        role="scientific_condition",
        subject=miniature_config.excluded_subject,
        version="simnibs4",
        target="left_hippocampus",
        condition="remesh",
    )
    poisoned = replace(plan, items=plan.items + (excluded,))
    with pytest.raises(ScopeValidationError, match="excluded subject"):
        validate_scientific_scope(poisoned)
