from dataclasses import replace
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.planning import (
    PlanningError,
    build_copy_plan,
    plan_sha256,
)


def test_plan_maps_every_cell_and_supporting_bundle_once(miniature_config) -> None:
    plan = build_copy_plan(miniature_config)
    scientific = [item for item in plan.items if item.role == "scientific_condition"]

    assert len(scientific) == 16
    assert len({item.destination for item in plan.items}) == len(plan.items)
    assert {
        (item.version, item.target, item.condition, item.subject)
        for item in scientific
    } == {
        (version, target, condition, subject)
        for version in ("simnibs326", "simnibs4")
        for target in ("left_hippocampus", "right_m1")
        for condition in ("remesh", "fixed_mesh")
        for subject in miniature_config.subjects
    }
    assert len([item for item in plan.items if item.role == "v3_headmodel_subject"]) == 2
    assert len([item for item in plan.items if item.role == "v4_scaffold_subject"]) == 2
    assert len([item for item in plan.items if item.role == "v4_scaffold_provenance"]) == 2
    assert {item.role for item in plan.items} >= {
        "paper_analysis",
        "nested_dataset",
        "source_metadata",
        "replacement_metadata",
        "v3_headmodel_cohort_metadata",
    }
    assert plan.expected_cells == 16
    assert plan.expected_simulations == 32
    assert len(plan_sha256(plan)) == 64
    assert plan_sha256(plan) == plan_sha256(plan)


def test_plan_rejects_output_overlap_with_a_source(miniature_config) -> None:
    overlapping = replace(
        miniature_config,
        output_root=miniature_config.sources.v3_left / "combined",
    )

    with pytest.raises(PlanningError, match="overlaps source"):
        build_copy_plan(overlapping)


def test_plan_rejects_duplicate_destinations(miniature_config, monkeypatch) -> None:
    import repeatability_dataset_consolidation.planning as planning

    original = planning._scientific_destination
    monkeypatch.setattr(
        planning,
        "_scientific_destination",
        lambda *args, **kwargs: Path("data/collision"),
    )
    try:
        with pytest.raises(PlanningError, match="Duplicate destination"):
            build_copy_plan(miniature_config)
    finally:
        monkeypatch.setattr(planning, "_scientific_destination", original)


def test_plan_rejects_broken_and_escaping_symlinks(miniature_config) -> None:
    source = (
        miniature_config.sources.v3_left
        / "sub-CC110174_repeatability"
        / "remesh"
    )
    broken = source / "broken"
    broken.symlink_to(source / "missing")
    with pytest.raises(PlanningError, match="Broken symlink"):
        build_copy_plan(miniature_config)
    broken.unlink()

    outside = miniature_config.output_root.parent.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    escaping = source / "escaping"
    escaping.symlink_to(outside)
    with pytest.raises(PlanningError, match="outside approved sources"):
        build_copy_plan(miniature_config)


def test_plan_rejects_traversal_in_generated_destination(
    miniature_config, monkeypatch
) -> None:
    import repeatability_dataset_consolidation.planning as planning

    monkeypatch.setattr(
        planning,
        "_scientific_destination",
        lambda *args, **kwargs: Path("../escape"),
    )
    with pytest.raises(PlanningError, match="Unsafe destination"):
        build_copy_plan(miniature_config)
