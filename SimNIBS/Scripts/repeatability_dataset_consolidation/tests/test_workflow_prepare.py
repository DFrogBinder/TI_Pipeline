from dataclasses import replace
import json
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.workflow import WorkflowError, prepare


def test_prepare_atomically_creates_owned_dataset_skeleton(miniature_config) -> None:
    source_probe = miniature_config.sources.paper_analysis / "analysis.json"
    before = (source_probe.read_bytes(), source_probe.stat().st_mtime_ns)

    result = prepare(miniature_config)

    assert result["status"] == "prepared"
    assert result["action"] == "created"
    assert result["source_outputs_modified"] is False
    root = miniature_config.output_root
    required = {
        "README.md",
        "subjects.txt",
        "dataset_manifest.json",
        "data/simnibs326/left_hippocampus",
        "data/simnibs326/right_m1",
        "data/simnibs4/left_hippocampus",
        "data/simnibs4/right_m1",
        "headmodels/simnibs326",
        "headmodels/simnibs4",
        "analyses",
        "supplemental",
        "provenance/configs/historical",
        "provenance/configs/canonical",
        "_consolidation/owner.json",
        "_consolidation/planned_items.json",
        "_consolidation/verification",
    }
    assert all((root / relative).exists() for relative in required)
    assert (root / "subjects.txt").read_text().splitlines() == list(
        miniature_config.subjects
    )
    owner = json.loads((root / "_consolidation/owner.json").read_text())
    planned = json.loads((root / "_consolidation/planned_items.json").read_text())
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    assert owner["plan_sha256"] == planned["plan_sha256"]
    assert manifest["plan_sha256"] == owner["plan_sha256"]
    assert manifest["scope"]["simulations"] == 32
    assert (source_probe.read_bytes(), source_probe.stat().st_mtime_ns) == before


def test_prepare_reuses_only_an_exact_owned_state(miniature_config) -> None:
    created = prepare(miniature_config)
    reused = prepare(miniature_config)

    assert reused == {**created, "action": "reused"}

    (miniature_config.output_root / "_consolidation/planned_items.json").unlink()
    with pytest.raises(WorkflowError, match="incomplete owned dataset root"):
        prepare(miniature_config)


def test_prepare_rejects_unowned_and_symlinked_roots(miniature_config) -> None:
    miniature_config.output_root.mkdir()
    (miniature_config.output_root / "unowned.txt").write_text("no\n", encoding="utf-8")
    with pytest.raises(WorkflowError, match="unowned dataset root"):
        prepare(miniature_config)

    for child in miniature_config.output_root.iterdir():
        child.unlink()
    miniature_config.output_root.rmdir()
    target = miniature_config.output_root.with_name("link-target")
    target.mkdir()
    miniature_config.output_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(WorkflowError, match="symbolic-link dataset root"):
        prepare(miniature_config)


def test_prepare_rejects_existing_owner_after_source_identity_changes(
    miniature_config,
) -> None:
    prepare(miniature_config)
    source = miniature_config.sources.paper_analysis
    original = source / "analysis.json"
    original.unlink()
    source.rmdir()
    source.mkdir()
    (source / "analysis.json").write_text("changed\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="different consolidation plan"):
        prepare(miniature_config)


def test_prepare_propagates_source_overlap_failure(miniature_config) -> None:
    config = replace(
        miniature_config,
        output_root=miniature_config.sources.paper_analysis / "dataset",
    )
    with pytest.raises(Exception, match="overlaps source"):
        prepare(config)
