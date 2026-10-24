from __future__ import annotations

from pathlib import Path

import pytest

import simnibs_subject_replacement.v4_workflow as workflow
from simnibs_subject_replacement.replacement import load_prepared_manifest, prepare
from simnibs_subject_replacement.settings import ReplacementConfig, replacement_paths


def _config(layout: dict[str, Path | str]) -> ReplacementConfig:
    return ReplacementConfig(
        subject=str(layout["subject"]),
        replaces=str(layout["replaces"]),
        final132_subjects_file=Path(layout["subjects_file"]),
        corrected_v4_scaffold_root=Path(layout["scaffold"]),
        legacy_v3_root=Path(layout["legacy"]),
        atlas_root=Path(layout["atlas"]),
        targets_csv=Path(layout["targets"]),
        work_root=Path(layout["work"]),
    )


def test_initialize_v4_target_delegates_exact_singleton_contract(
    replacement_layout: dict[str, Path | str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    prepare(config)
    manifest = load_prepared_manifest(paths.manifest)
    from simnibs_subject_replacement.tests.test_replacement import _write_v3_promotion

    _write_v3_promotion(manifest)
    calls: list[list[str]] = []
    monkeypatch.setattr(workflow.v4_pipeline, "main", lambda argv: calls.append(argv) or 0)

    result = workflow.initialize_v4_target(manifest, "left-hippocampus")

    assert result == 0
    assert calls == [[
        "init",
        "--source-root", str(paths.v4_corrected_root),
        "--experiment-root", str(paths.v4_left_root),
        "--subjects", config.subject,
        "--repeat-count", "40",
        "--roi-preset", "left-hippocampus",
        "--montage-preset", "left-hippocampus",
        "--targets-csv", str(config.targets_csv.resolve()),
        "--atlas-dir", str(config.atlas_root.resolve()),
        "--compare-metric", "median_roi",
    ]]


def test_v4_run_requires_promotion_and_right_requires_left_completion(
    replacement_layout: dict[str, Path | str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simnibs_subject_replacement.tests.test_replacement import (
        _write_left_completion,
        _write_v3_promotion,
    )

    config = _config(replacement_layout)
    paths = replacement_paths(config)
    prepare(config)
    manifest = load_prepared_manifest(paths.manifest)
    calls: list[list[str]] = []
    monkeypatch.setattr(workflow.v4_pipeline, "main", lambda argv: calls.append(argv) or 0)

    with pytest.raises(RuntimeError, match="promotion"):
        workflow.run_v4_target(manifest, "left-hippocampus", dry_run=True)
    _write_v3_promotion(manifest)
    assert workflow.run_v4_target(manifest, "left-hippocampus", dry_run=True) == 0
    assert calls[-1] == [
        "submit-all", "--experiment-root", str(paths.v4_left_root),
        "--max-concurrent", "50", "--analysis-max-concurrent", "1", "--dry-run",
    ]

    with pytest.raises(RuntimeError, match="left-hippocampus"):
        workflow.run_v4_target(manifest, "right-m1", dry_run=False)
    _write_left_completion(manifest, "v4")
    assert workflow.run_v4_target(manifest, "right-m1", dry_run=False) == 0
    assert calls[-1] == [
        "submit-all", "--experiment-root", str(paths.v4_right_root),
        "--max-concurrent", "50", "--analysis-max-concurrent", "1",
    ]


def test_v4_rejects_unknown_target(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    prepare(config)
    manifest = load_prepared_manifest(paths.manifest)

    with pytest.raises(ValueError, match="target"):
        workflow.initialize_v4_target(manifest, "other")


def test_status_reports_receipt_presence_without_claiming_validation(
    replacement_layout: dict[str, Path | str],
) -> None:
    config = _config(replacement_layout)
    paths = replacement_paths(config)
    prepare(config)
    manifest = load_prepared_manifest(paths.manifest)
    receipt = paths.v4_left_root / "_pipeline/workflow/complete.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")

    status = workflow.replacement_status(manifest)
    left = status["targets"]["v4:left-hippocampus"]

    assert left["terminal_receipt_present"] is True
    assert "complete" not in left
