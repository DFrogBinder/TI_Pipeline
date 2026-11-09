from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import simnibs_segmentation_determinism.workflow as workflow
from simnibs_segmentation_determinism.settings import scope_payload
from simnibs_segmentation_determinism.workflow import (
    WorkflowError,
    collect,
    initialize,
    run_repeat,
    status,
)


def _save_map(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.full((2, 2, 2), value, dtype=np.uint8), np.eye(4)),
        str(path),
    )


def _install_synthetic_receipts(config, monkeypatch, missing=None, retry=None):
    receipts = {}
    for arm in ("simnibs326_headreco", "simnibs401_charm"):
        for repeat in range(1, 11):
            key = (arm, repeat)
            if key == missing:
                continue
            root = config.output_root / arm / f"repeat_{repeat:03d}"
            attempt = 2 if key == retry else 1
            if arm == "simnibs326_headreco":
                path = root / "artifacts/labels.nii.gz"
                _save_map(path, 1)
                receipt = {
                    "status": "complete",
                    "arm": arm,
                    "repeat_index": repeat,
                    "attempt": attempt,
                    "artifacts": [
                        {"relative_path": "labels.nii.gz", "sha256": "fixture", "size_bytes": path.stat().st_size}
                    ],
                }
            else:
                path = root / "tissue_labeling_upsampled.nii.gz"
                _save_map(path, 2)
                receipt = {
                    "status": "complete",
                    "arm": arm,
                    "repeat_index": repeat,
                    "attempt": attempt,
                    "artifact": {
                        "relative_path": "tissue_labeling_upsampled.nii.gz",
                        "sha256": "fixture",
                        "size_bytes": path.stat().st_size,
                    },
                }
            root.mkdir(parents=True, exist_ok=True)
            (root / "complete.json").write_text(json.dumps(receipt), encoding="utf-8")
            receipts[key] = receipt

    def validate_v3(request):
        key = ("simnibs326_headreco", request.repeat_index)
        if key not in receipts:
            raise workflow.HeadrecoRunError("missing")
        return receipts[key]

    def validate_v4(request):
        key = ("simnibs401_charm", request.repeat_index)
        if key not in receipts:
            raise workflow.CharmRunError("missing")
        return receipts[key]

    monkeypatch.setattr(workflow, "validate_headreco_repeat", validate_v3)
    monkeypatch.setattr(workflow, "validate_charm_repeat", validate_v4)
    return receipts


def test_scope_and_preflight_are_exact_and_read_only(configured_experiment):
    config = configured_experiment
    payload = workflow.workflow_preflight(config)

    assert scope_payload(config)["scientific_segmentation_tasks"] == 20
    assert scope_payload(config)["scheduler_tasks"] == 23
    assert payload["status"] == "ready"
    assert not config.output_root.exists()


def test_initialize_is_idempotent_only_for_identical_plan(configured_experiment):
    config = configured_experiment

    created = initialize(config)
    reused = initialize(config)

    assert created["action"] == "created"
    assert reused["action"] == "reused"
    assert created["plan_sha256"] == reused["plan_sha256"]
    assert (config.output_root / "_pipeline/config.json").is_file()
    assert (config.output_root / "_pipeline/source_manifest.json").is_file()

    with pytest.raises(WorkflowError, match="plan"):
        initialize(replace(config, repeats_per_arm=9))


@pytest.mark.parametrize(
    ("arm", "index", "expected_repeat"),
    [("simnibs326_headreco", 0, 1), ("simnibs401_charm", 9, 10)],
)
def test_run_repeat_maps_array_index_without_changing_subject(
    configured_experiment, monkeypatch, arm, index, expected_repeat
):
    config = configured_experiment
    initialize(config)
    captured = []

    def fake_runner(request):
        captured.append(request)
        return {"status": "complete", "repeat_index": request.repeat_index}

    monkeypatch.setattr(
        workflow,
        "run_headreco_repeat" if arm == "simnibs326_headreco" else "run_charm_repeat",
        fake_runner,
    )

    result = run_repeat(config, arm=arm, array_index=index, attempt=1)

    assert result["repeat_index"] == expected_repeat
    assert captured[0].inputs.subject == "sub-CC110174"
    assert captured[0].repeat_root.name == f"repeat_{expected_repeat:03d}"


def test_collect_with_19_receipts_is_incomplete_and_has_no_terminal_receipt(
    configured_experiment, monkeypatch
):
    config = configured_experiment
    initialize(config)
    _install_synthetic_receipts(
        config,
        monkeypatch,
        missing=("simnibs401_charm", 10),
    )

    result = collect(config)

    assert result["status"] == "incomplete"
    assert result["validated_repeats"] == 19
    assert len(result["failures"]) == 1
    assert (config.output_root / "analysis/incomplete_audit.json").is_file()
    assert not (config.output_root / "_pipeline/complete.json").exists()
    assert status(config)["state"] == "incomplete"


def test_collect_20_receipts_writes_outputs_and_preserves_retry_metadata(
    configured_experiment, monkeypatch
):
    config = configured_experiment
    initialize(config)
    _install_synthetic_receipts(
        config,
        monkeypatch,
        retry=("simnibs326_headreco", 4),
    )

    result = collect(config)

    assert result["status"] == "complete"
    assert result["validated_repeats"] == 20
    assert result["retried_repeats"] == [
        {"arm": "simnibs326_headreco", "repeat_index": 4, "attempt": 2}
    ]
    expected = (
        "determinism_manifest.json",
        "artifact_inventory.tsv",
        "repeat_comparisons.tsv",
        "tissue_summary.tsv",
        "changed_voxel_summary.tsv",
        "report.md",
    )
    for name in expected:
        assert (config.output_root / "analysis" / name).is_file()
    terminal = config.output_root / "_pipeline/complete.json"
    assert terminal.is_file()
    assert status(config)["state"] == "complete"


def test_status_distinguishes_uninitialized_prepared_and_in_progress(
    configured_experiment
):
    config = configured_experiment
    assert status(config)["state"] == "uninitialized"
    initialize(config)
    assert status(config)["state"] == "prepared"
    repeat_root = config.output_root / "simnibs326_headreco/repeat_001"
    repeat_root.mkdir(parents=True)
    (repeat_root / "complete.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    assert status(config)["state"] == "in_progress"
