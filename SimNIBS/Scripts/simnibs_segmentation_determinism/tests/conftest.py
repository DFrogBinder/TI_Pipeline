from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest


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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def configured_experiment(tmp_path: Path):
    from simnibs_segmentation_determinism.settings import ExperimentConfig

    dataset = tmp_path / "final_132_repeatability_combined_dataset_v1"
    cohort = tmp_path / "cohort_subjects.txt"
    cohort.write_text("\n".join(SUBJECTS) + "\n", encoding="utf-8")
    plan_sha256 = "a" * 64
    manifest = dataset / "dataset_manifest.json"
    completion = dataset / "_consolidation/complete.json"
    source = tmp_path / "ti_dataset_final_132_balanced_10_corrected"
    anat = source / SUBJECTS[0] / "anat"
    anat.mkdir(parents=True)
    (anat / f"{SUBJECTS[0]}_T1w.nii").write_bytes(b"T1 fixture")
    (anat / f"{SUBJECTS[0]}_T2w.nii.gz").write_bytes(b"T2 fixture")
    output = tmp_path / "segmentation_determinism_staging_v1"
    protected = (source, dataset, tmp_path / "protected_completed_campaign")
    config = replace(
        ExperimentConfig.defaults(repo_scripts_root=tmp_path),
        dataset_root=dataset,
        dataset_manifest_file=manifest,
        dataset_completion_file=completion,
        dataset_plan_sha256=plan_sha256,
        cohort_file=cohort,
        cohort_sha256=digest(cohort),
        source_root=source,
        output_root=output,
        protected_roots=protected,
    )
    return config


def write_dataset_receipts(config) -> None:
    dataset = config.dataset_root
    dataset.mkdir(parents=True, exist_ok=True)
    scope = {"subjects": 10, "simulations": 3200}
    manifest_payload = {
        "schema_version": 1,
        "status": "complete",
        "workflow": "repeatability_dataset_consolidation",
        "output_root": str(dataset),
        "plan_sha256": config.dataset_plan_sha256,
        "subjects": list(SUBJECTS),
        "scope": scope,
        "source_outputs_modified": False,
    }
    config.dataset_manifest_file.write_text(
        json.dumps(manifest_payload) + "\n", encoding="utf-8"
    )
    config.dataset_completion_file.parent.mkdir(parents=True, exist_ok=True)
    config.dataset_completion_file.write_text(
        json.dumps(
            {
                **manifest_payload,
                "dataset_manifest": {
                    "path": "dataset_manifest.json",
                    "sha256": digest(config.dataset_manifest_file),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def dataset_receipt_writer():
    return write_dataset_receipts


def write_completed_staging(config) -> Path:
    from simnibs_segmentation_determinism.provenance import (
        sha256_file,
        write_json_atomic,
        write_sha256_sidecar,
    )
    from simnibs_segmentation_determinism.settings import resolve_staged_subject_inputs
    from simnibs_segmentation_determinism.workflow import initialize

    initialized = initialize(config)
    inputs = resolve_staged_subject_inputs(config)
    repeat_receipts = []
    for index in range(20):
        arm = "simnibs326_headreco" if index < 10 else "simnibs401_charm"
        repeat_index = index % 10 + 1
        root = config.output_root / arm / f"repeat_{repeat_index:03d}"
        artifact = (
            root / "artifacts/artifact.bin"
            if arm == "simnibs326_headreco"
            else root / "artifact.bin"
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact-{index}".encode())
        artifact_record = {
            "relative_path": "artifact.bin",
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
        manifest = root / "artifact_manifest.json"
        manifest_payload = {
            "status": "complete",
            "arm": arm,
            "repeat_index": repeat_index,
        }
        write_json_atomic(manifest, manifest_payload)
        write_sha256_sidecar(manifest)
        log = root / "logs/attempt_001.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("complete\n", encoding="utf-8")
        receipt_payload = {
            **manifest_payload,
            "source_outputs_modified": False,
            "inputs": {
                "t1": {"path": str(inputs.t1), "sha256": inputs.t1_sha256},
                "t2": {"path": str(inputs.t2), "sha256": inputs.t2_sha256},
            },
            "artifact_manifest": str(manifest),
            "artifact_manifest_sha256": sha256_file(manifest),
            "log": str(log),
            "log_sha256": sha256_file(log),
        }
        if arm == "simnibs326_headreco":
            receipt_payload["artifacts"] = [artifact_record]
        else:
            receipt_payload["artifact"] = artifact_record
        receipt = root / "complete.json"
        write_json_atomic(receipt, receipt_payload)
        write_sha256_sidecar(receipt)
        repeat_receipts.append(
            {
                "arm": arm,
                "repeat_index": repeat_index,
                "path": str(receipt),
                "sha256": sha256_file(receipt),
            }
        )
    analysis = config.output_root / "analysis/determinism_manifest.json"
    write_json_atomic(analysis, {"status": "complete"})
    terminal = config.output_root / "_pipeline/complete.json"
    write_json_atomic(
        terminal,
        {
            "schema_version": 1,
            "status": "complete",
            "workflow": "simnibs_segmentation_determinism",
            "plan_sha256": initialized["plan_sha256"],
            "scope": {
                "staging_root": str(config.output_root.resolve()),
                "future_extension_root": str(config.extension_root.resolve()),
                "scientific_segmentation_tasks": 20,
            },
            "validated_repeats": 20,
            "repeat_receipts": repeat_receipts,
            "analysis_artifacts": {
                "determinism_manifest.json": sha256_file(analysis)
            },
            "source_outputs_modified": False,
        },
    )
    write_sha256_sidecar(terminal)
    return terminal


@pytest.fixture
def staging_completer():
    return write_completed_staging
