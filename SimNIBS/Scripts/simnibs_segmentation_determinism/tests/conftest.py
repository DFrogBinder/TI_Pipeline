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
    dataset.mkdir()
    cohort = dataset / "subjects.txt"
    cohort.write_text("\n".join(SUBJECTS) + "\n", encoding="utf-8")
    plan_sha256 = "a" * 64
    manifest = dataset / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "workflow": "repeatability_dataset_consolidation",
                "output_root": str(dataset),
                "plan_sha256": plan_sha256,
                "subjects": list(SUBJECTS),
                "scope": {"subjects": 10, "simulations": 3200},
                "source_outputs_modified": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completion = dataset / "_consolidation/complete.json"
    completion.parent.mkdir()
    completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "workflow": "repeatability_dataset_consolidation",
                "output_root": str(dataset),
                "plan_sha256": plan_sha256,
                "dataset_manifest": {
                    "path": "dataset_manifest.json",
                    "sha256": digest(manifest),
                },
                "scope": {"subjects": 10, "simulations": 3200},
                "source_outputs_modified": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = (
        dataset
        / "data/simnibs4/left_hippocampus/"
        "sub-CC110174_repeatability/remesh/repeats/repeat_001"
    )
    anat = source / SUBJECTS[0] / "anat"
    anat.mkdir(parents=True)
    (anat / f"{SUBJECTS[0]}_T1w.nii").write_bytes(b"T1 fixture")
    (anat / f"{SUBJECTS[0]}_T2w.nii.gz").write_bytes(b"T2 fixture")
    output = dataset / "extensions/segmentation_determinism_v1"
    protected = (source, dataset / "protected_completed_campaign")
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
