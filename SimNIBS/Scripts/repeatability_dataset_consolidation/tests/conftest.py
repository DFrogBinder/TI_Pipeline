from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from repeatability_dataset_consolidation.settings import DatasetConfig, SourceRoots


@pytest.fixture
def miniature_config(tmp_path: Path) -> DatasetConfig:
    parscratch = tmp_path / "parscratch"
    shared = tmp_path / "shared"
    parscratch.mkdir()
    shared.mkdir()

    def root(name: str) -> Path:
        path = parscratch / name
        path.mkdir(parents=True)
        return path

    sources = SourceRoots(
        v4_left_remesh=root("v4-left-remesh"),
        v4_right_remesh=root("v4-right-remesh"),
        v4_left_fixed=root("v4-left-fixed"),
        v4_right_fixed=root("v4-right-fixed"),
        v3_left=root("v3-left"),
        v3_right=root("v3-right"),
        v3_headmodels=root("v3-headmodels"),
        v4_scaffolds=root("v4-scaffolds"),
        replacement_subject=root("replacement-subject"),
        paper_analysis=root("paper-analysis"),
        nested_analysis=root("nested-analysis"),
        legacy_v3_left=root("legacy-v3-left"),
        legacy_v3_right=root("legacy-v3-right"),
        legacy_v3_headreco=root("legacy-v3-headreco"),
        replacement_v3_scaffolds=root("replacement-v3-scaffolds"),
    )
    subjects = ("sub-CC110174", "sub-CC510415")
    replacement = subjects[-1]

    def completion_payload(
        target: str,
        receipt_subjects: tuple[str, ...],
        conditions: tuple[str, ...],
    ) -> dict[str, object]:
        subject_count = len(receipt_subjects)
        repeats = 2
        condition_tasks = subject_count * repeats
        total_tasks = condition_tasks * len(conditions)
        scope: dict[str, object] = {
            "status": "complete",
            "target": target.replace("_", "-"),
            "roi": target.replace("_", "-"),
            "stage": "final",
            "subjects": list(receipt_subjects),
            "subject_count": subject_count,
            "repeats_per_condition": repeats,
            "conditions": {
                condition: condition_tasks for condition in conditions
            },
            "expected_ti_msh": total_tasks,
            "expected_ti_nifti": total_tasks,
        }
        for condition in conditions:
            scope[f"{condition}_tasks"] = condition_tasks
        return {
            "schema_version": 1,
            "status": "complete",
            "stage": "final",
            "scope": scope,
        }

    base_sources = {
        ("simnibs4", "left_hippocampus", "remesh"): sources.v4_left_remesh,
        ("simnibs4", "left_hippocampus", "fixed_mesh"): sources.v4_left_fixed,
        ("simnibs4", "right_m1", "remesh"): sources.v4_right_remesh,
        ("simnibs4", "right_m1", "fixed_mesh"): sources.v4_right_fixed,
        ("simnibs326", "left_hippocampus", "remesh"): sources.v3_left,
        ("simnibs326", "left_hippocampus", "fixed_mesh"): sources.v3_left,
        ("simnibs326", "right_m1", "remesh"): sources.v3_right,
        ("simnibs326", "right_m1", "fixed_mesh"): sources.v3_right,
    }
    base_receipts: dict[Path, tuple[str, set[str]]] = {}
    for (_, target, condition), experiment_root in base_sources.items():
        condition_root = (
            experiment_root
            / f"{subjects[0]}_repeatability"
            / condition
        )
        condition_root.mkdir(parents=True, exist_ok=True)
        for repeat_index in range(1, 3):
            ti_mesh = (
                condition_root
                / "repeats"
                / f"repeat_{repeat_index:03d}"
                / subjects[0]
                / "anat/SimNIBS/Output"
                / subjects[0]
                / "TI.msh"
            )
            ti_mesh.parent.mkdir(parents=True)
            ti_mesh.write_text(f"field {repeat_index}\n", encoding="utf-8")
        receipt_target, receipt_conditions = base_receipts.setdefault(
            experiment_root, (target, set())
        )
        assert receipt_target == target
        receipt_conditions.add(condition)

    for experiment_root, (target, conditions) in base_receipts.items():
        workflow = experiment_root / "_pipeline/workflow"
        workflow.mkdir(parents=True, exist_ok=True)
        (workflow / "complete.json").write_text(
            json.dumps(
                completion_payload(target, (subjects[0],), tuple(sorted(conditions)))
            ),
            encoding="utf-8",
        )

    historical_config = sources.v3_left / "_pipeline/configs/paired_analysis.json"
    historical_config.parent.mkdir(parents=True, exist_ok=True)

    for version in ("simnibs326", "simnibs4"):
        for target in ("left_hippocampus", "right_m1"):
            experiment_root = sources.replacement_subject / version / target
            for condition in ("remesh", "fixed_mesh"):
                condition_root = (
                    experiment_root
                    / f"{replacement}_repeatability"
                    / condition
                )
                condition_root.mkdir(parents=True, exist_ok=True)
                for repeat_index in range(1, 3):
                    ti_mesh = (
                        condition_root
                        / "repeats"
                        / f"repeat_{repeat_index:03d}"
                        / replacement
                        / "anat/SimNIBS/Output"
                        / replacement
                        / "TI.msh"
                    )
                    ti_mesh.parent.mkdir(parents=True)
                    ti_mesh.write_text(f"field {repeat_index}\n", encoding="utf-8")
            workflow = experiment_root / "_pipeline/workflow"
            workflow.mkdir(parents=True, exist_ok=True)
            (workflow / "complete.json").write_text(
                json.dumps(
                    completion_payload(
                        target, (replacement,), ("remesh", "fixed_mesh")
                    )
                ),
                encoding="utf-8",
            )

    def seed_v3_headmodel(head_root: Path, subject: str, cohort: str) -> None:
        subject_tree = head_root / subject / "anat"
        subject_tree.mkdir(parents=True)
        (subject_tree / f"{subject}.msh").write_text("mesh\n", encoding="utf-8")
        labels = head_root / "harmonized_labels"
        (labels / "qc").mkdir(parents=True, exist_ok=True)
        (labels / "figures").mkdir(parents=True, exist_ok=True)
        (labels / f"{subject}_v3_harmonized_t1fs_grid.nii.gz").write_bytes(b"label")
        (labels / "qc" / f"{subject}.json").write_text("{}\n", encoding="utf-8")
        (labels / "figures" / f"{subject}.png").write_bytes(b"png")
        receipts = head_root / "_pipeline" / "subjects" / subject
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / "headmodel_complete.json").write_text(
            json.dumps({"status": "complete", "subject": subject}), encoding="utf-8"
        )
        cohort_root = head_root / "_pipeline" / "headmodels" / cohort
        cohort_root.mkdir(parents=True, exist_ok=True)
        (cohort_root / "manifest.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        (cohort_root / "promoted.json").write_text(
            json.dumps({"status": "promoted"}), encoding="utf-8"
        )
        collection = head_root / "_pipeline" / "reference_stage_collection.json"
        collection.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    seed_v3_headmodel(sources.v3_headmodels, subjects[0], "base9")
    replacement_headmodels = sources.replacement_subject / "simnibs326" / "headmodels"
    replacement_headmodels.mkdir(parents=True)
    seed_v3_headmodel(replacement_headmodels, replacement, "external1")
    corrected_inputs = sources.replacement_subject / "inputs/corrected_v4"
    corrected_inputs.mkdir(parents=True)
    (corrected_inputs / "input.nii").write_bytes(b"corrected input")
    historical_config.write_text(
        json.dumps(
            {
                "experiment_root": str(sources.v3_left),
                "headmodel_root": str(sources.v3_headmodels),
                "headmodel_promotion_path": str(
                    replacement_headmodels
                    / "_pipeline/headmodels/external1/promoted.json"
                ),
                "source_root": str(corrected_inputs),
                "unchanged_parameter": "median_roi",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    replacement_analysis = (
        replacement_headmodels
        / "_pipeline/analysis/repeatability_optimizer_roi_paired_external1_v2"
    )
    (replacement_analysis / "results").mkdir(parents=True)
    (replacement_analysis / "results/analysis_manifest.json").write_text(
        json.dumps({"status": "complete", "rows": 160}), encoding="utf-8"
    )
    archive_content = b"verified archive"
    (replacement_analysis / "simnibs326h_repeatability_analysis.tar.gz").write_bytes(
        archive_content
    )
    (
        replacement_analysis
        / "simnibs326h_repeatability_analysis.tar.gz.sha256"
    ).write_text(
        hashlib.sha256(archive_content).hexdigest()
        + "  simnibs326h_repeatability_analysis.tar.gz\n",
        encoding="utf-8",
    )

    for subject in subjects:
        scaffold = sources.v4_scaffolds / "subjects" / subject / "anat"
        scaffold.mkdir(parents=True)
        (scaffold / f"{subject}.msh").write_text("v4 mesh\n", encoding="utf-8")
        results = sources.v4_scaffolds / "results"
        results.mkdir(exist_ok=True)
        (results / f"{subject}.json").write_text(
            json.dumps({"status": "complete", "subject": subject}), encoding="utf-8"
        )

    (sources.paper_analysis / "analysis.json").write_text("{}\n", encoding="utf-8")
    (sources.nested_analysis / "nested.msh").write_text("nested\n", encoding="utf-8")
    replacement_pipeline = sources.replacement_subject / "_pipeline"
    replacement_pipeline.mkdir(exist_ok=True)
    (replacement_pipeline / "replacement_manifest.json").write_text(
        json.dumps({"status": "prepared"}), encoding="utf-8"
    )

    return DatasetConfig(
        output_root=parscratch / "combined-dataset",
        shared_archive_root=shared / "archive",
        shared_receipt_root=shared / "receipts",
        deletion_log_root=parscratch / "deletion-logs",
        sources=sources,
        subjects=subjects,
        replacement_subject=replacement,
        excluded_subject="sub-CC520083",
        repeat_count=2,
    )
