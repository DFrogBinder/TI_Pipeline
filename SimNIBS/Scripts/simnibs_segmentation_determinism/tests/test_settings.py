from __future__ import annotations

import random

import pytest

from simnibs_segmentation_determinism.provenance import ProvenanceError
from simnibs_segmentation_determinism.settings import ExperimentConfig, scope_payload


def test_defaults_pin_approved_scientific_and_hpc_scope():
    config = ExperimentConfig.defaults()

    dataset_root = (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_combined_dataset_v1"
    )
    assert config.subject == "sub-CC110174"
    assert config.selection_seed == 20261106
    assert config.repeats_per_arm == 10
    assert config.dataset_root.as_posix() == dataset_root
    assert config.cohort_file.name == "cohort_subjects.txt"
    assert config.cohort_file.parent.name == "simnibs_segmentation_determinism"
    assert config.dataset_manifest_file.as_posix() == (
        f"{dataset_root}/dataset_manifest.json"
    )
    assert config.dataset_completion_file.as_posix() == (
        f"{dataset_root}/_consolidation/complete.json"
    )
    assert config.source_root.as_posix() == (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_corrected"
    )
    assert config.output_root.as_posix() == (
        "/mnt/parscratch/users/cop23bi/segmentation_determinism_staging_v1"
    )
    assert config.extension_root.as_posix() == (
        f"{dataset_root}/extensions/segmentation_determinism_v1"
    )
    assert config.simnibs326_module == "SimNIBS/3.2.6-foss-2023a"
    assert config.matlab_module == "MATLAB/2023b"
    assert config.simnibs401_module == "SimNIBS/4.0.1-foss-2023a"
    assert config.partition == "sheffield"
    assert config.cpus_per_task == 8
    assert config.memory == "32G"
    assert config.time_limit == "08:00:00"
    assert config.max_concurrent == 10

    payload = scope_payload(config)
    assert payload["subjects"] == 1
    assert payload["versions"] == 2
    assert payload["repeats_per_arm"] == 10
    assert payload["scientific_segmentation_tasks"] == 20
    assert payload["scheduler_tasks"] == 23
    assert payload["surface_tasks"] == 0
    assert payload["meshing_tasks"] == 0
    assert payload["fem_simulation_tasks"] == 0


def test_seeded_selection_is_the_pinned_subject(configured_experiment):
    subjects = configured_experiment.cohort_file.read_text(encoding="utf-8").splitlines()
    assert random.Random(configured_experiment.selection_seed).choice(subjects) == configured_experiment.subject


def test_from_env_only_changes_nonscientific_path_overrides(monkeypatch, tmp_path):
    dataset = tmp_path / "combined"
    staging = tmp_path / "staging"
    monkeypatch.setenv("SEGDET_DATASET_ROOT", str(dataset))
    monkeypatch.setenv("SEGDET_STAGING_ROOT", str(staging))

    config = ExperimentConfig.from_env()

    assert config.dataset_root == dataset
    assert config.output_root == staging
    assert config.extension_root == dataset / "extensions/segmentation_determinism_v1"
    assert config.source_root.as_posix().endswith(
        "/ti_dataset_final_132_balanced_10_corrected"
    )
    assert config.repeats_per_arm == 10
    assert config.subject == "sub-CC110174"


def test_from_env_rejects_scientific_scope_overrides(monkeypatch):
    monkeypatch.setenv("SEGDET_REPEATS_PER_ARM", "7")

    with pytest.raises(ProvenanceError, match="ten-repeat scope"):
        ExperimentConfig.from_env()
