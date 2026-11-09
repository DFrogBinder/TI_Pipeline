from __future__ import annotations

import random

import pytest

from simnibs_segmentation_determinism.provenance import ProvenanceError
from simnibs_segmentation_determinism.settings import ExperimentConfig, scope_payload


def test_defaults_pin_approved_scientific_and_hpc_scope():
    config = ExperimentConfig.defaults()

    assert config.subject == "sub-CC110174"
    assert config.selection_seed == 20261106
    assert config.repeats_per_arm == 10
    assert config.source_root.as_posix() == (
        "/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected"
    )
    assert config.output_root.as_posix() == (
        "/mnt/parscratch/users/cop23bi/simnibs_segmentation_determinism_v1"
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
    monkeypatch.setenv("SEGDET_OUTPUT_ROOT", str(tmp_path / "custom"))

    config = ExperimentConfig.from_env()

    assert config.output_root == tmp_path / "custom"
    assert config.repeats_per_arm == 10
    assert config.subject == "sub-CC110174"


def test_from_env_rejects_scientific_scope_overrides(monkeypatch):
    monkeypatch.setenv("SEGDET_REPEATS_PER_ARM", "7")

    with pytest.raises(ProvenanceError, match="ten-repeat scope"):
        ExperimentConfig.from_env()
