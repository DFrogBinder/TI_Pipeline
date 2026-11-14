from __future__ import annotations

from dataclasses import replace

import pytest

from simnibs_segmentation_determinism.provenance import ProvenanceError
from simnibs_segmentation_determinism.settings import preflight, resolve_subject_inputs


def test_preflight_resolves_one_pair_and_is_read_only(configured_experiment):
    config = configured_experiment
    assert not config.output_root.exists()

    payload = preflight(config)

    assert payload["status"] == "ready"
    assert payload["subject"] == "sub-CC110174"
    assert payload["selected_subject"] == "sub-CC110174"
    assert payload["source_outputs_modified"] is False
    assert payload["staging_root"] == str(config.output_root)
    assert payload["future_extension_root"] == str(config.extension_root)
    assert payload["inputs"]["t1"]["path"].endswith("_T1w.nii")
    assert payload["inputs"]["t2"]["path"].endswith("_T2w.nii.gz")
    assert len(payload["inputs"]["t1"]["sha256"]) == 64
    assert not config.output_root.exists()


def test_ambiguous_nifti_extensions_fail_closed(configured_experiment):
    config = configured_experiment
    anat = config.source_root / config.subject / "anat"
    (anat / f"{config.subject}_T1w.nii.gz").write_bytes(b"second T1")

    with pytest.raises(ProvenanceError, match="Ambiguous T1"):
        resolve_subject_inputs(config)


def test_missing_input_fails_closed(configured_experiment):
    config = configured_experiment
    (config.source_root / config.subject / "anat" / f"{config.subject}_T2w.nii.gz").unlink()

    with pytest.raises(ProvenanceError, match="Missing T2"):
        resolve_subject_inputs(config)


def test_cohort_hash_drift_fails(configured_experiment):
    config = configured_experiment
    config.cohort_file.write_text("sub-CC999999\n", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="cohort SHA-256"):
        preflight(config)


def test_preflight_does_not_require_combined_dataset_receipts(configured_experiment):
    config = configured_experiment
    assert not config.dataset_manifest_file.exists()
    assert not config.dataset_completion_file.exists()
    assert preflight(config)["status"] == "ready"


@pytest.mark.parametrize("collision_kind", ["equal", "inside"])
def test_output_may_not_collide_with_protected_roots(configured_experiment, collision_kind):
    config = configured_experiment
    protected = config.protected_roots[1]
    output = protected if collision_kind == "equal" else protected / "child"
    config = replace(config, output_root=output)

    with pytest.raises(ProvenanceError, match="output root|protected root"):
        preflight(config)


def test_effective_source_root_is_always_protected(configured_experiment):
    config = configured_experiment
    config = replace(
        config,
        protected_roots=tuple(config.protected_roots[1:]),
        output_root=config.source_root / "forbidden-output",
    )

    with pytest.raises(ProvenanceError, match="output root|source root|protected root"):
        preflight(config)


def test_staging_root_must_not_overlap_combined_dataset(configured_experiment):
    config = replace(
        configured_experiment,
        output_root=configured_experiment.dataset_root / "extensions/forbidden",
    )
    with pytest.raises(ProvenanceError, match="combined dataset"):
        preflight(config)

    config = replace(
        configured_experiment,
        output_root=configured_experiment.source_root / "forbidden",
    )
    with pytest.raises(ProvenanceError, match="source root|protected root"):
        preflight(config)
