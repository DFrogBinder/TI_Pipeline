from __future__ import annotations

from dataclasses import replace
import json

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
    assert payload["dataset"]["root"] == str(config.dataset_root)
    assert payload["dataset"]["manifest"]["path"] == str(
        config.dataset_manifest_file
    )
    assert payload["dataset"]["completion"]["path"] == str(
        config.dataset_completion_file
    )
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


def test_dataset_manifest_status_drift_fails(configured_experiment):
    config = configured_experiment
    payload = json.loads(config.dataset_manifest_file.read_text())
    payload["status"] = "prepared"
    config.dataset_manifest_file.write_text(json.dumps(payload))

    with pytest.raises(ProvenanceError, match="dataset manifest"):
        preflight(config)


def test_dataset_completion_plan_drift_fails(configured_experiment):
    config = configured_experiment
    payload = json.loads(config.dataset_completion_file.read_text())
    payload["plan_sha256"] = "b" * 64
    config.dataset_completion_file.write_text(json.dumps(payload))

    with pytest.raises(ProvenanceError, match="dataset completion"):
        preflight(config)


def test_dataset_completion_manifest_checksum_drift_fails(configured_experiment):
    config = configured_experiment
    payload = json.loads(config.dataset_completion_file.read_text())
    payload["dataset_manifest"]["sha256"] = "b" * 64
    config.dataset_completion_file.write_text(json.dumps(payload))

    with pytest.raises(ProvenanceError, match="manifest binding"):
        preflight(config)


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


def test_all_scientific_paths_must_remain_inside_dataset(configured_experiment, tmp_path):
    config = replace(configured_experiment, output_root=tmp_path / "outside")

    with pytest.raises(ProvenanceError, match="dataset root"):
        preflight(config)

    config = replace(configured_experiment, source_root=tmp_path / "outside-source")
    with pytest.raises(ProvenanceError, match="dataset root"):
        preflight(config)
