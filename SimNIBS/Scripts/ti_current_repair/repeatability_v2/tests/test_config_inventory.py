from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from ti_current_repair.repeatability_v2.config import (
    CANONICAL_SUBJECTS,
    StudyConfigV2,
)
from ti_current_repair.repeatability_v2.inventory import (
    HistoricalRun,
    inventory_historical_sources,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def valid_config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "simnibs_version": "4.6.0",
        "simnibs_module": "SimNIBS/4.6.0",
        "subjects": list(CANONICAL_SUBJECTS),
        "targets": ["left_hippocampus", "right_m1"],
        "conditions": ["remesh", "fixed_mesh"],
        "repeat_count": 40,
        "corrected_segmentation_sha256": {
            subject: _sha(f"seg-{subject}") for subject in CANONICAL_SUBJECTS
        },
        "atlas_sha256": {
            subject: _sha(f"atlas-{subject}") for subject in CANONICAL_SUBJECTS
        },
        "targets_csv_sha256": _sha("targets"),
        "head_mesh_policy": {
            "remesh": "rebuild_per_repeat",
            "fixed_mesh": "reuse_representative_geometry",
        },
        "electrode_mesh_policy": {
            "remesh": "rebuild_unseeded_per_repeat",
            "fixed_mesh": "reuse_pair_specific_geometry",
        },
        "authoritative_field_format": "native_msh",
        "nifti_role": "derived_qc",
    }


def test_study_config_accepts_only_canonical_full_scope() -> None:
    config = StudyConfigV2.from_mapping(valid_config())

    assert config.subjects == CANONICAL_SUBJECTS
    assert config.expected_simulations == 1600
    assert config.expected_component_solves == 3200
    assert config.outcomes == (
        "ti_target",
        "tdcs_pair1_matched_target",
        "tdcs_pair1_under_anode",
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("simnibs_version", "4.0.1", "4.6.0"),
        ("subjects", [*CANONICAL_SUBJECTS[:-1], "sub-CC520083"], "canonical"),
        ("repeat_count", 2, "40"),
        ("authoritative_field_format", "nifti", "native_msh"),
        ("nifti_role", "analysis", "derived_qc"),
    ],
)
def test_study_config_rejects_scope_or_precision_drift(
    key: str, value: object, message: str
) -> None:
    payload = valid_config()
    payload[key] = value

    with pytest.raises(ValueError, match=message):
        StudyConfigV2.from_mapping(payload)


def test_study_config_rejects_legacy_mesh_mode() -> None:
    payload = valid_config()
    payload["mesh_mode"] = "fixed"

    with pytest.raises(ValueError, match="Legacy mesh_mode"):
        StudyConfigV2.from_mapping(payload)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_inventory_hashes_all_native_meshes_and_receipts_without_writes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "historical"
    output = tmp_path / "must-not-exist"
    runs = []
    for version in ("3.2.6", "4.0.1"):
        root = source / version
        runs.append(
            HistoricalRun(
                version=version,
                subject="sub-CC510415",
                target="left_hippocampus",
                condition="remesh",
                repeat=1,
                pair1_msh=_write(root / "pair1.msh", f"{version}-p1"),
                pair2_msh=_write(root / "pair2.msh", f"{version}-p2"),
                ti_msh=_write(root / "TI.msh", f"{version}-ti"),
                completion_receipt=_write(root / "complete.json", "complete"),
                source_group="replacement",
            )
        )

    manifest = inventory_historical_sources(runs, expected_per_version=1)

    assert manifest["status"] == "complete"
    assert manifest["counts_by_version"] == {"3.2.6": 1, "4.0.1": 1}
    assert len(manifest["files"]) == 8
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    assert not output.exists()


def test_inventory_fails_closed_on_incomplete_or_duplicate_scope(tmp_path: Path) -> None:
    path = _write(tmp_path / "one.msh", "one")
    receipt = _write(tmp_path / "complete.json", "complete")
    run = HistoricalRun(
        version="3.2.6",
        subject="sub-CC510415",
        target="left_hippocampus",
        condition="remesh",
        repeat=1,
        pair1_msh=path,
        pair2_msh=path,
        ti_msh=path,
        completion_receipt=receipt,
        source_group="replacement",
    )

    with pytest.raises(ValueError, match="4.0.1"):
        inventory_historical_sources([run], expected_per_version=1)
    with pytest.raises(ValueError, match="Duplicate"):
        inventory_historical_sources(
            [run, run], expected_per_version=2, expected_versions=("3.2.6",)
        )


def test_inventory_rejects_noncanonical_or_misrouted_cohort(tmp_path: Path) -> None:
    path = _write(tmp_path / "one.msh", "one")
    receipt = _write(tmp_path / "complete.json", "complete")
    replacement = HistoricalRun(
        version="3.2.6",
        subject="sub-CC510415",
        target="left_hippocampus",
        condition="remesh",
        repeat=1,
        pair1_msh=path,
        pair2_msh=path,
        ti_msh=path,
        completion_receipt=receipt,
        source_group="replacement",
    )

    with pytest.raises(ValueError, match="canonical 9\\+1"):
        inventory_historical_sources(
            [replace(replacement, subject="sub-CC520083")],
            expected_per_version=1,
            expected_versions=("3.2.6",),
        )
    with pytest.raises(ValueError, match="source_group"):
        inventory_historical_sources(
            [replace(replacement, source_group="base9")],
            expected_per_version=1,
            expected_versions=("3.2.6",),
        )

