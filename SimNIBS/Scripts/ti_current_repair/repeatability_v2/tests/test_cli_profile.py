from __future__ import annotations

import json
from pathlib import Path

import pytest

from ti_current_repair.repeatability_v2.cli import main
from ti_current_repair.repeatability_v2.profile import (
    OperationalProfile,
    verify_profile_source_inputs,
)
from ti_current_repair.repeatability_v2.tests.test_config_inventory import valid_config
from ti_current_repair.repeatability_v2.validation import PILOT_SCOPE


def _profile(tmp_path: Path) -> dict[str, object]:
    config = valid_config()
    targets_csv = tmp_path / "targets.csv"
    targets_csv.write_text("targets", encoding="utf-8")
    for subject in config["subjects"]:
        subject_root = tmp_path / subject
        subject_root.mkdir(parents=True, exist_ok=True)
        (subject_root / "T1.nii.gz").write_text("t1", encoding="utf-8")
        (subject_root / "T2.nii.gz").write_text("t2", encoding="utf-8")
        (subject_root / "seg.nii.gz").write_text(
            f"seg-{subject}", encoding="utf-8"
        )
        (subject_root / "atlas.nii.gz").write_text(
            f"atlas-{subject}", encoding="utf-8"
        )
    return {
        "schema_version": 2,
        "study_config": config,
        "output_root": str(tmp_path / "v2-production"),
        "cache_root": str(tmp_path / "v2-caches"),
        "targets_csv": str(targets_csv),
        "montage_presets": {
            "left_hippocampus": "left-hippocampus",
            "right_m1": "right-m1",
        },
        "subject_inputs": {
            subject: {
                "t1": str(tmp_path / subject / "T1.nii.gz"),
                "t2": str(tmp_path / subject / "T2.nii.gz"),
                "corrected_segmentation": str(tmp_path / subject / "seg.nii.gz"),
                "atlas": str(tmp_path / subject / "atlas.nii.gz"),
            }
            for subject in config["subjects"]
        },
        "historical_read_only_roots": [
            "/historical/simnibs326",
            "/historical/simnibs401",
        ],
    }


def test_operational_profile_keeps_output_isolated_from_historical_roots(
    tmp_path: Path,
) -> None:
    profile = OperationalProfile.from_mapping(_profile(tmp_path))
    assert profile.output_root.name == "v2-production"
    payload = _profile(tmp_path)
    payload["output_root"] = "/historical/simnibs401/new"
    with pytest.raises(ValueError, match="historical read-only"):
        OperationalProfile.from_mapping(payload)


@pytest.mark.parametrize("field", ("output_root", "cache_root", "targets_csv"))
def test_operational_profile_rejects_missing_required_paths(
    tmp_path: Path, field: str
) -> None:
    payload = _profile(tmp_path)
    payload[field] = ""

    with pytest.raises(ValueError, match=field):
        OperationalProfile.from_mapping(payload)


def test_operational_profile_rejects_missing_subject_input_path(
    tmp_path: Path,
) -> None:
    payload = _profile(tmp_path)
    subject = payload["study_config"]["subjects"][0]
    payload["subject_inputs"][subject]["atlas"] = ""

    with pytest.raises(ValueError, match=f"subject_inputs\\[{subject}\\]\\.atlas"):
        OperationalProfile.from_mapping(payload)


def test_profile_source_verification_hashes_inputs_instead_of_trusting_a_flag(
    tmp_path: Path,
) -> None:
    profile = OperationalProfile.from_mapping(_profile(tmp_path))
    result = verify_profile_source_inputs(profile)
    assert result["status"] == "verified"
    assert result["hashed_inputs"] == 21

    profile.subject_inputs[profile.study_config.subjects[0]].atlas.write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_profile_source_inputs(profile)


def test_cli_preflight_and_release_gate_are_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(tmp_path)), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "simnibs_version": "4.6.0",
                "simnibs_module": "SimNIBS/4.6.0",
                "scientific_python_ready": True,
                "meshing_api": True,
                "fem_tdcs_api": True,
                "native_msh_float64_round_trip": True,
                "input_hashes_valid": True,
                "available_storage_bytes": 20_000,
                "required_storage_bytes": 10_000,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "v2-production"

    assert main(
        [
            "preflight",
            "--profile",
            str(profile_path),
            "--evidence",
            str(evidence_path),
            "--output-root",
            str(output_root),
        ]
    ) == 0
    preflight_result = json.loads(capsys.readouterr().out)
    assert preflight_result["status"] == "ready_for_pilot"
    assert not output_root.exists()

    local = tmp_path / "local.json"
    pilot = tmp_path / "pilot.json"
    identity = {
        "schema_version": 2,
        "status": "ready",
        "campaign_identity_sha256": "a" * 64,
        "operational_profile_sha256": "b" * 64,
        "code_commit": "c" * 40,
        "simnibs_version": "4.6.0",
        "scope": PILOT_SCOPE,
        "source_outputs_modified": False,
    }
    local.write_text(json.dumps(identity), encoding="utf-8")
    pilot.write_text(
        json.dumps(
            {
                **identity,
                "slurm_jobs": [
                    {
                        "job_id": "123456",
                        "state": "COMPLETED",
                        "exit_code": "0:0",
                    }
                ],
                "local_hpc_comparison": {
                    "status": "passed",
                    "comparison_rows": 12,
                    "maximum_absolute_difference": 1e-12,
                    "atol": 1e-10,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(
        [
            "release-gate",
            "--local-validation",
            str(local),
            "--hpc-pilot",
            str(pilot),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready_for_production"
