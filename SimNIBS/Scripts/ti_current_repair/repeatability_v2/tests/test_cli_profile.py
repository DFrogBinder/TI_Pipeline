from __future__ import annotations

import json
from pathlib import Path

import pytest

from ti_current_repair.repeatability_v2.cli import main
from ti_current_repair.repeatability_v2.profile import OperationalProfile
from ti_current_repair.repeatability_v2.tests.test_config_inventory import valid_config


def _profile(tmp_path: Path) -> dict[str, object]:
    config = valid_config()
    return {
        "schema_version": 2,
        "study_config": config,
        "output_root": str(tmp_path / "v2-production"),
        "cache_root": str(tmp_path / "v2-caches"),
        "targets_csv": str(tmp_path / "targets.csv"),
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
    local.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    pilot.write_text(
        json.dumps({"status": "ready", "local_hpc_comparison": "passed"}),
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
