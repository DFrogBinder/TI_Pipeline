from __future__ import annotations

import json
from pathlib import Path

import pytest

from ti_current_repair.repeatability_v2.config import StudyConfigV2
from ti_current_repair.repeatability_v2.orchestration import (
    RetryPolicy,
    build_production_stage_plan,
    preflight,
    production_release_gate,
)
from ti_current_repair.repeatability_v2.tests.test_config_inventory import valid_config
from ti_current_repair.repeatability_v2.validation import validate_local_pilot


PILOT_IDENTITY = {
    "campaign_identity_sha256": "a" * 64,
    "operational_profile_sha256": "b" * 64,
    "code_commit": "c" * 40,
    "simnibs_version": "4.6.0",
}


def _ready_local_receipt() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "ready",
        **PILOT_IDENTITY,
        "scope": {
            "subject": "sub-CC110174",
            "target": "left_hippocampus",
            "remesh_simulations": 2,
            "fixed_simulations": 2,
            "outcome_rows": 12,
        },
        "source_outputs_modified": False,
    }


def _ready_hpc_receipt() -> dict[str, object]:
    return {
        **_ready_local_receipt(),
        "slurm_jobs": [
            {"job_id": "123456", "state": "COMPLETED", "exit_code": "0:0"}
        ],
        "local_hpc_comparison": {
            "status": "passed",
            "comparison_rows": 12,
            "maximum_absolute_difference": 1e-12,
            "atol": 1e-10,
        },
    }


def _manifest(condition: str, repeat: int) -> dict[str, object]:
    if condition == "remesh":
        events = [
            {"operation": "head_mesh_rebuilt"},
            {"operation": "electrode_geometry_constructed", "pair": 1, "seed": None},
            {"operation": "component_solved", "pair": 1},
            {"operation": "electrode_geometry_constructed", "pair": 2, "seed": None},
            {"operation": "component_solved", "pair": 2},
            {"operation": "ti_combined"},
        ]
        return {
            "status": "complete",
            "subject": "sub-CC110174",
            "target": "left_hippocampus",
            "condition": condition,
            "repeat": repeat,
            "events": events,
            "geometry_sha256": {"1": f"r{repeat}p1", "2": f"r{repeat}p2"},
        }
    return {
        "status": "complete",
        "subject": "sub-CC110174",
        "target": "left_hippocampus",
        "condition": condition,
        "repeat": repeat,
        "events": [
            {"operation": "existing_mesh_component_solved", "pair": 1},
            {"operation": "existing_mesh_component_solved", "pair": 2},
            {"operation": "ti_combined"},
        ],
        "geometry_sha256": {"1": "fixed-pair1", "2": "fixed-pair2"},
        "electrode_placement_invoked": False,
    }


def _metric_rows() -> list[dict[str, object]]:
    rows = []
    for condition in ("remesh", "fixed_mesh"):
        for repeat in (1, 2):
            common = {
                "subject": "sub-CC110174",
                "target": "left_hippocampus",
                "condition": condition,
                "repeat": repeat,
                "stored_field_dtype": "float64",
                "analysis_dtype": "float64",
                "finite_volume_fraction": 1.0,
            }
            rows.extend(
                [
                    {
                        **common,
                        "outcome_mode": "ti_target",
                        "roi_provenance_sha256": f"target-{condition}-{repeat}",
                    },
                    {
                        **common,
                        "outcome_mode": "tdcs_pair1_matched_target",
                        "roi_provenance_sha256": f"target-{condition}-{repeat}",
                    },
                    {
                        **common,
                        "outcome_mode": "tdcs_pair1_under_anode",
                        "roi_provenance_sha256": f"under-{condition}-{repeat}",
                        "electrode_name": "F8",
                    },
                ]
            )
    return rows


def test_local_pilot_requires_two_remesh_two_fixed_and_all_acceptance_gates() -> None:
    receipt = validate_local_pilot(
        manifests=[
            _manifest("remesh", 1),
            _manifest("remesh", 2),
            _manifest("fixed_mesh", 1),
            _manifest("fixed_mesh", 2),
        ],
        cache_manifest={
            "status": "ready",
            "geometry_sha256": {"1": "fixed-pair1", "2": "fixed-pair2"},
        },
        metric_rows=_metric_rows(),
        qc_comparisons=[
            {"native": 1.00000001, "nifti": 1.0, "atol": 2e-7},
            {"native": 2.00000001, "nifti": 2.0, "atol": 2e-7},
        ],
        provenance=PILOT_IDENTITY,
    )

    assert receipt["status"] == "ready"
    assert receipt["scope"] == {
        "subject": "sub-CC110174",
        "target": "left_hippocampus",
        "remesh_simulations": 2,
        "fixed_simulations": 2,
        "outcome_rows": 12,
    }


def test_local_pilot_rejects_fixed_electrode_placement() -> None:
    manifests = [
        _manifest("remesh", 1),
        _manifest("remesh", 2),
        _manifest("fixed_mesh", 1),
        _manifest("fixed_mesh", 2),
    ]
    manifests[-1]["electrode_placement_invoked"] = True
    with pytest.raises(ValueError, match="electrode placement"):
        validate_local_pilot(
            manifests=manifests,
            cache_manifest={
                "status": "ready",
                "geometry_sha256": {"1": "fixed-pair1", "2": "fixed-pair2"},
            },
            metric_rows=_metric_rows(),
            qc_comparisons=[{"native": 1.0, "nifti": 1.0, "atol": 1e-7}],
            provenance=PILOT_IDENTITY,
        )


def test_production_stage_plan_enforces_full_scope_dependencies_and_resources() -> None:
    stages = build_production_stage_plan()
    assert [(stage.name, stage.task_count) for stage in stages] == [
        ("left_hippocampus_remesh", 400),
        ("right_m1_remesh", 400),
        ("remesh_native_extraction_validation", 800),
        ("fixed_cache_preparation", 20),
        ("left_hippocampus_fixed", 400),
        ("right_m1_fixed", 400),
        ("final_native_extraction_geometry_validation", 800),
        ("derived_nifti_qc", 1600),
    ]
    assert stages[3].dependencies == ("remesh_native_extraction_validation",)
    assert stages[4].dependencies == ("fixed_cache_preparation",)
    simulation_stages = [stage for stage in stages if stage.task_count == 400]
    assert all(
        (stage.cpus, stage.memory_gb, stage.walltime, stage.concurrency)
        == (8, 32, "08:00:00", 50)
        for stage in simulation_stages
    )


def test_preflight_is_read_only_and_release_requires_both_pilots(tmp_path: Path) -> None:
    config = StudyConfigV2.from_mapping(valid_config())
    output = tmp_path / "production-must-not-exist"
    evidence = {
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

    result = preflight(config=config, evidence=evidence, output_root=output)

    assert result["status"] == "ready_for_pilot"
    assert result["scope"]["simulations"] == 1600
    assert result["scope"]["component_solves"] == 3200
    assert not output.exists()
    assert production_release_gate(
        local_validation=_ready_local_receipt(),
        hpc_pilot=_ready_hpc_receipt(),
    )["status"] == "ready_for_production"
    with pytest.raises(ValueError, match="bound|identity|scope"):
        production_release_gate(
            local_validation={"status": "ready"},
            hpc_pilot={"status": "ready", "local_hpc_comparison": "passed"},
        )
    with pytest.raises(ValueError, match="HPC pilot"):
        production_release_gate(
            local_validation=_ready_local_receipt(), hpc_pilot={"status": "pending"}
        )


def test_retry_policy_allows_exactly_one_automatic_retry() -> None:
    policy = RetryPolicy(max_automatic_retries=1)
    assert policy.action_for_attempt(0) == "run"
    assert policy.action_for_attempt(1) == "retry"
    assert policy.action_for_attempt(2) == "terminal_failure"
