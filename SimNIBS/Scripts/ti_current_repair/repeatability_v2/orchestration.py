"""Fail-closed production plan, preflight, release, and retry semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import StudyConfigV2


@dataclass(frozen=True)
class StageSpec:
    name: str
    task_count: int
    dependencies: tuple[str, ...]
    cpus: int
    memory_gb: int
    walltime: str
    concurrency: int


def _stage(
    name: str,
    count: int,
    dependencies: tuple[str, ...] = (),
    *,
    simulation: bool = False,
) -> StageSpec:
    return StageSpec(
        name=name,
        task_count=count,
        dependencies=dependencies,
        cpus=8 if simulation else 1,
        memory_gb=32 if simulation else 8,
        walltime="08:00:00" if simulation else "02:00:00",
        concurrency=50 if simulation else min(count, 50),
    )


def build_production_stage_plan() -> tuple[StageSpec, ...]:
    return (
        _stage("left_hippocampus_remesh", 400, simulation=True),
        _stage("right_m1_remesh", 400, simulation=True),
        _stage(
            "remesh_native_extraction_validation",
            800,
            ("left_hippocampus_remesh", "right_m1_remesh"),
        ),
        _stage(
            "fixed_cache_preparation",
            20,
            ("remesh_native_extraction_validation",),
        ),
        _stage(
            "left_hippocampus_fixed",
            400,
            ("fixed_cache_preparation",),
            simulation=True,
        ),
        _stage(
            "right_m1_fixed",
            400,
            ("fixed_cache_preparation",),
            simulation=True,
        ),
        _stage(
            "final_native_extraction_geometry_validation",
            800,
            ("left_hippocampus_fixed", "right_m1_fixed"),
        ),
        _stage(
            "derived_nifti_qc",
            1600,
            ("final_native_extraction_geometry_validation",),
        ),
    )


@dataclass(frozen=True)
class RetryPolicy:
    max_automatic_retries: int = 1

    def action_for_attempt(self, attempt: int) -> str:
        if attempt < 0:
            raise ValueError("attempt must be non-negative")
        if attempt == 0:
            return "run"
        if attempt <= self.max_automatic_retries:
            return "retry"
        return "terminal_failure"


def preflight(
    *,
    config: StudyConfigV2,
    evidence: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    """Validate a full-scope pilot candidate without creating its output root."""

    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"V2 output root must be new and isolated: {output_root}")
    checks = {
        "simnibs_version": evidence.get("simnibs_version") == "4.6.0",
        "simnibs_module": evidence.get("simnibs_module") == config.simnibs_module,
        "scientific_python_ready": evidence.get("scientific_python_ready") is True,
        "meshing_api": evidence.get("meshing_api") is True,
        "fem_tdcs_api": evidence.get("fem_tdcs_api") is True,
        "native_msh_float64_round_trip": evidence.get(
            "native_msh_float64_round_trip"
        )
        is True,
        "input_hashes_valid": evidence.get("input_hashes_valid") is True,
        "storage_available": int(evidence.get("available_storage_bytes", -1))
        >= int(evidence.get("required_storage_bytes", 0))
        > 0,
    }
    failed = [name for name, ready in checks.items() if not ready]
    if failed:
        raise ValueError(f"V2 preflight failed checks: {failed}")
    return {
        "schema_version": 2,
        "status": "ready_for_pilot",
        "output_root": str(output_root.resolve(strict=False)),
        "output_root_created": False,
        "scope": {
            "subjects": len(config.subjects),
            "targets": len(config.targets),
            "conditions": len(config.conditions),
            "repeats": config.repeat_count,
            "simulations": config.expected_simulations,
            "component_solves": config.expected_component_solves,
        },
        "checks": checks,
        "resources": {
            "cpus": 8,
            "memory_gb": 32,
            "walltime": "08:00:00",
            "concurrency": 50,
            "status": "candidate_pending_pilot",
        },
        "retry_policy": {
            "max_automatic_retries": 1,
            "terminal_failure_logs_immutable": True,
        },
    }


def production_release_gate(
    *,
    local_validation: Mapping[str, object],
    hpc_pilot: Mapping[str, object],
) -> dict[str, str]:
    if local_validation.get("status") != "ready":
        raise ValueError("Local validation receipt is not ready")
    if hpc_pilot.get("status") != "ready":
        raise ValueError("HPC pilot receipt is not ready")
    if hpc_pilot.get("local_hpc_comparison") != "passed":
        raise ValueError("HPC pilot lacks a passed local-versus-HPC comparison")
    return {
        "status": "ready_for_production",
        "local_validation": "passed",
        "hpc_pilot": "passed",
        "local_hpc_comparison": "passed",
    }
