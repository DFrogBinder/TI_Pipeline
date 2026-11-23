"""Slurm array mapping and command construction for the v2 stage graph."""

from __future__ import annotations

from pathlib import Path

from .config import CANONICAL_SUBJECTS
from .orchestration import StageSpec


SIMULATION_STAGE = {
    "left_hippocampus_remesh": ("left_hippocampus", "remesh"),
    "right_m1_remesh": ("right_m1", "remesh"),
    "left_hippocampus_fixed": ("left_hippocampus", "fixed_mesh"),
    "right_m1_fixed": ("right_m1", "fixed_mesh"),
}


def _simulation_task(target: str, condition: str, index: int) -> dict[str, object]:
    if index < 0 or index >= len(CANONICAL_SUBJECTS) * 40:
        raise IndexError(f"Simulation task index outside 0..399: {index}")
    return {
        "subject_index": index // 40,
        "target": target,
        "condition": condition,
        "repeat": index % 40 + 1,
    }


def task_for_stage(stage: str, index: int) -> dict[str, object]:
    if stage in SIMULATION_STAGE:
        target, condition = SIMULATION_STAGE[stage]
        return _simulation_task(target, condition, index)
    if stage == "fixed_cache_preparation":
        if index < 0 or index >= 20:
            raise IndexError(f"Fixed-cache task index outside 0..19: {index}")
        return {
            "subject_index": index // 2,
            "target": ("left_hippocampus", "right_m1")[index % 2],
        }
    if stage in {
        "remesh_native_extraction_validation",
        "final_native_extraction_geometry_validation",
    }:
        if index < 0 or index >= 800:
            raise IndexError(f"Extraction task index outside 0..799: {index}")
        condition = (
            "remesh"
            if stage == "remesh_native_extraction_validation"
            else "fixed_mesh"
        )
        target = "left_hippocampus" if index < 400 else "right_m1"
        return _simulation_task(target, condition, index % 400)
    if stage == "derived_nifti_qc":
        if index < 0 or index >= 1600:
            raise IndexError(f"NIfTI QC task index outside 0..1599: {index}")
        block, local = divmod(index, 400)
        condition, target = (
            ("remesh", "left_hippocampus"),
            ("remesh", "right_m1"),
            ("fixed_mesh", "left_hippocampus"),
            ("fixed_mesh", "right_m1"),
        )[block]
        return _simulation_task(target, condition, local)
    raise ValueError(f"Unknown v2 production stage: {stage}")


def build_sbatch_command(
    *,
    stage: StageSpec,
    worker_script: Path,
    dependency_job_ids: tuple[str, ...],
    profile: Path,
) -> list[str]:
    if not profile.is_absolute():
        profile = profile.resolve()
    command = [
        "sbatch",
        "--parsable",
        f"--array=0-{stage.task_count - 1}%{stage.concurrency}",
        f"--cpus-per-task={stage.cpus}",
        f"--mem={stage.memory_gb}G",
        f"--time={stage.walltime}",
        f"--job-name=repv2-{stage.name}",
        (
            "--export=ALL,"
            f"V2_STAGE={stage.name},V2_PROFILE={profile}"
        ),
    ]
    if dependency_job_ids:
        if any(not value.isdigit() for value in dependency_job_ids):
            raise ValueError("Slurm dependency job IDs must be numeric")
        command.append("--dependency=afterok:" + ":".join(dependency_job_ids))
    command.append(str(worker_script))
    return command

