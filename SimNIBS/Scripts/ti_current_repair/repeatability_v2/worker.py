"""Execute one indexed stage task from the full-scope Slurm graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ti_current_repair.stimulation_config import resolve_confirmed_stimulation

from .cache import prepare_pair_caches, select_representative_repeat
from .config import CANONICAL_SUBJECTS
from .execution import (
    SimulationTask,
    SourceInputs,
    run_fixed_task,
    run_remesh_task,
)
from .hpc import task_for_stage
from .native_pipeline import extract_task_manifest
from .precision import read_metric_table
from .profile import OperationalProfile
from .qc import export_manifest_qc
from .simnibs_adapter import (
    CorrectedCharmHeadBuilder,
    SimNIBS46Adapter,
    SimNIBS46RemeshAdapter,
)


def simulation_manifest_path(
    *,
    output_root: Path,
    subject: str,
    target: str,
    condition: str,
    repeat: int,
) -> Path:
    return (
        output_root
        / subject
        / target
        / condition
        / f"repeat_{repeat:03d}"
        / "task_manifest.json"
    )


def _task(profile: OperationalProfile, mapping: dict[str, object]) -> SimulationTask:
    subject = CANONICAL_SUBJECTS[int(mapping["subject_index"])]
    return SimulationTask(
        subject=subject,
        target=str(mapping["target"]),
        condition=str(mapping["condition"]),
        repeat=int(mapping["repeat"]),
    )


def _run_remesh(profile: OperationalProfile, task: SimulationTask) -> dict[str, object]:
    subject_inputs = profile.subject_inputs[task.subject]
    stimulation = resolve_confirmed_stimulation(
        profile.montage_presets[task.target], targets_csv=profile.targets_csv
    )
    workspace = (
        profile.output_root
        / "_head_workspaces"
        / task.subject
        / task.target
        / task.repeat_tag
    )
    builder = CorrectedCharmHeadBuilder(
        subject=task.subject,
        t1_path=subject_inputs.t1,
        t2_path=subject_inputs.t2,
        workspace=workspace,
    )
    adapter = SimNIBS46RemeshAdapter(
        stimulation=stimulation,
        head_builder=builder,
        eeg_cap_path=builder.eeg_cap_path,
        n_workers=8,
    )
    inputs = SourceInputs(
        corrected_segmentation=subject_inputs.corrected_segmentation,
        atlas=subject_inputs.atlas,
        targets_csv=profile.targets_csv,
    )
    config = profile.study_config
    return run_remesh_task(
        task=task,
        inputs=inputs,
        expected_hashes={
            inputs.corrected_segmentation: config.corrected_segmentation_sha256[
                task.subject
            ],
            inputs.atlas: config.atlas_sha256[task.subject],
            inputs.targets_csv: config.targets_csv_sha256,
        },
        output_root=profile.output_root,
        adapter=adapter,
    )


def _run_fixed(profile: OperationalProfile, task: SimulationTask) -> dict[str, object]:
    cache_manifest = (
        profile.cache_root / task.subject / task.target / "cache_manifest.json"
    )
    return run_fixed_task(
        task=task,
        cache_manifest=cache_manifest,
        output_root=profile.output_root,
        adapter=SimNIBS46Adapter(),
    )


def _extract(profile: OperationalProfile, task: SimulationTask) -> dict[str, object]:
    manifest = simulation_manifest_path(
        output_root=profile.output_root,
        subject=task.subject,
        target=task.target,
        condition=task.condition,
        repeat=task.repeat,
    )
    if task.condition == "fixed_mesh":
        expected: dict[str, str] | None = None
        for repeat in range(1, 41):
            candidate = json.loads(
                simulation_manifest_path(
                    output_root=profile.output_root,
                    subject=task.subject,
                    target=task.target,
                    condition="fixed_mesh",
                    repeat=repeat,
                ).read_text(encoding="utf-8")
            )
            geometry = {
                str(key): str(value)
                for key, value in candidate["geometry_sha256"].items()
            }
            if expected is None:
                expected = geometry
            elif geometry != expected:
                raise ValueError(
                    f"Fixed geometry changed across repeats for {task.subject} {task.target}"
                )
    return extract_task_manifest(
        manifest_path=manifest,
        atlas_path=profile.subject_inputs[task.subject].atlas,
        output_root=manifest.parent / "_native_metrics",
    )


def _prepare_cache(
    profile: OperationalProfile, mapping: dict[str, object]
) -> dict[str, object]:
    subject = CANONICAL_SUBJECTS[int(mapping["subject_index"])]
    target = str(mapping["target"])
    rows: list[dict[str, object]] = []
    for repeat in range(1, 41):
        manifest = simulation_manifest_path(
            output_root=profile.output_root,
            subject=subject,
            target=target,
            condition="remesh",
            repeat=repeat,
        )
        table = manifest.parent / "_native_metrics" / "ti_native_mesh_metrics.csv"
        loaded = read_metric_table(
            table,
            float_fields=("roi_median_v_per_m",),
            float_precision="round_trip",
        )
        if len(loaded) != 1:
            raise ValueError(f"Expected one TI metric row in {table}")
        rows.append({**loaded[0], "repeat": repeat})
    selected = select_representative_repeat(rows)
    source_manifest_path = simulation_manifest_path(
        output_root=profile.output_root,
        subject=subject,
        target=target,
        condition="remesh",
        repeat=int(selected["repeat"]),
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["selection"] = {
        "method": "nearest_native_ti_target_roi_median",
        "tie_break": "lowest_repeat_number",
        "cell_median_v_per_m": selected["cell_median_v_per_m"],
    }
    return prepare_pair_caches(
        remesh_manifest=source_manifest,
        cache_root=profile.cache_root,
        adapter=SimNIBS46Adapter(),
    )


def _qc(profile: OperationalProfile, task: SimulationTask) -> dict[str, object]:
    manifest = simulation_manifest_path(
        output_root=profile.output_root,
        subject=task.subject,
        target=task.target,
        condition=task.condition,
        repeat=task.repeat,
    )
    return export_manifest_qc(
        manifest_path=manifest,
        reference_path=profile.subject_inputs[task.subject].atlas,
        output_root=manifest.parent,
    )


def run_stage(
    *, profile: OperationalProfile, stage: str, task_index: int
) -> dict[str, object]:
    mapping = task_for_stage(stage, task_index)
    if stage in {"left_hippocampus_remesh", "right_m1_remesh"}:
        return _run_remesh(profile, _task(profile, mapping))
    if stage in {"left_hippocampus_fixed", "right_m1_fixed"}:
        return _run_fixed(profile, _task(profile, mapping))
    if stage in {
        "remesh_native_extraction_validation",
        "final_native_extraction_geometry_validation",
    }:
        return _extract(profile, _task(profile, mapping))
    if stage == "fixed_cache_preparation":
        return _prepare_cache(profile, mapping)
    if stage == "derived_nifti_qc":
        return _qc(profile, _task(profile, mapping))
    raise ValueError(f"Unknown v2 production stage: {stage}")
