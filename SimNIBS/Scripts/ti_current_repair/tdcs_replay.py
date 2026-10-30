#!/usr/bin/env python3
"""Configuration, task mapping, and provenance checks for tDCS mesh replay."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from stimulation_config import TDCSStimulationConfig, resolve_provisional_tdcs_stimulation


SCHEMA_VERSION = 1
VERSION_V4 = "v4"
VERSION_V3 = "v3"
VERSIONS = (VERSION_V4, VERSION_V3)
VERSION_AUDIT_LABELS = {
    VERSION_V4: "simnibs_v4",
    VERSION_V3: "simnibs_v3_2_6",
}
VERSION_MODULES = {
    VERSION_V4: "SimNIBS/4.0.1-foss-2023a",
    VERSION_V3: "SimNIBS/3.2.6-foss-2023a",
}
VERSION_PREFIXES = {
    VERSION_V4: "4.0.1",
    VERSION_V3: "3.2.6",
}
REQUIRED_STIMULATION_FIELDS = (
    "montage_preset",
    "anode",
    "cathode",
    "current_ma",
    "electrode_radius_mm",
    "electrode_thickness_mm",
    "electrode_shape",
    "electrode_conductivity_s_per_m",
)
EXPECTED_ARM_SOLVES = {
    "simnibs_v4_remesh_heads": 400,
    "simnibs_v4_fixed_head_control": 400,
    "simnibs_v3_deterministic_head_control": 400,
}


class ReplayConfigurationError(RuntimeError):
    """Raised when a replay plan or provenance gate is incomplete."""


class ReplayInputError(RuntimeError):
    """Raised when an audited source mesh has drifted or disappeared."""


@dataclass(frozen=True)
class ReplayTask:
    version: str
    task_index: int
    arm: str
    subject: str
    condition: str
    repeat_index: int
    repeat_tag: str
    source_mesh: Path
    source_anat_dir: Path
    output_root: Path
    task_manifest: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "task_index": self.task_index,
            "arm": self.arm,
            "subject": self.subject,
            "condition": self.condition,
            "repeat_index": self.repeat_index,
            "repeat_tag": self.repeat_tag,
            "source_mesh": str(self.source_mesh),
            "source_anat_dir": str(self.source_anat_dir),
            "output_root": str(self.output_root),
            "task_manifest": str(self.task_manifest),
        }


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ReplayConfigurationError(f"Expected a JSON object: {resolved}")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _path(value: object, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ReplayConfigurationError(f"Expected non-empty path {field_name}.")
    return Path(value).expanduser().resolve()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def subjects_from_plan(plan: dict[str, Any]) -> tuple[str, ...]:
    cohort = plan.get("cohort")
    if not isinstance(cohort, dict):
        raise ReplayConfigurationError("Missing plan.cohort object.")
    raw = cohort.get("subjects")
    if not isinstance(raw, list) or not raw:
        raise ReplayConfigurationError("plan.cohort.subjects must be a non-empty list.")
    subjects = tuple(str(item).strip() for item in raw)
    if any(not item for item in subjects) or len(set(subjects)) != len(subjects):
        raise ReplayConfigurationError("Plan subjects must be non-empty and unique.")
    if int(cohort.get("subject_count", -1)) != len(subjects):
        raise ReplayConfigurationError("Plan subject_count does not match subjects.")
    return subjects


def repeat_count_from_plan(plan: dict[str, Any]) -> int:
    try:
        repeat_count = int(plan["cohort"]["repeats_per_arm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayConfigurationError("Invalid plan repeats_per_arm.") from exc
    if repeat_count != 40:
        raise ReplayConfigurationError(
            f"The full replay design requires 40 repeats per arm; observed {repeat_count}."
        )
    return repeat_count


def input_root(plan: dict[str, Any], version: str) -> Path:
    key = "simnibs_v4_root" if version == VERSION_V4 else "simnibs_v3_root"
    return _path(plan.get("existing_inputs", {}).get(key), field_name=f"existing_inputs.{key}")


def output_root(plan: dict[str, Any], version: str) -> Path:
    key = "simnibs_v4" if version == VERSION_V4 else "simnibs_v3"
    return _path(plan.get("proposed_output_roots", {}).get(key), field_name=f"proposed_output_roots.{key}")


def plan_blockers(plan: dict[str, Any], *, require_submission: bool = False) -> list[str]:
    blockers: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        blockers.append(f"schema_version must be {SCHEMA_VERSION}")
    target = plan.get("target", {})
    if target.get("roi_preset") != "right-m1":
        blockers.append("target.roi_preset must be right-m1")
    if target.get("optimizer_roi_name") != "Right_M1":
        blockers.append("target.optimizer_roi_name must be Right_M1")
    if target.get("roi_definition") != "optimizer-matched-spherical":
        blockers.append("target.roi_definition must be optimizer-matched-spherical")
    analysis = plan.get("analysis", {})
    if not isinstance(analysis, dict) or not analysis.get("atlas_dir"):
        blockers.append("analysis.atlas_dir is required")
    if plan.get("existing_inputs", {}).get("policy") != "read_only":
        blockers.append("existing_inputs.policy must be read_only")
    try:
        subjects = subjects_from_plan(plan)
        if len(subjects) != 10:
            blockers.append(f"full scope requires 10 subjects, observed {len(subjects)}")
    except ReplayConfigurationError as exc:
        blockers.append(str(exc))
    try:
        repeat_count_from_plan(plan)
    except ReplayConfigurationError as exc:
        blockers.append(str(exc))

    arms = plan.get("design", {}).get("arms", [])
    observed_arms = {
        str(item.get("name")): int(item.get("tdcs_solves", -1))
        for item in arms
        if isinstance(item, dict)
    }
    if observed_arms != EXPECTED_ARM_SOLVES:
        blockers.append(
            "design arms must be exactly 400 v4 remesh, 400 v4 fixed, and 400 v3 deterministic solves"
        )
    if plan.get("design", {}).get("total_tdcs_solves") != 1200:
        blockers.append("design.total_tdcs_solves must be 1200")

    stimulation = plan.get("stimulation_parameters")
    if not isinstance(stimulation, dict):
        blockers.append("stimulation_parameters must be an object")
    else:
        for field in REQUIRED_STIMULATION_FIELDS:
            if stimulation.get(field) is None:
                blockers.append(f"stimulation_parameters.{field} is pending")

    try:
        inputs = [input_root(plan, version) for version in VERSIONS]
        outputs = [output_root(plan, version) for version in VERSIONS]
        if len(set(outputs)) != len(outputs):
            blockers.append("v3 and v4 output roots must be distinct")
        for output in outputs:
            for source in inputs:
                if _paths_overlap(output, source):
                    blockers.append(f"output root overlaps read-only input: {output} and {source}")
    except ReplayConfigurationError as exc:
        blockers.append(str(exc))

    if require_submission:
        if plan.get("status") != "ready_for_submission":
            blockers.append("plan.status must be ready_for_submission")
        if plan.get("submission_allowed") is not True:
            blockers.append("plan.submission_allowed must be true")
    return blockers


def validate_plan(
    plan: dict[str, Any],
    *,
    require_parameters: bool = True,
    require_submission: bool = False,
) -> None:
    blockers = plan_blockers(plan, require_submission=require_submission)
    if not require_parameters:
        blockers = [
            item for item in blockers
            if not item.startswith("stimulation_parameters.")
        ]
    if blockers:
        raise ReplayConfigurationError("Replay plan is blocked: " + "; ".join(blockers))


def stimulation_from_plan(plan: dict[str, Any]) -> TDCSStimulationConfig:
    validate_plan(plan, require_parameters=True, require_submission=False)
    values = plan["stimulation_parameters"]
    return resolve_provisional_tdcs_stimulation(
        anode=str(values["anode"]),
        cathode=str(values["cathode"]),
        current_ma=float(values["current_ma"]),
        montage_preset=str(values["montage_preset"]),
        target_roi="Right_M1",
        electrode_radius_mm=float(values["electrode_radius_mm"]),
        electrode_thickness_mm=float(values["electrode_thickness_mm"]),
        electrode_shape=str(values["electrode_shape"]),
        electrode_conductivity=float(values["electrode_conductivity_s_per_m"]),
    )


def _audit_version(audit: dict[str, Any], version: str) -> dict[str, Any]:
    label = VERSION_AUDIT_LABELS[version]
    matches = [
        item
        for item in audit.get("versions", [])
        if isinstance(item, dict) and item.get("label") == label
    ]
    if len(matches) != 1:
        raise ReplayConfigurationError(
            f"Audit must contain exactly one version record labelled {label}."
        )
    return matches[0]


def validate_audit(plan: dict[str, Any], audit: dict[str, Any]) -> None:
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise ReplayConfigurationError("Mesh audit schema version is unsupported.")
    if audit.get("mode") != "read_only_stdout_only" or audit.get("status") != "pass":
        raise ReplayConfigurationError("Mesh audit must be a passing read-only receipt.")
    subjects = subjects_from_plan(plan)
    repeat_count = repeat_count_from_plan(plan)
    for version in VERSIONS:
        item = _audit_version(audit, version)
        expected_unique = repeat_count if version == VERSION_V4 else 1
        if _path(item.get("root"), field_name=f"audit.{version}.root") != input_root(plan, version):
            raise ReplayConfigurationError(f"Audit {version} root differs from the plan.")
        if item.get("status") != "pass":
            raise ReplayConfigurationError(f"Audit {version} status is not pass.")
        if item.get("root_present") is not True:
            raise ReplayConfigurationError(f"Audit {version} root was not present.")
        if int(item.get("subjects_expected", -1)) != len(subjects):
            raise ReplayConfigurationError(f"Audit {version} subject count differs.")
        if int(item.get("subjects_passed", -1)) != len(subjects):
            raise ReplayConfigurationError(f"Audit {version} did not pass every subject.")
        if int(item.get("expected_unique_remesh_meshes_per_subject", -1)) != expected_unique:
            raise ReplayConfigurationError(f"Audit {version} mesh expectation differs.")
        rows = item.get("subjects")
        if not isinstance(rows, list):
            raise ReplayConfigurationError(f"Audit {version} has no subject records.")
        by_subject = {row.get("subject"): row for row in rows if isinstance(row, dict)}
        if set(by_subject) != set(subjects):
            raise ReplayConfigurationError(f"Audit {version} subjects differ from the plan.")
        for subject in subjects:
            row = by_subject[subject]
            remesh = row.get("remesh", {})
            records = remesh.get("mesh_records")
            fixed = row.get("fixed_cache", {})
            if row.get("status") != "pass":
                raise ReplayConfigurationError(f"Audit failed for {version}/{subject}.")
            checks = row.get("checks")
            if not isinstance(checks, dict) or not checks or not all(checks.values()):
                raise ReplayConfigurationError(f"Audit checks failed for {version}/{subject}.")
            if int(remesh.get("present", -1)) != repeat_count:
                raise ReplayConfigurationError(f"Audit mesh count differs for {version}/{subject}.")
            if int(remesh.get("readable", -1)) != repeat_count:
                raise ReplayConfigurationError(f"Audit readable count differs for {version}/{subject}.")
            if int(remesh.get("unique_sha256_count", -1)) != expected_unique:
                raise ReplayConfigurationError(f"Audit uniqueness differs for {version}/{subject}.")
            if not isinstance(records, list) or len(records) != repeat_count:
                raise ReplayConfigurationError(
                    f"Audit lacks per-repeat mesh records for {version}/{subject}."
                )
            expected_root = input_root(plan, version)
            record_hashes = set()
            for repeat_index, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    raise ReplayConfigurationError(
                        f"Audit mesh record is invalid for {version}/{subject}."
                    )
                expected_path = _remesh_mesh(expected_root, subject, repeat_index)
                if _path(record.get("path"), field_name="audit remesh path") != expected_path:
                    raise ReplayConfigurationError(
                        f"Audit remesh path differs for {version}/{subject}/repeat_{repeat_index:03d}."
                    )
                if not record.get("present") or not record.get("readable"):
                    raise ReplayConfigurationError(
                        f"Audit remesh record is unusable for {version}/{subject}."
                    )
                if not record.get("sha256") or int(record.get("size_bytes", 0)) <= 0:
                    raise ReplayConfigurationError(
                        f"Audit remesh content provenance is incomplete for {version}/{subject}."
                    )
                record_hashes.add(str(record["sha256"]))
            if len(record_hashes) != expected_unique:
                raise ReplayConfigurationError(
                    f"Audit remesh hashes differ from the expected uniqueness for {version}/{subject}."
                )
            if not fixed.get("present") or not fixed.get("readable") or not fixed.get("sha256"):
                raise ReplayConfigurationError(f"Audit fixed-cache mesh is incomplete for {version}/{subject}.")
            if _path(fixed.get("path"), field_name="audit fixed-cache path") != _fixed_mesh(expected_root, subject):
                raise ReplayConfigurationError(
                    f"Audit fixed-cache path differs for {version}/{subject}."
                )
            if str(fixed["sha256"]) not in record_hashes:
                raise ReplayConfigurationError(
                    f"Audit fixed-cache hash is not a remesh hash for {version}/{subject}."
                )


def _audit_subject_record(
    audit: dict[str, Any],
    version: str,
    subject: str,
) -> dict[str, Any]:
    item = _audit_version(audit, version)
    for row in item["subjects"]:
        if row.get("subject") == subject:
            return row
    raise ReplayConfigurationError(f"Audit subject missing: {version}/{subject}")


def _remesh_mesh(root: Path, subject: str, repeat_index: int) -> Path:
    return (
        root
        / f"{subject}_repeatability"
        / "remesh"
        / "repeats"
        / f"repeat_{repeat_index:03d}"
        / subject
        / "anat"
        / f"m2m_{subject}"
        / f"{subject}.msh"
    )


def _fixed_mesh(root: Path, subject: str) -> Path:
    return (
        root
        / f"{subject}_repeatability"
        / "fixed_mesh"
        / "mesh_cache"
        / subject
        / "anat"
        / f"m2m_{subject}"
        / f"{subject}.msh"
    )


def _task_output_root(
    root: Path,
    subject: str,
    condition: str,
    repeat_tag: str,
) -> Path:
    return (
        root
        / f"{subject}_repeatability"
        / condition
        / "repeats"
        / repeat_tag
        / subject
        / "anat"
        / "SimNIBS"
    )


def build_tasks(plan: dict[str, Any], version: str) -> list[ReplayTask]:
    if version not in VERSIONS:
        raise ReplayConfigurationError(f"Unsupported version {version!r}.")
    validate_plan(plan, require_parameters=False, require_submission=False)
    subjects = subjects_from_plan(plan)
    repeat_count = repeat_count_from_plan(plan)
    source_root = input_root(plan, version)
    destination_root = output_root(plan, version)
    arms = (
        (
            ("simnibs_v4_remesh_heads", "remesh", "remesh"),
            ("simnibs_v4_fixed_head_control", "fixed_mesh", "fixed"),
        )
        if version == VERSION_V4
        else (("simnibs_v3_deterministic_head_control", "fixed_mesh", "fixed"),)
    )
    tasks: list[ReplayTask] = []
    for arm, condition, source_kind in arms:
        for subject in subjects:
            for repeat_index in range(1, repeat_count + 1):
                repeat_tag = f"repeat_{repeat_index:03d}"
                mesh = (
                    _remesh_mesh(source_root, subject, repeat_index)
                    if source_kind == "remesh"
                    else _fixed_mesh(source_root, subject)
                )
                destination = _task_output_root(
                    destination_root,
                    subject,
                    condition,
                    repeat_tag,
                )
                tasks.append(
                    ReplayTask(
                        version=version,
                        task_index=len(tasks),
                        arm=arm,
                        subject=subject,
                        condition=condition,
                        repeat_index=repeat_index,
                        repeat_tag=repeat_tag,
                        source_mesh=mesh,
                        source_anat_dir=mesh.parent.parent,
                        output_root=destination,
                        task_manifest=destination.parent.parent / "task_manifest.json",
                    )
                )
    expected = 800 if version == VERSION_V4 else 400
    if len(tasks) != expected:
        raise ReplayConfigurationError(
            f"Resolved {len(tasks)} {version} tasks; expected {expected}."
        )
    return tasks


def task_for_index(plan: dict[str, Any], version: str, task_index: int) -> ReplayTask:
    tasks = build_tasks(plan, version)
    if task_index < 0 or task_index >= len(tasks):
        raise ReplayConfigurationError(
            f"Task index {task_index} outside 0..{len(tasks) - 1} for {version}."
        )
    return tasks[task_index]


def audit_mesh_record(
    audit: dict[str, Any],
    task: ReplayTask,
) -> dict[str, Any]:
    row = _audit_subject_record(audit, task.version, task.subject)
    if task.arm == "simnibs_v4_remesh_heads":
        records = row["remesh"]["mesh_records"]
        record = records[task.repeat_index - 1]
    else:
        record = row["fixed_cache"]
    if _path(record.get("path"), field_name="audit mesh path") != task.source_mesh:
        raise ReplayConfigurationError(
            f"Audit/source path mismatch for {task.version} task {task.task_index}."
        )
    if not record.get("present") or not record.get("readable"):
        raise ReplayConfigurationError(f"Audited source mesh is not usable: {task.source_mesh}")
    if not record.get("sha256") or int(record.get("size_bytes", 0)) <= 0:
        raise ReplayConfigurationError(f"Audited source mesh lacks content provenance: {task.source_mesh}")
    return record


def validate_task_source(audit: dict[str, Any], task: ReplayTask) -> dict[str, Any]:
    record = audit_mesh_record(audit, task)
    if not task.source_mesh.is_file():
        raise ReplayInputError(f"Source mesh is missing: {task.source_mesh}")
    observed_size = task.source_mesh.stat().st_size
    expected_size = int(record["size_bytes"])
    if observed_size != expected_size:
        raise ReplayInputError(
            f"Source mesh size drifted after audit: {task.source_mesh}: "
            f"{observed_size} != {expected_size}"
        )
    t1 = task.source_anat_dir / f"{task.subject}_T1w.nii"
    if not t1.is_file() or t1.stat().st_size <= 0:
        raise ReplayInputError(f"Source T1 is missing or empty: {t1}")
    return record


def required_output_paths(task: ReplayTask) -> dict[str, Path]:
    field_root = task.output_root / "Output" / task.subject
    return {
        "field_mesh": field_root / "TDCS.msh",
        "brain_nifti": task.output_root / "tdcs_brain_only.nii.gz",
        "raw_scalar_mesh": field_root / f"{task.subject}_TDCS_1_scalar.msh",
    }


def validate_completed_task(
    *,
    plan_path: Path,
    audit_path: Path,
    plan: dict[str, Any],
    audit: dict[str, Any],
    task: ReplayTask,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    if not task.task_manifest.is_file():
        raise ReplayInputError(f"Missing task manifest: {task.task_manifest}")
    manifest = load_json(task.task_manifest)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task": task.to_dict(),
        "plan_sha256": file_sha256(plan_path),
        "audit_sha256": file_sha256(audit_path),
        "stimulation": stimulation_from_plan(plan).to_dict(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ReplayInputError(
                f"Task manifest {key} differs for {task.version}/{task.task_index}: "
                f"{task.task_manifest}"
            )
    recorded_outputs = manifest.get("outputs")
    if not isinstance(recorded_outputs, dict):
        raise ReplayInputError(f"Task manifest has no output inventory: {task.task_manifest}")
    for label, path in required_output_paths(task).items():
        record = recorded_outputs.get(label)
        if not isinstance(record, dict) or record.get("path") != str(path):
            raise ReplayInputError(f"Output provenance differs for {label}: {task.task_manifest}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ReplayInputError(f"Required output is missing or empty: {path}")
        if int(record.get("size_bytes", -1)) != path.stat().st_size:
            raise ReplayInputError(f"Output size differs from task manifest: {path}")
        recorded_hash = str(record.get("sha256", ""))
        if len(recorded_hash) != 64:
            raise ReplayInputError(f"Output SHA-256 is missing from task manifest: {path}")
        if verify_hashes and file_sha256(path) != recorded_hash:
            raise ReplayInputError(f"Output SHA-256 differs from task manifest: {path}")
    return manifest


def analysis_config(plan: dict[str, Any], version: str, *, plan_sha256: str, audit_sha256: str) -> dict[str, Any]:
    stimulation = stimulation_from_plan(plan)
    conditions = (
        [
            {
                "name": "remesh",
                "mesh_mode": "remesh",
                "repeat_count": 40,
                "description": "tDCS replay on 40 audited content-distinct SimNIBS v4 meshes",
            },
            {
                "name": "fixed_mesh",
                "mesh_mode": "fixed_mesh",
                "repeat_count": 40,
                "description": "40 tDCS solver repeats on the audited v4 fixed-cache mesh",
            },
        ]
        if version == VERSION_V4
        else [
            {
                "name": "fixed_mesh",
                "mesh_mode": "fixed_mesh",
                "repeat_count": 40,
                "description": "40 tDCS solver repeats on one audited deterministic SimNIBS 3.2.6 mesh",
            }
        ]
    )
    return {
        "source_root": str(input_root(plan, version)),
        "experiment_root": str(output_root(plan, version)),
        "subjects": list(subjects_from_plan(plan)),
        "conditions": conditions,
        "stimulation": stimulation.to_dict(),
        "analysis": {
            "roi_preset": "right-m1",
            "roi_definition": "optimizer-matched-spherical",
            "atlas_dir": str(_path(plan["analysis"]["atlas_dir"], field_name="analysis.atlas_dir")),
            "compare_metric": str(plan["analysis"].get("primary_metric", "mean_roi")),
        },
        "tdcs_replay": {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "source_policy": "read_only",
            "plan_sha256": plan_sha256,
            "audit_sha256": audit_sha256,
        },
    }


def count_tasks(tasks: Iterable[ReplayTask]) -> int:
    return sum(1 for _ in tasks)
