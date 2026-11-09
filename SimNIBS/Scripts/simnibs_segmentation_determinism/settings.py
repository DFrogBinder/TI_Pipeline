from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

from .provenance import ProvenanceError, path_is_within, require_sha256, sha256_file


@dataclass(frozen=True)
class SubjectInputs:
    subject: str
    t1: Path
    t2: Path
    t1_sha256: str
    t2_sha256: str


@dataclass(frozen=True)
class ExperimentConfig:
    subject: str
    selection_seed: int
    repeats_per_arm: int
    cohort_file: Path
    cohort_sha256: str
    selection_audit_file: Path
    selection_audit_sha256: str
    source_root: Path
    output_root: Path
    protected_roots: Tuple[Path, ...]
    simnibs326_module: str
    matlab_module: str
    simnibs401_module: str
    partition: str
    cpus_per_task: int
    memory: str
    time_limit: str
    max_concurrent: int
    max_retries: int

    @classmethod
    def defaults(cls, repo_scripts_root: Optional[Path] = None) -> "ExperimentConfig":
        scripts_root = (
            Path(repo_scripts_root).expanduser().resolve(strict=False)
            if repo_scripts_root is not None
            else Path(__file__).resolve().parents[1]
        )
        cohort_root = scripts_root / "output/spreadsheet/final_132_repeatability_balanced_10"
        source_root = Path(
            "/mnt/parscratch/users/cop23bi/ti_dataset_final_132_balanced_10_corrected"
        )
        protected = (
            source_root,
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10"),
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1"),
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_spherical_fixed_v1"),
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_right_m1_spherical_fixed_v1"),
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_base9_left_hippocampus_v2"),
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10_simnibs326_harmonized_base9_right_m1_v2"),
            Path("/mnt/parscratch/users/cop23bi/simnibs_subject_replacements"),
            Path("/mnt/parscratch/users/cop23bi/final_132_repeatability_combined_dataset_v1"),
        )
        return cls(
            subject="sub-CC110174",
            selection_seed=20261106,
            repeats_per_arm=10,
            cohort_file=cohort_root / "subjects.txt",
            cohort_sha256="3cd591021466d414c5ab5019944b6cb698dd9de69577cf3d7cd10cc878027767",
            selection_audit_file=cohort_root / "selection_audit.xlsx",
            selection_audit_sha256="3d802d1308aee36dc46f9a6c547b35ef9eb4a5f1b96023032afb52513952868d",
            source_root=source_root,
            output_root=Path(
                "/mnt/parscratch/users/cop23bi/simnibs_segmentation_determinism_v1"
            ),
            protected_roots=protected,
            simnibs326_module="SimNIBS/3.2.6-foss-2023a",
            matlab_module="MATLAB/2023b",
            simnibs401_module="SimNIBS/4.0.1-foss-2023a",
            partition="sheffield",
            cpus_per_task=8,
            memory="32G",
            time_limit="08:00:00",
            max_concurrent=10,
            max_retries=1,
        )

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        config = cls.defaults()
        path_fields = {
            "SEGDET_SOURCE_ROOT": "source_root",
            "SEGDET_OUTPUT_ROOT": "output_root",
            "SEGDET_COHORT_FILE": "cohort_file",
            "SEGDET_SELECTION_AUDIT_FILE": "selection_audit_file",
        }
        changes = {
            field: Path(os.environ[name]).expanduser()
            for name, field in path_fields.items()
            if name in os.environ
        }
        integer_fields = {
            "SEGDET_MAX_CONCURRENT": "max_concurrent",
        }
        changes.update(
            {
                field: int(os.environ[name])
                for name, field in integer_fields.items()
                if name in os.environ
            }
        )
        if os.environ.get("SEGDET_SUBJECT", config.subject) != config.subject:
            raise ProvenanceError("SEGDET_SUBJECT cannot change the pinned experiment subject")
        if int(os.environ.get("SEGDET_REPEATS_PER_ARM", config.repeats_per_arm)) != config.repeats_per_arm:
            raise ProvenanceError("SEGDET_REPEATS_PER_ARM cannot change the pinned ten-repeat scope")
        if "source_root" in changes:
            changes["protected_roots"] = (
                changes["source_root"],
                *config.protected_roots[1:],
            )
        return replace(config, **changes)


def scope_payload(config: ExperimentConfig) -> Dict[str, object]:
    return {
        "workflow": "simnibs_segmentation_determinism",
        "subject": config.subject,
        "subjects": 1,
        "versions": 2,
        "repeats_per_arm": config.repeats_per_arm,
        "scientific_segmentation_tasks": 2 * config.repeats_per_arm,
        "scheduler_tasks": 2 * config.repeats_per_arm + 3,
        "surface_tasks": 0,
        "meshing_tasks": 0,
        "fem_simulation_tasks": 0,
        "source_outputs_modified": False,
        "resources": {
            "partition": config.partition,
            "cpus_per_task": config.cpus_per_task,
            "memory": config.memory,
            "time_limit": config.time_limit,
            "max_concurrent_per_arm": config.max_concurrent,
            "max_retries": config.max_retries,
        },
        "modules": {
            "simnibs326": config.simnibs326_module,
            "matlab": config.matlab_module,
            "simnibs401": config.simnibs401_module,
        },
    }


def _resolve_modality(anat: Path, subject: str, modality: str) -> Path:
    stem = f"{subject}_{modality}w"
    candidates = [path for path in (anat / f"{stem}.nii", anat / f"{stem}.nii.gz") if path.is_file()]
    if not candidates:
        raise ProvenanceError(f"Missing {modality} input for {subject} under {anat}")
    if len(candidates) != 1:
        raise ProvenanceError(
            f"Ambiguous {modality} input for {subject}: " + ", ".join(str(path) for path in candidates)
        )
    return candidates[0].resolve()


def resolve_subject_inputs(config: ExperimentConfig) -> SubjectInputs:
    anat = config.source_root / config.subject / "anat"
    t1 = _resolve_modality(anat, config.subject, "T1")
    t2 = _resolve_modality(anat, config.subject, "T2")
    return SubjectInputs(
        subject=config.subject,
        t1=t1,
        t2=t2,
        t1_sha256=sha256_file(t1),
        t2_sha256=sha256_file(t2),
    )


def _ordered_subjects(config: ExperimentConfig) -> list[str]:
    require_sha256(config.cohort_file, config.cohort_sha256, "cohort")
    subjects = [line.strip() for line in config.cohort_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not subjects:
        raise ProvenanceError("Pinned cohort is empty")
    if len(subjects) != len(set(subjects)):
        raise ProvenanceError("Pinned cohort contains duplicate subject IDs")
    return subjects


def _validate_output_isolation(config: ExperimentConfig) -> None:
    for root in (config.source_root, *config.protected_roots):
        if path_is_within(config.output_root, root):
            raise ProvenanceError(
                f"Output root collides with protected root: {config.output_root} within {root}"
            )


def preflight(config: ExperimentConfig) -> Dict[str, object]:
    _validate_output_isolation(config)
    subjects = _ordered_subjects(config)
    require_sha256(
        config.selection_audit_file,
        config.selection_audit_sha256,
        "selection audit",
    )
    if subjects[0] != config.subject:
        raise ProvenanceError(
            f"Pinned subject is not first in the ordered cohort: {subjects[0]} != {config.subject}"
        )
    selected = random.Random(config.selection_seed).choice(subjects)
    if selected != config.subject:
        raise ProvenanceError(
            f"Seeded selection drift: seed {config.selection_seed} selected {selected}, expected {config.subject}"
        )
    inputs = resolve_subject_inputs(config)
    payload = scope_payload(config)
    payload.update(
        {
            "status": "ready",
            "selected_subject": selected,
            "selection_seed": config.selection_seed,
            "cohort": {
                "path": str(config.cohort_file.resolve()),
                "sha256": config.cohort_sha256,
                "subjects": len(subjects),
            },
            "selection_audit": {
                "path": str(config.selection_audit_file.resolve()),
                "sha256": config.selection_audit_sha256,
            },
            "inputs": {
                "t1": {
                    "path": str(inputs.t1),
                    "size_bytes": inputs.t1.stat().st_size,
                    "sha256": inputs.t1_sha256,
                },
                "t2": {
                    "path": str(inputs.t2),
                    "size_bytes": inputs.t2.stat().st_size,
                    "sha256": inputs.t2_sha256,
                },
            },
            "output_root": str(config.output_root.resolve(strict=False)),
        }
    )
    return payload
