from __future__ import annotations

import json
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
    dataset_root: Path
    dataset_manifest_file: Path
    dataset_completion_file: Path
    dataset_plan_sha256: str
    cohort_file: Path
    cohort_sha256: str
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
        del repo_scripts_root
        dataset_root = Path(
            "/mnt/parscratch/users/cop23bi/"
            "final_132_repeatability_combined_dataset_v1"
        )
        source_root = (
            dataset_root
            / "data/simnibs4/left_hippocampus/"
            "sub-CC110174_repeatability/remesh/repeats/repeat_001"
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
        )
        return cls(
            subject="sub-CC110174",
            selection_seed=20261106,
            repeats_per_arm=10,
            dataset_root=dataset_root,
            dataset_manifest_file=dataset_root / "dataset_manifest.json",
            dataset_completion_file=dataset_root / "_consolidation/complete.json",
            dataset_plan_sha256=(
                "1ab5c604e4d965ae70a63ca9984c878db2631f89a1f99b5927196e25d696a32c"
            ),
            cohort_file=dataset_root / "subjects.txt",
            cohort_sha256=(
                "a6df80c94e83d2eb5ff85809f789a13de75d53b05eb14918e1abceebc6ccf007"
            ),
            source_root=source_root,
            output_root=dataset_root / "extensions/segmentation_determinism_v1",
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
        if "SEGDET_DATASET_ROOT" in os.environ:
            config = config.with_dataset_root(
                Path(os.environ["SEGDET_DATASET_ROOT"]).expanduser()
            )
        changes = {}
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
        return replace(config, **changes)

    def with_dataset_root(self, dataset_root: Path) -> "ExperimentConfig":
        dataset_root = dataset_root.expanduser().resolve(strict=False)
        source_root = (
            dataset_root
            / "data/simnibs4/left_hippocampus/"
            "sub-CC110174_repeatability/remesh/repeats/repeat_001"
        )
        return replace(
            self,
            dataset_root=dataset_root,
            dataset_manifest_file=dataset_root / "dataset_manifest.json",
            dataset_completion_file=dataset_root / "_consolidation/complete.json",
            cohort_file=dataset_root / "subjects.txt",
            source_root=source_root,
            output_root=dataset_root / "extensions/segmentation_determinism_v1",
            protected_roots=(source_root, *self.protected_roots[1:]),
        )

    @property
    def staged_input_root(self) -> Path:
        return self.output_root / "inputs"


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
        "dataset_root": str(config.dataset_root.resolve(strict=False)),
        "source_root": str(config.source_root.resolve(strict=False)),
        "output_root": str(config.output_root.resolve(strict=False)),
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


def resolve_staged_subject_inputs(config: ExperimentConfig) -> SubjectInputs:
    anat = config.staged_input_root / config.subject / "anat"
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


def validate_output_isolation(config: ExperimentConfig) -> None:
    dataset_root = config.dataset_root.resolve(strict=False)
    expected_source = (
        dataset_root
        / "data/simnibs4/left_hippocampus/"
        "sub-CC110174_repeatability/remesh/repeats/repeat_001"
    )
    expected_output = dataset_root / "extensions/segmentation_determinism_v1"
    expected_paths = {
        "source root": (config.source_root, expected_source),
        "output root": (config.output_root, expected_output),
        "cohort": (config.cohort_file, dataset_root / "subjects.txt"),
        "dataset manifest": (
            config.dataset_manifest_file,
            dataset_root / "dataset_manifest.json",
        ),
        "dataset completion": (
            config.dataset_completion_file,
            dataset_root / "_consolidation/complete.json",
        ),
    }
    for label, (observed, expected) in expected_paths.items():
        if observed.resolve(strict=False) != expected.resolve(strict=False):
            raise ProvenanceError(
                f"{label} is outside the approved consolidated dataset root: "
                f"{observed} != {expected}"
            )
    if not path_is_within(config.output_root, dataset_root):
        raise ProvenanceError("Output root is outside the consolidated dataset root")
    if not path_is_within(config.source_root, dataset_root):
        raise ProvenanceError("Source root is outside the consolidated dataset root")
    for root in (config.source_root, *config.protected_roots):
        if path_is_within(config.output_root, root):
            raise ProvenanceError(
                f"Output root collides with protected root: {config.output_root} within {root}"
            )


def _read_dataset_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ProvenanceError(f"Missing or unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Unreadable {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError(f"Invalid {label}: {path}")
    return payload


def _dataset_receipts(config: ExperimentConfig) -> Dict[str, object]:
    expected_root = str(config.dataset_root.resolve())
    manifest = _read_dataset_json(config.dataset_manifest_file, "dataset manifest")
    completion = _read_dataset_json(
        config.dataset_completion_file, "dataset completion receipt"
    )
    for label, payload in (
        ("dataset manifest", manifest),
        ("dataset completion receipt", completion),
    ):
        if (
            payload.get("status") != "complete"
            or payload.get("workflow") != "repeatability_dataset_consolidation"
            or payload.get("output_root") != expected_root
            or payload.get("plan_sha256") != config.dataset_plan_sha256
            or payload.get("source_outputs_modified") is not False
        ):
            raise ProvenanceError(f"Invalid {label} binding")
    manifest_binding = completion.get("dataset_manifest")
    if (
        not isinstance(manifest_binding, dict)
        or manifest_binding.get("path") != "dataset_manifest.json"
        or manifest_binding.get("sha256") != sha256_file(config.dataset_manifest_file)
    ):
        raise ProvenanceError("Invalid dataset completion manifest binding")
    if manifest.get("scope") != completion.get("scope"):
        raise ProvenanceError("Dataset manifest and completion scope differ")
    return {
        "root": expected_root,
        "plan_sha256": config.dataset_plan_sha256,
        "manifest": {
            "path": str(config.dataset_manifest_file.resolve()),
            "sha256": sha256_file(config.dataset_manifest_file),
        },
        "completion": {
            "path": str(config.dataset_completion_file.resolve()),
            "sha256": sha256_file(config.dataset_completion_file),
        },
    }


def preflight(config: ExperimentConfig) -> Dict[str, object]:
    validate_output_isolation(config)
    dataset = _dataset_receipts(config)
    subjects = _ordered_subjects(config)
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
            "dataset": dataset,
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
