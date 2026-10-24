"""Pinned configuration and path isolation for the harmonized v3 campaign."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ENV_PREFIX = "SIMNIBS326H_"
PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = PACKAGE_ROOT.parent
TARGETS_CSV = Path(
    os.environ.get(
        f"{ENV_PREFIX}TARGETS_CSV",
        str(SCRIPTS_ROOT / "utils" / "targets.csv"),
    )
).expanduser().resolve(strict=False)
SIMNIBS_MODULE = "SimNIBS/3.2.6-foss-2023a"
PARTITION = "sheffield"
CPUS_PER_TASK = 8
MEMORY = "32G"
TIME_LIMIT = "08:00:00"
MAX_CONCURRENT = 50
MAX_RETRIES = 1
REPEAT_COUNT = 40

SUBJECTS = (
    "sub-CC110174",
    "sub-CC121144",
    "sub-CC310407",
    "sub-CC320616",
    "sub-CC420071",
    "sub-CC410432",
    "sub-CC520083",
    "sub-CC520127",
    "sub-CC610631",
    "sub-CC720941",
)
BASE9_EXCLUDED_SUBJECT = "sub-CC520083"
BASE9_SUBJECTS = tuple(
    subject for subject in SUBJECTS if subject != BASE9_EXCLUDED_SUBJECT
)
COHORT_ENV = f"{ENV_PREFIX}COHORT"
EXTERNAL_MANIFEST_ENV = f"{ENV_PREFIX}EXTERNAL_MANIFEST"
COHORT_PROFILES = ("full10", "base9", "external1")
_CANONICAL_SUBJECT_RE = re.compile(r"^sub-CC[0-9]{6}$")


@dataclass(frozen=True)
class CampaignPaths:
    corrected_root: Path
    legacy_reference_root: Path
    atlas_root: Path
    headmodel_root: Path
    left_root: Path
    right_root: Path
    completed_v3_left_root: Path
    completed_v3_right_root: Path
    completed_v4_left_root: Path
    completed_v4_right_root: Path


@dataclass(frozen=True)
class SubjectPaths:
    subject: str
    corrected_label: Path
    corrected_t1: Path
    corrected_t2: Path
    legacy_anat: Path
    legacy_m2m: Path
    legacy_ready_marker: Path
    staged_anat: Path
    staged_m2m: Path
    harmonized_label: Path
    harmonization_qc: Path
    harmonization_figure: Path
    head_mesh: Path
    compatibility_mesh: Path
    reference_receipt: Path
    headmodel_receipt: Path


@dataclass(frozen=True)
class TargetSettings:
    key: str
    label: str
    experiment_root: Path


_DEFAULTS = {
    "CORRECTED_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_corrected"
    ),
    "LEGACY_REFERENCE_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_headreco"
    ),
    "ATLAS_ROOT": "/mnt/parscratch/users/cop23bi/ZIPs/atlases",
    "HEADMODEL_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "ti_dataset_final_132_balanced_10_simnibs326_harmonized_v2"
    ),
    "LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "left_hippocampus_v2"
    ),
    "RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_right_m1_v2"
    ),
    "COMPLETED_V3_LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_left_hippocampus_v1"
    ),
    "COMPLETED_V3_RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_right_m1_v1"
    ),
    "COMPLETED_V4_LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/final_132_repeatability_balanced_10"
    ),
    "COMPLETED_V4_RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_right_m1"
    ),
}

_BASE9_ROOT_DEFAULTS = {
    "LEFT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_left_hippocampus_v2"
    ),
    "RIGHT_ROOT": (
        "/mnt/parscratch/users/cop23bi/"
        "final_132_repeatability_balanced_10_simnibs326_harmonized_"
        "base9_right_m1_v2"
    ),
}


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve(strict=False)
    right = right.resolve(strict=False)
    return left == right or left in right.parents or right in left.parents


def cohort_name(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    name = source.get(COHORT_ENV, "full10").strip().lower()
    if name not in COHORT_PROFILES:
        raise ValueError(
            f"Unsupported cohort profile {name!r}; expected one of {COHORT_PROFILES}"
        )
    return name


def load_external_manifest(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Load and validate the manifest that authorizes one external subject."""
    source = os.environ if environ is None else environ
    raw_path = source.get(EXTERNAL_MANIFEST_ENV, "").strip()
    if not raw_path:
        raise ValueError(
            f"{EXTERNAL_MANIFEST_ENV} is required for the external1 cohort"
        )
    manifest_path = Path(raw_path).expanduser()
    if not manifest_path.is_absolute():
        raise ValueError(f"{EXTERNAL_MANIFEST_ENV} must be an absolute path")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{EXTERNAL_MANIFEST_ENV} must reference readable, valid JSON: "
            f"{manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("External manifest must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("External manifest schema_version must be 1")
    if payload.get("workflow") != "simnibs_subject_replacement":
        raise ValueError(
            "External manifest workflow must be simnibs_subject_replacement"
        )
    if payload.get("profile") != "external1":
        raise ValueError("External manifest profile must be external1")
    if payload.get("status") != "prepared":
        raise ValueError("External manifest status must be prepared")
    subjects = payload.get("subjects")
    if not isinstance(subjects, list) or len(subjects) != 1:
        raise ValueError("External manifest must bind exactly one subject")
    subject = payload.get("subject")
    if subject != subjects[0]:
        raise ValueError("External manifest subject must match subjects[0]")
    if not isinstance(subject, str) or not _CANONICAL_SUBJECT_RE.fullmatch(subject):
        raise ValueError("External manifest subject must be a canonical subject ID")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("External manifest must contain path bindings")
    bound_manifest = paths.get("manifest")
    if not isinstance(bound_manifest, str) or _resolved(bound_manifest) != _resolved(
        manifest_path
    ):
        raise ValueError("External manifest is bound to a different path")
    return payload


def _validate_external_root_bindings(
    payload: dict[str, object],
    paths: CampaignPaths,
    source: Mapping[str, str],
) -> None:
    """Fail closed if runtime roots differ from the prepared replacement manifest."""
    manifest_paths = payload.get("paths")
    inputs = payload.get("inputs")
    if not isinstance(manifest_paths, dict) or not isinstance(inputs, dict):
        raise ValueError("External manifest lacks required path/input bindings")

    expected_roots = {
        "CORRECTED_ROOT": (paths.corrected_root, "v4_corrected_root"),
        "HEADMODEL_ROOT": (paths.headmodel_root, "v3_headmodel_root"),
        "LEFT_ROOT": (paths.left_root, "v3_left_root"),
        "RIGHT_ROOT": (paths.right_root, "v3_right_root"),
    }
    for environment_name, (observed, manifest_name) in expected_roots.items():
        bound = manifest_paths.get(manifest_name)
        if not isinstance(bound, str) or observed != _resolved(bound):
            raise ValueError(
                f"{ENV_PREFIX}{environment_name} does not match the external "
                f"manifest binding {manifest_name}"
            )

    subject = str(payload["subject"])
    legacy = inputs.get("legacy_reference")
    atlas = inputs.get("atlas")
    targets = inputs.get("targets_csv")
    if not isinstance(legacy, dict) or not isinstance(legacy.get("anat_root"), str):
        raise ValueError("External manifest lacks the legacy reference binding")
    if _resolved(str(legacy["anat_root"])) != (
        paths.legacy_reference_root / subject / "anat"
    ):
        raise ValueError(
            f"{ENV_PREFIX}LEGACY_REFERENCE_ROOT does not match the external "
            "manifest binding"
        )
    if not isinstance(atlas, dict) or not isinstance(atlas.get("path"), str):
        raise ValueError("External manifest lacks the atlas binding")
    if _resolved(str(atlas["path"])) != paths.atlas_root / f"{subject}.nii.gz":
        raise ValueError(
            f"{ENV_PREFIX}ATLAS_ROOT does not match the external manifest binding"
        )
    if not isinstance(targets, dict) or not isinstance(targets.get("path"), str):
        raise ValueError("External manifest lacks the targets CSV binding")
    runtime_targets = _resolved(
        source.get(f"{ENV_PREFIX}TARGETS_CSV", str(TARGETS_CSV))
    )
    if _resolved(str(targets["path"])) != runtime_targets:
        raise ValueError(
            f"{ENV_PREFIX}TARGETS_CSV does not match the external manifest binding"
        )


def cohort_subjects(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    profile = cohort_name(environ)
    if profile == "base9":
        return BASE9_SUBJECTS
    if profile == "external1":
        return (str(load_external_manifest(environ)["subject"]),)
    return SUBJECTS


def campaign_paths_from_environment(
    environ: Mapping[str, str] | None = None,
) -> CampaignPaths:
    """Resolve every campaign root from one namespaced environment contract."""
    source = os.environ if environ is None else environ

    profile = cohort_name(source)

    def get(key: str) -> Path:
        return _resolved(source.get(f"{ENV_PREFIX}{key}", _DEFAULTS[key]))

    def get_result_root(key: str) -> Path:
        if profile == "base9":
            return _resolved(
                source.get(
                    f"{ENV_PREFIX}BASE9_{key}",
                    _BASE9_ROOT_DEFAULTS[key],
                )
            )
        return get(key)

    full10_roots = {
        "left": get("LEFT_ROOT"),
        "right": get("RIGHT_ROOT"),
    }
    base9_roots = {
        "left": _resolved(
            source.get(
                f"{ENV_PREFIX}BASE9_LEFT_ROOT",
                _BASE9_ROOT_DEFAULTS["LEFT_ROOT"],
            )
        ),
        "right": _resolved(
            source.get(
                f"{ENV_PREFIX}BASE9_RIGHT_ROOT",
                _BASE9_ROOT_DEFAULTS["RIGHT_ROOT"],
            )
        ),
    }
    for base9_name, base9_root in base9_roots.items():
        for full10_name, full10_root in full10_roots.items():
            if _paths_overlap(base9_root, full10_root):
                raise ValueError(
                    "base9 writable result root overlaps a full10 writable result "
                    f"root: base9_{base9_name}={base9_root}; "
                    f"full10_{full10_name}={full10_root}"
                )

    resolved_paths = CampaignPaths(
        corrected_root=get("CORRECTED_ROOT"),
        legacy_reference_root=get("LEGACY_REFERENCE_ROOT"),
        atlas_root=get("ATLAS_ROOT"),
        headmodel_root=get("HEADMODEL_ROOT"),
        left_root=get_result_root("LEFT_ROOT"),
        right_root=get_result_root("RIGHT_ROOT"),
        completed_v3_left_root=get("COMPLETED_V3_LEFT_ROOT"),
        completed_v3_right_root=get("COMPLETED_V3_RIGHT_ROOT"),
        completed_v4_left_root=get("COMPLETED_V4_LEFT_ROOT"),
        completed_v4_right_root=get("COMPLETED_V4_RIGHT_ROOT"),
    )
    if profile == "external1":
        _validate_external_root_bindings(
            load_external_manifest(source), resolved_paths, source
        )
    return resolved_paths


def subject_paths(paths: CampaignPaths, subject: str) -> SubjectPaths:
    profile = cohort_name()
    allowed_subjects = cohort_subjects()
    if subject not in allowed_subjects:
        if profile == "external1":
            raise ValueError(
                f"Unsupported subject {subject!r}; expected the active external1 "
                f"subject {allowed_subjects[0]!r}"
            )
        raise ValueError(
            f"Unsupported subject {subject!r}; expected one of {allowed_subjects}"
        )
    corrected_anat = paths.corrected_root / subject / "anat"
    legacy_anat = paths.legacy_reference_root / subject / "anat"
    staged_anat = paths.headmodel_root / subject / "anat"
    receipt_root = paths.headmodel_root / "_pipeline" / "subjects" / subject
    return SubjectPaths(
        subject=subject,
        corrected_label=(
            corrected_anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"
        ),
        corrected_t1=corrected_anat / f"{subject}_T1w.nii",
        corrected_t2=corrected_anat / f"{subject}_T2w.nii",
        legacy_anat=legacy_anat,
        legacy_m2m=legacy_anat / f"m2m_{subject}",
        legacy_ready_marker=legacy_anat / ".simnibs326_scaffold_ready.json",
        staged_anat=staged_anat,
        staged_m2m=staged_anat / f"m2m_{subject}",
        harmonized_label=(
            paths.headmodel_root
            / "harmonized_labels"
            / f"{subject}_v3_harmonized_t1fs_grid.nii.gz"
        ),
        harmonization_qc=(
            paths.headmodel_root / "harmonized_labels" / "qc" / f"{subject}.json"
        ),
        harmonization_figure=(
            paths.headmodel_root / "harmonized_labels" / "figures" / f"{subject}.png"
        ),
        head_mesh=staged_anat / f"{subject}.msh",
        compatibility_mesh=staged_anat / f"m2m_{subject}" / f"{subject}.msh",
        reference_receipt=receipt_root / "reference_stage.json",
        headmodel_receipt=receipt_root / "headmodel_complete.json",
    )


def target_settings(value: str, paths: CampaignPaths | None = None) -> TargetSettings:
    resolved = paths or campaign_paths_from_environment()
    targets = {
        "left-hippocampus": TargetSettings(
            "left-hippocampus", "left hippocampus", resolved.left_root
        ),
        "right-m1": TargetSettings("right-m1", "right M1", resolved.right_root),
    }
    try:
        return targets[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported target {value!r}; expected one of {sorted(targets)}"
        ) from exc


def assert_root_isolation(paths: CampaignPaths) -> dict[str, dict[str, str]]:
    """Fail before writes if a new root overlaps any writable/protected root."""
    writable = {
        "headmodel_root": paths.headmodel_root,
        "left_root": paths.left_root,
        "right_root": paths.right_root,
    }
    protected = {
        "corrected_root": paths.corrected_root,
        "legacy_reference_root": paths.legacy_reference_root,
        "atlas_root": paths.atlas_root,
        "completed_v3_left_root": paths.completed_v3_left_root,
        "completed_v3_right_root": paths.completed_v3_right_root,
        "completed_v4_left_root": paths.completed_v4_left_root,
        "completed_v4_right_root": paths.completed_v4_right_root,
    }
    writable_items = list(writable.items())
    for index, (left_name, left_path) in enumerate(writable_items):
        for right_name, right_path in writable_items[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise RuntimeError(
                    "Writable roots overlap: "
                    f"{left_name}={left_path}; {right_name}={right_path}"
                )
    for writable_name, writable_path in writable.items():
        for protected_name, protected_path in protected.items():
            if _paths_overlap(writable_path, protected_path):
                raise RuntimeError(
                    "Writable and protected roots overlap: "
                    f"{writable_name}={writable_path}; "
                    f"{protected_name}={protected_path}"
                )
    return {
        "writable_roots": {name: str(path) for name, path in writable.items()},
        "protected_read_only_roots": {
            name: str(path) for name, path in protected.items()
        },
    }


def campaign_scope(
    paths: CampaignPaths | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    resolved = paths or campaign_paths_from_environment(environ)
    subjects = cohort_subjects(environ)
    per_target_condition = len(subjects) * REPEAT_COUNT
    return {
        "cohort": cohort_name(environ),
        "subjects": list(subjects),
        "subject_count": len(subjects),
        "targets": ["left-hippocampus", "right-m1"],
        "conditions": ["remesh", "fixed_mesh"],
        "repeats_per_condition": REPEAT_COUNT,
        "total_simulation_tasks": per_target_condition * 2 * 2,
        "component_fem_solves": per_target_condition * 2 * 2 * 2,
        "arrays": {
            "left_hippocampus_remesh": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
            "left_hippocampus_fixed": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
            "right_m1_remesh": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
            "right_m1_fixed": f"0-{per_target_condition - 1}%{MAX_CONCURRENT}",
        },
        "resources": {
            "simnibs_module": SIMNIBS_MODULE,
            "partition": PARTITION,
            "cpus_per_task": CPUS_PER_TASK,
            "memory": MEMORY,
            "time_limit": TIME_LIMIT,
            "max_concurrent": MAX_CONCURRENT,
            "max_retries": MAX_RETRIES,
        },
        "root_isolation": assert_root_isolation(resolved),
    }
