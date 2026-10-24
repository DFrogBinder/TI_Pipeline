"""Identity, paths, and isolation for one SimNIBS 3.2.6 scaffold."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

from simnibs_326_corrected_label_repeatability.settings import (
    campaign_paths_from_environment,
)
from simnibs_326_repeatability.settings import (
    ATLAS_DIR,
    PROTECTED_CHARM_ROOTS,
    SCAFFOLD_ROOT as LEGACY_SCAFFOLD_ROOT,
    SOURCE_ROOT as LEGACY_SOURCE_ROOT,
)


SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FINAL132_SUBJECTS = (
    SCRIPTS_ROOT
    / "CamCan_Experiment"
    / "cohort_pipeline"
    / "cohorts"
    / "final_132"
    / "subjects.txt"
)
DEFAULT_CORRECTED_SOURCE_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/CamCan_Corrected_v4_Scaffolds"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/parscratch/users/cop23bi/simnibs326_subject_scaffolds_v1"
)
SUBJECT_PATTERN = re.compile(r"^sub-CC[0-9]{6}$")


@dataclass(frozen=True)
class CreatorConfig:
    subject: str
    corrected_source_root: Path = DEFAULT_CORRECTED_SOURCE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    final132_subjects_file: Path = DEFAULT_FINAL132_SUBJECTS


@dataclass(frozen=True)
class CreatorPaths:
    subject_anat: Path
    module_receipt: Path
    owner_receipt: Path
    completion_receipt: Path
    log_dir: Path


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _overlap(left: Path, right: Path) -> bool:
    left = _resolved(left)
    right = _resolved(right)
    return left == right or left in right.parents or right in left.parents


def validate_subject(config: CreatorConfig) -> tuple[str, ...]:
    if not SUBJECT_PATTERN.fullmatch(config.subject):
        raise ValueError("subject must be a canonical sub-CC###### subject ID")
    try:
        cohort = tuple(
            line.strip()
            for line in config.final132_subjects_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise ValueError(
            "Authoritative final-132 subject list is unreadable: "
            f"{config.final132_subjects_file}"
        ) from exc
    invalid = [value for value in cohort if not SUBJECT_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(
            f"Authoritative subject list contains non-canonical IDs: {invalid}"
        )
    duplicates = sorted(
        value for value, count in Counter(cohort).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Authoritative subject list contains duplicate IDs: {duplicates}"
        )
    if cohort.count(config.subject) != 1:
        raise ValueError(
            f"Subject {config.subject} must occur exactly once in final-132"
        )
    return cohort


def creator_paths(config: CreatorConfig) -> CreatorPaths:
    if not SUBJECT_PATTERN.fullmatch(config.subject):
        raise ValueError("subject must be a canonical sub-CC###### subject ID")
    output = _resolved(config.output_root)
    metadata = output / "_simnibs326"
    subject_metadata = metadata / "subjects" / config.subject
    return CreatorPaths(
        subject_anat=output / config.subject / "anat",
        module_receipt=metadata / "module_preflight.json",
        owner_receipt=subject_metadata / "owner.json",
        completion_receipt=subject_metadata / "complete.json",
        log_dir=metadata / "logs" / config.subject,
    )


def production_paths():
    """Return the default corrected-v3 production roots for audit and tests."""
    return campaign_paths_from_environment({})


def assert_output_root_isolation(
    config: CreatorConfig,
) -> dict[str, dict[str, str]]:
    output = _resolved(config.output_root)
    production = production_paths()
    protected = {
        "repository_scripts": SCRIPTS_ROOT,
        "authoritative_subjects": config.final132_subjects_file,
        "corrected_source": config.corrected_source_root,
        "legacy_source": LEGACY_SOURCE_ROOT,
        "legacy_scaffolds": LEGACY_SCAFFOLD_ROOT,
        "atlas_root": ATLAS_DIR,
        **PROTECTED_CHARM_ROOTS,
        **{
            name: getattr(production, name)
            for name in (
                "corrected_root",
                "legacy_reference_root",
                "atlas_root",
                "headmodel_root",
                "left_root",
                "right_root",
                "completed_v3_left_root",
                "completed_v3_right_root",
                "completed_v4_left_root",
                "completed_v4_right_root",
            )
        },
    }
    for name, path in protected.items():
        if _overlap(output, Path(path)):
            raise ValueError(
                "Scaffold creator writable/protected roots overlap: "
                f"output_root={output}; {name}={_resolved(Path(path))}"
            )
    return {
        "writable_roots": {"output_root": str(output)},
        "protected_read_only_roots": {
            name: str(_resolved(Path(path))) for name, path in protected.items()
        },
    }
