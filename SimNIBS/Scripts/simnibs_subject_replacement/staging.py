"""Adapter around the existing final-132 corrected-input staging utility."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
STAGING_SCRIPT = (
    SCRIPTS_ROOT
    / "output"
    / "spreadsheet"
    / "final_132_repeatability_balanced_10"
    / "stage_repeatability_dataset.py"
)


def _load_staging_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_replacement_final132_stage", STAGING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load staging utility: {STAGING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_corrected_sources(source_root: Path, subject: str):
    module = _load_staging_module()
    return module.audit(source_root, [subject], "final132-scaffold")


def stage_corrected_dataset(
    source_root: Path,
    output_root: Path,
    subjects_file: Path,
    subject: str,
    ready,
) -> None:
    module = _load_staging_module()
    module.stage(
        source_root,
        output_root,
        subjects_file,
        [subject],
        ready,
        "final132-scaffold",
    )
