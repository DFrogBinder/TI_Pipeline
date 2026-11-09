from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest


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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def configured_experiment(tmp_path: Path):
    from simnibs_segmentation_determinism.settings import ExperimentConfig

    cohort = tmp_path / "subjects.txt"
    cohort.write_text("\n".join(SUBJECTS) + "\n", encoding="utf-8")
    audit = tmp_path / "selection_audit.xlsx"
    audit.write_bytes(b"tracked selection audit fixture\n")
    source = tmp_path / "source"
    anat = source / SUBJECTS[0] / "anat"
    anat.mkdir(parents=True)
    (anat / f"{SUBJECTS[0]}_T1w.nii").write_bytes(b"T1 fixture")
    (anat / f"{SUBJECTS[0]}_T2w.nii.gz").write_bytes(b"T2 fixture")
    output = tmp_path / "output"
    protected = (source, tmp_path / "completed_campaign")
    config = replace(
        ExperimentConfig.defaults(repo_scripts_root=tmp_path),
        cohort_file=cohort,
        cohort_sha256=digest(cohort),
        selection_audit_file=audit,
        selection_audit_sha256=digest(audit),
        source_root=source,
        output_root=output,
        protected_roots=protected,
    )
    return config
