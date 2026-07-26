import csv
import gzip
import json
from pathlib import Path

import pytest

from cohorts import stage_repeatability_dataset as staging


def _subjects() -> list[str]:
    return [f"sub-CC{index:06d}" for index in range(1, 11)]


def _write_subjects(path: Path) -> list[str]:
    subjects = _subjects()
    path.write_text("\n".join(subjects) + "\n", encoding="utf-8")
    return subjects


def _write_final132_scaffolds(root: Path, subjects: list[str]) -> None:
    for index, subject in enumerate(subjects, start=1):
        anat = root / "subjects" / subject / "anat"
        label_dir = anat / f"m2m_{subject}" / "label_prep"
        label_dir.mkdir(parents=True)
        t1 = anat / f"{subject}_T1w.nii"
        t2 = anat / f"{subject}_T2w.nii"
        label = label_dir / "tissue_labeling_upsampled.nii.gz"
        t1.write_bytes(f"T1-{index}\n".encode())
        t2.write_bytes(f"T2-{index}\n".encode())
        with gzip.open(label, "wb") as handle:
            handle.write(f"LABEL-{index}\n".encode())
        record = {
            "status": "complete",
            "subject": subject,
            "task_t1_path": str(t1),
            "task_t2_path": str(t2),
            "source_t1_sha256": staging.sha256_file(t1),
            "source_t2_sha256": staging.sha256_file(t2),
            "installed_label": str(label),
            "installed_label_sha256": staging.sha256_file(label),
        }
        result = root / "results" / f"{subject}.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_final132_scaffold_audit_and_atomic_stage(tmp_path):
    subjects_file = tmp_path / "subjects.txt"
    subjects = _write_subjects(subjects_file)
    scaffold_root = tmp_path / "scaffolds"
    _write_final132_scaffolds(scaffold_root, subjects)

    ready, issues = staging.audit(
        scaffold_root / "subjects",
        subjects,
        "final132-scaffold",
    )

    assert issues == []
    assert len(ready) == 30
    assert {item.transform for item in ready if item.kind == "segmentation"} == {"gunzip"}
    assert all(item.expected_source_sha256 == item.source_sha256 for item in ready)

    output_root = tmp_path / "staged"
    staging.stage(
        scaffold_root,
        output_root,
        subjects_file,
        subjects,
        ready,
        "final132-scaffold",
    )

    assert not any(tmp_path.glob(".staged.staging-*"))
    assert (output_root / "subjects.txt").read_text(encoding="utf-8") == subjects_file.read_text(
        encoding="utf-8"
    )
    first = subjects[0]
    staged_label = (
        output_root
        / first
        / "anat"
        / f"{first}_T1w_ras_1mm_T1andT2_masks.nii"
    )
    assert staged_label.read_bytes() == b"LABEL-1\n"
    with (output_root / "dataset_manifest.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 30
    assert sum(row["transform"] == "gunzip" for row in rows) == 10
    assert all(row["expected_source_sha256"] for row in rows)
    assert all(Path(row["provenance_record"]).name == f"{row['subject']}.json" for row in rows)
    metadata = json.loads((output_root / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_layout"] == "final132-scaffold"
    assert metadata["source_hashes_verified"] is True
    assert metadata["subject_count"] == 10
    assert metadata["required_file_count"] == 30

    with pytest.raises(FileExistsError):
        staging.stage(
            scaffold_root,
            output_root,
            subjects_file,
            subjects,
            ready,
            "final132-scaffold",
        )


def test_final132_scaffold_audit_rejects_recorded_hash_mismatch(tmp_path):
    subjects_file = tmp_path / "subjects.txt"
    subjects = _write_subjects(subjects_file)
    scaffold_root = tmp_path / "scaffolds"
    _write_final132_scaffolds(scaffold_root, subjects)
    result_path = scaffold_root / "results" / f"{subjects[0]}.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["installed_label_sha256"] = "0" * 64
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    ready, issues = staging.audit(scaffold_root, subjects, "final132-scaffold")

    assert len(ready) == 29
    assert len(issues) == 1
    assert subjects[0] in issues[0]
    assert "segmentation SHA-256 mismatch" in issues[0]


def test_legacy_layout_remains_supported(tmp_path):
    subjects_file = tmp_path / "subjects.txt"
    subjects = _write_subjects(subjects_file)
    source_root = tmp_path / "legacy"
    for subject in subjects:
        anat = source_root / subject / "anat"
        anat.mkdir(parents=True)
        for _, suffix in staging.REQUIRED_INPUTS:
            (anat / f"{subject}{suffix}").write_bytes(suffix.encode())

    ready, issues = staging.audit(source_root, subjects)

    assert issues == []
    assert len(ready) == 30
    assert all(item.transform == "copy" for item in ready)
    assert all(item.expected_source_sha256 == "" for item in ready)
