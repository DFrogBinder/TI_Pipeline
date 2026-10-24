from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def creator_layout(tmp_path: Path) -> dict[str, Path | str]:
    subject = "sub-CC510415"
    subjects = tmp_path / "subjects.txt"
    subjects.write_text(f"sub-CC110056\n{subject}\n", encoding="utf-8")

    source = tmp_path / "corrected-v4"
    anat = source / "subjects" / subject / "anat"
    anat.mkdir(parents=True)
    t1 = anat / f"{subject}_T1w.nii"
    t1.write_bytes(b"t1")
    t2 = anat / f"{subject}_T2w.nii.gz"
    with gzip.open(t2, "wb") as handle:
        handle.write(b"t2")
    labels = anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii.gz"
    with gzip.open(labels, "wb") as handle:
        handle.write(b"labels")
    record = source / "results" / f"{subject}.json"
    record.parent.mkdir()
    record.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject": subject,
                "task_t1_path": str(t1),
                "source_t1_sha256": _sha256(t1),
                "task_t2_path": str(t2),
                "source_t2_sha256": _sha256(t2),
                "installed_label": str(labels),
                "installed_label_sha256": _sha256(labels),
            }
        ),
        encoding="utf-8",
    )
    return {
        "subject": subject,
        "subjects": subjects,
        "source": source,
        "record": record,
        "t1": t1,
        "t2": t2,
        "labels": labels,
        "output": tmp_path / "simnibs326-scaffolds",
    }
