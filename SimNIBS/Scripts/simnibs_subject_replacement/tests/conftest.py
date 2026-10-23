from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def replacement_layout(tmp_path: Path) -> dict[str, Path | str]:
    subject = "sub-CC110056"
    replaces = "sub-CC520083"
    subjects_file = tmp_path / "final132-subjects.txt"
    subjects_file.write_text(f"{subject}\nsub-CC110098\n", encoding="utf-8")

    scaffold = tmp_path / "scaffold"
    anat = scaffold / "subjects" / subject / "anat"
    anat.mkdir(parents=True)
    t1 = anat / f"{subject}_T1w.nii"
    t2 = anat / f"{subject}_T2w.nii"
    label = anat / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"
    t1.write_bytes(b"t1")
    t2.write_bytes(b"t2")
    label.write_bytes(b"labels")
    result = scaffold / "results" / f"{subject}.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "status": "complete",
                "subject": subject,
                "task_t1_path": str(t1),
                "source_t1_sha256": sha256(t1),
                "task_t2_path": str(t2),
                "source_t2_sha256": sha256(t2),
                "installed_label": str(label),
                "installed_label_sha256": sha256(label),
            }
        ),
        encoding="utf-8",
    )

    legacy = tmp_path / "legacy"
    legacy_anat = legacy / subject / "anat"
    m2m = legacy_anat / f"m2m_{subject}"
    (m2m / "eeg_positions").mkdir(parents=True)
    (m2m / "toMNI").mkdir()
    (m2m / "T1fs_conform.nii.gz").write_bytes(b"conform")
    (m2m / "T1fs_nu_conform.nii.gz").write_bytes(b"nu")
    (m2m / "eeg_positions" / "positions.csv").write_text("C3,1,2,3\n")
    (m2m / "toMNI" / "warp.nii.gz").write_bytes(b"warp")
    (legacy_anat / ".simnibs326_scaffold_ready.json").write_text(
        json.dumps({"status": "ready", "subject": subject}), encoding="utf-8"
    )

    atlas = tmp_path / "atlases"
    atlas.mkdir()
    (atlas / f"{subject}.nii.gz").write_bytes(b"atlas")
    targets = tmp_path / "targets.csv"
    targets.write_text("target,configuration\nright-m1,1\n", encoding="utf-8")

    return {
        "subject": subject,
        "replaces": replaces,
        "subjects_file": subjects_file,
        "scaffold": scaffold,
        "legacy": legacy,
        "atlas": atlas,
        "targets": targets,
        "work": tmp_path / "replacement-work",
    }
