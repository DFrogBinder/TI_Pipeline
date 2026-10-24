from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from simnibs_326_corrected_label_repeatability.settings import BASE9_SUBJECTS
from simnibs_headmodel_supervisor_review.settings import ReviewPaths


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def review_layout(tmp_path: Path) -> tuple[ReviewPaths, tuple[str, ...]]:
    paths = ReviewPaths(
        original_charm_root=tmp_path / "original-charm",
        corrected_v4_scaffold_root=tmp_path / "corrected-v4",
        legacy_v3_root=tmp_path / "legacy-v3",
        harmonized_v3_root=tmp_path / "harmonized-v3",
        atlas_root=tmp_path / "atlases",
        completed_v4_left_root=tmp_path / "v4-left",
        output_root=tmp_path / "review-output",
    )
    selection_rows: list[dict[str, str]] = []
    for subject in BASE9_SUBJECTS:
        original_charm = write_file(
            paths.original_charm_root
            / "maps"
            / f"{subject}_CHARM_tissue_labeling_upsampled.nii.gz",
            f"original-charm-{subject}".encode(),
        )
        write_json(
            paths.original_charm_root / "metadata" / f"{subject}.json",
            {
                "status": "complete",
                "subject": subject,
                "archived_map": str(original_charm),
                "archived_map_sha256": file_hash(original_charm),
            },
        )

        corrected = write_file(
            paths.corrected_v4_scaffold_root
            / "subjects"
            / subject
            / "anat"
            / f"m2m_{subject}"
            / "label_prep"
            / "tissue_labeling_upsampled.nii.gz",
            f"corrected-charm-{subject}".encode(),
        )
        write_json(
            paths.corrected_v4_scaffold_root / "results" / f"{subject}.json",
            {
                "status": "complete",
                "subject": subject,
                "installed_label": str(corrected),
                "installed_label_sha256": file_hash(corrected),
            },
        )

        legacy_anat = paths.legacy_v3_root / subject / "anat"
        legacy_mesh = write_file(
            legacy_anat / f"{subject}.msh", f"legacy-mesh-{subject}".encode()
        )
        legacy_reference = write_file(
            legacy_anat / f"m2m_{subject}" / "T1fs_conform.nii.gz",
            f"legacy-reference-{subject}".encode(),
        )
        write_json(
            legacy_anat / ".simnibs326_scaffold_ready.json",
            {
                "schema_version": 1,
                "status": "ready",
                "subject": subject,
                "simnibs_module": "SimNIBS/3.2.6-foss-2023a",
                "native_mesh": str(legacy_mesh),
                "native_mesh_sha256": file_hash(legacy_mesh),
            },
        )

        harmonized_map = write_file(
            paths.harmonized_v3_root
            / "harmonized_labels"
            / f"{subject}_v3_harmonized_t1fs_grid.nii.gz",
            f"harmonized-map-{subject}".encode(),
        )
        harmonization_qc = write_json(
            paths.harmonized_v3_root
            / "harmonized_labels"
            / "qc"
            / f"{subject}.json",
            {
                "status": "complete",
                "subject": subject,
                "output": {
                    "path": str(harmonized_map),
                    "sha256": file_hash(harmonized_map),
                },
            },
        )
        harmonized_mesh = write_file(
            paths.harmonized_v3_root / subject / "anat" / f"{subject}.msh",
            f"harmonized-mesh-{subject}".encode(),
        )
        write_json(
            paths.harmonized_v3_root
            / "_pipeline"
            / "subjects"
            / subject
            / "headmodel_complete.json",
            {
                "schema_version": 1,
                "status": "complete",
                "subject": subject,
                "harmonized_map": {
                    "path": str(harmonized_map),
                    "sha256": file_hash(harmonized_map),
                },
                "harmonization_receipt": {
                    "path": str(harmonization_qc),
                    "sha256": file_hash(harmonization_qc),
                },
                "mesh": {
                    "path": str(harmonized_mesh),
                    "sha256": file_hash(harmonized_mesh),
                },
                "api_probe": {"simnibs_version": "3.2.6"},
            },
        )

        atlas = write_file(
            paths.atlas_root / f"{subject}.nii.gz", f"atlas-{subject}".encode()
        )
        assert atlas.is_file()

        v4_mesh = write_file(
            paths.completed_v4_left_root
            / f"{subject}_repeatability"
            / "remesh"
            / "repeats"
            / "repeat_001"
            / subject
            / "anat"
            / f"m2m_{subject}"
            / f"{subject}.msh",
            f"v4-mesh-{subject}".encode(),
        )
        selection_rows.append(
            {
                "subject": subject,
                "selection_status": "selected",
                "selected_repeat_tag": "repeat_001",
                "selected_mesh_path": str(v4_mesh),
                "mesh_checksum": file_hash(v4_mesh),
            }
        )

    selection_csv = (
        paths.completed_v4_left_root
        / "_pipeline"
        / "median_mesh_selection"
        / "median_representative_remesh_repeats.csv"
    )
    selection_csv.parent.mkdir(parents=True, exist_ok=True)
    with selection_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selection_rows[0].keys())
        writer.writeheader()
        writer.writerows(selection_rows)
    write_json(
        paths.completed_v4_left_root / "_pipeline" / "workflow" / "complete.json",
        {"status": "complete", "stage": "final"},
    )
    return paths, BASE9_SUBJECTS
