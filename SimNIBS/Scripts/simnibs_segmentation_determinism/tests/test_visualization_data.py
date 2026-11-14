from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simnibs_segmentation_determinism.visualization_data import (
    VisualizationDataError,
    build_pairwise_matrix,
    compute_modal_disagreement,
    load_visualization_data,
)


def _write_tsv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _completed_analysis(root: Path) -> Path:
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "validated_repeats": 20,
        "retried_repeats": [],
        "arms": {
            "simnibs326_headreco": {
                "repeat_count": 10,
                "operational_deterministic": True,
                "geometric_deterministic": True,
                "discrete_segmentation_deterministic": True,
                "probability_map_deterministic": True,
                "intensity_image_deterministic": True,
            },
            "simnibs401_charm": {
                "repeat_count": 10,
                "operational_deterministic": True,
                "geometric_deterministic": True,
                "discrete_segmentation_deterministic": False,
                "probability_map_deterministic": None,
                "intensity_image_deterministic": None,
            },
        },
    }
    (analysis / "determinism_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    pair_rows = []
    for left in range(1, 11):
        for right in range(left + 1, 11):
            fraction = (left + right) / 10_000
            pair_rows.append(
                {
                    "arm": "simnibs401_charm",
                    "relative_path": "tissue_labeling_upsampled.nii.gz",
                    "kind": "discrete",
                    "repeat_a": left,
                    "repeat_b": right,
                    "file_bytes_equal": "False",
                    "array_equal": "False",
                    "geometry_equal": "True",
                    "header_equal": "True",
                    "changed_voxels": int(fraction * 100_000),
                    "changed_fraction": fraction,
                    "dice_by_label": json.dumps(
                        {"0": 1 - fraction, "1": 1 - 2 * fraction}
                    ),
                }
            )
    _write_tsv(
        analysis / "changed_voxel_summary.tsv",
        pair_rows,
        list(pair_rows[0]),
    )

    volume_rows = []
    affine = np.eye(4)
    for repeat in range(1, 11):
        volume_rows.append(
            {
                "arm": "simnibs401_charm",
                "repeat_index": repeat,
                "relative_path": "tissue_labeling_upsampled.nii.gz",
                "label_counts": json.dumps({"0": 6, "1": 2}),
                "label_volumes_mm3": json.dumps(
                    {"0": 6.0 + repeat / 10, "1": 2.0 - repeat / 100}
                ),
            }
        )
        labels = np.zeros((2, 2, 2), dtype=np.uint8)
        labels[0, 0, 0] = repeat % 2
        labels[1, 1, 1] = 1
        repeat_root = root / "simnibs401_charm" / f"repeat_{repeat:03d}"
        repeat_root.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(labels, affine),
            repeat_root / "tissue_labeling_upsampled.nii.gz",
        )
    _write_tsv(
        analysis / "tissue_summary.tsv",
        volume_rows,
        list(volume_rows[0]),
    )
    return root


def test_build_pairwise_matrix_requires_every_unique_pair():
    rows = [
        {"repeat_a": left, "repeat_b": right, "changed_fraction": 0.001}
        for left in range(1, 11)
        for right in range(left + 1, 11)
    ]
    matrix = build_pairwise_matrix(rows, repeat_count=10)
    assert matrix.shape == (10, 10)
    assert np.allclose(np.diag(matrix), 0)
    assert np.allclose(matrix, matrix.T)

    with pytest.raises(VisualizationDataError, match="45 unique repeat pairs"):
        build_pairwise_matrix(rows[:-1], repeat_count=10)
    with pytest.raises(VisualizationDataError, match="canonical"):
        build_pairwise_matrix([*rows[:-1], {**rows[-1], "repeat_a": 10, "repeat_b": 9}], 10)


def test_compute_modal_disagreement_returns_mode_and_frequency():
    arrays = np.zeros((10, 2, 2, 1), dtype=np.uint8)
    arrays[:, 1, 1, 0] = 2
    arrays[:3, 0, 0, 0] = 1

    modal, disagreement = compute_modal_disagreement(arrays)

    assert modal[0, 0, 0] == 0
    assert disagreement[0, 0, 0] == pytest.approx(0.3)
    assert disagreement[1, 1, 0] == 0


def test_load_visualization_data_validates_and_summarizes(tmp_path):
    root = _completed_analysis(tmp_path / "staging")

    data = load_visualization_data(root)

    assert data.pairwise_changed_percent.shape == (10, 10)
    assert len(data.changed_percent) == 45
    assert sorted(data.dice_by_label) == ["0", "1"]
    assert all(len(values) == 45 for values in data.dice_by_label.values())
    assert sorted(data.volume_mm3_by_label) == ["0", "1"]
    assert all(len(values) == 10 for values in data.volume_mm3_by_label.values())
    assert data.modal_labels.shape == (2, 2, 2)
    assert data.disagreement_frequency.max() == pytest.approx(0.5)
    assert data.summary_statistics["pairwise_comparisons"] == 45
    assert data.summary_statistics["changed_label_percent"]["maximum"] > 0


def test_load_visualization_data_rejects_mismatched_affine(tmp_path):
    root = _completed_analysis(tmp_path / "staging")
    path = root / "simnibs401_charm/repeat_010/tissue_labeling_upsampled.nii.gz"
    image = nib.load(str(path))
    affine = image.affine.copy()
    affine[0, 3] = 2
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), path)

    with pytest.raises(VisualizationDataError, match="affine"):
        load_visualization_data(root)


def test_load_visualization_data_rejects_duplicate_pair(tmp_path):
    root = _completed_analysis(tmp_path / "staging")
    table = root / "analysis/changed_voxel_summary.tsv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[-1] = dict(rows[0])
    _write_tsv(table, rows, list(rows[0]))

    with pytest.raises(VisualizationDataError, match="duplicate"):
        load_visualization_data(root)
