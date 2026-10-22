from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simnibs_326_corrected_label_repeatability.label_io import (
    LabelImageError,
    harmonize_subject_labels,
    resample_geometry_to_reference,
    temporary_surface_masks,
)
from simnibs_326_corrected_label_repeatability.settings import (
    SUBJECTS,
    campaign_paths_from_environment,
    subject_paths,
)


def test_resample_is_nearest_neighbour_categorical_and_uint8() -> None:
    data = _complete_geometry((6, 6, 6))
    source = nib.Nifti1Image(data, np.eye(4))
    reference_affine = np.diag([0.5, 0.5, 0.5, 1.0])
    reference = nib.Nifti1Image(np.zeros((12, 12, 12)), reference_affine)

    result = resample_geometry_to_reference(source, reference)

    values = np.asanyarray(result.dataobj)
    assert values.dtype == np.uint8
    assert set(np.unique(values)) <= {0, 1, 2, 3, 4, 5, 6, 11}
    assert set(range(1, 7)) <= set(np.unique(values))
    assert np.array_equal(result.affine, reference_affine)


def test_resample_preserves_reference_qform_and_sform() -> None:
    source = nib.Nifti1Image(_complete_geometry((6, 6, 6)), np.eye(4))
    reference = nib.Nifti1Image(np.zeros((6, 6, 6)), np.eye(4))
    qform = np.eye(4)
    sform = np.eye(4)
    reference.set_qform(qform, code=2)
    reference.set_sform(sform, code=4)

    result = resample_geometry_to_reference(source, reference)

    result_qform, result_qcode = result.get_qform(coded=True)
    result_sform, result_scode = result.get_sform(coded=True)
    assert result_qcode == 2
    assert result_scode == 4
    assert np.allclose(result_qform, qform)
    assert np.allclose(result_sform, sform)


def test_resample_rejects_non_overlapping_grids() -> None:
    source = nib.Nifti1Image(_complete_geometry((6, 6, 6)), np.eye(4))
    shifted = np.eye(4)
    shifted[:3, 3] = 100.0
    reference = nib.Nifti1Image(np.zeros((6, 6, 6)), shifted)

    with pytest.raises(LabelImageError, match="do not overlap"):
        resample_geometry_to_reference(source, reference)


def test_resample_rejects_empty_required_tissue() -> None:
    data = np.zeros((10, 2, 2), dtype=np.uint8)
    for x, label in enumerate((1, 2, 3, 4, 5)):
        data[x, 0, 0] = label
    data[9, 0, 0] = 6
    source = nib.Nifti1Image(data, np.eye(4))
    reference = nib.Nifti1Image(np.zeros((5, 2, 2)), np.eye(4))

    with pytest.raises(LabelImageError, match="required tissue.*6"):
        resample_geometry_to_reference(source, reference)


def test_failed_subject_conversion_leaves_no_retained_map(tmp_path: Path) -> None:
    paths, resolved = _subject_layout(tmp_path, crop_out_eyes=True)

    with pytest.raises(LabelImageError, match="required tissue.*6"):
        harmonize_subject_labels(resolved)

    assert not resolved.harmonized_label.exists()
    assert not resolved.harmonization_qc.exists()
    assert not (paths.headmodel_root / "harmonized_labels").exists()


def test_retains_only_one_map_and_records_source_without_copying(tmp_path: Path) -> None:
    paths, resolved = _subject_layout(tmp_path)
    source_hash = _sha256(resolved.corrected_label)

    receipt = harmonize_subject_labels(resolved)

    retained_images = sorted(
        path.relative_to(paths.headmodel_root / "harmonized_labels")
        for path in (paths.headmodel_root / "harmonized_labels").rglob("*.nii*")
    )
    assert retained_images == [Path(f"{resolved.subject}_v3_harmonized_t1fs_grid.nii.gz")]
    assert resolved.corrected_label.exists()
    assert source_hash == _sha256(resolved.corrected_label)
    assert not (paths.headmodel_root / resolved.corrected_label.name).exists()
    assert receipt.source_path == str(resolved.corrected_label.resolve())
    assert receipt.source_sha256 == source_hash
    assert receipt.output_path == str(resolved.harmonized_label.resolve())
    assert receipt.output_sha256 == _sha256(resolved.harmonized_label)

    payload = json.loads(resolved.harmonization_qc.read_text(encoding="utf-8"))
    assert payload["source"]["shape"] == [6, 6, 6]
    assert payload["output"]["shape"] == [6, 6, 6]
    assert payload["source"]["sha256"] == source_hash
    assert payload["output"]["sha256"] == receipt.output_sha256
    assert set(payload["harmonization"]["source_counts"]) >= {
        "1",
        "2",
        "3",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    }
    assert set(payload["harmonization"]["output_counts"]) >= {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    }


def test_temporary_masks_are_exclusive_reconstructable_and_removed(tmp_path: Path) -> None:
    _, resolved = _subject_layout(tmp_path)
    harmonize_subject_labels(resolved)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with temporary_surface_masks(resolved.harmonized_label, scratch) as masks:
        assert set(masks) == {"WM", "GM", "CSF", "BONE", "SKIN", "EYES", "AIR"}
        assert {path.name for path in masks.values()} == {
            "MASK_WM.nii.gz",
            "MASK_GM.nii.gz",
            "MASK_CSF.nii.gz",
            "MASK_BONE.nii.gz",
            "MASK_SKIN.nii.gz",
            "MASK_EYES.nii.gz",
            "MASK_AIR.nii.gz",
        }
        arrays = {
            name: np.asanyarray(nib.load(path).dataobj).astype(bool)
            for name, path in masks.items()
        }
        total = sum(array.astype(np.uint8) for array in arrays.values())
        assert int(total.max()) == 1
        retained = np.asanyarray(nib.load(resolved.harmonized_label).dataobj)
        expected = {
            "WM": 1,
            "GM": 2,
            "CSF": 3,
            "BONE": 4,
            "SKIN": 5,
            "EYES": 6,
            "AIR": 11,
        }
        for name, label in expected.items():
            assert np.array_equal(arrays[name], retained == label)
        created = list(masks.values())

    assert all(not path.exists() for path in created)
    assert list(scratch.iterdir()) == []


def test_temporary_masks_cleanup_after_exception(tmp_path: Path) -> None:
    _, resolved = _subject_layout(tmp_path)
    harmonize_subject_labels(resolved)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    created: list[Path] = []

    with pytest.raises(RuntimeError, match="deliberate"):
        with temporary_surface_masks(resolved.harmonized_label, scratch) as masks:
            created = list(masks.values())
            raise RuntimeError("deliberate")

    assert all(not path.exists() for path in created)
    assert list(scratch.iterdir()) == []


def _complete_geometry(shape: tuple[int, int, int]) -> np.ndarray:
    data = np.zeros(shape, dtype=np.uint8)
    labels = (1, 2, 3, 4, 5, 6, 11)
    for index, label in enumerate(labels):
        data[index % shape[0], (index // shape[0]) % shape[1], 0] = label
    return data


def _subject_layout(tmp_path: Path, *, crop_out_eyes: bool = False):
    defaults = campaign_paths_from_environment({})
    paths = replace(
        defaults,
        corrected_root=(tmp_path / "corrected").resolve(),
        legacy_reference_root=(tmp_path / "legacy").resolve(),
        atlas_root=(tmp_path / "atlas").resolve(),
        headmodel_root=(tmp_path / "headmodels").resolve(),
        left_root=(tmp_path / "left").resolve(),
        right_root=(tmp_path / "right").resolve(),
        completed_v3_left_root=(tmp_path / "old-v3-left").resolve(),
        completed_v3_right_root=(tmp_path / "old-v3-right").resolve(),
        completed_v4_left_root=(tmp_path / "old-v4-left").resolve(),
        completed_v4_right_root=(tmp_path / "old-v4-right").resolve(),
    )
    resolved = subject_paths(paths, SUBJECTS[0])
    resolved.corrected_label.parent.mkdir(parents=True)
    resolved.staged_m2m.mkdir(parents=True)
    source = np.zeros((6, 6, 6), dtype=np.uint8)
    source[0, 0, 0] = 1
    source[1, 0, 0] = 2
    source[2, 0, 0] = 3
    source[3, 0, 0] = 7
    source[3, 1, 0] = 8
    source[4, 0, 0] = 5
    source[4, 1, 0] = 10
    source[5, 0, 0] = 6
    source[1, 1, 0] = 9
    source[5, 1, 0] = 11
    nib.save(nib.Nifti1Image(source, np.eye(4)), resolved.corrected_label)
    reference_shape = (5, 6, 6) if crop_out_eyes else source.shape
    nib.save(
        nib.Nifti1Image(np.zeros(reference_shape, dtype=np.float32), np.eye(4)),
        resolved.staged_m2m / "T1fs_conform.nii.gz",
    )
    return paths, resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
