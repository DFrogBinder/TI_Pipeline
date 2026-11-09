from __future__ import annotations

import gzip
import shutil
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np

from simnibs_segmentation_determinism.analysis import (
    analyze_arm,
    compare_niftis,
    infer_artifact_kind,
    inspect_nifti,
)


def _save(path: Path, data, affine=None):
    affine = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(np.asarray(data), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(path))
    return path


def test_identical_arrays_with_different_gzip_headers_are_voxel_identical(tmp_path):
    raw = _save(tmp_path / "map.nii", np.arange(8, dtype=np.int16).reshape(2, 2, 2))
    paths = []
    for index, mtime in enumerate((1, 2), start=1):
        destination = tmp_path / f"map_{index}.nii.gz"
        with raw.open("rb") as source, destination.open("wb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", mtime=mtime) as compressed:
                shutil.copyfileobj(source, compressed)
        paths.append(destination)

    comparison = compare_niftis(paths[0], paths[1], "discrete")

    assert inspect_nifti(paths[0]).file_sha256 != inspect_nifti(paths[1]).file_sha256
    assert comparison.file_bytes_equal is False
    assert comparison.array_equal is True
    assert comparison.geometry_equal is True
    assert comparison.changed_voxels == 0


def test_affine_only_change_fails_geometric_determinism(tmp_path):
    data = np.ones((2, 2, 2), dtype=np.uint8)
    first = _save(tmp_path / "first.nii.gz", data)
    affine = np.eye(4)
    affine[0, 3] = 2.0
    second = _save(tmp_path / "second.nii.gz", data, affine)

    comparison = compare_niftis(first, second, "discrete")

    assert comparison.array_equal is True
    assert comparison.geometry_equal is False
    assert comparison.effective_affine_equal is False


def test_single_label_transition_reports_counts_and_dice(tmp_path):
    first_data = np.array([[[0, 1], [1, 2]]], dtype=np.uint8)
    second_data = first_data.copy()
    second_data[0, 0, 1] = 2
    first = _save(tmp_path / "first.nii.gz", first_data)
    second = _save(tmp_path / "second.nii.gz", second_data)

    comparison = compare_niftis(first, second, "discrete")

    assert comparison.changed_voxels == 1
    assert comparison.transition_counts == {"1->2": 1}
    assert comparison.dice_by_label["1"] == 2 / 3
    assert comparison.dice_by_label["2"] == 2 / 3


def test_probability_differences_and_stable_nans_use_fixed_absolute_thresholds(tmp_path):
    first_data = np.zeros((1, 1, 5), dtype=np.float32)
    first_data[0, 0, 4] = np.nan
    second_data = first_data.copy()
    second_data[0, 0, :4] = (0.5e-7, 2e-7, 2e-6, 2e-5)
    first = _save(tmp_path / "first.nii.gz", first_data)
    second = _save(tmp_path / "second.nii.gz", second_data)

    comparison = compare_niftis(first, second, "probability")

    assert comparison.nonfinite_pattern_equal is True
    assert comparison.changed_voxels == 4
    assert comparison.absolute_threshold_counts == {
        "1e-07": 3,
        "1e-06": 2,
        "1e-05": 1,
    }
    assert np.isclose(comparison.max_absolute_difference, 2e-5)
    assert comparison.mean_absolute_difference > 0
    assert comparison.transition_counts is None
    assert comparison.dice_by_label is None


def test_differing_reference_comparison_triggers_all_45_pairs(tmp_path):
    repeat_artifacts = {}
    for repeat in range(1, 11):
        data = np.zeros((2, 2, 2), dtype=np.uint8)
        if repeat == 10:
            data[0, 0, 0] = 1
        path = _save(tmp_path / f"repeat_{repeat:03d}.nii.gz", data)
        repeat_artifacts[repeat] = {"labels.nii.gz": path}

    result = analyze_arm(
        "simnibs401_charm",
        repeat_artifacts,
        artifact_kinds={"labels.nii.gz": "discrete"},
    )

    artifact = result["artifacts"]["labels.nii.gz"]
    assert artifact["comparison_mode"] == "all_pairs"
    assert len(artifact["comparisons"]) == len(list(combinations(range(1, 11), 2))) == 45
    assert result["discrete_segmentation_deterministic"] is False
    assert result["geometric_deterministic"] is True


def test_scaled_integer_probability_uses_decoded_values_and_proxy_scaling(tmp_path):
    stored = np.ones((2, 2, 2), dtype=np.int16)
    first_image = nib.Nifti1Image(stored, np.eye(4))
    first_image.header.set_slope_inter(1.0, 0.0)
    first = tmp_path / "first.nii"
    nib.save(first_image, str(first))
    second_image = nib.Nifti1Image(stored, np.eye(4))
    second_image.header.set_slope_inter(2.0, 0.0)
    second = tmp_path / "second.nii"
    nib.save(second_image, str(second))

    comparison = compare_niftis(first, second, "probability")

    assert inspect_nifti(first).scaling_slope == 1.0
    assert inspect_nifti(second).scaling_slope == 2.0
    assert comparison.array_equal is False
    assert comparison.header_equal is False
    assert comparison.changed_voxels == stored.size
    assert comparison.max_absolute_difference == 1.0
    assert comparison.deterministic is False


def test_semantic_artifact_classes_do_not_follow_storage_dtype(tmp_path):
    scaled_probability = _save(
        tmp_path / "c1T1fs_conform.nii", np.ones((2, 2, 2), dtype=np.uint8)
    )
    intensity = _save(
        tmp_path / "T1fs_conform.nii.gz", np.ones((2, 2, 2), dtype=np.float32)
    )

    assert infer_artifact_kind("simnibs326_headreco", "segment/spm/c1T1fs_conform.nii", scaled_probability) == "probability"
    assert infer_artifact_kind("simnibs326_headreco", "T1fs_conform.nii.gz", intensity) == "intensity"
    assert infer_artifact_kind("simnibs326_headreco", "segment/spm/binarized/c1T1fs_conform_bin.nii.gz", scaled_probability) == "discrete"
    assert infer_artifact_kind("simnibs401_charm", "tissue_labeling_upsampled.nii.gz", scaled_probability) == "discrete"

    repeats = {
        index: {"segment/spm/c1T1fs_conform.nii": scaled_probability}
        for index in range(1, 11)
    }
    result = analyze_arm("simnibs326_headreco", repeats)
    artifact = result["artifacts"]["segment/spm/c1T1fs_conform.nii"]
    assert artifact["kind"] == "probability"
    assert all(
        fingerprint["label_counts"] is None
        for fingerprint in artifact["fingerprints"].values()
    )
