from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from .provenance import sha256_file


ABSOLUTE_THRESHOLDS = (1e-7, 1e-6, 1e-5)


class AnalysisError(RuntimeError):
    """Raised when repeat outputs cannot be compared safely."""


def _matrix_tuple(matrix) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    return tuple(tuple(float(value) for value in row) for row in array)


def _normalized_header_float(value) -> Optional[float]:
    number = float(value)
    return None if not np.isfinite(number) else number


def _unscaled_array(image) -> np.ndarray:
    dataobj = image.dataobj
    if hasattr(dataobj, "get_unscaled"):
        data = dataobj.get_unscaled()
    else:
        data = np.asanyarray(dataobj)
    return np.ascontiguousarray(data)


def _decoded_array(image) -> np.ndarray:
    return np.ascontiguousarray(np.asanyarray(image.dataobj))


def _proxy_scaling(image) -> Tuple[float, float]:
    slope = getattr(image.dataobj, "slope", None)
    intercept = getattr(image.dataobj, "inter", None)
    return (
        1.0 if slope is None or not np.isfinite(float(slope)) else float(slope),
        0.0
        if intercept is None or not np.isfinite(float(intercept))
        else float(intercept),
    )


def _canonical_array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _label_key(value) -> str:
    scalar = value.item() if hasattr(value, "item") else value
    return str(scalar)


@dataclass(frozen=True)
class NiftiFingerprint:
    path: str
    file_sha256: str
    array_sha256: str
    decoded_array_sha256: str
    shape: Tuple[int, ...]
    voxel_sizes: Tuple[float, ...]
    dtype: str
    scaling_slope: Optional[float]
    scaling_intercept: Optional[float]
    qform: Optional[Tuple[Tuple[float, ...], ...]]
    qform_code: int
    sform: Optional[Tuple[Tuple[float, ...], ...]]
    sform_code: int
    effective_affine: Tuple[Tuple[float, ...], ...]
    finite_voxels: int
    nonfinite_voxels: int
    label_counts: Optional[Dict[str, int]]
    label_volumes_mm3: Optional[Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NiftiComparison:
    reference: str
    candidate: str
    kind: str
    file_bytes_equal: bool
    array_equal: bool
    geometry_equal: bool
    header_equal: bool
    shape_equal: bool
    voxel_sizes_equal: bool
    qform_equal: bool
    sform_equal: bool
    effective_affine_equal: bool
    nonfinite_pattern_equal: bool
    changed_voxels: int
    changed_fraction: float
    transition_counts: Optional[Dict[str, int]]
    dice_by_label: Optional[Dict[str, float]]
    max_absolute_difference: Optional[float]
    mean_absolute_difference: Optional[float]
    absolute_threshold_counts: Optional[Dict[str, int]]

    @property
    def deterministic(self) -> bool:
        return self.array_equal and self.geometry_equal and self.header_equal

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def inspect_nifti(path: Path) -> NiftiFingerprint:
    path = Path(path)
    image = nib.load(str(path))
    array = _unscaled_array(image)
    decoded = _decoded_array(image)
    header = image.header
    scaling_slope, scaling_intercept = _proxy_scaling(image)
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    finite = np.isfinite(decoded) if np.issubdtype(decoded.dtype, np.number) else np.ones(decoded.shape, bool)
    label_counts = None
    label_volumes = None
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        labels, counts = np.unique(array, return_counts=True)
        label_counts = {_label_key(label): int(count) for label, count in zip(labels, counts)}
        voxel_volume = float(np.prod(header.get_zooms()[: min(3, array.ndim)]))
        label_volumes = {
            key: float(count * voxel_volume) for key, count in label_counts.items()
        }
    return NiftiFingerprint(
        path=str(path.resolve()),
        file_sha256=sha256_file(path),
        array_sha256=_canonical_array_sha256(array),
        decoded_array_sha256=_canonical_array_sha256(decoded),
        shape=tuple(int(value) for value in array.shape),
        voxel_sizes=tuple(float(value) for value in header.get_zooms()[: array.ndim]),
        dtype=array.dtype.str,
        scaling_slope=scaling_slope,
        scaling_intercept=scaling_intercept,
        qform=_matrix_tuple(qform),
        qform_code=int(qform_code),
        sform=_matrix_tuple(sform),
        sform_code=int(sform_code),
        effective_affine=_matrix_tuple(image.affine),  # type: ignore[arg-type]
        finite_voxels=int(np.count_nonzero(finite)),
        nonfinite_voxels=int(array.size - np.count_nonzero(finite)),
        label_counts=label_counts,
        label_volumes_mm3=label_volumes,
    )


def _array_equal_mask(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    equal = reference == candidate
    if np.issubdtype(reference.dtype, np.inexact) or np.issubdtype(candidate.dtype, np.inexact):
        equal = equal | (np.isnan(reference) & np.isnan(candidate))
    return np.asarray(equal, dtype=bool)


def _discrete_details(reference: np.ndarray, candidate: np.ndarray, changed: np.ndarray):
    transitions: Dict[str, int] = {}
    if np.any(changed):
        pairs = np.column_stack((reference[changed], candidate[changed]))
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        transitions = {
            f"{_label_key(pair[0])}->{_label_key(pair[1])}": int(count)
            for pair, count in zip(unique, counts)
        }
    dice = {}
    for label in np.union1d(np.unique(reference), np.unique(candidate)):
        reference_mask = reference == label
        candidate_mask = candidate == label
        denominator = int(reference_mask.sum() + candidate_mask.sum())
        dice[_label_key(label)] = (
            1.0
            if denominator == 0
            else float(2 * np.count_nonzero(reference_mask & candidate_mask) / denominator)
        )
    return transitions, dice


def compare_niftis(reference: Path, candidate: Path, kind: str) -> NiftiComparison:
    if kind not in {"discrete", "probability", "intensity"}:
        raise ValueError(f"Unsupported NIfTI comparison kind: {kind}")
    reference = Path(reference)
    candidate = Path(candidate)
    reference_image = nib.load(str(reference))
    candidate_image = nib.load(str(candidate))
    reference_array = _unscaled_array(reference_image)
    candidate_array = _unscaled_array(candidate_image)
    reference_decoded = _decoded_array(reference_image)
    candidate_decoded = _decoded_array(candidate_image)
    comparison_reference = reference_array if kind == "discrete" else reference_decoded
    comparison_candidate = candidate_array if kind == "discrete" else candidate_decoded
    reference_fingerprint = inspect_nifti(reference)
    candidate_fingerprint = inspect_nifti(candidate)

    shape_equal = comparison_reference.shape == comparison_candidate.shape
    if shape_equal:
        equal_mask = _array_equal_mask(comparison_reference, comparison_candidate)
        changed = ~equal_mask
        changed_voxels = int(np.count_nonzero(changed))
        changed_fraction = float(changed_voxels / comparison_reference.size) if comparison_reference.size else 0.0
        finite_reference = np.isfinite(comparison_reference)
        finite_candidate = np.isfinite(comparison_candidate)
        nonfinite_pattern_equal = bool(np.array_equal(finite_reference, finite_candidate))
    else:
        changed = np.ones(comparison_reference.shape, dtype=bool)
        changed_voxels = max(comparison_reference.size, comparison_candidate.size)
        changed_fraction = 1.0
        nonfinite_pattern_equal = False

    qform_equal = (
        reference_fingerprint.qform_code == candidate_fingerprint.qform_code
        and reference_fingerprint.qform == candidate_fingerprint.qform
    )
    sform_equal = (
        reference_fingerprint.sform_code == candidate_fingerprint.sform_code
        and reference_fingerprint.sform == candidate_fingerprint.sform
    )
    effective_affine_equal = (
        reference_fingerprint.effective_affine == candidate_fingerprint.effective_affine
    )
    voxel_sizes_equal = reference_fingerprint.voxel_sizes == candidate_fingerprint.voxel_sizes
    geometry_equal = bool(
        shape_equal
        and voxel_sizes_equal
        and qform_equal
        and sform_equal
        and effective_affine_equal
    )
    header_equal = bool(
        geometry_equal
        and reference_fingerprint.dtype == candidate_fingerprint.dtype
        and reference_fingerprint.scaling_slope == candidate_fingerprint.scaling_slope
        and reference_fingerprint.scaling_intercept == candidate_fingerprint.scaling_intercept
    )
    array_equal = bool(
        shape_equal
        and (
            reference_fingerprint.array_sha256 == candidate_fingerprint.array_sha256
            if kind == "discrete"
            else reference_fingerprint.decoded_array_sha256
            == candidate_fingerprint.decoded_array_sha256
        )
    )

    transitions = None
    dice = None
    maximum = None
    mean = None
    threshold_counts = None
    if shape_equal and kind == "discrete":
        transitions, dice = _discrete_details(reference_array, candidate_array, changed)
    elif shape_equal:
        finite_both = np.isfinite(reference_decoded) & np.isfinite(candidate_decoded)
        differences = np.abs(
            reference_decoded[finite_both].astype(np.float64)
            - candidate_decoded[finite_both].astype(np.float64)
        )
        maximum = float(differences.max()) if differences.size else 0.0
        mean = float(differences.mean()) if differences.size else 0.0
        threshold_counts = {
            f"{threshold:.0e}": int(np.count_nonzero(differences > threshold))
            for threshold in ABSOLUTE_THRESHOLDS
        }

    return NiftiComparison(
        reference=str(reference.resolve()),
        candidate=str(candidate.resolve()),
        kind=kind,
        file_bytes_equal=reference_fingerprint.file_sha256 == candidate_fingerprint.file_sha256,
        array_equal=array_equal,
        geometry_equal=geometry_equal,
        header_equal=header_equal,
        shape_equal=shape_equal,
        voxel_sizes_equal=voxel_sizes_equal,
        qform_equal=qform_equal,
        sform_equal=sform_equal,
        effective_affine_equal=effective_affine_equal,
        nonfinite_pattern_equal=nonfinite_pattern_equal,
        changed_voxels=changed_voxels,
        changed_fraction=changed_fraction,
        transition_counts=transitions,
        dice_by_label=dice,
        max_absolute_difference=maximum,
        mean_absolute_difference=mean,
        absolute_threshold_counts=threshold_counts,
    )


def infer_artifact_kind(arm: str, relative_path: str, path: Path) -> str:
    del path
    name = relative_path.replace("\\", "/")
    if arm == "simnibs401_charm" and name == "tissue_labeling_upsampled.nii.gz":
        return "discrete"
    if arm == "simnibs326_headreco":
        if "_bin.nii" in name:
            return "discrete"
        if re.fullmatch(
            r"segment/(?:spm/c\d+|cat/mri/p\d+)T1fs_conform\.nii(?:\.gz)?",
            name,
        ):
            return "probability"
        if name in {"T1fs_conform.nii.gz", "T1fs_nu_conform.nii.gz", "T2_conform.nii.gz"}:
            return "intensity"
    raise AnalysisError(f"{arm}: no pinned semantic class for artifact {relative_path}")


def analyze_arm(
    arm: str,
    repeat_artifacts: Mapping[int, Mapping[str, Path]],
    artifact_kinds: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    if not repeat_artifacts:
        raise AnalysisError(f"{arm}: no repeat artifacts")
    repeat_indices = sorted(repeat_artifacts)
    expected_indices = list(range(1, len(repeat_indices) + 1))
    if repeat_indices != expected_indices:
        raise AnalysisError(
            f"{arm}: repeat indices are not contiguous from 1: {repeat_indices}"
        )
    inventory = set(repeat_artifacts[repeat_indices[0]])
    for repeat in repeat_indices[1:]:
        observed = set(repeat_artifacts[repeat])
        if observed != inventory:
            raise AnalysisError(
                f"{arm}: artifact inventory differs at repeat {repeat}: "
                f"missing={sorted(inventory - observed)}, extra={sorted(observed - inventory)}"
            )

    artifact_results: Dict[str, object] = {}
    all_comparisons = []
    for name in sorted(inventory):
        paths = {repeat: Path(repeat_artifacts[repeat][name]) for repeat in repeat_indices}
        kind = (artifact_kinds or {}).get(name) or infer_artifact_kind(arm, name, paths[1])
        reference_comparisons = [
            (1, repeat, compare_niftis(paths[1], paths[repeat], kind))
            for repeat in repeat_indices[1:]
        ]
        differs = any(not comparison.deterministic for _, _, comparison in reference_comparisons)
        selected_pairs = (
            [
                (left, right, compare_niftis(paths[left], paths[right], kind))
                for left, right in combinations(repeat_indices, 2)
            ]
            if differs
            else reference_comparisons
        )
        serialized = [
            {"repeat_a": left, "repeat_b": right, **comparison.to_dict()}
            for left, right, comparison in selected_pairs
        ]
        all_comparisons.extend((kind, comparison) for _, _, comparison in selected_pairs)
        fingerprints = {}
        for repeat, path in paths.items():
            fingerprint = inspect_nifti(path).to_dict()
            if kind != "discrete":
                fingerprint["label_counts"] = None
                fingerprint["label_volumes_mm3"] = None
            fingerprints[str(repeat)] = fingerprint
        artifact_results[name] = {
            "kind": kind,
            "comparison_mode": "all_pairs" if differs else "repeat_001_reference",
            "fingerprints": fingerprints,
            "comparisons": serialized,
            "deterministic": all(comparison.deterministic for _, _, comparison in selected_pairs),
        }

    discrete = [comparison for kind, comparison in all_comparisons if kind == "discrete"]
    probability = [comparison for kind, comparison in all_comparisons if kind == "probability"]
    intensity = [comparison for kind, comparison in all_comparisons if kind == "intensity"]
    return {
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "repeat_count": len(repeat_indices),
        "artifact_count": len(inventory),
        "operational_deterministic": None,
        "geometric_deterministic": all(comparison.geometry_equal for _, comparison in all_comparisons),
        "discrete_segmentation_deterministic": all(
            comparison.array_equal for comparison in discrete
        ),
        "probability_map_deterministic": (
            all(comparison.array_equal and comparison.header_equal for comparison in probability)
            if probability
            else None
        ),
        "intensity_image_deterministic": (
            all(comparison.array_equal and comparison.header_equal for comparison in intensity)
            if intensity
            else None
        ),
        "artifacts": artifact_results,
    }
