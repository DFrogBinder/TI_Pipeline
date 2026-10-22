"""Pure corrected-label validation and deterministic v3 harmonization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


ACCEPTED_SOURCE_LABELS = frozenset({0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11})
BLOOD_RECIPIENT_PRIORITY = (1, 2, 3, 5)
NONBLOOD_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 5: 5, 6: 6, 7: 4, 8: 4, 10: 5, 11: 11}


class LabelContractError(ValueError):
    """Raised when a label image violates the approved scientific contract."""


@dataclass(frozen=True)
class HarmonizationStats:
    observed_source_labels: tuple[int, ...]
    source_counts: dict[int, int]
    output_counts: dict[int, int]
    spacing_mm: tuple[float, float, float]
    blood_voxels: int
    blood_recipient_counts: dict[int, int]
    blood_tie_voxels: int
    recipient_classes_present: dict[int, bool]


def _counts(data: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(data, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def validate_source_labels(data: np.ndarray) -> np.ndarray:
    """Return an integral copy after validating the entire source label set."""
    array = np.asarray(data)
    if array.ndim != 3:
        raise LabelContractError(
            f"Source label data must be three-dimensional, got shape {array.shape}"
        )
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise LabelContractError("Source labels must be finite numeric values") from exc
    if not bool(np.all(finite)):
        raise LabelContractError("Source labels must contain only finite values")
    rounded = np.rint(array)
    if not bool(np.all(array == rounded)):
        invalid = np.unique(array[array != rounded]).tolist()
        raise LabelContractError(f"Source labels must be exactly integer-valued: {invalid}")
    labels = rounded.astype(np.int64, copy=False)
    observed = {int(value) for value in np.unique(labels)}
    unsupported = sorted(observed - ACCEPTED_SOURCE_LABELS)
    if unsupported:
        raise LabelContractError(f"Source label image contains unsupported labels: {unsupported}")
    return labels.copy()


def _validate_spacing(spacing_mm: tuple[float, ...]) -> tuple[float, float, float]:
    if len(spacing_mm) != 3:
        raise LabelContractError("Voxel spacing must contain exactly three values")
    spacing = tuple(float(value) for value in spacing_mm)
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise LabelContractError(f"Voxel spacing must be finite and positive: {spacing}")
    return spacing  # type: ignore[return-value]


def harmonize_labels(
    labels: np.ndarray,
    spacing_mm: tuple[float, ...],
) -> tuple[np.ndarray, HarmonizationStats]:
    """Map corrected-v4 tissue labels into the approved v3 geometry labels."""
    source = validate_source_labels(labels)
    spacing = _validate_spacing(spacing_mm)
    output = np.zeros(source.shape, dtype=np.uint8)
    for source_label, output_label in NONBLOOD_MAPPING.items():
        output[source == source_label] = output_label
    blood_mask = source == 9
    recipient_present = {
        label: bool(np.any(output == label)) for label in BLOOD_RECIPIENT_PRIORITY
    }
    recipient_counts = {label: 0 for label in BLOOD_RECIPIENT_PRIORITY}
    tie_count = 0
    if np.any(blood_mask):
        present_labels = [
            label for label in BLOOD_RECIPIENT_PRIORITY if recipient_present[label]
        ]
        if not present_labels:
            raise LabelContractError(
                "Blood voxels exist but there is no eligible recipient tissue"
            )
        distances = np.stack(
            [
                distance_transform_edt(output != label, sampling=spacing)
                for label in present_labels
            ],
            axis=0,
        )
        blood_distances = distances[:, blood_mask]
        winning_indices = np.argmin(blood_distances, axis=0)
        minimum = np.min(blood_distances, axis=0)
        tie_count = int(np.count_nonzero(np.sum(blood_distances == minimum, axis=0) > 1))
        assignments = np.asarray(present_labels, dtype=np.uint8)[winning_indices]
        output[blood_mask] = assignments
        for label in present_labels:
            recipient_counts[label] = int(np.count_nonzero(assignments == label))
    stats = HarmonizationStats(
        observed_source_labels=tuple(sorted(_counts(source))),
        source_counts=_counts(source),
        output_counts=_counts(output),
        spacing_mm=spacing,
        blood_voxels=int(np.count_nonzero(blood_mask)),
        blood_recipient_counts=recipient_counts,
        blood_tie_voxels=tie_count,
        recipient_classes_present=recipient_present,
    )
    return output, stats
