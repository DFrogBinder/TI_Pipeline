from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from .provenance import sha256_file, validate_sha256_sidecar
from .settings import (
    ExperimentConfig,
    resolve_staged_subject_inputs,
)
from .visualization_data import CHARM_ARM, VisualizationDataError, load_visualization_data


ORIENTATIONS = ("sagittal", "coronal", "axial")
MAX_DISPLAY_DISAGREEMENT = 6


class JournalSpatialDataError(RuntimeError):
    """Raised when validated experiment outputs cannot support spatial figures."""


@dataclass(frozen=True)
class JournalSpatialData:
    staging_root: Path
    subject: str
    repeat_count: int
    t1_path: Path
    t1_sha256: str
    modal_labels: np.ndarray
    disagreement_counts: np.ndarray
    t1_normalized: np.ndarray
    affine: np.ndarray
    crop_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    main_slice_indices: Mapping[str, int]
    atlas_slice_indices: Mapping[str, Tuple[int, ...]]
    spatial_summary: Mapping[str, object]


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JournalSpatialDataError(f"Could not read provenance JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise JournalSpatialDataError(f"Expected a JSON object: {path}")
    return payload


def _validated_staged_t1(config: ExperimentConfig) -> tuple[Path, str]:
    try:
        inputs = resolve_staged_subject_inputs(config)
    except Exception as exc:
        raise JournalSpatialDataError("Could not resolve the staged anatomical inputs") from exc

    terminal = config.output_root / "_pipeline/complete.json"
    try:
        validate_sha256_sidecar(terminal)
    except (OSError, RuntimeError) as exc:
        raise JournalSpatialDataError("Experiment terminal receipt is missing or invalid") from exc
    records = _read_json(terminal).get("repeat_receipts")
    if not isinstance(records, list) or len(records) != 20:
        raise JournalSpatialDataError("Terminal repeat-receipt inventory is incomplete")

    expected_path = inputs.t1.resolve()
    matching = 0
    for record in records:
        if not isinstance(record, dict) or record.get("arm") != CHARM_ARM:
            continue
        receipt = Path(str(record.get("path", "")))
        if not receipt.is_file() or sha256_file(receipt) != record.get("sha256"):
            raise JournalSpatialDataError(f"CHARM repeat receipt is missing or changed: {receipt}")
        receipt_t1 = _read_json(receipt).get("inputs", {}).get("t1", {})
        if not isinstance(receipt_t1, dict):
            raise JournalSpatialDataError(f"CHARM repeat receipt lacks T1 provenance: {receipt}")
        try:
            recorded_path = Path(str(receipt_t1.get("path", ""))).resolve()
        except OSError as exc:
            raise JournalSpatialDataError(f"Invalid T1 path in repeat receipt: {receipt}") from exc
        if recorded_path != expected_path or receipt_t1.get("sha256") != inputs.t1_sha256:
            raise JournalSpatialDataError(
                f"Staged T1 does not match CHARM repeat provenance: {receipt}"
            )
        matching += 1
    if matching != 10:
        raise JournalSpatialDataError(f"Expected 10 CHARM T1 records, observed {matching}")
    return inputs.t1, inputs.t1_sha256


def weighted_slice_indices(weights: np.ndarray, count: int) -> tuple[int, ...]:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise JournalSpatialDataError("Slice weights must be a non-empty vector")
    if count < 1:
        raise JournalSpatialDataError("Slice count must be positive")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise JournalSpatialDataError("Slice weights must be finite and non-negative")

    target_count = min(int(count), int(values.size))
    if float(values.sum()) == 0.0:
        return tuple(
            sorted(
                set(
                    int(index)
                    for index in np.rint(
                        np.linspace(0, values.size - 1, target_count)
                    ).astype(int)
                )
            )
        )

    cumulative = np.cumsum(values) / float(values.sum())
    quantiles = np.linspace(0.0, 1.0, target_count + 2)[1:-1]
    selected = {
        min(int(np.searchsorted(cumulative, quantile, side="left")), values.size - 1)
        for quantile in quantiles
    }
    if len(selected) < target_count:
        candidates = sorted(range(values.size), key=lambda index: (-values[index], index))
        selected.update(candidates[: target_count - len(selected)])
    if len(selected) < target_count:
        for index in np.rint(np.linspace(0, values.size - 1, values.size)).astype(int):
            selected.add(int(index))
            if len(selected) == target_count:
                break
    return tuple(sorted(selected)[:target_count])


def _canonical_arrays(modal: np.ndarray, counts: np.ndarray, affine: np.ndarray):
    modal_image = nib.as_closest_canonical(nib.Nifti1Image(modal, affine))
    count_image = nib.as_closest_canonical(nib.Nifti1Image(counts, affine))
    canonical_modal = np.asanyarray(modal_image.dataobj).astype(modal.dtype, copy=False)
    canonical_counts = np.asanyarray(count_image.dataobj).astype(np.uint8, copy=False)
    if canonical_modal.shape != canonical_counts.shape or not np.array_equal(
        modal_image.affine, count_image.affine
    ):
        raise JournalSpatialDataError("Canonical label and disagreement grids differ")
    return canonical_modal, canonical_counts, np.asarray(modal_image.affine, dtype=np.float64)


def _normalised_t1(t1_path: Path, shape: tuple[int, ...], affine: np.ndarray) -> np.ndarray:
    try:
        image = nib.load(str(t1_path), keep_file_open=True)
        resampled = resample_from_to(image, (shape, affine), order=1)
        values = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
        raise JournalSpatialDataError(f"Could not load and resample staged T1: {t1_path}") from exc
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise JournalSpatialDataError("Staged T1 contains no finite positive intensities")
    low, high = np.percentile(positive, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high):
        raise JournalSpatialDataError("Staged T1 robust intensity range is invalid")
    if high <= low:
        normalised = np.where(np.isfinite(values) & (values > 0), 1.0, 0.0)
    else:
        normalised = np.clip((values - low) / (high - low), 0.0, 1.0)
        normalised[~np.isfinite(normalised)] = 0.0
    return np.asarray(normalised, dtype=np.float32)


def _crop_bounds(
    modal: np.ndarray, affine: np.ndarray, margin_mm: float = 5.0
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    coordinates = np.argwhere(modal != 0)
    if coordinates.size == 0:
        raise JournalSpatialDataError("Consensus label map contains no non-background voxels")
    voxel_sizes = nib.affines.voxel_sizes(affine)
    margins = np.ceil(margin_mm / voxel_sizes).astype(int)
    starts = np.maximum(coordinates.min(axis=0) - margins, 0)
    stops = np.minimum(coordinates.max(axis=0) + margins + 1, modal.shape)
    bounds = tuple((int(start), int(stop)) for start, stop in zip(starts, stops))
    if any(stop <= start for start, stop in bounds):
        raise JournalSpatialDataError("Computed spatial crop is empty")
    return bounds  # type: ignore[return-value]


def _slice_selections(
    counts: np.ndarray,
    bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
):
    slices = tuple(slice(start, stop) for start, stop in bounds)
    cropped = counts[slices]
    plane_weights = {
        "sagittal": cropped.sum(axis=(1, 2), dtype=np.uint64),
        "coronal": cropped.sum(axis=(0, 2), dtype=np.uint64),
        "axial": cropped.sum(axis=(0, 1), dtype=np.uint64),
    }
    main = {}
    atlas = {}
    for axis, orientation in enumerate(ORIENTATIONS):
        start, stop = bounds[axis]
        weights = np.asarray(plane_weights[orientation], dtype=np.float64)
        if float(weights.sum()) == 0.0:
            main[orientation] = (start + stop - 1) // 2
        else:
            main[orientation] = start + int(np.argmax(weights))
        atlas[orientation] = tuple(
            start + index for index in weighted_slice_indices(weights, count=4)
        )
    return main, atlas


def _world_coordinate(affine: np.ndarray, orientation: str, index: int) -> float:
    axis = ORIENTATIONS.index(orientation)
    voxel = np.zeros(3, dtype=np.float64)
    voxel[axis] = index
    world = nib.affines.apply_affine(affine, voxel)
    return float(world[axis])


def prepare_journal_spatial_data(config: ExperimentConfig) -> JournalSpatialData:
    if config.repeats_per_arm != 10:
        raise JournalSpatialDataError("Journal figures require the pinned ten-repeat scope")
    try:
        source = load_visualization_data(config.output_root)
    except VisualizationDataError as exc:
        raise JournalSpatialDataError(str(exc)) from exc
    t1_path, t1_sha256 = _validated_staged_t1(config)

    counts = np.rint(source.disagreement_frequency * config.repeats_per_arm).astype(
        np.uint8
    )
    maximum = int(counts.max())
    if maximum > MAX_DISPLAY_DISAGREEMENT:
        raise JournalSpatialDataError(
            f"Observed disagreement count {maximum} exceeds the approved 1-6 scale"
        )
    modal, counts, affine = _canonical_arrays(source.modal_labels, counts, source.affine)
    t1 = _normalised_t1(t1_path, modal.shape, affine)
    bounds = _crop_bounds(modal, affine)
    main, atlas = _slice_selections(counts, bounds)

    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    distribution = {
        str(value): int(np.count_nonzero(counts == value))
        for value in range(1, MAX_DISPLAY_DISAGREEMENT + 1)
    }
    summary = {
        "repeat_count": config.repeats_per_arm,
        "maximum_disagreement_count": maximum,
        "nonzero_disagreement_voxels": int(np.count_nonzero(counts)),
        "disagreement_count_distribution": distribution,
        "voxel_volume_mm3": voxel_volume,
        "changed_volume_mm3": float(np.count_nonzero(counts) * voxel_volume),
        "crop_bounds_voxels": [list(bound) for bound in bounds],
        "main_slice_indices": dict(main),
        "main_slice_coordinates_mm": {
            orientation: _world_coordinate(affine, orientation, index)
            for orientation, index in main.items()
        },
        "atlas_slice_indices": {
            orientation: list(indices) for orientation, indices in atlas.items()
        },
        "atlas_slice_coordinates_mm": {
            orientation: [
                _world_coordinate(affine, orientation, index) for index in indices
            ]
            for orientation, indices in atlas.items()
        },
    }
    return JournalSpatialData(
        staging_root=config.output_root.resolve(),
        subject=config.subject,
        repeat_count=config.repeats_per_arm,
        t1_path=t1_path,
        t1_sha256=t1_sha256,
        modal_labels=modal,
        disagreement_counts=counts,
        t1_normalized=t1,
        affine=affine,
        crop_bounds=bounds,
        main_slice_indices=main,
        atlas_slice_indices=atlas,
        spatial_summary=summary,
    )
