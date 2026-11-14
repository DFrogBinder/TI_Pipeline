from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import nibabel as nib
import numpy as np


CHARM_ARM = "simnibs401_charm"
CHARM_MAP = "tissue_labeling_upsampled.nii.gz"


class VisualizationDataError(RuntimeError):
    """Raised when completed experiment outputs cannot support honest figures."""


@dataclass(frozen=True)
class VisualizationData:
    staging_root: Path
    analysis_manifest: Mapping[str, object]
    analysis_manifest_sha256: str
    pair_labels: Tuple[Tuple[int, int], ...]
    changed_percent: np.ndarray
    pairwise_changed_percent: np.ndarray
    dice_by_label: Mapping[str, np.ndarray]
    volume_mm3_by_label: Mapping[str, np.ndarray]
    charm_map_paths: Tuple[Path, ...]
    modal_labels: np.ndarray
    disagreement_frequency: np.ndarray
    affine: np.ndarray
    summary_statistics: Mapping[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualizationDataError(f"Could not read JSON source: {path}") from exc
    if not isinstance(payload, dict):
        raise VisualizationDataError(f"Expected a JSON object: {path}")
    return payload


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise VisualizationDataError(f"Could not read table: {path}") from exc


def build_pairwise_matrix(
    rows: Sequence[Mapping[str, object]], repeat_count: int = 10
) -> np.ndarray:
    expected = repeat_count * (repeat_count - 1) // 2
    seen = set()
    matrix = np.zeros((repeat_count, repeat_count), dtype=np.float64)
    for row in rows:
        try:
            left = int(row["repeat_a"])
            right = int(row["repeat_b"])
            fraction = float(row["changed_fraction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VisualizationDataError("Pairwise table contains an invalid row") from exc
        if not 1 <= left < right <= repeat_count:
            raise VisualizationDataError(
                "Repeat pairs must use canonical indices with repeat_a < repeat_b"
            )
        pair = (left, right)
        if pair in seen:
            raise VisualizationDataError(f"Pairwise table contains duplicate pair {pair}")
        if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
            raise VisualizationDataError(f"Invalid changed fraction for pair {pair}")
        seen.add(pair)
        matrix[left - 1, right - 1] = fraction * 100.0
        matrix[right - 1, left - 1] = fraction * 100.0
    if len(seen) != expected:
        raise VisualizationDataError(
            f"Expected {expected} unique repeat pairs, observed {len(seen)}"
        )
    return matrix


def compute_modal_disagreement(label_arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arrays = np.asarray(label_arrays)
    if arrays.ndim < 2 or arrays.shape[0] < 2:
        raise VisualizationDataError("At least two repeated label arrays are required")
    if not np.issubdtype(arrays.dtype, np.integer):
        raise VisualizationDataError("Repeated label arrays must have an integer dtype")
    output_shape = arrays.shape[1:]
    modal = np.zeros(output_shape, dtype=arrays.dtype)
    maximum_count = np.zeros(output_shape, dtype=np.uint8)
    for label in np.unique(arrays):
        count = np.count_nonzero(arrays == label, axis=0).astype(np.uint8)
        update = count > maximum_count
        modal[update] = label
        maximum_count[update] = count[update]
    disagreement = 1.0 - maximum_count.astype(np.float32) / arrays.shape[0]
    return modal, disagreement


def _spatial_disagreement(paths: Sequence[Path], slab_depth: int = 8):
    if len(paths) != 10:
        raise VisualizationDataError(f"Expected 10 CHARM label maps, observed {len(paths)}")
    images = []
    for path in paths:
        try:
            image = nib.load(str(path), keep_file_open=True)
        except (OSError, nib.filebasedimages.ImageFileError) as exc:
            raise VisualizationDataError(f"Could not load CHARM label map: {path}") from exc
        if not np.issubdtype(image.get_data_dtype(), np.integer):
            raise VisualizationDataError(f"CHARM label map is not integer-valued: {path}")
        images.append(image)
    shape = images[0].shape
    affine = np.asarray(images[0].affine, dtype=np.float64)
    if len(shape) != 3:
        raise VisualizationDataError(f"Expected three-dimensional CHARM maps, observed {shape}")
    for path, image in zip(paths[1:], images[1:]):
        if image.shape != shape:
            raise VisualizationDataError(f"CHARM label-map shape differs: {path}")
        if not np.array_equal(np.asarray(image.affine, dtype=np.float64), affine):
            raise VisualizationDataError(f"CHARM label-map affine differs: {path}")

    dtype = np.result_type(*(image.get_data_dtype() for image in images))
    modal = np.zeros(shape, dtype=dtype)
    disagreement = np.zeros(shape, dtype=np.float32)
    for start in range(0, shape[2], slab_depth):
        stop = min(start + slab_depth, shape[2])
        slab = np.stack(
            [np.asanyarray(image.dataobj[..., start:stop]) for image in images],
            axis=0,
        )
        slab_modal, slab_disagreement = compute_modal_disagreement(slab)
        modal[..., start:stop] = slab_modal
        disagreement[..., start:stop] = slab_disagreement
    return modal, disagreement, affine


def _finite_summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise VisualizationDataError("Summary values must be finite and non-empty")
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _parse_dice(rows: Sequence[Mapping[str, str]]) -> Mapping[str, np.ndarray]:
    values: Dict[str, List[float]] = {}
    for row in rows:
        try:
            payload = json.loads(row["dice_by_label"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise VisualizationDataError("Pairwise Dice payload is invalid") from exc
        if not isinstance(payload, dict):
            raise VisualizationDataError("Pairwise Dice payload must be an object")
        for label, value in payload.items():
            values.setdefault(str(label), []).append(float(value))
    for label, observed in values.items():
        if len(observed) != 45:
            raise VisualizationDataError(
                f"Tissue label {label} has {len(observed)} Dice values; expected 45"
            )
    return {label: np.asarray(observed, dtype=np.float64) for label, observed in values.items()}


def _parse_volumes(rows: Sequence[Mapping[str, str]]) -> Mapping[str, np.ndarray]:
    selected = [
        row
        for row in rows
        if row.get("arm") == CHARM_ARM and row.get("relative_path") == CHARM_MAP
    ]
    try:
        selected.sort(key=lambda row: int(row["repeat_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualizationDataError("Tissue-volume repeat index is invalid") from exc
    indices = [int(row["repeat_index"]) for row in selected]
    if indices != list(range(1, 11)):
        raise VisualizationDataError(f"Expected CHARM volume repeats 1-10, observed {indices}")
    values: Dict[str, List[float]] = {}
    expected_labels = None
    for row in selected:
        try:
            payload = json.loads(row["label_volumes_mm3"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise VisualizationDataError("Tissue-volume payload is invalid") from exc
        if not isinstance(payload, dict):
            raise VisualizationDataError("Tissue-volume payload must be an object")
        labels = set(map(str, payload))
        if expected_labels is None:
            expected_labels = labels
        elif labels != expected_labels:
            raise VisualizationDataError("Tissue labels differ across CHARM repeats")
        for label, value in payload.items():
            values.setdefault(str(label), []).append(float(value))
    return {label: np.asarray(observed, dtype=np.float64) for label, observed in values.items()}


def _summary_statistics(
    changed_percent: np.ndarray,
    dice_by_label: Mapping[str, np.ndarray],
    volumes: Mapping[str, np.ndarray],
    disagreement: np.ndarray,
) -> Mapping[str, object]:
    volume_summary = {}
    for label, values in volumes.items():
        mean = float(np.mean(values))
        volume_summary[label] = {
            **_finite_summary(values),
            "coefficient_of_variation_percent": (
                float(np.std(values, ddof=0) / mean * 100.0) if mean else 0.0
            ),
        }
    return {
        "pairwise_comparisons": int(changed_percent.size),
        "changed_label_percent": _finite_summary(changed_percent),
        "dice_by_label": {
            label: _finite_summary(values) for label, values in dice_by_label.items()
        },
        "volume_mm3_by_label": volume_summary,
        "spatial_disagreement": {
            "maximum_frequency": float(np.max(disagreement)),
            "nonzero_voxels": int(np.count_nonzero(disagreement)),
        },
    }


def load_visualization_data(staging_root: Path) -> VisualizationData:
    staging_root = Path(staging_root).resolve()
    analysis_root = staging_root / "analysis"
    manifest_path = analysis_root / "determinism_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("validated_repeats") != 20:
        raise VisualizationDataError("Determinism analysis is not complete with 20 repeats")
    arms = manifest.get("arms")
    if not isinstance(arms, dict) or set(arms) != {
        "simnibs326_headreco",
        CHARM_ARM,
    }:
        raise VisualizationDataError("Determinism manifest arm inventory is invalid")

    pair_rows = [
        row
        for row in _read_tsv(analysis_root / "changed_voxel_summary.tsv")
        if row.get("arm") == CHARM_ARM and row.get("relative_path") == CHARM_MAP
    ]
    matrix = build_pairwise_matrix(pair_rows, repeat_count=10)
    pair_labels = tuple((int(row["repeat_a"]), int(row["repeat_b"])) for row in pair_rows)
    changed_percent = np.asarray(
        [float(row["changed_fraction"]) * 100.0 for row in pair_rows], dtype=np.float64
    )
    dice_by_label = _parse_dice(pair_rows)
    volumes = _parse_volumes(_read_tsv(analysis_root / "tissue_summary.tsv"))
    map_paths = tuple(
        staging_root / CHARM_ARM / f"repeat_{index:03d}" / CHARM_MAP
        for index in range(1, 11)
    )
    modal, disagreement, affine = _spatial_disagreement(map_paths)
    statistics = _summary_statistics(changed_percent, dice_by_label, volumes, disagreement)
    return VisualizationData(
        staging_root=staging_root,
        analysis_manifest=manifest,
        analysis_manifest_sha256=_sha256(manifest_path),
        pair_labels=pair_labels,
        changed_percent=changed_percent,
        pairwise_changed_percent=matrix,
        dice_by_label=dice_by_label,
        volume_mm3_by_label=volumes,
        charm_map_paths=map_paths,
        modal_labels=modal,
        disagreement_frequency=disagreement,
        affine=affine,
        summary_statistics=statistics,
    )
