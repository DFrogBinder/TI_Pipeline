"""NIfTI I/O for the single retained harmonized geometry map."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from . import __version__
from .labels import harmonize_labels
from .settings import SubjectPaths


ALLOWED_GEOMETRY_LABELS = frozenset({0, 1, 2, 3, 4, 5, 6, 11})
REQUIRED_TISSUE_LABELS = (1, 2, 3, 4, 5, 6)


class LabelImageError(ValueError):
    """Raised when a retained label image would be unsafe or incomplete."""


@dataclass(frozen=True)
class SubjectLabelReceipt:
    subject: str
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    receipt_path: str


def _world_bounds(image: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(image.shape[:3], dtype=float)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise LabelImageError(f"Image must have a nonempty 3D grid, got {image.shape}")
    edges = [(-0.5, size - 0.5) for size in shape]
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in edges[0]
            for y in edges[1]
            for z in edges[2]
        ]
    )
    world = (np.asarray(image.affine, dtype=float) @ corners.T).T[:, :3]
    if not np.all(np.isfinite(world)):
        raise LabelImageError("Image affine must contain only finite values")
    return world.min(axis=0), world.max(axis=0)


def _grids_overlap(
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> bool:
    source_min, source_max = _world_bounds(source)
    reference_min, reference_max = _world_bounds(reference)
    return bool(np.all(source_max >= reference_min) and np.all(reference_max >= source_min))


def _validated_geometry_data(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 3:
        raise LabelImageError(f"Geometry map must be three-dimensional, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise LabelImageError("Geometry map contains non-finite values")
    rounded = np.rint(array)
    if not np.all(array == rounded):
        raise LabelImageError("Geometry map contains fractional labels")
    integers = rounded.astype(np.int64, copy=False)
    observed = {int(value) for value in np.unique(integers)}
    unexpected = sorted(observed - ALLOWED_GEOMETRY_LABELS)
    if unexpected:
        raise LabelImageError(f"Geometry map contains unsupported labels: {unexpected}")
    return integers


def resample_geometry_to_reference(
    source_img: nib.spatialimages.SpatialImage,
    reference_img: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    """Nearest-neighbour resample of one categorical map to the v3 T1fs grid."""
    _validated_geometry_data(np.asanyarray(source_img.dataobj))
    if not _grids_overlap(source_img, reference_img):
        raise LabelImageError("Source and reference image grids do not overlap")
    resampled = resample_from_to(
        source_img,
        (reference_img.shape[:3], reference_img.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    data = _validated_geometry_data(np.asanyarray(resampled.dataobj))
    missing = [label for label in REQUIRED_TISSUE_LABELS if not np.any(data == label)]
    if missing:
        raise LabelImageError(
            f"Resampling emptied required tissue labels: {missing}"
        )
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(data.astype(np.uint8), reference_img.affine, header=header)
    qform, qcode = reference_img.get_qform(coded=True)
    sform, scode = reference_img.get_sform(coded=True)
    if qform is not None:
        result.set_qform(qform, code=int(qcode))
    if sform is not None:
        result.set_sform(sform, code=int(scode))
    return result


def harmonize_subject_labels(subject_paths: SubjectPaths) -> SubjectLabelReceipt:
    """Create the one retained map and a compact provenance receipt."""
    source_path = subject_paths.corrected_label.resolve(strict=True)
    reference_path = (
        subject_paths.staged_m2m / "T1fs_conform.nii.gz"
    ).resolve(strict=True)
    source_sha256 = _sha256(source_path)
    source_img = nib.load(source_path)
    source_data = np.asanyarray(source_img.dataobj)
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(source_img.affine))
    harmonized, stats = harmonize_labels(source_data, spacing)
    source_header = source_img.header.copy()
    source_header.set_data_dtype(np.uint8)
    source_geometry = nib.Nifti1Image(
        harmonized,
        source_img.affine,
        header=source_header,
    )
    source_qform, source_qcode = source_img.get_qform(coded=True)
    source_sform, source_scode = source_img.get_sform(coded=True)
    if source_qform is not None:
        source_geometry.set_qform(source_qform, code=int(source_qcode))
    if source_sform is not None:
        source_geometry.set_sform(source_sform, code=int(source_scode))
    reference_img = nib.load(reference_path)
    retained_img = resample_geometry_to_reference(source_geometry, reference_img)
    if _sha256(source_path) != source_sha256:
        raise LabelImageError(f"Corrected source changed while being read: {source_path}")

    output_path = subject_paths.harmonized_label
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_path(output_path.parent, suffix=".nii.gz")
    try:
        nib.save(retained_img, temporary_output)
        _validated_geometry_data(np.asanyarray(nib.load(temporary_output).dataobj))
        output_sha256 = _sha256(temporary_output)
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    receipt_payload = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "subject": subject_paths.subject,
        "source": _image_record(source_path, source_img, source_sha256),
        "reference": _image_record(reference_path, reference_img, _sha256(reference_path)),
        "output": _image_record(output_path, retained_img, output_sha256),
        "harmonization": asdict(stats),
        "software": {
            "pipeline_version": __version__,
            "numpy": np.__version__,
            "nibabel": nib.__version__,
        },
    }
    _atomic_write_json(subject_paths.harmonization_qc, receipt_payload)
    return SubjectLabelReceipt(
        subject=subject_paths.subject,
        source_path=str(source_path),
        source_sha256=source_sha256,
        output_path=str(output_path.resolve()),
        output_sha256=output_sha256,
        receipt_path=str(subject_paths.harmonization_qc.resolve()),
    )


@contextmanager
def temporary_surface_masks(
    harmonized_path: Path,
    scratch_root: Path,
) -> Iterator[dict[str, Path]]:
    """Materialize v3 mask inputs in an automatically removed task directory."""
    image = nib.load(harmonized_path)
    data = _validated_geometry_data(np.asanyarray(image.dataobj))
    missing = [label for label in REQUIRED_TISSUE_LABELS if not np.any(data == label)]
    if missing:
        raise LabelImageError(f"Retained map is missing required tissue labels: {missing}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    label_by_name = {
        "WM": 1,
        "GM": 2,
        "CSF": 3,
        "BONE": 4,
        "SKIN": 5,
        "EYES": 6,
        "AIR": 11,
    }
    with tempfile.TemporaryDirectory(
        prefix=f"{harmonized_path.stem}-masks-",
        dir=scratch_root,
    ) as directory:
        working = Path(directory)
        masks: dict[str, Path] = {}
        for name, label in label_by_name.items():
            mask_path = working / f"MASK_{name}.nii.gz"
            _save_like(image, (data == label).astype(np.uint8), mask_path)
            masks[name] = mask_path
        yield masks


def _save_like(
    reference: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    destination: Path,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(data.astype(np.uint8), reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, code=int(qcode))
    if sform is not None:
        image.set_sform(sform, code=int(scode))
    nib.save(image, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(directory: Path, *, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=".tmp-", suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(raw_path)


def _image_record(
    path: Path,
    image: nib.spatialimages.SpatialImage,
    checksum: str,
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": checksum,
        "shape": [int(value) for value in image.shape[:3]],
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "voxel_sizes_mm": [
            float(value) for value in nib.affines.voxel_sizes(image.affine)
        ],
    }


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path.parent, suffix=".json")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
