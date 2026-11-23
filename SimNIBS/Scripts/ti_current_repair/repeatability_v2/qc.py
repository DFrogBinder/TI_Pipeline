"""Explicitly non-authoritative float32 NIfTI derivatives and QC checks."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .precision import require_float64


QC_DERIVATIVE_DIR = "_derived_qc"


def write_float32_qc_nifti(
    *, reference: nib.spatialimages.SpatialImage, values: object, output: Path
) -> Path:
    if QC_DERIVATIVE_DIR not in output.parts:
        raise ValueError(f"QC NIfTI must live under {QC_DERIVATIVE_DIR}")
    data = np.asarray(values)
    if tuple(data.shape) != tuple(reference.shape[:3]):
        raise ValueError("QC values do not match the reference grid")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32), reference.affine, header
    )
    image.header.set_data_dtype(np.float32)
    nib.save(image, str(output))
    return output


def validate_native_qc_values(
    *, native: object, qc: object, atol: float
) -> dict[str, object]:
    native_values = require_float64(native, "native QC comparison values")
    qc_values = np.asarray(qc)
    if qc_values.dtype != np.dtype(np.float32):
        raise ValueError("QC comparison values must be float32")
    if native_values.shape != qc_values.shape:
        raise ValueError("Native and QC comparison shapes differ")
    difference = np.abs(native_values - qc_values.astype(np.float64))
    if not np.isfinite(difference).all() or np.any(difference > float(atol)):
        raise ValueError("Native and NIfTI QC values exceed the expected float32 tolerance")
    return {
        "status": "within_float32_tolerance",
        "atol": float(atol),
        "maximum_absolute_difference": float(np.max(difference, initial=0.0)),
        "sample_count": int(difference.size),
    }


def _mesh_io_module():
    try:
        from simnibs import mesh_io  # type: ignore
    except ImportError:
        from simnibs.msh import mesh_io  # type: ignore
    return mesh_io


def export_manifest_qc(
    *, manifest_path: Path, reference_path: Path, output_root: Path
) -> dict[str, object]:
    """Interpolate native fields solely for visualization under `_derived_qc`."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Simulation manifest is not complete: {manifest_path}")
    mesh_io = _mesh_io_module()
    reference = nib.load(str(reference_path))
    derived = output_root / QC_DERIVATIVE_DIR
    derived.mkdir(parents=True, exist_ok=True)
    specifications = (
        (Path(manifest["ti_msh_path"]), "TImax", "TImax.nii.gz"),
        (
            Path(manifest["component_field_paths"]["1"]),
            "E",
            "pair1_magnitude.nii.gz",
        ),
    )
    outputs: list[dict[str, object]] = []
    for mesh_path, field_name, filename in specifications:
        mesh = mesh_io.read_msh(str(mesh_path))
        field = mesh.field[field_name]
        values = require_float64(field.value, f"native {field_name} field")
        if values.ndim == 2:
            if values.shape[1] != 3:
                raise ValueError(f"Unexpected vector field shape: {values.shape}")
            values = np.sqrt(
                np.sum(values * values, axis=1, dtype=np.float64), dtype=np.float64
            )
            field = mesh_io.ElementData(values, name="pair1_magnitude", mesh=mesh)
        destination = derived / filename
        field.to_nifti(
            reference.shape[:3],
            reference.affine,
            fn=str(destination),
            qform=reference.header.get_qform(),
            method="linear",
        )
        interpolated = nib.load(str(destination))
        data = np.asarray(interpolated.dataobj, dtype=np.float64)
        write_float32_qc_nifti(
            reference=interpolated, values=data, output=destination
        )
        outputs.append(
            {
                "path": str(destination),
                "dtype": "float32",
                "role": "derived_qc",
                "source_msh": str(mesh_path),
                "source_field": field_name,
            }
        )
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "role": "derived_qc_only",
        "authoritative_metric_source": False,
        "outputs": outputs,
    }
    (derived / "qc_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
