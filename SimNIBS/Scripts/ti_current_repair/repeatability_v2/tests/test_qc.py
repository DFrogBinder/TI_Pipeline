from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ti_current_repair.repeatability_v2.qc import (
    QC_DERIVATIVE_DIR,
    validate_native_qc_values,
    write_float32_qc_nifti,
)


def test_qc_nifti_is_float32_and_confined_to_derived_directory(
    tmp_path: Path,
) -> None:
    reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    values = np.asarray(
        [[[1.0, 1.0 + 1e-9], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
        dtype=np.float64,
    )
    output = tmp_path / QC_DERIVATIVE_DIR / "TImax.nii.gz"

    write_float32_qc_nifti(reference=reference, values=values, output=output)

    loaded = nib.load(output)
    assert loaded.get_data_dtype() == np.dtype(np.float32)
    assert QC_DERIVATIVE_DIR in output.parts
    assert np.float32(values[0, 0, 0]) == np.float32(values[0, 0, 1])


def test_native_qc_comparison_uses_explicit_float32_tolerance() -> None:
    result = validate_native_qc_values(
        native=np.asarray([1.00000001, 2.0], dtype=np.float64),
        qc=np.asarray([1.0, 2.0], dtype=np.float32),
        atol=2e-7,
    )
    assert result["status"] == "within_float32_tolerance"
    with pytest.raises(ValueError, match="tolerance"):
        validate_native_qc_values(
            native=np.asarray([1.001], dtype=np.float64),
            qc=np.asarray([1.0], dtype=np.float32),
            atol=2e-7,
        )
