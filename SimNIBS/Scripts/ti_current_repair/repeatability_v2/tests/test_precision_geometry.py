from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ti_current_repair.repeatability_v2.geometry import geometry_fingerprint
from ti_current_repair.repeatability_v2.precision import (
    PrecisionContractError,
    assert_authoritative_metric_row,
    read_metric_table,
    write_metric_table,
)


def _bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def test_scientific_csv_round_trip_preserves_sub_float32_difference(tmp_path: Path) -> None:
    first = np.float64(1.0)
    second = np.nextafter(first, np.float64(2.0))
    assert np.float32(first) == np.float32(second)
    path = tmp_path / "metrics.csv"
    rows = [
        {"source_msh_path": "/native/a.msh", "roi_median_v_per_m": first},
        {"source_msh_path": "/native/b.msh", "roi_median_v_per_m": second},
    ]

    write_metric_table(path, rows)
    loaded = read_metric_table(
        path,
        float_fields=("roi_median_v_per_m",),
        float_precision="round_trip",
    )

    assert _bits(loaded[0]["roi_median_v_per_m"]) == _bits(first)
    assert _bits(loaded[1]["roi_median_v_per_m"]) == _bits(second)


def test_metric_table_requires_round_trip_reader_and_native_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv"
    write_metric_table(path, [{"value": np.float64(0.1)}])

    with pytest.raises(PrecisionContractError, match="round_trip"):
        read_metric_table(path, float_fields=("value",), float_precision="legacy")
    with pytest.raises(PrecisionContractError, match="NIfTI"):
        assert_authoritative_metric_row(
            {
                "source_msh_path": "/qc/value.nii.gz",
                "stored_field_dtype": "float64",
                "analysis_dtype": "float64",
            }
        )
    with pytest.raises(PrecisionContractError, match="float64"):
        assert_authoritative_metric_row(
            {
                "source_msh_path": "/native/value.msh",
                "stored_field_dtype": "float32",
                "analysis_dtype": "float64",
            }
        )


def test_geometry_fingerprint_ignores_fields_but_detects_geometry_and_tags() -> None:
    nodes = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    tags = np.asarray([2], dtype=np.int64)

    baseline = geometry_fingerprint(nodes, tetrahedra, tags)
    assert baseline == geometry_fingerprint(nodes, tetrahedra, tags)
    assert baseline != geometry_fingerprint(nodes + 0.01, tetrahedra, tags)
    assert baseline != geometry_fingerprint(nodes, tetrahedra, tags + 1)

