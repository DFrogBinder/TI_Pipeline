from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ti_current_repair.repeatability_v2.extraction import (
    NativeMeshField,
    extract_three_outcomes,
    write_extraction_tables,
)
from ti_current_repair.repeatability_v2.mesh_io import load_native_element_field
from ti_current_repair.repeatability_v2.native_pipeline import (
    map_ti_target_mask_to_pair1,
    parcel_mask_from_atlas,
    under_anode_name,
)
from ti_current_repair.repeatability_v2.precision import read_metric_table
from ti_current_repair.repeatability_v2.roi import (
    build_parcel_clipped_roi,
    build_under_anode_roi,
    volume_weighted_metrics,
)


def test_volume_weighted_metrics_match_hand_calculated_fixture() -> None:
    result = volume_weighted_metrics(
        values=np.asarray([1.0, 3.0, 9.0, np.nan], dtype=np.float64),
        volumes=np.asarray([1.0, 2.0, 1.0, 1.0], dtype=np.float64),
        roi_mask=np.asarray([True, True, True, True]),
    )

    assert result["mean"] == pytest.approx(4.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["p95"] == pytest.approx(9.0)
    assert result["achieved_roi_volume_mm3"] == pytest.approx(5.0)
    assert result["finite_volume_mm3"] == pytest.approx(4.0)
    assert result["finite_volume_fraction"] == pytest.approx(0.8)


def test_volume_weighted_metrics_reject_empty_finite_support() -> None:
    with pytest.raises(ValueError, match="finite support"):
        volume_weighted_metrics(
            values=np.asarray([np.nan], dtype=np.float64),
            volumes=np.asarray([1.0], dtype=np.float64),
            roi_mask=np.asarray([True]),
        )


def test_target_and_under_anode_rois_enforce_volume_location_and_gm() -> None:
    centres = np.asarray([[float(index), 0.0, 0.0] for index in range(12)])
    volumes = np.full(12, 25.0, dtype=np.float64)
    parcel = np.asarray([True] * 10 + [False, False])
    target = build_parcel_clipped_roi(
        centres=centres,
        volumes=volumes,
        parcel_mask=parcel,
        target="left_hippocampus",
        start_radius_mm=1.0,
        radius_step_mm=1.0,
        radius_cap_mm=20.0,
    )
    assert target.provenance["requested_volume_mm3"] == 200.0
    assert not np.any(target.mask & ~parcel)

    nodes = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    )
    gm = np.asarray([False, True, True, True, True, True] + [False] * 6)
    under = build_under_anode_roi(
        electrode_name="F8",
        surface_nodes=nodes,
        surface_triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        tetra_centres=centres,
        tetra_volumes=volumes,
        cortical_gm_mask=gm,
        target_volume_mm3=100.0,
    )
    assert under.provenance["electrode_name"] == "F8"
    assert under.provenance["projected_to_cortical_gm"] is True
    assert not np.any(under.mask & ~gm)
    assert under.provenance["requested_volume_mm3"] == 100.0


def _field(path: Path, name: str, values: np.ndarray) -> NativeMeshField:
    path.write_text(name, encoding="utf-8")
    centres = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    return NativeMeshField(
        path=path,
        field_name=name,
        values=values,
        tetra_centres=centres,
        tetra_volumes=np.asarray([1.0, 1.0], dtype=np.float64),
        geometry_sha256="a" * 64,
        stored_dtype="float64",
    )


def test_extract_three_outcomes_keeps_matched_roi_and_float64_provenance(
    tmp_path: Path,
) -> None:
    ti = _field(tmp_path / "TI.msh", "TImax", np.asarray([1.0, 2.0]))
    pair = _field(
        tmp_path / "pair1.msh",
        "E",
        np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64),
    )
    matched = np.asarray([True, False])
    under = np.asarray([False, True])

    rows = extract_three_outcomes(
        ti_field=ti,
        pair1_field=pair,
        matched_target_mask=matched,
        matched_roi_provenance={"roi_id": "same-target"},
        under_anode_mask=under,
        under_anode_provenance={"electrode_name": "F8"},
        common={"subject": "sub-CC510415", "target": "left_hippocampus"},
    )

    assert tuple(row["outcome_mode"] for row in rows) == (
        "ti_target",
        "tdcs_pair1_matched_target",
        "tdcs_pair1_under_anode",
    )
    assert rows[0]["roi_provenance_sha256"] == rows[1]["roi_provenance_sha256"]
    assert rows[1]["roi_median_v_per_m"] == pytest.approx(5.0)
    assert rows[2]["roi_median_v_per_m"] == pytest.approx(2.0)
    assert all(row["analysis_dtype"] == "float64" for row in rows)
    assert all(str(row["source_msh_path"]).endswith(".msh") for row in rows)
    assert [row["electrode_name"] for row in rows] == ["", "", "F8"]


def test_extract_maps_same_target_roi_across_cropped_ti_and_augmented_pair_mesh(
    tmp_path: Path,
) -> None:
    ti = _field(tmp_path / "TI.msh", "TImax", np.asarray([1.0, 2.0]))
    pair_path = tmp_path / "pair1.msh"
    pair_path.write_text("E", encoding="utf-8")
    pair = NativeMeshField(
        path=pair_path,
        field_name="E",
        values=np.asarray(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [99.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        tetra_centres=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        tetra_volumes=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        geometry_sha256="b" * 64,
        stored_dtype="float64",
        tetra_tags=np.asarray([2, 2, 100], dtype=np.int64),
    )

    rows = extract_three_outcomes(
        ti_field=ti,
        pair1_field=pair,
        matched_target_mask=np.asarray([True, False]),
        pair1_matched_target_mask=np.asarray([True, False, False]),
        matched_roi_provenance={"roi_id": "same-target"},
        under_anode_mask=np.asarray([False, True, False]),
        under_anode_provenance={"electrode_name": "F8"},
        common={},
    )

    assert rows[0]["roi_median_v_per_m"] == pytest.approx(1.0)
    assert rows[1]["roi_median_v_per_m"] == pytest.approx(5.0)
    assert rows[0]["roi_provenance_sha256"] == rows[1]["roi_provenance_sha256"]


def test_target_mask_mapping_verifies_shared_biological_geometry(
    tmp_path: Path,
) -> None:
    ti = NativeMeshField(
        **{
            **_field(
                tmp_path / "TI-map.msh", "TImax", np.asarray([1.0, 2.0])
            ).__dict__,
            "tetra_tags": np.asarray([2, 3], dtype=np.int64),
        }
    )
    pair = NativeMeshField(
        path=_field(
            tmp_path / "pair-map.msh",
            "E",
            np.ones((2, 3), dtype=np.float64),
        ).path,
        field_name="E",
        values=np.ones((3, 3), dtype=np.float64),
        tetra_centres=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        tetra_volumes=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        geometry_sha256="b" * 64,
        stored_dtype="float64",
        tetra_tags=np.asarray([2, 3, 100], dtype=np.int64),
    )

    mapped = map_ti_target_mask_to_pair1(
        ti=ti, pair1=pair, ti_target_mask=np.asarray([True, False])
    )
    assert mapped.tolist() == [True, False, False]

    shifted = NativeMeshField(
        **{
            **pair.__dict__,
            "tetra_centres": pair.tetra_centres
            + np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        }
    )
    with pytest.raises(ValueError, match="biological geometry"):
        map_ti_target_mask_to_pair1(
            ti=ti, pair1=shifted, ti_target_mask=np.asarray([True, False])
        )


def test_extract_rejects_float32_or_geometry_mismatch(tmp_path: Path) -> None:
    ti = _field(tmp_path / "TI.msh", "TImax", np.asarray([1.0, 2.0]))
    pair = _field(tmp_path / "pair1.msh", "E", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="float64"):
        extract_three_outcomes(
            ti_field=ti,
            pair1_field=pair,
            matched_target_mask=np.asarray([True, False]),
            matched_roi_provenance={"roi_id": "target"},
            under_anode_mask=np.asarray([False, True]),
            under_anode_provenance={"electrode_name": "F8"},
            common={},
        )

    pair64 = _field(
        tmp_path / "pair1b.msh", "E", np.ones((2, 3), dtype=np.float64)
    )
    pair64 = NativeMeshField(**{**pair64.__dict__, "geometry_sha256": "b" * 64})
    with pytest.raises(ValueError, match="geometry"):
        extract_three_outcomes(
            ti_field=ti,
            pair1_field=pair64,
            matched_target_mask=np.asarray([True, False]),
            matched_roi_provenance={"roi_id": "target"},
            under_anode_mask=np.asarray([False, True]),
            under_anode_provenance={"electrode_name": "F8"},
            common={},
        )


class _Data:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value


class _Nodes:
    node_coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


class _Elements:
    elm_type = np.asarray([2, 4, 4], dtype=np.int64)
    node_number_list = np.asarray(
        [[1, 2, 3, -1], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int64
    )
    tag1 = np.asarray([1005, 2, 2], dtype=np.int64)


class _Mesh:
    nodes = _Nodes()
    elm = _Elements()
    field = {"TImax": _Data(np.asarray([99.0, 1.0, 2.0], dtype=np.float64))}


def test_native_mesh_loader_selects_tetrahedra_and_preserves_float64(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TI.msh"
    path.write_text("fixture", encoding="utf-8")

    field = load_native_element_field(
        path, "TImax", mesh_reader=lambda _: _Mesh()
    )

    assert field.values.tolist() == [1.0, 2.0]
    assert field.values.dtype == np.float64
    assert field.tetra_centres.shape == (2, 3)
    assert np.all(field.tetra_volumes > 0)
    assert len(field.geometry_sha256) == 64


def test_native_mesh_loader_rejects_float32_field(tmp_path: Path) -> None:
    class Mesh32(_Mesh):
        field = {"TImax": _Data(np.asarray([99.0, 1.0, 2.0], dtype=np.float32))}

    path = tmp_path / "TI.msh"
    path.write_text("fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="float64"):
        load_native_element_field(path, "TImax", mesh_reader=lambda _: Mesh32())


def test_extraction_writer_creates_three_schema_compatible_tables(
    tmp_path: Path,
) -> None:
    ti = _field(tmp_path / "TI.msh", "TImax", np.asarray([1.0, 2.0]))
    pair = _field(
        tmp_path / "pair1.msh",
        "E",
        np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64),
    )
    rows = extract_three_outcomes(
        ti_field=ti,
        pair1_field=pair,
        matched_target_mask=np.asarray([True, False]),
        matched_roi_provenance={"roi_id": "same-target"},
        under_anode_mask=np.asarray([False, True]),
        under_anode_provenance={"electrode_name": "F8"},
        common={"subject": "sub-CC510415"},
    )
    geometry = [
        {
            "subject": "sub-CC510415",
            "condition": "remesh",
            "pair": 1,
            "geometry_sha256": "a" * 64,
        }
    ]

    written = write_extraction_tables(tmp_path / "tables", rows, geometry)

    assert set(written) == {
        "ti_native_mesh_metrics.csv",
        "tdcs_pair1_matched_target_metrics.csv",
        "tdcs_pair1_under_anode_metrics.csv",
        "geometry_audit.csv",
    }
    loaded = read_metric_table(
        written["ti_native_mesh_metrics.csv"],
        float_fields=("roi_median_v_per_m",),
        float_precision="round_trip",
    )
    assert loaded[0]["source_msh_path"].endswith("TI.msh")


def test_mesh_centres_map_to_integer_atlas_labels_without_field_resampling(
    tmp_path: Path,
) -> None:
    import nibabel as nib

    labels = np.zeros((3, 3, 3), dtype=np.int16)
    labels[0, 0, 0] = 17
    labels[1, 1, 1] = 17
    path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), path)
    centres = np.asarray(
        [[0.1, 0.1, 0.1], [1.1, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=np.float64,
    )

    mask = parcel_mask_from_atlas(centres, path, label_ids=(17,))

    assert mask.tolist() == [True, True, False]
    assert under_anode_name("left_hippocampus") == "F8"
    assert under_anode_name("right_m1") == "Fp2"
