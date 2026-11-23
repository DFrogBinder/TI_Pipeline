from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from ti_current_repair.repeatability_v2.cache import (
    prepare_pair_caches,
    select_representative_repeat,
)
from ti_current_repair.repeatability_v2.execution import (
    SimulationTask,
    SourceInputs,
    run_fixed_task,
    run_remesh_task,
)
from ti_current_repair.repeatability_v2.simnibs_adapter import (
    CorrectedCharmHeadBuilder,
    SimNIBS46Adapter,
    SimNIBS46RemeshAdapter,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeAdapter:
    version = "4.6.0"

    def rebuild_head(self, segmentation: Path, output: Path) -> Path:
        return _write(output, f"head:{_sha(segmentation)}")

    def construct_electrode_geometry(
        self, head_mesh: Path, *, pair: int, target: str, output: Path, seed: int | None
    ) -> Path:
        assert seed is None
        return _write(output, f"geometry:{head_mesh.read_text()}:{target}:pair{pair}")

    def solve_component(self, geometry: Path, output: Path) -> Path:
        return _write(output, f"field:{geometry.read_text()}")

    def geometry_metadata(self, pair: int) -> dict[str, object]:
        return {
            "pair": pair,
            "currents_a": [0.001, -0.001],
            "conductivity_by_tag": {"2": 0.275, "100": 1.4},
            "electrode_surface_tags": [1000 + pair * 2 + 3, 1000 + pair * 2 + 4],
            "electrode_names": ["F8", "P7"] if pair == 1 else ["F7", "P8"],
            "solver_options": "-ksp_type cg",
            "n_workers": 8,
        }

    def solve_existing_mesh(self, geometry: Path, output: Path, metadata: dict) -> Path:
        return _write(output, f"fixed-field:{geometry.read_text()}:{metadata['pair']}")

    def combine_ti(self, pair1: Path, pair2: Path, output: Path) -> Path:
        return _write(output, f"ti:{pair1.read_text()}:{pair2.read_text()}")

    def geometry_sha256(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        text = text.removeprefix("field:").removeprefix("fixed-field:")
        return hashlib.sha256(text.split(":1")[0].encode()).hexdigest()

    def strip_result_fields(self, source: Path, output: Path) -> Path:
        text = source.read_text(encoding="utf-8").removeprefix("field:")
        return _write(output, text)


def _inputs(tmp_path: Path) -> SourceInputs:
    return SourceInputs(
        corrected_segmentation=_write(tmp_path / "seg.nii.gz", "segmentation"),
        atlas=_write(tmp_path / "atlas.nii.gz", "atlas"),
        targets_csv=_write(tmp_path / "targets.csv", "targets"),
    )


def test_remesh_rebuilds_head_and_both_electrode_geometries_unseeded(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    before = {path: _sha(path) for path in inputs.paths}
    task = SimulationTask(
        subject="sub-CC110174", target="left_hippocampus", condition="remesh", repeat=1
    )

    manifest = run_remesh_task(
        task=task,
        inputs=inputs,
        expected_hashes=before,
        output_root=tmp_path / "v2-output",
        adapter=FakeAdapter(),
    )

    assert [event["operation"] for event in manifest["events"]] == [
        "head_mesh_rebuilt",
        "electrode_geometry_constructed",
        "component_solved",
        "electrode_geometry_constructed",
        "component_solved",
        "ti_combined",
    ]
    electrode_events = [
        event for event in manifest["events"]
        if event["operation"] == "electrode_geometry_constructed"
    ]
    assert [event["pair"] for event in electrode_events] == [1, 2]
    assert all(event["seed"] is None for event in electrode_events)
    assert manifest["nifti_role"] == "not_generated_authoritative_stage"
    assert manifest["currents_a"] == {
        "1": [0.001, -0.001],
        "2": [0.001, -0.001],
    }
    assert manifest["electrode_surface_tags"] == {
        "1": [1005, 1006],
        "2": [1007, 1008],
    }
    assert manifest["electrode_names"]["1"] == ["F8", "P7"]
    assert {path: _sha(path) for path in inputs.paths} == before


def test_median_repeat_selection_uses_nearest_value_then_lowest_repeat() -> None:
    rows = [
        {"repeat": 1, "roi_median_v_per_m": 1.0},
        {"repeat": 2, "roi_median_v_per_m": 3.0},
        {"repeat": 3, "roi_median_v_per_m": 5.0},
        {"repeat": 4, "roi_median_v_per_m": 7.0},
    ]
    selected = select_representative_repeat(rows)
    assert selected["repeat"] == 2


def test_pair_caches_are_separate_immutable_and_fixed_runs_never_place_electrodes(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    pair1 = _write(tmp_path / "remesh" / "pair1.msh", "field:geometry-pair1")
    pair2 = _write(tmp_path / "remesh" / "pair2.msh", "field:geometry-pair2")
    manifest = {
        "subject": "sub-CC110174",
        "target": "left_hippocampus",
        "repeat": 7,
        "component_geometry_paths": {"1": str(pair1), "2": str(pair2)},
        "currents_a": {"1": [0.001, -0.001], "2": [0.001, -0.001]},
        "conductivity_by_tag": {"2": 0.275, "100": 1.4},
        "electrode_surface_tags": {"1": [1005, 1006], "2": [1007, 1008]},
        "solver_options": "-ksp_type cg",
    }
    cache = prepare_pair_caches(
        remesh_manifest=manifest,
        cache_root=tmp_path / "cache",
        adapter=adapter,
    )

    assert cache["pair_caches"]["1"] != cache["pair_caches"]["2"]
    for path in map(Path, cache["pair_caches"].values()):
        assert path.is_file()
        assert path.stat().st_mode & 0o222 == 0

    fixed = run_fixed_task(
        task=SimulationTask(
            subject="sub-CC110174",
            target="left_hippocampus",
            condition="fixed_mesh",
            repeat=2,
        ),
        cache_manifest=Path(cache["manifest_path"]),
        output_root=tmp_path / "fixed-output",
        adapter=adapter,
    )
    assert fixed["electrode_placement_invoked"] is False
    assert [event["operation"] for event in fixed["events"]] == [
        "existing_mesh_component_solved",
        "existing_mesh_component_solved",
        "ti_combined",
    ]
    assert fixed["geometry_sha256"] == cache["geometry_sha256"]


def test_simnibs46_adapter_calls_existing_mesh_fem_tdcs_contract(tmp_path: Path) -> None:
    calls: list[tuple] = []

    class Elements:
        tag1 = np.asarray([2, 100, 1005, 1006], dtype=np.int64)

    class Mesh:
        elm = Elements()

    class ElementData:
        def __init__(self, value, name, mesh):
            self.value = np.asarray(value)
            self.name = name
            self.mesh = mesh

    class MeshIO:
        @staticmethod
        def read_msh(path: str):
            return Mesh()

        @staticmethod
        def write_msh(mesh, path: str) -> None:
            _write(Path(path), "solved")

    MeshIO.ElementData = ElementData

    class Fem:
        @staticmethod
        def tdcs(mesh, cond, currents, electrode_surface_tags, **kwargs):
            calls.append((mesh, cond, currents, electrode_surface_tags, kwargs))
            return "potential"

        @staticmethod
        def calc_fields(potential, fields, **kwargs):
            assert potential == "potential"
            assert fields == "E"
            return Mesh()

    geometry = _write(tmp_path / "pair1-cache.msh", "geometry")
    output = tmp_path / "pair1-result.msh"
    adapter = SimNIBS46Adapter(fem_module=Fem, mesh_io_module=MeshIO)
    adapter.solve_existing_mesh(
        geometry,
        output,
        {
            "pair": 1,
            "currents_a": [0.001, -0.001],
            "conductivity_by_tag": {"2": 0.275, "100": 1.4},
            "electrode_surface_tags": [1005, 1006],
            "solver_options": "-ksp_type cg",
            "n_workers": 8,
        },
    )

    assert output.read_text(encoding="utf-8") == "solved"
    assert len(calls) == 1
    _, cond, currents, tags, kwargs = calls[0]
    assert cond.value.tolist() == [0.275, 1.4, 0.0, 0.0]
    assert currents == [0.001, -0.001]
    assert tags == [1005, 1006]
    assert kwargs == {
        "n_workers": 8,
        "units": "mm",
        "solver_options": "-ksp_type cg",
    }


def test_remesh_adapter_places_each_pair_once_then_solves_existing_geometry(
    tmp_path: Path,
) -> None:
    placements: list[tuple[str, str]] = []

    class Elements:
        tag1 = np.asarray([2, 100, 1005, 1006], dtype=np.int64)
        tag2 = tag1.copy()
        elm_type = np.asarray([4, 4, 2, 2], dtype=np.int64)
        node_number_list = np.asarray(
            [[1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 3, -1], [2, 3, 4, -1]],
            dtype=np.int64,
        )

    class Nodes:
        node_coord = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=np.float64,
        )

    class Mesh:
        elm = Elements()
        nodes = Nodes()

    class ElementData:
        def __init__(self, value, name="field", mesh=None):
            self.value = np.asarray(value)
            self.name = name
            self.mesh = mesh

    class MeshIO:
        meshes: dict[str, object] = {}

        @classmethod
        def read_msh(cls, path: str):
            return cls.meshes.get(path, Mesh())

        @classmethod
        def write_msh(cls, mesh, path: str) -> None:
            cls.meshes[path] = mesh
            _write(Path(path), "mesh")

    MeshIO.ElementData = ElementData

    class TDCSList:
        def __init__(self):
            self.mesh = None
            self.currents = []
            self.solver_options = "-ksp_type cg"
            self.cond = [SimpleNamespace(name="electrode", value=1.4)]
            self.electrode = []

        def add_electrode(self):
            electrode = SimpleNamespace()
            self.electrode.append(electrode)
            return electrode

        def _prepare(self):
            assert Path(self.eeg_cap).name == "cap.csv"
            return None

        def _place_electrodes(self):
            placements.append((self.electrode[0].centre, self.electrode[1].centre))
            return self.mesh, [1005, 1006]

        def cond2elmdata(self, mesh):
            return ElementData(
                np.asarray([0.275, 1.4, 0.0, 0.0], dtype=np.float64),
                name="conductivity",
                mesh=mesh,
            )

    class SimStruct:
        TDCSLIST = TDCSList

    class Fem:
        @staticmethod
        def tdcs(mesh, cond, currents, electrode_surface_tags, **kwargs):
            return SimpleNamespace(mesh=mesh)

        @staticmethod
        def calc_fields(potential, fields, **kwargs):
            return potential.mesh

    stimulation = SimpleNamespace(
        pair1=SimpleNamespace(anode="F8", cathode="P7", current_a=0.001),
        pair2=SimpleNamespace(anode="F7", cathode="P8", current_a=0.0012),
        electrode_shape="ellipse",
        electrode_radius_mm=10.0,
        electrode_thickness_mm=2.0,
        electrode_conductivity=1.4,
    )

    def build_head(segmentation: Path, output: Path) -> Path:
        return _write(output, f"head:{segmentation.read_text()}")

    adapter = SimNIBS46RemeshAdapter(
        stimulation=stimulation,
        head_builder=build_head,
        fem_module=Fem,
        mesh_io_module=MeshIO,
        sim_struct_module=SimStruct,
        eeg_cap_path=_write(tmp_path / "cap.csv", "positions"),
        n_workers=8,
    )
    segmentation = _write(tmp_path / "corrected.nii.gz", "corrected")
    head = adapter.rebuild_head(segmentation, tmp_path / "head.msh")
    geometry = adapter.construct_electrode_geometry(
        head,
        pair=1,
        target="left_hippocampus",
        output=tmp_path / "pair1-geometry.msh",
        seed=None,
    )
    result = adapter.solve_component(geometry, tmp_path / "pair1-field.msh")

    assert placements == [("F8", "P7")]
    assert result.is_file()
    metadata = adapter.geometry_metadata(1)
    assert metadata["currents_a"] == [0.001, -0.001]
    assert metadata["electrode_surface_tags"] == [1005, 1006]
    assert metadata["conductivity_by_tag"] == {"2": 0.275, "100": 1.4}


def test_corrected_charm_builder_replaces_label_map_before_meshing(
    tmp_path: Path,
) -> None:
    import nibabel as nib

    affine = np.eye(4)
    t1 = tmp_path / "T1.nii.gz"
    t2 = tmp_path / "T2.nii.gz"
    corrected = tmp_path / "corrected.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine), t1)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine), t2)
    corrected_values = np.full((2, 2, 2), 7, dtype=np.uint16)
    nib.save(nib.Nifti1Image(corrected_values, affine), corrected)
    workspace = tmp_path / "workspace"
    m2m = workspace / "m2m_sub-CC110174"
    observed_at_mesh: list[np.ndarray] = []

    def command_runner(command: list[str], cwd: Path) -> None:
        assert cwd == workspace
        if command[-1] == "--forceqform":
            label = m2m / "label_prep" / "tissue_labeling_upsampled.nii.gz"
            label.parent.mkdir(parents=True, exist_ok=True)
            nib.save(
                nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint16), affine),
                label,
            )
        else:
            label = m2m / "label_prep" / "tissue_labeling_upsampled.nii.gz"
            observed_at_mesh.append(np.asarray(nib.load(label).dataobj))
            _write(m2m / "sub-CC110174.msh", "head-mesh")

    builder = CorrectedCharmHeadBuilder(
        subject="sub-CC110174",
        t1_path=t1,
        t2_path=t2,
        workspace=workspace,
        command_runner=command_runner,
    )
    output = builder(corrected, tmp_path / "output" / "head.msh")

    assert output.read_text(encoding="utf-8") == "head-mesh"
    assert len(observed_at_mesh) == 1
    assert np.array_equal(observed_at_mesh[0], corrected_values)
