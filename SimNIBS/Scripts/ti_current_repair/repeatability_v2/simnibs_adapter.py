"""SimNIBS 4.6 existing-mesh solver adapter.

The fixed branch calls :func:`simnibs.simulation.fem.tdcs` directly on an
electrode-augmented cache.  It deliberately exposes no electrode-placement
operation in that code path.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

import numpy as np

from .precision import require_float64


HEAD_VOLUME_TAGS = np.arange(1, 100, dtype=np.int64)


def _run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


class CorrectedCharmHeadBuilder:
    """Build one fresh CHARM mesh after replacing its label map with the correction."""

    def __init__(
        self,
        *,
        subject: str,
        t1_path: Path,
        t2_path: Path,
        workspace: Path,
        command_runner: Callable[[list[str], Path], None] = _run_command,
    ) -> None:
        self.subject = subject
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.workspace = workspace
        self.command_runner = command_runner

    @property
    def eeg_cap_path(self) -> Path:
        return (
            self.workspace
            / f"m2m_{self.subject}"
            / "eeg_positions"
            / "EEG10-10_UI_Jurak_2007.csv"
        )

    def __call__(self, corrected_segmentation: Path, output: Path) -> Path:
        import nibabel as nib

        for path in (self.t1_path, self.t2_path, corrected_segmentation):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"Missing or unsafe CHARM input: {path}")
        if self.workspace.is_symlink():
            raise ValueError(f"Refusing symbolic-link CHARM workspace: {self.workspace}")
        if self.workspace.exists() and any(self.workspace.iterdir()):
            raise RuntimeError(
                f"Refusing to reuse non-empty remesh workspace: {self.workspace}"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.command_runner(
            [
                "charm",
                self.subject,
                str(self.t1_path),
                str(self.t2_path),
                "--forcerun",
                "--forceqform",
            ],
            self.workspace,
        )
        m2m = self.workspace / f"m2m_{self.subject}"
        label_path = m2m / "label_prep" / "tissue_labeling_upsampled.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"CHARM initialization did not create {label_path}")
        corrected = nib.load(str(corrected_segmentation))
        initialized = nib.load(str(label_path))
        if corrected.shape != initialized.shape or not np.allclose(
            corrected.affine, initialized.affine, atol=1e-5
        ):
            raise ValueError(
                "Corrected segmentation grid differs from the initialized CHARM label grid"
            )
        values = np.asarray(corrected.dataobj, dtype=np.float64)
        if not np.isfinite(values).all() or not np.allclose(values, np.rint(values)):
            raise ValueError("Corrected segmentation must contain finite integer labels")
        header = initialized.header.copy()
        header.set_data_dtype(np.uint16)
        nib.save(
            nib.Nifti1Image(
                np.rint(values).astype(np.uint16), initialized.affine, header
            ),
            str(label_path),
        )
        self.command_runner(["charm", self.subject, "--mesh"], self.workspace)
        generated = m2m / f"{self.subject}.msh"
        if not generated.is_file() or generated.stat().st_size == 0:
            raise FileNotFoundError(f"CHARM did not create the head mesh: {generated}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, output)
        return output


class SimNIBS46Adapter:
    version = "4.6.0"

    def __init__(
        self,
        *,
        fem_module: Any = None,
        mesh_io_module: Any = None,
        ti_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if fem_module is None or mesh_io_module is None:
            from simnibs import mesh_io  # type: ignore
            from simnibs.simulation import fem  # type: ignore

            fem_module = fem
            mesh_io_module = mesh_io
        self.fem = fem_module
        self.mesh_io = mesh_io_module
        self.ti_function = ti_function

    @staticmethod
    def _hash_array(digest: Any, label: str, value: object) -> None:
        array = np.ascontiguousarray(value)
        digest.update(label.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())

    def geometry_sha256(self, path: Path) -> str:
        mesh = self.mesh_io.read_msh(str(path))
        digest = hashlib.sha256()
        self._hash_array(digest, "nodes", np.asarray(mesh.nodes.node_coord, dtype="<f8"))
        for label in ("elm_type", "node_number_list", "tag1", "tag2"):
            value = getattr(mesh.elm, label)
            self._hash_array(digest, label, np.asarray(value, dtype="<i8"))
        return digest.hexdigest()

    def strip_result_fields(self, source: Path, output: Path) -> Path:
        mesh = deepcopy(self.mesh_io.read_msh(str(source)))
        mesh.elmdata = []
        mesh.nodedata = []
        output.parent.mkdir(parents=True, exist_ok=True)
        self.mesh_io.write_msh(mesh, str(output))
        return output

    def solve_existing_mesh(
        self, geometry: Path, output: Path, metadata: dict
    ) -> Path:
        mesh = self.mesh_io.read_msh(str(geometry))
        conductivity_by_tag = {
            int(tag): float(value)
            for tag, value in metadata["conductivity_by_tag"].items()
        }
        conductivities = np.asarray(
            [conductivity_by_tag.get(int(tag), 0.0) for tag in mesh.elm.tag1],
            dtype=np.float64,
        )
        cond = self.mesh_io.ElementData(
            conductivities, name="conductivity", mesh=mesh
        )
        tdcs_parameters = inspect.signature(self.fem.tdcs).parameters
        tdcs_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in tdcs_parameters.values()
        )
        tdcs_kwargs: dict[str, object] = {}
        if "n_workers" in tdcs_parameters or tdcs_var_kwargs:
            tdcs_kwargs["n_workers"] = int(metadata.get("n_workers", 1))
        if "units" in tdcs_parameters or tdcs_var_kwargs:
            tdcs_kwargs["units"] = "mm"
        if "solver_options" in tdcs_parameters or tdcs_var_kwargs:
            tdcs_kwargs["solver_options"] = metadata.get("solver_options")
        potential = self.fem.tdcs(
            mesh,
            cond,
            list(metadata["currents_a"]),
            list(metadata["electrode_surface_tags"]),
            **tdcs_kwargs,
        )
        calc_parameters = inspect.signature(self.fem.calc_fields).parameters
        calc_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in calc_parameters.values()
        )
        calc_kwargs: dict[str, object] = {"cond": cond}
        if "units" in calc_parameters or calc_var_kwargs:
            calc_kwargs["units"] = "mm"
        result = self.fem.calc_fields(potential, "E", **calc_kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.mesh_io.write_msh(result, str(output))
        return output

    def combine_ti(self, pair1: Path, pair2: Path, output: Path) -> Path:
        first = self.mesh_io.read_msh(str(pair1)).crop_mesh(tags=HEAD_VOLUME_TAGS)
        second = self.mesh_io.read_msh(str(pair2)).crop_mesh(tags=HEAD_VOLUME_TAGS)
        if not np.array_equal(first.nodes.node_coord, second.nodes.node_coord):
            raise ValueError("Pair head-volume node geometry differs after electrode removal")
        for label in ("elm_type", "node_number_list", "tag1", "tag2"):
            if not np.array_equal(getattr(first.elm, label), getattr(second.elm, label)):
                raise ValueError(
                    f"Pair head-volume element geometry differs after electrode removal: {label}"
                )
        e1 = require_float64(first.field["E"].value, "pair-1 E field")
        e2 = require_float64(second.field["E"].value, "pair-2 E field")
        if e1.shape != e2.shape or e1.ndim != 2 or e1.shape[1] != 3:
            raise ValueError("Pair E fields must have matching N x 3 float64 arrays")
        ti_function = self.ti_function
        if ti_function is None:
            from simnibs.utils.TI_utils import get_maxTI  # type: ignore

            ti_function = get_maxTI
        ti_values = np.asarray(ti_function(e1, e2), dtype=np.float64)
        require_float64(ti_values, "TImax field")
        result = deepcopy(first)
        result.elmdata = []
        result.nodedata = []
        result.add_element_field(ti_values, "TImax")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.mesh_io.write_msh(result, str(output))
        return output


class SimNIBS326Adapter(SimNIBS46Adapter):
    """SimNIBS 3.2.6 existing-mesh adapter with signature-aware kwargs."""

    version = "3.2.6"


class SimNIBS46RemeshAdapter(SimNIBS46Adapter):
    """Place each montage pair independently, then solve its augmented mesh."""

    def __init__(
        self,
        *,
        stimulation: Any,
        head_builder: Callable[[Path, Path], Path],
        fem_module: Any = None,
        mesh_io_module: Any = None,
        sim_struct_module: Any = None,
        eeg_cap_path: Path | None = None,
        n_workers: int = 8,
    ) -> None:
        super().__init__(fem_module=fem_module, mesh_io_module=mesh_io_module)
        if sim_struct_module is None:
            from simnibs.simulation import sim_struct  # type: ignore

            sim_struct_module = sim_struct
        self.sim_struct = sim_struct_module
        self.stimulation = stimulation
        self.head_builder = head_builder
        self.n_workers = int(n_workers)
        self.eeg_cap_path = eeg_cap_path
        self._metadata_by_pair: dict[int, dict[str, object]] = {}
        self._pair_by_geometry: dict[str, int] = {}

    def rebuild_head(self, segmentation: Path, output: Path) -> Path:
        result = self.head_builder(segmentation, output)
        if result != output or not output.is_file():
            raise RuntimeError(f"Head builder did not create the requested mesh: {output}")
        return result

    def _pair(self, pair: int) -> Any:
        if pair == 1:
            return self.stimulation.pair1
        if pair == 2:
            return self.stimulation.pair2
        raise ValueError(f"Pair must be 1 or 2, observed {pair}")

    def construct_electrode_geometry(
        self,
        head_mesh: Path,
        *,
        pair: int,
        target: str,
        output: Path,
        seed: int | None,
    ) -> Path:
        if seed is not None:
            raise ValueError("V2 remesh electrode placement must be unseeded")
        montage = self._pair(pair)
        tdcs = self.sim_struct.TDCSLIST()
        tdcs.mesh = self.mesh_io.read_msh(str(head_mesh))
        if any(
            isinstance(value, str) for value in (montage.anode, montage.cathode)
        ):
            if self.eeg_cap_path is None or not self.eeg_cap_path.is_file():
                raise FileNotFoundError(
                    "String electrode positions require the generated CHARM EEG cap"
                )
            tdcs.eeg_cap = str(self.eeg_cap_path)
        tdcs.currents = [float(montage.current_a), -float(montage.current_a)]
        for conductivity in tdcs.cond:
            if getattr(conductivity, "name", None) == "electrode":
                conductivity.value = float(self.stimulation.electrode_conductivity)
        for channel, centre in ((1, montage.anode), (2, montage.cathode)):
            electrode = tdcs.add_electrode()
            electrode.channelnr = channel
            electrode.centre = centre
            electrode.shape = self.stimulation.electrode_shape
            electrode.dimensions = [
                float(self.stimulation.electrode_radius_mm) * 2.0,
                float(self.stimulation.electrode_radius_mm) * 2.0,
            ]
            electrode.thickness = float(self.stimulation.electrode_thickness_mm)
            electrode.mesh_element_size = 0.1
        tdcs._prepare()
        placed_mesh, surfaces = tdcs._place_electrodes()
        surface_tags = [int(value) for value in np.unique(surfaces)]
        if len(surface_tags) != 2:
            raise ValueError(
                f"Pair {pair} placement returned {len(surface_tags)} surface tags"
            )
        conductivity = tdcs.cond2elmdata(placed_mesh)
        conductivity_values = require_float64(
            conductivity.value, "placed-mesh conductivity"
        )
        element_tags = np.asarray(placed_mesh.elm.tag1, dtype=np.int64)
        if conductivity_values.shape != element_tags.shape:
            raise ValueError("Placed-mesh conductivity does not cover every element")
        conductivity_by_tag: dict[str, float] = {}
        for tag in np.unique(element_tags):
            values = conductivity_values[element_tags == tag]
            unique = np.unique(values)
            if len(unique) != 1:
                raise ValueError(f"Conductivity varies within element tag {int(tag)}")
            value = float(unique[0])
            if value != 0.0:
                conductivity_by_tag[str(int(tag))] = value
        output.parent.mkdir(parents=True, exist_ok=True)
        self.mesh_io.write_msh(placed_mesh, str(output))
        metadata: dict[str, object] = {
            "pair": pair,
            "target": target,
            "currents_a": list(tdcs.currents),
            "conductivity_by_tag": conductivity_by_tag,
            "electrode_surface_tags": surface_tags,
            "electrode_names": [str(montage.anode), str(montage.cathode)],
            "solver_options": getattr(tdcs, "solver_options", None),
            "n_workers": self.n_workers,
            "placement_seed": None,
        }
        self._metadata_by_pair[pair] = metadata
        self._pair_by_geometry[str(output.resolve())] = pair
        return output

    def geometry_metadata(self, pair: int) -> dict[str, object]:
        try:
            return dict(self._metadata_by_pair[pair])
        except KeyError as error:
            raise ValueError(f"No placed geometry metadata recorded for pair {pair}") from error

    def solve_component(self, geometry: Path, output: Path) -> Path:
        try:
            pair = self._pair_by_geometry[str(geometry.resolve())]
        except KeyError as error:
            raise ValueError(f"Geometry was not placed by this adapter: {geometry}") from error
        return self.solve_existing_mesh(
            geometry, output, self.geometry_metadata(pair)
        )
