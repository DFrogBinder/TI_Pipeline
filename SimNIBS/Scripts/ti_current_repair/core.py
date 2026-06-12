#!/usr/bin/env python3
"""Shared helpers for repairing pair-2 current errors in TI simulations."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


TI_CROP_TAGS = np.hstack((np.arange(0, 499), np.arange(1000, 1499)))
DEFAULT_MANIFEST_NAME = "current_repair_manifest.json"
PAIR2_RERUN_STAGING_DIR = "current_repair_pair2_rerun"


@dataclass(frozen=True)
class CurrentSpec:
    pair1_label: str
    pair1_current_ma: float
    pair2_label: str
    pair2_intended_current_ma: float
    pair2_original_current_ma: float

    @property
    def pair1_current_a(self) -> float:
        return self.pair1_current_ma * 1e-3

    @property
    def pair2_intended_current_a(self) -> float:
        return self.pair2_intended_current_ma * 1e-3

    @property
    def pair2_original_current_a(self) -> float:
        return self.pair2_original_current_ma * 1e-3

    @property
    def scale_factor(self) -> float:
        if self.pair2_original_current_ma == 0:
            raise ZeroDivisionError("pair2_original_current_ma must be non-zero")
        return self.pair2_intended_current_ma / self.pair2_original_current_ma


@dataclass(frozen=True)
class ElectrodePair:
    anode: str
    cathode: str
    current_a: float


@dataclass(frozen=True)
class SimulationSpec:
    subject: str
    head_mesh: Path
    pair2: ElectrodePair
    electrode_radius_mm: float
    electrode_thickness_mm: float
    electrode_shape: str
    electrode_conductivity: float
    element_size: float = 0.1
    conductivity_by_name: Mapping[str, float] = field(default_factory=dict)
    conductivity_by_index: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairKey:
    experiment: str
    subject: str
    dataset: str | None = None
    condition: str | None = None
    repeat_tag: str | None = None

    def identity_tuple(self) -> tuple[str, str, str | None, str | None, str | None]:
        return (
            self.experiment,
            self.subject,
            self.dataset,
            self.condition,
            self.repeat_tag,
        )


@dataclass(frozen=True)
class RepairTask:
    key: RepairKey
    original_subject_root: Path
    output_subject_root: Path
    original_anat_dir: Path
    output_anat_dir: Path
    current_spec: CurrentSpec
    montage_name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def original_sim_root(self) -> Path:
        return self.original_anat_dir / "SimNIBS"

    @property
    def output_sim_root(self) -> Path:
        return self.output_anat_dir / "SimNIBS"

    @property
    def original_dir(self) -> Path:
        return self.original_sim_root / "Output" / self.key.subject

    @property
    def output_dir(self) -> Path:
        return self.output_sim_root / "Output" / self.key.subject

    @property
    def reference_t1(self) -> Path:
        nii = self.output_anat_dir / f"{self.key.subject}_T1w.nii"
        if nii.is_file():
            return nii
        gz = self.output_anat_dir / f"{self.key.subject}_T1w.nii.gz"
        if gz.is_file():
            return gz
        return nii

    @property
    def manifest_path(self) -> Path:
        return self.output_sim_root / DEFAULT_MANIFEST_NAME


@dataclass(frozen=True)
class AgreementMetrics:
    max_abs: float
    mean_abs: float
    rmse: float
    max_rel: float
    mean_rel: float
    n_values: int


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: object) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def copy_subject_tree(original_subject_root: Path, output_subject_root: Path, *, overwrite: bool) -> bool:
    original_subject_root = Path(original_subject_root)
    output_subject_root = Path(output_subject_root)
    if not original_subject_root.is_dir():
        raise FileNotFoundError(f"Original subject root not found: {original_subject_root}")
    if output_subject_root.exists():
        if not overwrite:
            return False
        shutil.rmtree(output_subject_root)
    output_subject_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original_subject_root, output_subject_root, symlinks=True)
    return True


def get_pair_mesh_paths(output_dir: Path, subject: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{subject}_TDCS_1_scalar.msh",
        output_dir / f"{subject}_TDCS_2_scalar.msh",
    )


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _load_simnibs_modules():
    from simnibs import mesh_io, sim_struct  # type: ignore
    import simnibs as sim_module  # type: ignore
    from simnibs.utils import TI_utils  # type: ignore

    return sim_module, mesh_io, sim_struct, TI_utils


def _default_get_max_ti(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    _, _, _, ti_utils = _load_simnibs_modules()
    return ti_utils.get_maxTI(e1, e2)


def _field_values(mesh: object, field_name: str) -> np.ndarray:
    try:
        return np.asarray(mesh.field[field_name].value)
    except Exception as exc:
        raise KeyError(f"Mesh does not contain element field {field_name!r}") from exc


def scale_mesh_e_field(mesh: object, scale_factor: float) -> object:
    """Return a deep-copied mesh with only its E vector field scaled."""
    scaled = deepcopy(mesh)
    scaled.field["E"].value = np.asarray(scaled.field["E"].value) * float(scale_factor)
    return scaled


def crop_for_ti(mesh: object, tags: np.ndarray | None = None) -> object:
    return mesh.crop_mesh(tags=TI_CROP_TAGS if tags is None else tags)


def build_ti_mesh(
    pair1_mesh: object,
    pair2_mesh: object,
    *,
    get_max_ti: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[object, np.ndarray]:
    """Build a TImax mesh from pair-1 and corrected pair-2 scalar meshes."""
    pair1_cropped = crop_for_ti(pair1_mesh)
    pair2_cropped = crop_for_ti(pair2_mesh)
    e1 = _field_values(pair1_cropped, "E")
    e2 = _field_values(pair2_cropped, "E")
    max_ti = (get_max_ti or _default_get_max_ti)(e1, e2)

    out_mesh = deepcopy(pair1_cropped)
    out_mesh.elmdata = []
    out_mesh.add_element_field(max_ti, "TImax")
    return out_mesh, np.asarray(max_ti)


def array_agreement_metrics(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> AgreementMetrics:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Shape mismatch: {left_arr.shape} vs {right_arr.shape}")

    finite = np.isfinite(left_arr) & np.isfinite(right_arr)
    if not np.any(finite):
        return AgreementMetrics(
            max_abs=math.nan,
            mean_abs=math.nan,
            rmse=math.nan,
            max_rel=math.nan,
            mean_rel=math.nan,
            n_values=0,
        )

    left_flat = left_arr[finite]
    right_flat = right_arr[finite]
    diff = right_flat - left_flat
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(left_flat), np.finfo(float).eps)
    rel = abs_diff / denom
    return AgreementMetrics(
        max_abs=float(np.max(abs_diff)),
        mean_abs=float(np.mean(abs_diff)),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        max_rel=float(np.max(rel)),
        mean_rel=float(np.mean(rel)),
        n_values=int(left_flat.size),
    )


def validate_same_repair_key(left: RepairKey, right: RepairKey, *, experiment: str) -> None:
    if left.experiment != experiment or right.experiment != experiment:
        raise ValueError(
            f"Expected two {experiment} repair keys; got {left.experiment!r} and {right.experiment!r}."
        )
    if left.identity_tuple() != right.identity_tuple():
        raise ValueError(f"Cannot compare different runs: {left!r} vs {right!r}")


def run_cmd(cmd: Sequence[str], *, cwd: Path | None = None, label: str = "cmd") -> None:
    result = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stdout = result.stdout[-2000:] if result.stdout else ""
        stderr = result.stderr[-2000:] if result.stderr else ""
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}: {cmd}\n"
            f"stdout tail:\n{stdout}\n"
            f"stderr tail:\n{stderr}"
        )


def _clear_generated_volume_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_msh2nii_outputs(
    *,
    ti_msh: Path,
    reference_t1: Path,
    output_dir: Path,
    runner: Callable[[Sequence[str]], None] | None = None,
) -> dict[str, Path]:
    _require_existing_file(ti_msh, label="TI mesh")
    _require_existing_file(reference_t1, label="reference T1")

    volume_masks = output_dir / "Volume_Maks"
    volume_base = output_dir / "Volume_Base"
    volume_labels = output_dir / "Volume_Labels"
    for path in (volume_masks, volume_base, volume_labels):
        _clear_generated_volume_dir(path)

    labels_prefix = volume_labels / "TI_Volumetric_Labels"
    masks_prefix = volume_masks / "TI_Volumetric_Masks"
    base_prefix = volume_base / "TI_Volumetric_Base"
    command_runner = runner or (lambda cmd: run_cmd(cmd, label=cmd[0]))

    command_runner(["msh2nii", str(ti_msh), str(reference_t1), str(labels_prefix), "--create_label"])
    command_runner(["msh2nii", str(ti_msh), str(reference_t1), str(masks_prefix), "--create_masks"])
    command_runner(["msh2nii", str(ti_msh), str(reference_t1), str(base_prefix)])
    return {
        "volume_masks": volume_masks,
        "volume_base": volume_base,
        "volume_labels": volume_labels,
    }


def _is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def first_nifti_with_prefix(directory: Path, prefix: str) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing volume directory: {directory}")
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.name.startswith(prefix) and _is_nifti(path))
    if not matches:
        raise FileNotFoundError(f"No {prefix}*.nii* file found in {directory}")
    return matches[0]


def write_ti_brain_only(output_dir: Path, sim_root: Path) -> Path:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    label_path = first_nifti_with_prefix(output_dir / "Volume_Labels", "TI_Volumetric_")
    ti_volume_path = first_nifti_with_prefix(output_dir / "Volume_Base", "TI_Volumetric_")
    label_img = nib.load(str(label_path))
    ti_img = nib.load(str(ti_volume_path))
    if label_img.shape != ti_img.shape or not np.allclose(label_img.affine, ti_img.affine, atol=1e-3):
        label_img = resample_from_to(label_img, ti_img, order=0)

    labels = np.rint(np.asarray(label_img.dataobj)).astype(np.int32, copy=False)
    brain_mask = np.isin(labels, [1, 2])
    ti_data = ti_img.get_fdata(dtype=np.float32)
    masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)
    masked_img = nib.Nifti1Image(masked, ti_img.affine, ti_img.header)
    masked_img.header.set_data_dtype(np.float32)

    out_path = sim_root / "ti_brain_only.nii.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(masked_img, str(out_path))
    return out_path


def recompute_ti_outputs(
    *,
    output_dir: Path,
    subject: str,
    reference_t1: Path,
    mesh_io_module: object | None = None,
    get_max_ti: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    msh2nii_runner: Callable[[Sequence[str]], None] | None = None,
    regenerate_volumes: bool = True,
) -> dict[str, Path]:
    if mesh_io_module is None:
        _, mesh_io_module, _, _ = _load_simnibs_modules()

    pair1_path, pair2_path = get_pair_mesh_paths(output_dir, subject)
    _require_existing_file(pair1_path, label="pair-1 scalar mesh")
    _require_existing_file(pair2_path, label="pair-2 scalar mesh")
    pair1_mesh = mesh_io_module.read_msh(str(pair1_path))
    pair2_mesh = mesh_io_module.read_msh(str(pair2_path))
    ti_mesh, _ = build_ti_mesh(pair1_mesh, pair2_mesh, get_max_ti=get_max_ti)

    ti_path = output_dir / "TI.msh"
    mesh_io_module.write_msh(ti_mesh, str(ti_path))

    generated: dict[str, Path] = {"ti_msh": ti_path}
    if regenerate_volumes:
        generated.update(
            run_msh2nii_outputs(
                ti_msh=ti_path,
                reference_t1=reference_t1,
                output_dir=output_dir,
                runner=msh2nii_runner,
            )
        )
        generated["ti_brain_only"] = write_ti_brain_only(output_dir, output_dir.parents[1])
    return generated


def _write_manifest(
    task: RepairTask,
    *,
    method: str,
    generated_files: Mapping[str, Path],
    started_at: float,
    extra: Mapping[str, object] | None = None,
) -> Path:
    payload = {
        "repair_method": method,
        "experiment": task.key.experiment,
        "subject": task.key.subject,
        "dataset": task.key.dataset,
        "condition": task.key.condition,
        "repeat_tag": task.key.repeat_tag,
        "montage_name": task.montage_name,
        "original_subject_root": task.original_subject_root,
        "output_subject_root": task.output_subject_root,
        "original_sim_root": task.original_sim_root,
        "output_sim_root": task.output_sim_root,
        "currents": asdict(task.current_spec),
        "scale_factor": task.current_spec.scale_factor,
        "source_files": {
            "pair1_scalar_msh": task.original_dir / f"{task.key.subject}_TDCS_1_scalar.msh",
            "pair2_scalar_msh": task.original_dir / f"{task.key.subject}_TDCS_2_scalar.msh",
            "ti_msh": task.original_dir / "TI.msh",
            "ti_brain_only": task.original_sim_root / "ti_brain_only.nii.gz",
        },
        "generated_files": generated_files,
        "metadata": dict(task.metadata),
        "started_at_unix": started_at,
        "completed_at_unix": time.time(),
    }
    if extra:
        payload.update(extra)
    return write_json(task.manifest_path, payload)


def repair_scaled_run(
    task: RepairTask,
    *,
    overwrite: bool = False,
    mesh_io_module: object | None = None,
    get_max_ti: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    msh2nii_runner: Callable[[Sequence[str]], None] | None = None,
    regenerate_volumes: bool = True,
) -> Path:
    started = time.time()
    copied = copy_subject_tree(task.original_subject_root, task.output_subject_root, overwrite=overwrite)
    if not copied:
        if task.manifest_path.is_file():
            return task.manifest_path
        raise FileExistsError(
            f"Output subject root already exists without a repair manifest: {task.output_subject_root}. "
            "Pass --overwrite to rebuild it from the original run."
        )
    if mesh_io_module is None:
        _, mesh_io_module, _, _ = _load_simnibs_modules()

    pair1_path, pair2_path = get_pair_mesh_paths(task.output_dir, task.key.subject)
    _require_existing_file(pair1_path, label="pair-1 scalar mesh")
    _require_existing_file(pair2_path, label="pair-2 scalar mesh")
    pair2_mesh = mesh_io_module.read_msh(str(pair2_path))
    scaled_pair2 = scale_mesh_e_field(pair2_mesh, task.current_spec.scale_factor)
    mesh_io_module.write_msh(scaled_pair2, str(pair2_path))

    generated = recompute_ti_outputs(
        output_dir=task.output_dir,
        subject=task.key.subject,
        reference_t1=task.reference_t1,
        mesh_io_module=mesh_io_module,
        get_max_ti=get_max_ti,
        msh2nii_runner=msh2nii_runner,
        regenerate_volumes=regenerate_volumes,
    )
    return _write_manifest(
        task,
        method="scaled",
        generated_files=generated,
        started_at=started,
    )


def _apply_conductivities(tdcs: object, spec: SimulationSpec) -> None:
    for index, value in spec.conductivity_by_index.items():
        tdcs.cond[int(index)].value = float(value)
    if spec.conductivity_by_name:
        for conductivity in tdcs.cond:
            if conductivity.name in spec.conductivity_by_name:
                conductivity.value = float(spec.conductivity_by_name[conductivity.name])


def rerun_pair2_scalar_mesh(
    spec: SimulationSpec,
    *,
    staging_pathfem: Path,
    sim_module: object | None = None,
    sim_struct_module: object | None = None,
) -> Path:
    if sim_module is None or sim_struct_module is None:
        sim_module, _, sim_struct_module, _ = _load_simnibs_modules()

    staging_pathfem.mkdir(parents=True, exist_ok=True)
    session = sim_struct_module.SESSION()
    session.fnamehead = str(spec.head_mesh)
    session.pathfem = str(staging_pathfem)
    session.element_size = spec.element_size
    session.map_to_vol = True

    tdcs = session.add_tdcslist()
    _apply_conductivities(tdcs, spec)
    tdcs.currents = [spec.pair2.current_a, -spec.pair2.current_a]

    first = tdcs.add_electrode()
    first.channelnr = 1
    first.centre = spec.pair2.anode
    first.shape = spec.electrode_shape
    first.dimensions = [spec.electrode_radius_mm * 2, spec.electrode_radius_mm * 2]
    first.thickness = spec.electrode_thickness_mm
    first.mesh_element_size = spec.element_size

    second = tdcs.add_electrode()
    second.channelnr = 2
    second.centre = spec.pair2.cathode
    second.shape = spec.electrode_shape
    second.dimensions = [spec.electrode_radius_mm * 2, spec.electrode_radius_mm * 2]
    second.thickness = spec.electrode_thickness_mm
    second.mesh_element_size = spec.element_size

    sim_module.run_simnibs(session)
    pair2_path = staging_pathfem / f"{spec.subject}_TDCS_1_scalar.msh"
    return _require_existing_file(pair2_path, label="rerun pair-2 scalar mesh")


def repair_pair2_rerun_run(
    task: RepairTask,
    simulation_spec: SimulationSpec,
    *,
    overwrite: bool = False,
    mesh_io_module: object | None = None,
    get_max_ti: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    msh2nii_runner: Callable[[Sequence[str]], None] | None = None,
    regenerate_volumes: bool = True,
) -> Path:
    started = time.time()
    copied = copy_subject_tree(task.original_subject_root, task.output_subject_root, overwrite=overwrite)
    if not copied:
        if task.manifest_path.is_file():
            return task.manifest_path
        raise FileExistsError(
            f"Output subject root already exists without a repair manifest: {task.output_subject_root}. "
            "Pass --overwrite to rebuild it from the original run."
        )
    staging_pathfem = task.output_sim_root / PAIR2_RERUN_STAGING_DIR / "Output" / task.key.subject
    if staging_pathfem.exists() and overwrite:
        shutil.rmtree(staging_pathfem)
    rerun_pair2_path = rerun_pair2_scalar_mesh(simulation_spec, staging_pathfem=staging_pathfem)

    _, final_pair2_path = get_pair_mesh_paths(task.output_dir, task.key.subject)
    final_pair2_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rerun_pair2_path, final_pair2_path)

    generated = recompute_ti_outputs(
        output_dir=task.output_dir,
        subject=task.key.subject,
        reference_t1=task.reference_t1,
        mesh_io_module=mesh_io_module,
        get_max_ti=get_max_ti,
        msh2nii_runner=msh2nii_runner,
        regenerate_volumes=regenerate_volumes,
    )
    generated = {**generated, "pair2_rerun_scalar_msh": final_pair2_path, "pair2_rerun_staging_msh": rerun_pair2_path}
    return _write_manifest(
        task,
        method="pair2-rerun",
        generated_files=generated,
        started_at=started,
        extra={"pair2_rerun_staging_pathfem": staging_pathfem},
    )


def _mesh_field_array(mesh_path: Path, field_name: str, mesh_io_module: object | None = None) -> np.ndarray:
    if mesh_io_module is None:
        _, mesh_io_module, _, _ = _load_simnibs_modules()
    mesh = mesh_io_module.read_msh(str(mesh_path))
    return _field_values(mesh, field_name)


def compare_repaired_run_outputs(
    *,
    left_task: RepairTask,
    right_task: RepairTask,
    output_root: Path,
    mesh_io_module: object | None = None,
    experiment: str,
) -> dict[str, object]:
    validate_same_repair_key(left_task.key, right_task.key, experiment=experiment)
    left_pair2 = _mesh_field_array(
        left_task.output_dir / f"{left_task.key.subject}_TDCS_2_scalar.msh",
        "E",
        mesh_io_module=mesh_io_module,
    )
    right_pair2 = _mesh_field_array(
        right_task.output_dir / f"{right_task.key.subject}_TDCS_2_scalar.msh",
        "E",
        mesh_io_module=mesh_io_module,
    )
    left_ti = _mesh_field_array(left_task.output_dir / "TI.msh", "TImax", mesh_io_module=mesh_io_module)
    right_ti = _mesh_field_array(right_task.output_dir / "TI.msh", "TImax", mesh_io_module=mesh_io_module)

    import nibabel as nib

    left_vol = np.asarray(nib.load(str(left_task.output_sim_root / "ti_brain_only.nii.gz")).dataobj)
    right_vol = np.asarray(nib.load(str(right_task.output_sim_root / "ti_brain_only.nii.gz")).dataobj)

    row = {
        "experiment": experiment,
        "subject": left_task.key.subject,
        "dataset": left_task.key.dataset,
        "condition": left_task.key.condition,
        "repeat_tag": left_task.key.repeat_tag,
        "pair2_e": asdict(array_agreement_metrics(left_pair2, right_pair2)),
        "timax": asdict(array_agreement_metrics(left_ti, right_ti)),
        "ti_brain_only": asdict(array_agreement_metrics(left_vol, right_vol)),
        "left_root": left_task.output_subject_root,
        "right_root": right_task.output_subject_root,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    slug_parts = [
        part
        for part in (
            row["subject"],
            row["dataset"],
            row["condition"],
            row["repeat_tag"],
        )
        if part
    ]
    slug = "__".join(str(part).replace("/", "_") for part in slug_parts)
    write_json(output_root / f"{slug}.json", row)
    return row


def write_comparison_summary(rows: Sequence[Mapping[str, object]], output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_root / "summary.json", list(rows))
    csv_path = output_root / "summary.csv"
    fieldnames = [
        "experiment",
        "subject",
        "dataset",
        "condition",
        "repeat_tag",
        "pair2_e_max_abs",
        "pair2_e_rmse",
        "timax_max_abs",
        "timax_rmse",
        "ti_brain_only_max_abs",
        "ti_brain_only_rmse",
        "left_root",
        "right_root",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pair2 = row["pair2_e"]
            timax = row["timax"]
            volume = row["ti_brain_only"]
            writer.writerow(
                {
                    "experiment": row.get("experiment"),
                    "subject": row.get("subject"),
                    "dataset": row.get("dataset"),
                    "condition": row.get("condition"),
                    "repeat_tag": row.get("repeat_tag"),
                    "pair2_e_max_abs": pair2["max_abs"],
                    "pair2_e_rmse": pair2["rmse"],
                    "timax_max_abs": timax["max_abs"],
                    "timax_rmse": timax["rmse"],
                    "ti_brain_only_max_abs": volume["max_abs"],
                    "ti_brain_only_rmse": volume["rmse"],
                    "left_root": row.get("left_root"),
                    "right_root": row.get("right_root"),
                }
            )
    return {"summary_json": json_path, "summary_csv": csv_path}
