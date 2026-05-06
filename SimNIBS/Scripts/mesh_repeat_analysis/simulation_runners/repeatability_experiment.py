#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
"""Unified runner for paired remesh vs fixed-mesh repeatability experiments."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

HERE = Path(__file__).resolve()
PIPELINE_ROOT = HERE.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiment_config import (  # noqa: E402
    ExperimentConfig,
    ExperimentTask,
    condition_by_name,
    condition_manifest_path,
    iter_experiment_tasks,
    load_experiment_config,
    repeat_tag,
    subject_condition_mesh_cache_root,
    subject_condition_repeats_root,
    subject_condition_root,
    subject_repeatability_root,
    template_config_dict,
    write_template_config,
)


def _find_camcan_root() -> Path:
    candidates = [
        PIPELINE_ROOT / "CamCan_Experiment",
        PIPELINE_ROOT.parent / "CamCan_Experiment",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ModuleNotFoundError(
        "Could not resolve CamCan_Experiment relative to "
        f"{PIPELINE_ROOT}. Checked: {', '.join(str(c) for c in candidates)}"
    )


CAMCAN_ROOT = _find_camcan_root()
if str(CAMCAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMCAN_ROOT))

from utils.sim_utils import atomic_replace  # noqa: E402


# Deterministic meshing settings
USE_CUSTOM_LABELS_ONLY = True
SMOOTH_SCALP = True
SCALP_LABEL = 5
MORPH_KERNEL = (3, 3, 3)
CLOSE_ITERS = 1
OPEN_ITERS = 0
DILATE_ITERS = 0
ERODE_ITERS = 0

SIM_MODULE = None
SIM_MESH_IO = None
SIM_STRUCT = None
SIM_TI = None
MESH_LOCK_TIMEOUT_SEC = 12 * 60 * 60
MESH_LOCK_POLL_SEC = 5.0


@dataclass(frozen=True)
class SourceSubjectPaths:
    subject: str
    subject_root: Path
    anat_dir: Path
    t1_path: Path
    t2_path: Path
    seg_path: Path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    anat_dir: Path
    output_root: Path
    mesh_dir: Path
    mesh_path: Path


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str))


def _ensure_simnibs_imports() -> None:
    global SIM_MODULE, SIM_MESH_IO, SIM_STRUCT, SIM_TI
    if SIM_MODULE is not None:
        return
    import simnibs as sim_module  # type: ignore
    from simnibs import mesh_io, sim_struct  # type: ignore
    from simnibs.utils import TI_utils as ti_utils  # type: ignore

    SIM_MODULE = sim_module
    SIM_MESH_IO = mesh_io
    SIM_STRUCT = sim_struct
    SIM_TI = ti_utils


def log_file_info(label: str, path: Path) -> None:
    log_event(
        "file_info",
        label=label,
        path=str(path),
        exists=path.exists(),
        size_bytes=path.stat().st_size if path.exists() else None,
    )


@contextmanager
def _exclusive_lock(
    lock_path: Path,
    *,
    timeout_sec: float = MESH_LOCK_TIMEOUT_SEC,
    poll_interval_sec: float = MESH_LOCK_POLL_SEC,
):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    last_wait_log_sec = -60.0
    acquired = False
    with lock_path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                waited_sec = time.time() - start
                handle.seek(0)
                handle.truncate()
                handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "host": socket.gethostname(),
                            "acquired_at": time.time(),
                            "waited_sec": waited_sec,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                log_event("lock_acquired", lock_path=str(lock_path), waited_sec=round(waited_sec, 3))
                break
            except BlockingIOError:
                waited_sec = time.time() - start
                if waited_sec - last_wait_log_sec >= 60.0:
                    log_event("lock_wait", lock_path=str(lock_path), waited_sec=round(waited_sec, 3))
                    last_wait_log_sec = waited_sec
                if waited_sec >= timeout_sec:
                    raise TimeoutError(
                        f"Timed out waiting for lock {lock_path} after {waited_sec:.1f} sec"
                    )
                time.sleep(poll_interval_sec)
        try:
            yield
        finally:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                log_event("lock_released", lock_path=str(lock_path))


def run_cmd(cmd: list[str], *, cwd: str | None = None, label: str = "cmd") -> None:
    log_event("run_cmd", label=label, cmd=cmd, cwd=cwd)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log_event(
        "cmd_result",
        label=label,
        returncode=result.returncode,
        stdout_tail=result.stdout[-2000:] if result.stdout else "",
        stderr_tail=result.stderr[-2000:] if result.stderr else "",
    )
    result.check_returncode()


def _smooth_scalp_labels(label_data: np.ndarray) -> np.ndarray:
    mask = label_data == SCALP_LABEL
    structure = np.ones(MORPH_KERNEL, dtype=bool)
    if CLOSE_ITERS > 0:
        mask = ndi.binary_closing(mask, structure=structure, iterations=CLOSE_ITERS)
    if OPEN_ITERS > 0:
        mask = ndi.binary_opening(mask, structure=structure, iterations=OPEN_ITERS)
    if DILATE_ITERS > 0:
        mask = ndi.binary_dilation(mask, structure=structure, iterations=DILATE_ITERS)
    if ERODE_ITERS > 0:
        mask = ndi.binary_erosion(mask, structure=structure, iterations=ERODE_ITERS)

    out = label_data.copy()
    out[out == SCALP_LABEL] = 0
    out[mask] = SCALP_LABEL
    return out


def _source_subject_paths(source_root: Path, subject: str) -> SourceSubjectPaths:
    anat_dir = source_root / subject / "anat"
    return SourceSubjectPaths(
        subject=subject,
        subject_root=source_root / subject,
        anat_dir=anat_dir,
        t1_path=anat_dir / f"{subject}_T1w.nii",
        t2_path=anat_dir / f"{subject}_T2w.nii",
        seg_path=anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
    )


def _validate_source_inputs(paths: SourceSubjectPaths) -> None:
    missing = [
        str(path)
        for path in (paths.anat_dir, paths.t1_path, paths.t2_path, paths.seg_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Subject '{paths.subject}' is missing required inputs: {', '.join(missing)}"
        )


def _ensure_link(dest: Path, src: Path) -> None:
    if dest.is_symlink():
        if dest.exists():
            return
        dest.unlink()
    elif dest.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Cannot create link; source does not exist: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dest)


def _reset_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _link_common_inputs(source_paths: SourceSubjectPaths, target_anat_dir: Path) -> None:
    _ensure_link(target_anat_dir / source_paths.t1_path.name, source_paths.t1_path)
    _ensure_link(target_anat_dir / source_paths.t2_path.name, source_paths.t2_path)
    _ensure_link(target_anat_dir / source_paths.seg_path.name, source_paths.seg_path)


def _workspace_from_anat_dir(anat_dir: Path, subject: str) -> WorkspacePaths:
    mesh_dir = anat_dir / f"m2m_{subject}"
    return WorkspacePaths(
        root=anat_dir.parent,
        anat_dir=anat_dir,
        output_root=anat_dir / "SimNIBS",
        mesh_dir=mesh_dir,
        mesh_path=mesh_dir / f"{subject}.msh",
    )


def _mesh_ready_marker(workspace: WorkspacePaths) -> Path:
    return workspace.anat_dir / ".mesh_ready.json"


def _find_generated_volume(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted(parent.glob("TI_Volumetric_*"))
    return candidates[0] if candidates else None


def _repeat_done_marker(workspace: WorkspacePaths) -> Path:
    return workspace.output_root / "ti_brain_only.nii.gz"


def _repeat_outputs_complete(workspace: WorkspacePaths, subject: str) -> bool:
    output_root = workspace.output_root / "Output" / subject
    if not workspace.mesh_path.is_file():
        return False
    if not _repeat_done_marker(workspace).is_file():
        return False
    if not (output_root / "TI.msh").is_file():
        return False
    if _find_generated_volume(output_root / "Volume_Labels") is None:
        return False
    if _find_generated_volume(output_root / "Volume_Base") is None:
        return False
    return True


def _prepare_repeat_workspace(
    source_paths: SourceSubjectPaths,
    *,
    repeat_root: Path,
    mesh_dir_source: Path | None,
    overwrite: bool,
) -> WorkspacePaths:
    anat_dir = repeat_root / source_paths.subject / "anat"
    if overwrite and anat_dir.exists():
        _reset_dir_contents(anat_dir)
    anat_dir.mkdir(parents=True, exist_ok=True)
    _link_common_inputs(source_paths, anat_dir)
    workspace = _workspace_from_anat_dir(anat_dir, source_paths.subject)
    workspace.output_root.mkdir(parents=True, exist_ok=True)
    if mesh_dir_source is not None:
        _ensure_link(workspace.mesh_dir, mesh_dir_source)
    return workspace


def _prepare_mesh_cache_workspace(
    source_paths: SourceSubjectPaths,
    *,
    mesh_cache_anat_dir: Path,
    overwrite: bool,
) -> WorkspacePaths:
    if overwrite and mesh_cache_anat_dir.exists():
        _reset_dir_contents(mesh_cache_anat_dir)
    mesh_cache_anat_dir.mkdir(parents=True, exist_ok=True)
    _link_common_inputs(source_paths, mesh_cache_anat_dir)
    workspace = _workspace_from_anat_dir(mesh_cache_anat_dir, source_paths.subject)
    workspace.output_root.mkdir(parents=True, exist_ok=True)
    return workspace


def _mesh_workspace(
    workspace: WorkspacePaths,
    *,
    subject: str,
    force_mesh: bool,
) -> Path:
    ready_marker = _mesh_ready_marker(workspace)
    lock_path = workspace.anat_dir / ".mesh_build.lock"

    log_file_info("workspace_t1", workspace.anat_dir / f"{subject}_T1w.nii")
    log_file_info("workspace_t2", workspace.anat_dir / f"{subject}_T2w.nii")
    log_file_info(
        "workspace_segmentation",
        workspace.anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii",
    )
    log_file_info("workspace_mesh", workspace.mesh_path)

    if workspace.mesh_path.exists() and ready_marker.exists() and not force_mesh:
        log_event("mesh_reuse", subject=subject, mesh_path=str(workspace.mesh_path))
        return workspace.mesh_path

    with _exclusive_lock(lock_path):
        if workspace.mesh_path.exists() and ready_marker.exists() and not force_mesh:
            log_event("mesh_reuse", subject=subject, mesh_path=str(workspace.mesh_path))
            return workspace.mesh_path

        if force_mesh and ready_marker.exists():
            ready_marker.unlink()

        if workspace.mesh_path.exists() and not ready_marker.exists():
            log_event(
                "mesh_rebuild_without_ready_marker",
                subject=subject,
                mesh_path=str(workspace.mesh_path),
                note=(
                    "Found a mesh file without a ready marker. Treating it as incomplete "
                    "and rebuilding instead of reusing it."
                ),
            )

        run_cmd(
            [
                "charm",
                subject,
                str(workspace.anat_dir / f"{subject}_T1w.nii"),
                str(workspace.anat_dir / f"{subject}_T2w.nii"),
                "--forcerun",
                "--forceqform",
            ],
            cwd=str(workspace.anat_dir),
            label="charm_init",
        )

        custom_seg_map = nib.load(str(workspace.anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"))
        charm_seg_map_path = (
            workspace.anat_dir
            / f"m2m_sub-{subject.split('-')[-1].upper()}"
            / "label_prep"
            / "tissue_labeling_upsampled.nii.gz"
        )
        log_file_info("charm_seg_map", charm_seg_map_path)

        data = custom_seg_map.get_fdata(dtype=np.float32)
        if not np.allclose(data, np.round(data)):
            log_event(
                "segmentation_rounding_warning",
                subject=subject,
                note="Custom segmentation contained non-integer values; rounding to nearest integers.",
            )
        data = np.rint(data).astype(np.int16)
        if USE_CUSTOM_LABELS_ONLY and SMOOTH_SCALP:
            data = _smooth_scalp_labels(data)

        merged_seg_img_path = workspace.anat_dir / f"{subject}_T1w_ras_1mm_T1andT2_masks_merged.nii"
        out_img = nib.Nifti1Image(data.astype(np.uint16), custom_seg_map.affine, custom_seg_map.header)
        nib.save(out_img, str(merged_seg_img_path))
        atomic_replace(
            str(merged_seg_img_path),
            str(charm_seg_map_path),
            force_int=True,
            int_dtype="uint16",
        )

        run_cmd(
            ["charm", subject, "--mesh"],
            cwd=str(workspace.anat_dir),
            label="charm_remesh",
        )

        if not workspace.mesh_path.exists():
            raise FileNotFoundError(f"Expected mesh was not created: {workspace.mesh_path}")
        _write_json(
            ready_marker,
            {
                "subject": subject,
                "mesh_path": str(workspace.mesh_path),
                "status": "mesh_ready",
                "created_at": time.time(),
            },
        )
        return workspace.mesh_path


def _run_ti_pipeline(
    *,
    subject: str,
    subject_dir: Path,
    output_root: Path,
    fnamehead: Path,
    repeat_tag_value: str,
    condition_name: str,
) -> float:
    _ensure_simnibs_imports()
    subject_start = time.time()
    log_event(
        "simulation_start",
        subject=subject,
        condition=condition_name,
        repeat_tag=repeat_tag_value,
        mesh_path=str(fnamehead),
    )
    _reset_dir_contents(output_root)

    electrode_size = [10, 1]
    electrode_shape = "ellipse"
    electrode_conductivity = 0.85

    montage_right = ("Fp2", 2, "P8", -2)
    montage_left = ("T7", 2, "P7", -2)

    S = SIM_STRUCT.SESSION()
    S.fnamehead = str(fnamehead)
    S.pathfem = str(output_root / "Output" / subject)
    Path(S.pathfem).mkdir(parents=True, exist_ok=True)
    S.element_size = 0.1
    S.map_to_vol = True

    tdcs1 = S.add_tdcslist()
    tdcs1.cond[2].value = electrode_conductivity
    tdcs1.currents = [montage_right[1], montage_right[3]]

    el1 = tdcs1.add_electrode()
    el1.channelnr = 1
    el1.centre = montage_right[0]
    el1.shape = electrode_shape
    el1.dimensions = [electrode_size[0] * 2, electrode_size[0] * 2]
    el1.thickness = electrode_size[1]

    el2 = tdcs1.add_electrode()
    el2.channelnr = 2
    el2.centre = montage_right[2]
    el2.shape = electrode_shape
    el2.dimensions = [electrode_size[0] * 2, electrode_size[0] * 2]
    el2.thickness = electrode_size[1]

    tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
    tdcs2.electrode[0].centre = montage_left[0]
    tdcs2.electrode[1].centre = montage_left[2]
    tdcs2.electrode[0].mesh_element_size = 0.1
    tdcs2.electrode[1].mesh_element_size = 0.1

    SIM_MODULE.run_simnibs(S)

    m1 = SIM_MESH_IO.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_1_scalar.msh"))
    m2 = SIM_MESH_IO.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_2_scalar.msh"))

    tags_keep = np.hstack((np.arange(0, 499), np.arange(1000, 1499)))
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    E1_vec = m1.field["E"]
    E2_vec = m2.field["E"]
    TImax = SIM_TI.get_maxTI(E1_vec.value, E2_vec.value)

    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TImax, "TImax")

    out_path = os.path.join(S.pathfem, "TI.msh")
    SIM_MESH_IO.write_msh(mout, out_path)

    volume_masks_path = Path(S.pathfem) / "Volume_Maks"
    volume_base_path = Path(S.pathfem) / "Volume_Base"
    volume_labels_path = Path(S.pathfem) / "Volume_Labels"
    volume_masks_path.mkdir(exist_ok=True)
    volume_base_path.mkdir(exist_ok=True)
    volume_labels_path.mkdir(exist_ok=True)

    labels_path = volume_labels_path / "TI_Volumetric_Labels"
    masks_path = volume_masks_path / "TI_Volumetric_Masks"
    ti_volume_path = volume_base_path / "TI_Volumetric_Base"
    t1_path = subject_dir / f"{subject}_T1w.nii"

    run_cmd(
        ["msh2nii", os.path.join(S.pathfem, "TI.msh"), str(t1_path), str(labels_path), "--create_label"],
        label="msh2nii_labels",
    )
    run_cmd(
        ["msh2nii", os.path.join(S.pathfem, "TI.msh"), str(t1_path), str(masks_path), "--create_masks"],
        label="msh2nii_masks",
    )
    run_cmd(
        ["msh2nii", os.path.join(S.pathfem, "TI.msh"), str(t1_path), str(ti_volume_path)],
        label="msh2nii_volume",
    )

    label_candidates = sorted(
        [f for f in os.listdir(volume_labels_path) if f.startswith("TI_Volumetric_")]
        or os.listdir(volume_labels_path)
    )
    volume_candidates = sorted(
        [f for f in os.listdir(volume_base_path) if f.startswith("TI_Volumetric_")]
        or os.listdir(volume_base_path)
    )
    label_file_path = label_candidates[0]
    ti_volume_file = volume_candidates[0]
    log_event(
        "volume_selection",
        subject=subject,
        condition=condition_name,
        repeat_tag=repeat_tag_value,
        label_file=label_file_path,
        ti_volume_file=ti_volume_file,
    )

    label_img = nib.load(str(volume_labels_path / label_file_path))
    ti_img = nib.load(str(volume_base_path / ti_volume_file))

    same_shape = ti_img.shape == label_img.shape
    same_affine = np.allclose(ti_img.affine, label_img.affine, atol=1e-3)
    log_event(
        "grid_check",
        subject=subject,
        condition=condition_name,
        repeat_tag=repeat_tag_value,
        same_shape=same_shape,
        same_affine=same_affine,
        ti_shape=ti_img.shape,
        label_shape=label_img.shape,
    )
    if not (same_shape and same_affine):
        label_img = resample_from_to(label_img, ti_img, order=0)

    labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)
    brain_mask = np.isin(labels, [1, 2])

    ti_data = ti_img.get_fdata(dtype=np.float32)
    masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)

    masked_img = nib.Nifti1Image(masked, ti_img.affine, ti_img.header)
    masked_img.header.set_data_dtype(np.float32)
    ti_brain_only_path = output_root / "ti_brain_only.nii.gz"
    nib.save(masked_img, str(ti_brain_only_path))

    elapsed = time.time() - subject_start
    log_event(
        "simulation_done",
        subject=subject,
        condition=condition_name,
        repeat_tag=repeat_tag_value,
        elapsed_sec=elapsed,
        ti_mesh=str(out_path),
        ti_brain_only=str(ti_brain_only_path),
    )
    return elapsed


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _write_condition_manifest(
    config: ExperimentConfig,
    *,
    subject: str,
    condition_name: str,
) -> None:
    condition = condition_by_name(config, condition_name)
    payload = {
        "subject": subject,
        "condition": condition.name,
        "mesh_mode": condition.mesh_mode,
        "repeat_count": condition.repeat_count,
        "description": condition.description,
        "source_root": str(config.source_root),
        "condition_root": str(subject_condition_root(config, subject, condition_name)),
        "repeats_root": str(subject_condition_repeats_root(config, subject, condition_name)),
        "mesh_cache_root": str(subject_condition_mesh_cache_root(config, subject, condition_name)),
    }
    _write_json(condition_manifest_path(config, subject, condition_name), payload)


def _write_task_manifest(
    workspace: WorkspacePaths,
    *,
    task: ExperimentTask,
    config: ExperimentConfig,
    shared_mesh_path: Path | None,
) -> None:
    payload = {
        "subject": task.subject,
        "condition": task.condition_name,
        "mesh_mode": task.mesh_mode,
        "repeat_index": task.repeat_index,
        "repeat_tag": task.repeat_tag,
        "source_root": str(config.source_root),
        "repeat_root": str(workspace.root),
        "repeat_anat_dir": str(workspace.anat_dir),
        "repeat_output_root": str(workspace.output_root),
        "repeat_mesh_path": str(workspace.mesh_path),
        "shared_mesh_path": str(shared_mesh_path) if shared_mesh_path is not None else None,
    }
    _write_json(workspace.root / "task_manifest.json", payload)


def execute_task(
    config: ExperimentConfig,
    task: ExperimentTask,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    force_mesh: bool = False,
) -> dict[str, object]:
    source_paths = _source_subject_paths(config.source_root, task.subject)
    _validate_source_inputs(source_paths)
    _write_condition_manifest(
        config,
        subject=task.subject,
        condition_name=task.condition_name,
    )

    repeat_root = subject_condition_repeats_root(config, task.subject, task.condition_name) / task.repeat_tag
    repeat_workspace = _workspace_from_anat_dir(repeat_root / task.subject / "anat", task.subject)
    shared_mesh_path: Path | None = None

    if task.mesh_mode == "fixed_mesh":
        mesh_cache_workspace = _workspace_from_anat_dir(
            subject_condition_mesh_cache_root(config, task.subject, task.condition_name),
            task.subject,
        )
        shared_mesh_path = mesh_cache_workspace.mesh_path

    result = {
        "subject": task.subject,
        "condition": task.condition_name,
        "mesh_mode": task.mesh_mode,
        "repeat_index": task.repeat_index,
        "repeat_tag": task.repeat_tag,
        "repeat_root": str(repeat_workspace.root),
        "repeat_anat_dir": str(repeat_workspace.anat_dir),
        "repeat_output_root": str(repeat_workspace.output_root),
        "repeat_mesh_path": str(repeat_workspace.mesh_path),
        "shared_mesh_path": str(shared_mesh_path) if shared_mesh_path is not None else None,
        "status": "planned" if dry_run else "completed",
        "skipped_existing": False,
    }

    if _repeat_outputs_complete(repeat_workspace, task.subject) and not overwrite and not dry_run:
        if not (repeat_workspace.root / "task_manifest.json").exists():
            _write_task_manifest(
                repeat_workspace,
                task=task,
                config=config,
                shared_mesh_path=shared_mesh_path,
            )
        log_event(
            "task_skip",
            subject=task.subject,
            condition=task.condition_name,
            repeat_tag=task.repeat_tag,
            reason="required_outputs_exist",
            done_marker=str(_repeat_done_marker(repeat_workspace)),
        )
        result["status"] = "skipped"
        result["skipped_existing"] = True
        return result

    repeat_workspace: WorkspacePaths

    if task.mesh_mode == "remesh":
        repeat_workspace = _prepare_repeat_workspace(
            source_paths,
            repeat_root=repeat_root,
            mesh_dir_source=None,
            overwrite=overwrite,
        )
        if not dry_run:
            _mesh_workspace(
                repeat_workspace,
                subject=task.subject,
                force_mesh=force_mesh or overwrite,
            )
    elif task.mesh_mode == "fixed_mesh":
        mesh_cache_workspace = _prepare_mesh_cache_workspace(
            source_paths,
            mesh_cache_anat_dir=subject_condition_mesh_cache_root(config, task.subject, task.condition_name),
            overwrite=force_mesh,
        )
        if not dry_run:
            shared_mesh_path = _mesh_workspace(
                mesh_cache_workspace,
                subject=task.subject,
                force_mesh=force_mesh,
            )
        else:
            shared_mesh_path = mesh_cache_workspace.mesh_path

        repeat_workspace = _prepare_repeat_workspace(
            source_paths,
            repeat_root=repeat_root,
            mesh_dir_source=mesh_cache_workspace.mesh_dir if not dry_run else None,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported mesh mode: {task.mesh_mode}")

    _write_task_manifest(
        repeat_workspace,
        task=task,
        config=config,
        shared_mesh_path=shared_mesh_path,
    )

    if dry_run:
        log_event(
            "task_planned",
            subject=task.subject,
            condition=task.condition_name,
            repeat_tag=task.repeat_tag,
            mesh_mode=task.mesh_mode,
            repeat_root=str(repeat_workspace.root),
            repeat_mesh_path=str(repeat_workspace.mesh_path),
            shared_mesh_path=str(shared_mesh_path) if shared_mesh_path is not None else None,
        )
        return result

    result["runtime_sec"] = _run_ti_pipeline(
        subject=task.subject,
        subject_dir=repeat_workspace.anat_dir,
        output_root=repeat_workspace.output_root,
        fnamehead=repeat_workspace.mesh_path,
        repeat_tag_value=task.repeat_tag,
        condition_name=task.condition_name,
    )
    return result


def _select_task(
    config: ExperimentConfig,
    *,
    task_index: int | None,
    subject: str | None,
    condition_name: str | None,
    repeat_index: int | None,
) -> ExperimentTask:
    tasks = iter_experiment_tasks(config)
    if task_index is not None:
        if task_index < 0 or task_index >= len(tasks):
            raise IndexError(
                f"task_index {task_index} is out of range. Valid range: 0..{len(tasks) - 1}"
            )
        return tasks[task_index]
    if not subject or not condition_name or repeat_index is None:
        raise ValueError(
            "Provide either --task-index or the full triplet "
            "--subject/--condition/--repeat-index."
        )
    condition = condition_by_name(config, condition_name)
    if repeat_index < 1 or repeat_index > condition.repeat_count:
        raise ValueError(
            f"Repeat index {repeat_index} is outside the configured range for "
            f"condition '{condition_name}' (1..{condition.repeat_count})."
        )
    return ExperimentTask(
        subject=subject,
        condition_name=condition.name,
        mesh_mode=condition.mesh_mode,
        repeat_index=repeat_index,
        repeat_tag=repeat_tag(repeat_index),
    )


def _command_show_plan(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config, validate_paths=False)
    tasks = iter_experiment_tasks(config)
    if args.count_only:
        print(len(tasks))
        return 0
    payload = {
        "config": str(config.config_path),
        "source_root": str(config.source_root),
        "experiment_root": str(config.experiment_root),
        "subjects": config.subjects,
        "conditions": [
            {
                "name": condition.name,
                "mesh_mode": condition.mesh_mode,
                "repeat_count": condition.repeat_count,
                "description": condition.description,
            }
            for condition in config.conditions
        ],
        "task_count": len(tasks),
        "tasks": [
            {
                "task_index": idx,
                "subject": task.subject,
                "condition": task.condition_name,
                "mesh_mode": task.mesh_mode,
                "repeat_index": task.repeat_index,
                "repeat_tag": task.repeat_tag,
            }
            for idx, task in enumerate(tasks)
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _command_run_task(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    task = _select_task(
        config,
        task_index=args.task_index,
        subject=args.subject,
        condition_name=args.condition,
        repeat_index=args.repeat_index,
    )
    result = execute_task(
        config,
        task,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        force_mesh=args.force_mesh,
    )
    print(json.dumps(result, indent=2))
    return 0


def _command_run_condition(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    subject = args.subject.strip()
    condition = condition_by_name(config, args.condition)
    results: list[dict[str, object]] = []
    for repeat_index in range(1, condition.repeat_count + 1):
        task = ExperimentTask(
            subject=subject,
            condition_name=condition.name,
            mesh_mode=condition.mesh_mode,
            repeat_index=repeat_index,
            repeat_tag=repeat_tag(repeat_index),
        )
        results.append(
            execute_task(
                config,
                task,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                force_mesh=args.force_mesh,
            )
        )
    print(json.dumps(results, indent=2))
    return 0


def _command_run_all(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    subjects = config.subjects[: args.max_subjects] if args.max_subjects is not None else config.subjects
    condition_names = (
        [name.strip() for name in args.conditions.split(",") if name.strip()]
        if args.conditions
        else None
    )
    tasks = iter_experiment_tasks(config, subjects=subjects, condition_names=condition_names)
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for task in tasks:
        try:
            results.append(
                execute_task(
                    config,
                    task,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                    force_mesh=args.force_mesh,
                )
            )
        except Exception as exc:
            failure = {
                "subject": task.subject,
                "condition": task.condition_name,
                "repeat_index": task.repeat_index,
                "repeat_tag": task.repeat_tag,
                "error": str(exc),
            }
            failures.append(failure)
            log_event("task_error", **failure)
            if args.fail_fast:
                raise
    payload = {"results": results, "failures": failures}
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


def _command_write_template(args: argparse.Namespace) -> int:
    out_path = write_template_config(args.output)
    print(str(out_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired repeatability experiments with two mesh strategies: "
            "fresh remeshing per repeat and shared fixed-mesh repeats."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_plan = subparsers.add_parser(
        "show-plan",
        help="Print the resolved subject/condition/repeat task plan from the JSON config.",
    )
    show_plan.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    show_plan.add_argument(
        "--count-only",
        action="store_true",
        help="Print only the total task count. Useful for Slurm submission wrappers.",
    )
    show_plan.set_defaults(func=_command_show_plan)

    run_task = subparsers.add_parser(
        "run-task",
        help="Run one subject/condition/repeat task or resolve it from a 0-based task index.",
    )
    run_task.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    run_task.add_argument("--task-index", type=int, default=None, help="0-based task index from show-plan.")
    run_task.add_argument("--subject", default=None, help="Subject ID when not using --task-index.")
    run_task.add_argument("--condition", default=None, help="Condition name when not using --task-index.")
    run_task.add_argument("--repeat-index", type=int, default=None, help="1-based repeat index when not using --task-index.")
    run_task.add_argument("--dry-run", action="store_true", help="Validate paths and print the resolved task without running SimNIBS.")
    run_task.add_argument("--overwrite", action="store_true", help="Reset the repeat workspace and rerun the task even if outputs already exist.")
    run_task.add_argument("--force-mesh", action="store_true", help="Force regeneration of the mesh for this task or its shared mesh cache.")
    run_task.set_defaults(func=_command_run_task)

    run_condition = subparsers.add_parser(
        "run-condition",
        help="Run all repeats for one subject and one configured condition.",
    )
    run_condition.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    run_condition.add_argument("--subject", required=True, help="Subject ID.")
    run_condition.add_argument("--condition", required=True, help="Condition name.")
    run_condition.add_argument("--dry-run", action="store_true", help="Validate and print the planned tasks without running SimNIBS.")
    run_condition.add_argument("--overwrite", action="store_true", help="Reset repeat workspaces before rerunning.")
    run_condition.add_argument("--force-mesh", action="store_true", help="Force regeneration of meshes.")
    run_condition.set_defaults(func=_command_run_condition)

    run_all = subparsers.add_parser(
        "run-all",
        help="Run every configured task sequentially. Useful for smoke tests or small local runs.",
    )
    run_all.add_argument("--config", required=True, help="Path to the experiment JSON config.")
    run_all.add_argument("--conditions", default=None, help="Optional comma-separated subset of condition names.")
    run_all.add_argument("--max-subjects", type=int, default=None, help="Optional cap on the number of subjects.")
    run_all.add_argument("--dry-run", action="store_true", help="Validate and print planned tasks without running SimNIBS.")
    run_all.add_argument("--overwrite", action="store_true", help="Reset repeat workspaces before rerunning.")
    run_all.add_argument("--force-mesh", action="store_true", help="Force regeneration of meshes.")
    run_all.add_argument("--fail-fast", action="store_true", help="Abort immediately on the first task failure.")
    run_all.set_defaults(func=_command_run_all)

    write_template = subparsers.add_parser(
        "write-template-config",
        help="Write a starter JSON config for the paired experiment.",
    )
    write_template.add_argument("--output", required=True, help="Output JSON path.")
    write_template.set_defaults(func=_command_write_template)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
