#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import os
import shutil
import signal
import sys
from pathlib import Path
import argparse
import concurrent.futures
import json
import numpy as np
import simnibs as sim
import subprocess
import nibabel as nib

from nibabel.processing import resample_from_to

from copy import deepcopy
#from simnibs import sim_struct, mesh_io, ElementTags
from simnibs import *
from simnibs.utils import TI_utils as TI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sim_utils import (
    format_output_dir,
    img_info,
)
from utils.subject_inputs import SubjectInputPaths, resolve_subject_input_paths
from simulation.mesh_reuse import (
    candidate_mesh_paths,
    resolve_existing_mesh,
)
from utils.camcan_dataset import (
    CONFIRMED_TARGETS_SHA256,
    REPEAT_DATASET_PATTERN,
    sha256_file,
    validate_dataset_montage,
)
from target_montages import (
    MONTAGE_PRESETS,
    TARGETS_CSV_PATH,
    resolve_individualized_montage,
    resolve_montage_preset,
    targets_csv_sha256,
)
import time



runMNI152 = False
rootDIR = os.environ.get("TI_SIM_ROOT", "/mnt/parscratch/users/cop23bi/LM1")
DEFAULT_MESH_TIMEOUT_HOURS = 4.0
MESH_TOTAL_TIMEOUT_SECONDS = DEFAULT_MESH_TIMEOUT_HOURS * 60 * 60
MESH_TIMEOUT_EXIT_CODE = 124
SIM_INPUT_EXIT_CODE = 126
DEFAULT_MONTAGE_PRESET = "right-m1"
SELECTED_MONTAGE = None
REUSE_EXISTING_MESH = False
GENERATE_CHARM_MESH = False


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, default=str))


def log_file_info(label: str, path: str) -> None:
    p = Path(path)
    log_event(
        "file_info",
        label=label,
        path=str(p),
        exists=p.exists(),
        size_bytes=p.stat().st_size if p.exists() else None,
    )


def list_montage_presets() -> None:
    print("Available montage presets:")
    for name in sorted(MONTAGE_PRESETS):
        preset = MONTAGE_PRESETS[name]
        print(
            f"- {name}: "
            f"pair1={preset.pair1.anode}->{preset.pair1.cathode} ({preset.pair1.current_a:.6g} A), "
            f"pair2={preset.pair2.anode}->{preset.pair2.cathode} ({preset.pair2.current_a:.6g} A), "
            f"electrode radius={preset.electrode_radius_mm:.1f} mm, "
            f"thickness={preset.electrode_thickness_mm:.1f} mm, "
            f"conductivity={preset.electrode_conductivity:.3g} S/m"
        )
        print(f"  {preset.description}")


class MeshTimeoutError(RuntimeError):
    def __init__(self, *, label: str, cmd: list[str], timeout_sec: float):
        super().__init__(f"{label} timed out after {timeout_sec:.0f} seconds")
        self.label = label
        self.cmd = cmd
        self.timeout_sec = timeout_sec


class SimulationInputError(RuntimeError):
    pass


def _ensure_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data


def _kill_process_group(process: subprocess.Popen, *, label: str, sig: int, name: str) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, sig)
        log_event("cmd_signal", label=label, signal=name, pid=process.pid)
    except ProcessLookupError:
        pass
    except Exception as exc:
        log_event("cmd_signal_error", label=label, signal=name, pid=process.pid, error=str(exc))


def cleanup_subject_generated_outputs(output_root: str, subject: str) -> None:
    """Remove generated TI simulation outputs so retries cannot validate stale data."""
    simnibs_path = Path(output_root)
    candidates = [
        simnibs_path / "Output" / subject,
        simnibs_path / "ti_brain_only.nii.gz",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
                log_event("sim_output_cleanup", kind="dir", path=str(path))
            else:
                path.unlink()
                log_event("sim_output_cleanup", kind="file", path=str(path))
        except Exception as exc:
            log_event(
                "sim_output_cleanup_error",
                kind="dir" if path.is_dir() else "file",
                path=str(path),
                error=str(exc),
            )
            raise


def validate_subject_inputs(subject_dir: str, subject: str) -> SubjectInputPaths:
    try:
        return resolve_subject_input_paths(subject_dir, subject)
    except FileNotFoundError as exc:
        message = str(exc)
        log_event("simulation_input_missing", subject=subject, missing=[message])
        raise SimulationInputError(message) from exc


def validate_charm_only_mesh_reuse(
    subject_inputs: SubjectInputPaths,
    mesh_path: Path,
) -> Path:
    if subject_inputs.custom_segmentation is not None:
        raise SimulationInputError(
            "CHARM-only mesh reuse refuses the live ROAST/custom segmentation: "
            f"{subject_inputs.custom_segmentation}"
        )
    label_path = mesh_path.parent / "label_prep" / "tissue_labeling_upsampled.nii.gz"
    if not label_path.is_file():
        raise SimulationInputError(
            "Installed CHARM label is missing beside the reused mesh: "
            f"{label_path}"
        )
    log_event(
        "charm_only_mesh_reuse_validated",
        subject=subject_inputs.subject,
        mesh_path=str(mesh_path),
        mesh_sha256=sha256_file(mesh_path),
        label_path=str(label_path),
        label_sha256=sha256_file(label_path),
        roast_custom_segmentation_present=False,
    )
    return label_path


def _remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def run_mesh_cmd(
    cmd: list[str],
    *,
    cwd: str,
    label: str,
    mesh_deadline: float | None,
) -> None:
    timeout_sec = _remaining_timeout(mesh_deadline)
    if timeout_sec is not None and timeout_sec <= 0:
        raise MeshTimeoutError(label=label, cmd=cmd, timeout_sec=0.0)
    run_cmd(cmd, cwd=cwd, label=label, timeout_sec=timeout_sec)


def run_cmd(
    cmd: list[str],
    *,
    cwd: str | None = None,
    label: str = "cmd",
    timeout_sec: float | None = None,
) -> None:
    log_event("run_cmd", label=label, cmd=cmd, cwd=cwd, timeout_sec=timeout_sec)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        stdout = _ensure_text(exc.stdout)
        stderr = _ensure_text(exc.stderr)
        _kill_process_group(process, label=label, sig=signal.SIGTERM, name="SIGTERM")
        try:
            extra_stdout, extra_stderr = process.communicate(timeout=30)
            stdout += _ensure_text(extra_stdout)
            stderr += _ensure_text(extra_stderr)
        except subprocess.TimeoutExpired:
            _kill_process_group(process, label=label, sig=signal.SIGKILL, name="SIGKILL")
            extra_stdout, extra_stderr = process.communicate()
            stdout += _ensure_text(extra_stdout)
            stderr += _ensure_text(extra_stderr)

        log_event(
            "cmd_timeout",
            label=label,
            timeout_sec=timeout_sec,
            returncode=process.returncode,
            stdout_tail=stdout[-2000:] if stdout else "",
            stderr_tail=stderr[-2000:] if stderr else "",
        )
        raise MeshTimeoutError(label=label, cmd=cmd, timeout_sec=timeout_sec) from exc

    result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    log_event(
        "cmd_result",
        label=label,
        returncode=result.returncode,
        stdout_tail=result.stdout[-2000:] if result.stdout else "",
        stderr_tail=result.stderr[-2000:] if result.stderr else "",
    )
    result.check_returncode()


def process_subject(subject_entry):
    """Run the TI pipeline for a single subject entry."""
    subject_source = subject_entry
    subject = subject_entry
    subject_start = time.time()
    log_event("subject_start", subject=subject_source)

    if runMNI152:
        #? Use MNI152 template mesh | Adjust paths as needed
        subject = 'MNI152'
        sadnboxDIR      = rootDIR.split('Jake_Data')[0]
        fnamehead    = os.path.join(sadnboxDIR,'simnibs4_exmaples','m2m_MNI152','MNI152.msh')

        output_root  = os.path.join(rootDIR, subject, 'anat','SimNIBS')
        subject_dir = os.path.join(rootDIR, subject, 'anat')

    else:
        fnamehead    = os.path.join(rootDIR, subject, 'anat', f'm2m_{subject}', f'{subject}.msh')
        output_root  = os.path.join(rootDIR,subject, 'anat','SimNIBS')
        subject_dir = os.path.join(rootDIR, subject, 'anat')

    print(f"[INFO] Starting TI pipeline for {subject_source} (using '{subject}' resources).")
    subject_inputs = None if runMNI152 else validate_subject_inputs(subject_dir, subject)
    if subject_inputs is not None:
        log_file_info("t1", str(subject_inputs.t1))
        log_file_info("t2", str(subject_inputs.t2))
        log_event(
            "custom_segmentation_mode",
            subject=subject,
            custom_segmentation_path=str(subject_inputs.custom_segmentation) if subject_inputs.custom_segmentation else None,
            uses_custom_segmentation=subject_inputs.custom_segmentation is not None,
        )

    if REUSE_EXISTING_MESH and not runMNI152:
        resolved_mesh = resolve_existing_mesh(subject_dir, subject)
        if resolved_mesh is None:
            candidates = [str(path) for path in candidate_mesh_paths(subject_dir, subject)]
            log_event(
                "simulation_input_missing",
                subject=subject,
                missing=candidates,
                reason="reuse_existing_mesh requested but no existing mesh was found",
            )
            raise SimulationInputError(
                "Missing existing mesh for "
                f"{subject}. Checked: {', '.join(candidates)}"
            )
        fnamehead = str(resolved_mesh)
        validate_charm_only_mesh_reuse(subject_inputs, resolved_mesh)
        print(f"[INFO] ({subject_source}) Reusing existing mesh: {fnamehead}")
        log_event(
            "mesh_reuse_enabled",
            subject=subject_source,
            mesh_path=fnamehead,
        )

    cleanup_subject_generated_outputs(output_root, subject)

    if REUSE_EXISTING_MESH:
        print(f"[INFO] ({subject_source}) CHARM-only mesh reuse enabled; no meshing command will run.")
    elif GENERATE_CHARM_MESH:
        if runMNI152:
            raise SimulationInputError("--generate-charm-mesh cannot be used for MNI152 mode")
        if subject_inputs.custom_segmentation is not None:
            raise SimulationInputError(
                "Pure CHARM generation refuses the ROAST/custom segmentation: "
                f"{subject_inputs.custom_segmentation}"
            )
        mesh_deadline = (
            time.monotonic() + MESH_TOTAL_TIMEOUT_SECONDS
            if MESH_TOTAL_TIMEOUT_SECONDS is not None
            else None
        )
        command = [
            "charm",
            subject,
            str(subject_inputs.t1),
            str(subject_inputs.t2),
            "--forcerun",
            "--forceqform",
        ]
        run_mesh_cmd(
            command,
            cwd=str(subject_dir),
            label="pure_charm_generation",
            mesh_deadline=mesh_deadline,
        )
        generated_mesh = resolve_existing_mesh(subject_dir, subject)
        if generated_mesh is None:
            raise SimulationInputError(
                f"Pure CHARM generation completed without a mesh for {subject}"
            )
        fnamehead = str(generated_mesh)
        label_path = generated_mesh.parent / "label_prep" / "tissue_labeling_upsampled.nii.gz"
        if not label_path.is_file():
            raise SimulationInputError(
                f"Pure CHARM generation completed without its label map: {label_path}"
            )
        log_event(
            "pure_charm_generation_complete",
            subject=subject,
            mesh_path=fnamehead,
            mesh_sha256=sha256_file(generated_mesh),
            label_path=str(label_path),
            label_sha256=sha256_file(label_path),
            roast_custom_segmentation_present=False,
        )
    else:
        raise AssertionError("no mesh mode selected")

    montage = SELECTED_MONTAGE or resolve_montage_preset(DEFAULT_MONTAGE_PRESET)
    electrode_size = [montage.electrode_radius_mm, montage.electrode_thickness_mm]
    electrode_shape = montage.electrode_shape
    electrode_conductivity = montage.electrode_conductivity

    montage_right = (
        montage.pair1.anode,
        montage.pair1.current_a,
        montage.pair1.cathode,
        -montage.pair1.current_a,
    )
    montage_left = (
        montage.pair2.anode,
        montage.pair2.current_a,
        montage.pair2.cathode,
        -montage.pair2.current_a,
    )
    log_event(
        "montage_config",
        subject=subject_source,
        preset=montage.name,
        roi=montage.roi,
        e_target=montage.e_target,
        stimulated_volume=montage.stimulated_volume,
        configuration=montage.configuration,
        pair1_anode=montage.pair1.anode,
        pair1_cathode=montage.pair1.cathode,
        pair1_current_a=montage.pair1.current_a,
        pair2_anode=montage.pair2.anode,
        pair2_cathode=montage.pair2.cathode,
        pair2_current_a=montage.pair2.current_a,
        electrode_radius_mm=montage.electrode_radius_mm,
        electrode_thickness_mm=montage.electrode_thickness_mm,
        electrode_shape=montage.electrode_shape,
        electrode_conductivity=montage.electrode_conductivity,
        individualized_targets_csv=os.environ.get(
            "TI_INDIVIDUALIZED_TARGETS_CSV"
        ),
        individualized_targets_csv_sha256=os.environ.get(
            "TI_EXPECTED_INDIVIDUALIZED_TARGETS_SHA256"
        ),
        individualized_pareto_selection=os.environ.get(
            "TI_INDIVIDUALIZED_PARETO_SELECTION"
        ),
        individualized_source_mat=os.environ.get(
            "TI_INDIVIDUALIZED_SOURCE_MAT"
        ),
        individualized_source_mat_sha256=os.environ.get(
            "TI_INDIVIDUALIZED_SOURCE_MAT_SHA256"
        ),
    )

    # Brain tissue tags (adjust if your labeling differs)
    brain_tags = np.hstack((np.arange(1, 100), np.arange(1001, 1100)))
    #region Simulation
    # ———— SET UP SESSION ————
    S = sim_struct.SESSION()
    S.fnamehead    = fnamehead
    S.pathfem      = os.path.join(output_root, 'Output',subject)
    os.makedirs(S.pathfem, exist_ok=True)
    format_output_dir(S.pathfem)
    S.element_size = 0.1
    S.map_to_vol   = True

    # ———— DEFINE FIRST TDCS MONTAGE ————
    tdcs1 = S.add_tdcslist()
    
    # -----------------------------
    # Custom tissue conductivities
    # -----------------------------
    custom_conductivities = {
        "WM": 0.126,        # white matter
        "GM": 0.276,        # gray matter
        "CSF": 1.65,
        "Skull": 0.01,
        "Scalp": 0.465,
        "Eye": 0.5,        # adjust if present
        "Muscle": 0.16,    # if present in your mesh
        "Saline": electrode_conductivity  # electrode/saline region
    }

    # Apply to first montage
    for c in tdcs1.cond:
        if c.name in custom_conductivities:
            c.value = float(custom_conductivities[c.name])
            print(f"[COND] {c.name} set to {c.value} S/m")    
    
    tdcs1.currents     = [montage_right[1], montage_right[3]]
    el1 = tdcs1.add_electrode()
    el1.channelnr  = 1
    el1.centre     = montage_right[0]
    el1.shape      = electrode_shape
    el1.dimensions = [electrode_size[0]*2, electrode_size[0]*2]
    el1.thickness  = electrode_size[1]

    el2 = tdcs1.add_electrode()
    el2.channelnr  = 2
    el2.centre     = montage_right[2]
    el2.shape      = electrode_shape
    el2.dimensions = [electrode_size[0]*2, electrode_size[0]*2]
    el2.thickness  = electrode_size[1]

    # ———— DEFINE SECOND TDCS MONTAGE ————
    tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
    tdcs2.currents = [montage_left[1], montage_left[3]]
    tdcs2.electrode[0].centre        = montage_left[0]
    tdcs2.electrode[1].centre        = montage_left[2]
    tdcs2.electrode[0].mesh_element_size = 0.1
    tdcs2.electrode[1].mesh_element_size = 0.1

    # ———— RUN SIMULATION ————
    print(f"Running SimNIBS for TI brain-only mesh… ({subject_source})")
    log_event("simnibs_start", subject=subject_source)
    sim.run_simnibs(S)
    log_event("simnibs_done", subject=subject_source)

    # ———— POST-PROCESS ————

    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_1_scalar.msh'))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_2_scalar.msh'))

    # Define tissue tags (replace with actual IDs from your head model)
    # white_tags = [1003]   # e.g. subcortical/white matter tag

    tags_keep = np.hstack((
        np.arange(0, 499),     # 0–498 inclusive
        np.arange(1000, 1499)  # 1000–1498 inclusive
        ))
   # tags_keep = np.hstack((np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1), np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1)))

    # # Crop to gray + white matter only
    # m1=m1.crop_mesh(tags = tags_keep)
    # m2=m2.crop_mesh(tags = tags_keep)

    m1 = m1.crop_mesh(tags = tags_keep)
    m2 = m2.crop_mesh(tags = tags_keep)

    # Extract field vectors on gray+white mesh
    E1_vec = m1.field['E']
    E2_vec = m2.field['E']

    # Compute TI metric
    TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

    # Build output mesh from gray+white region
    mout = deepcopy(m1)
    mout.elmdata = []

    # Add magnitude and TI fields
    # mout.add_element_field(E1_vec.norm(), 'magnE - pair 1')
    # mout.add_element_field(E2_vec.norm(), 'magnE - pair 2')
    mout.add_element_field(TImax,       'TImax')

    # Write out the gray+white TI mesh
    out_path = os.path.join(S.pathfem, 'TI.msh')
    mesh_io.write_msh(mout, out_path)
    print(f"Saved gray+white TI mesh to: {out_path}")
    #endregion
    #region Saving Results
    volume_masks_path = os.path.join(S.pathfem,'Volume_Maks')
    if not os.path.isdir(volume_masks_path):
        os.mkdir(volume_masks_path)

    volume_base_path = os.path.join(S.pathfem,'Volume_Base')
    if not os.path.isdir(volume_base_path):
        os.mkdir(volume_base_path)

    volume_labels_path = os.path.join(S.pathfem,'Volume_Labels')
    if not os.path.isdir(volume_labels_path):
        os.mkdir(volume_labels_path)

    labels_path = os.path.join(volume_labels_path, "TI_Volumetric_Labels")
    masks_path = os.path.join(volume_masks_path, "TI_Volumetric_Masks")
    ti_volume_path = os.path.join(volume_base_path, "TI_Volumetric_Base")

    print('Exporting volumetric meshes...')
    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, labels_path,"--create_label"], label="msh2nii_labels")
        else:
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), str(subject_inputs.t1), labels_path,"--create_label"], label="msh2nii_labels")
    except Exception as e:
        log_event("error", stage="msh2nii_labels", subject=subject, error=str(e))

    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, masks_path,"--create_masks"], label="msh2nii_masks")
        else:
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), str(subject_inputs.t1), masks_path,"--create_masks"], label="msh2nii_masks")
    except Exception as e:
        log_event("error", stage="msh2nii_masks", subject=subject, error=str(e))

    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, ti_volume_path], label="msh2nii_volume")
        else:
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), str(subject_inputs.t1), ti_volume_path], label="msh2nii_volume")
    except Exception as e:
        log_event("error", stage="msh2nii_volume", subject=subject, error=str(e))
    #endregion

    #region Post-process
    # Loads the label and TI volume files (prefer TI_Volumetric_* outputs)
    label_candidates = sorted(
        [f for f in os.listdir(volume_labels_path) if f.startswith("TI_Volumetric_")]
        or os.listdir(volume_labels_path)
    )
    volume_candidates = sorted(
        [f for f in os.listdir(volume_base_path) if f.startswith("TI_Volumetric_")]
        or os.listdir(volume_base_path)
    )
    label_file_path = label_candidates[0]
    ti_volume_path = volume_candidates[0]
    log_event(
        "volume_selection",
        label_file=label_file_path,
        ti_volume_file=ti_volume_path,
        label_candidates=label_candidates,
        volume_candidates=volume_candidates,
    )



    # Check that the file is a nifti file
    if not label_file_path.endswith('.nii') and not label_file_path.endswith('.nii.gz'):
        raise ValueError("The label file is not a NIfTI file.")

    label_img = nib.load(os.path.join(volume_labels_path,label_file_path))
    data = label_img.get_fdata(dtype=np.float32)  # read into RAM as float32
    affine = label_img.affine
    hdr = label_img.header

    #print(f'—— Label image info for: {label_file_path} ———')
    #print("shape:", data.shape)
    #print("voxel sizes (mm):", hdr.get_zooms()[:3])
    #print("units:", hdr.get_xyzt_units())
    #print(f'———'*19)

    ti_img = nib.load(os.path.join(volume_base_path,ti_volume_path))
    ti_data = ti_img.get_fdata(dtype=np.float32)  # read into RAM as float32
    ti_affine = ti_img.affine
    ti_hdr = ti_img.header

    #print(f'—— TI image info for: {ti_volume_path} ———')
    #print("shape:", ti_data.shape)
    #print("voxel sizes (mm):", ti_hdr.get_zooms()[:3])
    #print("units:", ti_hdr.get_xyzt_units())
    #print(f'———'*19)

    # Extract unique labels
    labels = np.asarray(label_img.dataobj)  # lazy; no copy unless needed
    labels = labels.astype(np.int32, copy=False)
    codes, counts = np.unique(labels, return_counts=True)

    GM_LABELS = {2}
    WM_LABELS = {1}
    brain_mask = np.isin(labels, list(GM_LABELS | WM_LABELS))

    same_shape = ti_img.shape == label_img.shape
    same_affine = np.allclose(ti_img.affine, label_img.affine, atol=1e-3)
    log_event(
        "grid_check",
        same_shape=same_shape,
        same_affine=same_affine,
        ti_shape=ti_img.shape,
        label_shape=label_img.shape,
    )
    if not (same_shape and same_affine):
        label_img = resample_from_to(label_img, ti_img, order=0)

    ti_data = ti_img.get_fdata(dtype=np.float32)
    masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)
    finite = np.isfinite(masked)
    log_event(
        "ti_stats",
        ti_min=float(np.nanmin(ti_data)),
        ti_max=float(np.nanmax(ti_data)),
        ti_mean=float(np.nanmean(ti_data)),
        brain_mask_voxels=int(brain_mask.sum()),
        masked_finite_voxels=int(finite.sum()),
    )

    # --- Save outputs ---
    masked_img = nib.Nifti1Image(masked, ti_img.affine, ti_img.header)
    masked_img.header.set_data_dtype(np.float32)
    nib.save(masked_img, os.path.join(output_root,"ti_brain_only.nii.gz"))
    log_file_info("ti_brain_only", os.path.join(output_root,"ti_brain_only.nii.gz"))

    #endregion

    elapsed = time.time() - subject_start
    print(f"[INFO] Completed TI pipeline for {subject_source} in {elapsed:.2f} seconds.")
    log_event("subject_done", subject=subject_source, elapsed_sec=elapsed)
    return elapsed




def run_many_subjects(max_workers: int | None = None):
    """
    Legacy / local mode: process all subjects found in rootDIR with a pool.

    This is NOT what we use on the Slurm array. On the array we always call
    process_subject() with a single --subject from the job script.
    """
    # Discover subject folders
    subjects = [
        d for d in os.listdir(rootDIR)
        if os.path.isdir(os.path.join(rootDIR, d))
    ]

    if not subjects:
        print(f"[WARN] No subjects found in root directory '{rootDIR}'; exiting.")
        return None

    # Default: as many workers as CPUs, capped at number of subjects
    if max_workers is None:
        max_workers = min(len(subjects), max(1, os.cpu_count() or 1))

    print(f"[INFO] Launching TI pipeline on {len(subjects)} subject(s) "
          f"with {max_workers} worker(s).")

    subject_durations: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {
            executor.submit(process_subject, subject): subject
            for subject in subjects
        }
        for future in concurrent.futures.as_completed(future_to_subject):
            subject = future_to_subject[future]
            try:
                duration = future.result()
                if duration is not None:
                    subject_durations[subject] = duration
            except Exception as exc:
                print(f"[ERROR] Failure while processing {subject}: {exc}")

    return subject_durations


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Interference pipeline runner (single- or multi-subject)."
    )
    parser.add_argument(
        "--subject",
        help=(
            "Run the pipeline for a single subject ID, e.g. 'sub-CC110056'. "
            "This is the mode to use from the Slurm job array."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Max number of worker threads when processing multiple subjects. "
            "Ignored when --subject is given. Defaults to #CPUs (capped by #subjects)."
        ),
    )
    parser.add_argument(
        "--montage-preset",
        default=os.environ.get("TI_MONTAGE_PRESET", DEFAULT_MONTAGE_PRESET),
        help=(
            "Named montage preset to run. Can also be set with TI_MONTAGE_PRESET. "
            "Use --list-montage-presets to print available values."
        ),
    )
    parser.add_argument(
        "--list-montage-presets",
        action="store_true",
        help="Print available montage presets and exit.",
    )
    parser.add_argument(
        "--reuse-existing-mesh",
        action="store_true",
        help=(
            "Run simulation only with an existing CHARM mesh. This mode also "
            "requires the installed CHARM label and refuses live ROAST/custom maps."
        ),
    )
    parser.add_argument(
        "--generate-charm-mesh",
        action="store_true",
        help=(
            "Explicit pure-CHARM mode for experiments that start from T1/T2 only. "
            "This mode refuses ROAST/custom maps and never merges segmentations."
        ),
    )
    parser.add_argument(
        "--mesh-timeout-hours",
        type=float,
        default=DEFAULT_MESH_TIMEOUT_HOURS,
        help="Timeout for explicit --generate-charm-mesh mode; ignored for mesh reuse.",
    )
    parser.add_argument(
        "--individualized-targets-csv",
        help=(
            "Subject/ROI-specific Pareto montage table. When supplied, the "
            "TI_free.Emin row for --subject and the dataset ROI replaces the "
            "fixed cohort montage."
        ),
    )
    parser.add_argument(
        "--expected-individualized-targets-sha256",
        help=(
            "Required SHA-256 for --individualized-targets-csv. The runner "
            "fails closed if it is absent or does not match."
        ),
    )

    args = parser.parse_args()
    if args.list_montage_presets:
        list_montage_presets()
        return

    global SELECTED_MONTAGE
    try:
        SELECTED_MONTAGE = resolve_montage_preset(args.montage_preset)
    except ValueError as exc:
        parser.error(str(exc))
    global REUSE_EXISTING_MESH
    REUSE_EXISTING_MESH = bool(args.reuse_existing_mesh)
    global GENERATE_CHARM_MESH
    GENERATE_CHARM_MESH = bool(args.generate_charm_mesh)
    if REUSE_EXISTING_MESH and GENERATE_CHARM_MESH:
        parser.error("--reuse-existing-mesh and --generate-charm-mesh are mutually exclusive.")
    if not REUSE_EXISTING_MESH and not GENERATE_CHARM_MESH:
        parser.error(
            "Select --reuse-existing-mesh for CamCan reruns or the explicit "
            "--generate-charm-mesh pure-CHARM experiment mode."
        )

    start = time.time()
    global MESH_TOTAL_TIMEOUT_SECONDS
    MESH_TOTAL_TIMEOUT_SECONDS = (
        args.mesh_timeout_hours * 60 * 60 if args.mesh_timeout_hours > 0 else None
    )
    print(f"[INFO] Montage preset: {SELECTED_MONTAGE.name}")
    actual_targets_hash = targets_csv_sha256()
    if actual_targets_hash != CONFIRMED_TARGETS_SHA256:
        parser.error(
            "targets.csv is not the confirmed optimized file: "
            f"{actual_targets_hash} != {CONFIRMED_TARGETS_SHA256}"
        )
    dataset_name = Path(rootDIR).name
    if REPEAT_DATASET_PATTERN.fullmatch(dataset_name):
        try:
            dataset_config = validate_dataset_montage(
                [dataset_name], SELECTED_MONTAGE.name
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        dataset_config = None
    if args.individualized_targets_csv:
        if not args.subject:
            parser.error(
                "--individualized-targets-csv requires single-subject "
                "--subject mode."
            )
        if dataset_config is None:
            parser.error(
                "--individualized-targets-csv requires a CamCan "
                "<ROI>_Data_<repeat> TI_SIM_ROOT."
            )
        if not args.expected_individualized_targets_sha256:
            parser.error(
                "--expected-individualized-targets-sha256 is required with "
                "--individualized-targets-csv."
            )
        try:
            SELECTED_MONTAGE, individualized_row = (
                resolve_individualized_montage(
                    args.individualized_targets_csv,
                    subject=args.subject,
                    dataset_roi=dataset_config.dataset_prefix,
                    expected_targets_roi=dataset_config.targets_roi,
                    expected_montage_preset=dataset_config.montage_preset,
                    expected_sha256=args.expected_individualized_targets_sha256,
                )
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        individualized_path = str(
            Path(args.individualized_targets_csv)
            .expanduser()
            .resolve(strict=True)
        )
        os.environ["TI_INDIVIDUALIZED_TARGETS_CSV"] = individualized_path
        os.environ[
            "TI_EXPECTED_INDIVIDUALIZED_TARGETS_SHA256"
        ] = args.expected_individualized_targets_sha256
        os.environ["TI_INDIVIDUALIZED_PARETO_SELECTION"] = individualized_row[
            "pareto_selection"
        ]
        os.environ["TI_INDIVIDUALIZED_SOURCE_MAT"] = individualized_row.get(
            "source_mat", ""
        )
        os.environ[
            "TI_INDIVIDUALIZED_SOURCE_MAT_SHA256"
        ] = individualized_row.get("source_mat_sha256", "")
        log_event(
            "individualized_montage_selected",
            subject=args.subject,
            dataset_roi=dataset_config.dataset_prefix,
            targets_roi=dataset_config.targets_roi,
            montage_preset=SELECTED_MONTAGE.name,
            pareto_selection=individualized_row["pareto_selection"],
            source_mat=individualized_row.get("source_mat"),
            source_mat_sha256=individualized_row.get("source_mat_sha256"),
            individualized_targets_csv=individualized_path,
            individualized_targets_csv_sha256=(
                args.expected_individualized_targets_sha256
            ),
        )
    log_event(
        "montage_preset_selected",
        preset=SELECTED_MONTAGE.name,
        requested=args.montage_preset,
        targets_csv=str(TARGETS_CSV_PATH),
        targets_csv_sha256=actual_targets_hash,
    )
    log_event(
        "mesh_reuse_enabled",
        enabled=REUSE_EXISTING_MESH,
    )
    log_event(
        "pure_charm_generation_enabled",
        enabled=GENERATE_CHARM_MESH,
        mesh_timeout_seconds=MESH_TOTAL_TIMEOUT_SECONDS if GENERATE_CHARM_MESH else None,
    )

    if args.subject:
        # ---------- Single-subject (Slurm array) mode ----------
        subject_id = args.subject.strip()
        print(f"[INFO] Running TI pipeline for single subject: {subject_id}")
        try:
            duration = process_subject(subject_id)
        except SimulationInputError as exc:
            total_runtime = time.time() - start
            log_event(
                "subject_input_error",
                subject=subject_id,
                error=str(exc),
                total_runtime_sec=total_runtime,
                exit_code=SIM_INPUT_EXIT_CODE,
            )
            print(f"[ERROR] {exc}")
            sys.exit(SIM_INPUT_EXIT_CODE)
        except MeshTimeoutError as exc:
            total_runtime = time.time() - start
            log_event(
                "subject_mesh_timeout",
                subject=subject_id,
                stage=exc.label,
                timeout_sec=exc.timeout_sec,
                total_runtime_sec=total_runtime,
                exit_code=MESH_TIMEOUT_EXIT_CODE,
            )
            print(f"[ERROR] Pure CHARM generation timed out for {subject_id}.")
            sys.exit(MESH_TIMEOUT_EXIT_CODE)
        total_runtime = time.time() - start

        print("Done.")
        print(f"[INFO] Subject {subject_id} runtime: "
              f"{(duration or 0.0):.2f} seconds.")
        print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")

    else:
        # ---------- Multi-subject / local mode ----------
        subject_durations = run_many_subjects(max_workers=args.max_workers)
        total_runtime = time.time() - start

        print("Done.")
        print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")

        if subject_durations:
            sequential_estimate = sum(subject_durations.values())
            print(
                f"[INFO] Sum of per-subject runtimes (sequential baseline): "
                f"{sequential_estimate:.2f} seconds"
            )
            if total_runtime > 0:
                speedup = sequential_estimate / total_runtime
                print(
                    f"[INFO] Approximate speed-up vs sequential: "
                    f"{speedup:.2f}x"
                )


if __name__ == "__main__":
    main()
