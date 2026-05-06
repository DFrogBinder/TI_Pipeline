#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import simnibs as sim
from nibabel.processing import resample_from_to
from simnibs import mesh_io, sim_struct
from simnibs.utils import TI_utils as TI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sim_utils import (
    atomic_replace,
    format_output_dir,
    img_info,
    merge_segmentation_maps,
)


# Set appropriate flags.
meshPresent = True
runMNI152 = True
rootDIR = "/home/boyan/sandbox/Jake_Data/Left_Thalamus_MNI152"

electrode_size = [10, 1]  # [radius_mm, thickness_mm]
electrode_shape = "ellipse"
electrode_conductivity = 0.85

# Left Thalamus
montage_right = ("F7", 1.588656e-3, "P7", -1.588656e-3)
montage_left = ("F8", 2e-3, "P8", -2e-3)

# Right Thalamus
# montage_right = ("AF7", 2e-3, "TP7", -2e-3)
# montage_left = ("T8", 2e-3, "PO8", -2e-3)

# Hippocampus montage
# montage_right = ("F10", 2e-3, "P8", -2e-3)
# montage_left = ("T7", 1.588656e-3, "P7", -1.588656e-3)

# M1 montage
# montage_right = ("C1", 1.34e-3, "Cz", -1.34e-3)
# montage_left = ("C3", 2.66e-3, "CP5", -2.66e-3)


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


def discover_subjects() -> list[str]:
    if runMNI152:
        return ["MNI152"]

    subjects = [
        entry
        for entry in os.listdir(rootDIR)
        if os.path.isdir(os.path.join(rootDIR, entry))
    ]
    return sorted(subjects)


def select_output_file(directory: str, prefix: str) -> tuple[str, list[str]]:
    candidates = sorted(
        [name for name in os.listdir(directory) if name.startswith(prefix)]
        or os.listdir(directory)
    )
    if not candidates:
        raise FileNotFoundError(f"No output files found in '{directory}'.")

    for candidate in candidates:
        if candidate.endswith((".nii", ".nii.gz")):
            return candidate, candidates

    raise ValueError(f"No NIfTI outputs found in '{directory}': {candidates}")


def get_reference_t1_path(fnamehead: str, subject: str, subject_dir: str) -> str:
    if runMNI152:
        return os.path.join(os.path.dirname(fnamehead), "T1.nii.gz")
    return os.path.join(subject_dir, f"{subject}_T1w.nii")


def process_subject(subject_entry: str) -> float | None:
    subject_source = subject_entry
    subject = subject_entry
    subject_start = time.time()
    log_event("subject_start", subject=subject_source)

    if runMNI152:
        subject = "MNI152"
        sandbox_dir = rootDIR.split("Jake_Data")[0]
        fnamehead = os.path.join(
            sandbox_dir, "simnibs4_exmaples", "m2m_MNI152", "MNI152.msh"
        )
        output_root = os.path.join(rootDIR, subject, "anat", "SimNIBS")
        subject_dir = os.path.join(rootDIR, subject, "anat")
    else:
        fnamehead = os.path.join(rootDIR, subject, "anat", f"m2m_{subject}", f"{subject}.msh")
        output_root = os.path.join(rootDIR, subject, "anat", "SimNIBS")
        subject_dir = os.path.join(rootDIR, subject, "anat")

    print(f"[INFO] Starting TI pipeline for {subject_source} (using '{subject}' resources).")
    log_file_info("head_mesh", fnamehead)
    log_file_info("t1", os.path.join(subject_dir, f"{subject}_T1w.nii"))
    log_file_info("t2", os.path.join(subject_dir, f"{subject}_T2w.nii"))

    if meshPresent:
        if not Path(fnamehead).exists():
            raise FileNotFoundError(f"Mesh marked present, but head mesh is missing: {fnamehead}")
        print(f"[INFO] ({subject_source}) Mesh present, skipping meshing step.")
    else:
        cmd = [
            "charm",
            subject,
            os.path.join(subject_dir, f"{subject}_T1w.nii"),
            os.path.join(subject_dir, f"{subject}_T2w.nii"),
            "--forcerun",
            "--forceqform",
        ]

        try:
            run_cmd(cmd, cwd=str(subject_dir), label="charm_init")
        except Exception as exc:
            log_event("error", stage="charm_init", subject=subject, error=str(exc))
            return None

        custom_seg_map_path = os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks.nii")
        charm_seg_map_path = os.path.join(
            subject_dir,
            f"m2m_sub-{subject.split('-')[-1].upper()}",
            "label_prep",
            "tissue_labeling_upsampled.nii.gz",
        )
        log_file_info("custom_seg_map", custom_seg_map_path)
        log_file_info("charm_seg_map", charm_seg_map_path)

        custom_seg_map = nib.load(custom_seg_map_path)
        charm_seg_map = nib.load(charm_seg_map_path)

        def to_int_img(img: nib.spatialimages.SpatialImage, like: nib.spatialimages.SpatialImage):
            data = img.get_fdata(dtype=np.float32)
            if not np.allclose(data, np.round(data)):
                print("[WARN] Segmentation contains non-integer values; rounding to nearest integers.")
            data = np.rint(data).astype(np.int16)
            return nib.Nifti1Image(data, like.affine, like.header)

        same_shape = custom_seg_map.shape == charm_seg_map.shape
        same_affine = np.allclose(custom_seg_map.affine, charm_seg_map.affine, atol=1e-5)

        if same_shape and same_affine:
            resampled = to_int_img(charm_seg_map, custom_seg_map)
        else:
            print("[INFO] Resampling CHARM segmentation to custom label grid (nearest-neighbor).")
            src_img_nn = nib.Nifti1Image(
                np.rint(charm_seg_map.get_fdata()).astype(np.int16),
                charm_seg_map.affine,
                charm_seg_map.header,
            )
            resampled = resample_from_to(src_img_nn, custom_seg_map, order=0)

        img_info("MAN", custom_seg_map)
        img_info("CHA", resampled)

        merged_img, debug = merge_segmentation_maps(
            custom_seg_map,
            resampled,
            manual_skin_id=5,
            dilate_envelope_voxels=1,
            background_label=0,
            output_path=os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_clipped.nii"),
            save_envelope_path=os.path.join(subject_dir, "skin_mask.nii.gz"),
        )
        _ = debug

        merged_seg_img_path = os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_merged.nii")
        nib.save(merged_img, merged_seg_img_path)
        atomic_replace(merged_seg_img_path, charm_seg_map_path, force_int=True, int_dtype="uint16")

        remesh_cmd = [
            "charm",
            subject,
            "--mesh",
        ]

        try:
            run_cmd(remesh_cmd, cwd=str(subject_dir), label="charm_remesh")
        except Exception as exc:
            log_event("error", stage="charm_remesh", subject=subject, error=str(exc))

    S = sim_struct.SESSION()
    S.fnamehead = fnamehead
    S.pathfem = os.path.join(output_root, "Output", subject)
    os.makedirs(S.pathfem, exist_ok=True)
    format_output_dir(S.pathfem)
    S.element_size = 0.1
    S.map_to_vol = True

    tdcs1 = S.add_tdcslist()
    custom_conductivities = {
        "WM": 0.126,
        "GM": 0.276,
        "CSF": 1.65,
        "Skull": 0.01,
        "Scalp": 0.465,
        "Eye": 0.5,
        "Muscle": 0.16,
        "Saline": electrode_conductivity,
    }

    for conductivity in tdcs1.cond:
        if conductivity.name in custom_conductivities:
            conductivity.value = float(custom_conductivities[conductivity.name])
            print(f"[COND] {conductivity.name} set to {conductivity.value} S/m")

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

    print(f"Running SimNIBS for TI brain-only mesh... ({subject_source})")
    log_event("simnibs_start", subject=subject_source)
    sim.run_simnibs(S)
    log_event("simnibs_done", subject=subject_source)

    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_1_scalar.msh"))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_2_scalar.msh"))

    tags_keep = np.hstack((
        np.arange(0, 499),
        np.arange(1000, 1499),
    ))

    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    E1_vec = m1.field["E"]
    E2_vec = m2.field["E"]
    TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TImax, "TImax")

    out_path = os.path.join(S.pathfem, "TI.msh")
    mesh_io.write_msh(mout, out_path)
    log_file_info("ti_mesh", out_path)
    print(f"Saved gray+white TI mesh to: {out_path}")

    volume_masks_path = os.path.join(S.pathfem, "Volume_Maks")
    volume_base_path = os.path.join(S.pathfem, "Volume_Base")
    volume_labels_path = os.path.join(S.pathfem, "Volume_Labels")
    os.makedirs(volume_masks_path, exist_ok=True)
    os.makedirs(volume_base_path, exist_ok=True)
    os.makedirs(volume_labels_path, exist_ok=True)

    labels_path = os.path.join(volume_labels_path, "TI_Volumetric_Labels")
    masks_path = os.path.join(volume_masks_path, "TI_Volumetric_Masks")
    ti_volume_prefix = os.path.join(volume_base_path, "TI_Volumetric_Base")
    reference_t1_path = get_reference_t1_path(fnamehead, subject, subject_dir)

    print("Exporting volumetric meshes...")
    try:
        run_cmd(
            ["msh2nii", out_path, reference_t1_path, labels_path, "--create_label"],
            label="msh2nii_labels",
        )
    except Exception as exc:
        log_event("error", stage="msh2nii_labels", subject=subject, error=str(exc))

    try:
        run_cmd(
            ["msh2nii", out_path, reference_t1_path, masks_path, "--create_masks"],
            label="msh2nii_masks",
        )
    except Exception as exc:
        log_event("error", stage="msh2nii_masks", subject=subject, error=str(exc))

    try:
        run_cmd(
            ["msh2nii", out_path, reference_t1_path, ti_volume_prefix],
            label="msh2nii_volume",
        )
    except Exception as exc:
        log_event("error", stage="msh2nii_volume", subject=subject, error=str(exc))

    label_file_name, label_candidates = select_output_file(volume_labels_path, "TI_Volumetric_")
    ti_volume_file_name, volume_candidates = select_output_file(volume_base_path, "TI_Volumetric_")
    log_event(
        "volume_selection",
        label_file=label_file_name,
        ti_volume_file=ti_volume_file_name,
        label_candidates=label_candidates,
        volume_candidates=volume_candidates,
    )

    label_file_path = os.path.join(volume_labels_path, label_file_name)
    ti_volume_path = os.path.join(volume_base_path, ti_volume_file_name)
    log_file_info("label_volume", label_file_path)
    log_file_info("ti_volume", ti_volume_path)

    label_img = nib.load(label_file_path)
    ti_img = nib.load(ti_volume_path)

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

    labels = np.rint(np.asarray(label_img.dataobj)).astype(np.int32, copy=False)
    GM_LABELS = {2}
    WM_LABELS = {1}
    brain_mask = np.isin(labels, list(GM_LABELS | WM_LABELS))

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

    masked_output_path = os.path.join(output_root, "ti_brain_only.nii.gz")
    masked_img = nib.Nifti1Image(masked, ti_img.affine, ti_img.header)
    masked_img.header.set_data_dtype(np.float32)
    nib.save(masked_img, masked_output_path)
    log_file_info("ti_brain_only", masked_output_path)

    elapsed = time.time() - subject_start
    print(f"[INFO] Completed TI pipeline for {subject_source} in {elapsed:.2f} seconds.")
    log_event("subject_done", subject=subject_source, elapsed_sec=elapsed)
    return elapsed


def run_subjects_sequentially(subjects: list[str] | None = None) -> dict[str, float]:
    subject_list = subjects or discover_subjects()
    if not subject_list:
        print(f"[WARN] No subjects found in root directory '{rootDIR}'; exiting.")
        return {}

    print(f"[INFO] Launching TI pipeline on {len(subject_list)} subject(s) with 1 worker.")
    subject_durations: dict[str, float] = {}

    for subject in subject_list:
        try:
            duration = process_subject(subject)
            if duration is not None:
                subject_durations[subject] = duration
        except Exception as exc:
            print(f"[ERROR] Failure while processing {subject}: {exc}")

    return subject_durations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal Interference pipeline runner (single-core / local debug)."
    )
    parser.add_argument(
        "--subject",
        help=(
            "Run the pipeline for a single subject ID, e.g. 'sub-CC110056'. "
            "When runMNI152=True the MNI152 template resources are used."
        ),
    )
    args = parser.parse_args()
    start = time.time()

    if args.subject:
        subject_id = args.subject.strip()
        print(f"[INFO] Running TI pipeline for single subject: {subject_id}")
        duration = process_subject(subject_id)
        total_runtime = time.time() - start

        print("Done.")
        print(f"[INFO] Subject {subject_id} runtime: {(duration or 0.0):.2f} seconds.")
        print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")
        return

    subject_durations = run_subjects_sequentially()
    total_runtime = time.time() - start

    print("Done.")
    print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")

    if subject_durations:
        sequential_estimate = sum(subject_durations.values())
        print(f"[INFO] Sum of per-subject runtimes: {sequential_estimate:.2f} seconds")


if __name__ == "__main__":
    main()
