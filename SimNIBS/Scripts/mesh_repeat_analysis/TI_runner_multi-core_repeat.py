#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import os
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
from simnibs import *
from simnibs.utils import TI_utils as TI

ROOT = Path(__file__).resolve().parents[1] / "CamCan_Experiment"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sim_utils import (
    format_output_dir,
    merge_segmentation_maps,
    atomic_replace,
    img_info,
)
import time

# Set appropriate flags
meshPresent = False
runMNI152 = False
rootDIR = '/mnt/parscratch/users/cop23bi/full-ti-dataset'

REPEAT_PREFIX = "repeat_"
DEFAULT_REPEAT_COUNT = 30


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


def _repeat_tag(prefix: str, index: int, width: int = 3) -> str:
    return f"{prefix}{index:0{width}d}"


def _select_repeat_tags(base_root: str, prefix: str, count: int, *, start_index: int = 1) -> list[str]:
    tags: list[str] = []
    idx = start_index
    while len(tags) < count:
        tag = _repeat_tag(prefix, idx)
        out_dir = os.path.join(base_root, tag)
        if os.path.exists(out_dir):
            print(f"[WARN] Repeat output exists, skipping: {out_dir}")
        else:
            tags.append(tag)
        idx += 1
    return tags


def process_subject(subject_entry, *, repeat_tag: str | None = None):
    """Run the TI pipeline for a single subject entry."""
    subject_source = subject_entry
    subject = subject_entry
    subject_start = time.time()
    log_event("subject_start", subject=subject_source, repeat_tag=repeat_tag)

    if runMNI152:
        # Use MNI152 template mesh | Adjust paths as needed
        subject = 'MNI152'
        sadnboxDIR      = rootDIR.split('Jake_Data')[0]
        fnamehead    = os.path.join(sadnboxDIR,'simnibs4_exmaples','m2m_MNI152','MNI152.msh')

        output_root  = os.path.join(rootDIR, subject, 'anat','SimNIBS')
        subject_dir = os.path.join(rootDIR, subject, 'anat')

    else:
        fnamehead    = os.path.join(rootDIR, subject, 'anat', f'm2m_{subject}', f'{subject}.msh')
        output_root  = os.path.join(rootDIR,subject, 'anat','SimNIBS')
        subject_dir = os.path.join(rootDIR, subject, 'anat')

    if repeat_tag:
        output_root = os.path.join(output_root, repeat_tag)
        os.makedirs(output_root, exist_ok=True)

    print(f"[INFO] Starting TI pipeline for {subject_source} (using '{subject}' resources).")
    log_file_info("t1", os.path.join(subject_dir, f"{subject}_T1w.nii"))
    log_file_info("t2", os.path.join(subject_dir, f"{subject}_T2w.nii"))

    # region Meshing
    if meshPresent:
        print(f"[INFO] ({subject_source}) Mesh present, skipping meshing step.")
    else:
        cmd = [
            "charm",
            subject,  # SUBJECT_ID must be first
            os.path.join(subject_dir, f"{subject}_T1w.nii"),
            os.path.join(subject_dir, f"{subject}_T2w.nii"),
            "--forcerun",
            "--forceqform"
            ]

        try:
            run_cmd(cmd, cwd=str(subject_dir), label="charm_init")
        except Exception as e:
            log_event("error", stage="charm_init", subject=subject, error=str(e))
            return

        # Load images
        custom_seg_map_path = os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks.nii")
        custom_seg_map = nib.load(custom_seg_map_path)

        charm_seg_map_path = os.path.join(subject_dir, f"m2m_sub-{subject.split('-')[-1].upper()}", 'label_prep', 'tissue_labeling_upsampled.nii.gz')
        charm_seg_map = nib.load(charm_seg_map_path)
        log_file_info("custom_seg_map", custom_seg_map_path)
        log_file_info("charm_seg_map", charm_seg_map_path)

        # Ensure integer labels; nibabel exposes floats via get_fdata().
        # We'll round+cast only if dtype isn't int-like.
        def to_int_img(img, like):
            data = img.get_fdata(dtype=np.float32)  # safe access; may be float
            # Detect non-integers
            if not np.allclose(data, np.round(data)):
                print("[WARN] Custom segmentation contains non-integer values; rounding to nearest integers.")
            data = np.rint(data).astype(np.int16)
            return nib.Nifti1Image(data, like.affine, like.header)

        # Resample to reference grid if needed
        same_shape = custom_seg_map.shape == charm_seg_map.shape
        same_affine = np.allclose(custom_seg_map.affine, charm_seg_map.affine, atol=1e-5)

        # High to low resampling
        if not (same_shape and same_affine):

            print("[INFO] Resampling CHARM segmentation to custom label grid (nearest-neighbor).")
            # order=0 enforces nearest-neighbor to preserve labels
            src_img_nn = nib.Nifti1Image(
                np.rint(charm_seg_map.get_fdata()).astype(np.int16), charm_seg_map.affine, charm_seg_map.header
            )
            resampled = resample_from_to(src_img_nn, custom_seg_map, order=0)

        merged_img, debug = merge_segmentation_maps(custom_seg_map, resampled,
            manual_skin_id=5,          # scalp ID in custom segmentation
            dilate_envelope_voxels=1,                  # dilate CHARM envelope by this many voxels
            background_label=0,              # background ID in custom segmentation
            output_path=os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_clipped.nii"),
            save_envelope_path=os.path.join(subject_dir,"skin_mask.nii.gz"))

        merged_seg_img_path = os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_merged.nii")
        nib.save(merged_img, merged_seg_img_path)

        atomic_replace(merged_seg_img_path, charm_seg_map_path, force_int=True, int_dtype="uint16")

        # Re-mesh with charm --mesh from the directory that contains m2m_<subject>
        remesh_cmd = [
            "charm",
            subject,
            "--mesh"
        ]

        try:
            run_cmd(remesh_cmd, cwd=str(subject_dir), label="charm_remesh")
        except Exception as e:
            log_event("error", stage="charm_remesh", subject=subject, error=str(e))

    electrode_size        = [10, 1]       # [radius_mm, thickness_mm]
    electrode_shape       = 'ellipse'
    electrode_conductivity = 0.85

    # Hippocampus montage
    montage_right = ('Fp2', 2, 'P8', -2)
    montage_left  = ('T7', 2, 'P7',  -2)

    # Brain tissue tags (adjust if your labeling differs)
    brain_tags = np.hstack((np.arange(1, 100), np.arange(1001, 1100)))
    # region Simulation
    # ---- SET UP SESSION ----
    S = sim_struct.SESSION()
    S.fnamehead    = fnamehead
    S.pathfem      = os.path.join(output_root, 'Output',subject)
    os.makedirs(S.pathfem, exist_ok=True)
    format_output_dir(S.pathfem)
    S.element_size = 0.1
    S.map_to_vol   = True

    # ---- DEFINE FIRST TDCS MONTAGE ----
    tdcs1 = S.add_tdcslist()
    tdcs1.cond[2].value = electrode_conductivity
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

    # ---- DEFINE SECOND TDCS MONTAGE ----
    tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
    tdcs2.electrode[0].centre        = montage_left[0]
    tdcs2.electrode[1].centre        = montage_left[2]
    tdcs2.electrode[0].mesh_element_size = 0.1
    tdcs2.electrode[1].mesh_element_size = 0.1

    # ---- RUN SIMULATION ----
    print(f"Running SimNIBS for TI brain-only mesh… ({subject_source})")
    log_event("simnibs_start", subject=subject_source, repeat_tag=repeat_tag)
    sim.run_simnibs(S)
    log_event("simnibs_done", subject=subject_source, repeat_tag=repeat_tag)

    # ---- POST-PROCESS ----

    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_1_scalar.msh'))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_2_scalar.msh'))

    # Define tissue tags (replace with actual IDs from your head model)
    gray_tags  = [1002]   # e.g. cortical gray matter tag

    tags_keep = np.hstack((
        np.arange(0, 499),     # 0–498 inclusive
        np.arange(1000, 1499)  # 1000–1498 inclusive
        ))

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
    mout.add_element_field(TImax,       'TImax')

    # Write out the gray+white TI mesh
    out_path = os.path.join(S.pathfem, 'TI.msh')
    mesh_io.write_msh(mout, out_path)
    print(f"Saved gray+white TI mesh to: {out_path}")
    # endregion
    # region Saving Results
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
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii'), labels_path,"--create_label"], label="msh2nii_labels")
    except Exception as e:
        log_event("error", stage="msh2nii_labels", subject=subject, error=str(e))

    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, masks_path,"--create_masks"], label="msh2nii_masks")
        else:
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii'), masks_path,"--create_masks"], label="msh2nii_masks")
    except Exception as e:
        log_event("error", stage="msh2nii_masks", subject=subject, error=str(e))

    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, ti_volume_path], label="msh2nii_volume")
        else:
            run_cmd(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii'), ti_volume_path], label="msh2nii_volume")
    except Exception as e:
        log_event("error", stage="msh2nii_volume", subject=subject, error=str(e))
    # endregion

    # region Post-process
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

    ti_img = nib.load(os.path.join(volume_base_path,ti_volume_path))
    ti_data = ti_img.get_fdata(dtype=np.float32)  # read into RAM as float32
    ti_affine = ti_img.affine
    ti_hdr = ti_img.header

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

    # endregion

    elapsed = time.time() - subject_start
    print(f"[INFO] Completed TI pipeline for {subject_source} in {elapsed:.2f} seconds.")
    log_event("subject_done", subject=subject_source, repeat_tag=repeat_tag, elapsed_sec=elapsed)
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
        "--repeats",
        type=int,
        default=None,
        help=(
            "Repeat the same subject simulation N times without overwriting outputs. "
            "Each repeat writes to SimNIBS/repeat_###. Defaults to 30."
        ),
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=None,
        help=(
            "Run only a specific repeat index (1-based). "
            "Writes to SimNIBS/repeat_### and errors if it already exists."
        ),
    )

    args = parser.parse_args()
    start = time.time()

    if args.subject:
        # ---------- Single-subject (Slurm array) mode ----------
        subject_id = args.subject.strip()
        env_repeat_count = int(os.environ.get("SIMNIBS_REPEAT_COUNT", str(DEFAULT_REPEAT_COUNT)))
        env_repeat_index = os.environ.get("SIMNIBS_REPEAT_INDEX")
        repeat_count = args.repeats if args.repeats is not None else env_repeat_count
        repeat_index = args.repeat_index if args.repeat_index is not None else (
            int(env_repeat_index) if env_repeat_index else None
        )

        if repeat_index is not None and repeat_count != DEFAULT_REPEAT_COUNT:
            raise ValueError("Use either --repeat-index or --repeats (count), not both.")

        if repeat_index is not None:
            repeat_tag = _repeat_tag(REPEAT_PREFIX, repeat_index)
            repeat_root = os.path.join(rootDIR, subject_id, "anat", "SimNIBS", repeat_tag)
            if os.path.exists(repeat_root):
                raise FileExistsError(f"Repeat output already exists: {repeat_root}")
            print(f"[INFO] Running TI pipeline for single subject: {subject_id} ({repeat_tag})")
            duration = process_subject(subject_id, repeat_tag=repeat_tag)
        else:
            base_root = os.path.join(rootDIR, subject_id, "anat", "SimNIBS")
            repeat_tags = _select_repeat_tags(base_root, REPEAT_PREFIX, repeat_count, start_index=1)
            print(
                f"[INFO] Running TI pipeline for single subject: {subject_id} "
                f"({repeat_count} repeats)"
            )
            total_elapsed = 0.0
            for tag in repeat_tags:
                print(f"[INFO] Starting repeat: {tag}")
                total_elapsed += process_subject(subject_id, repeat_tag=tag) or 0.0
            duration = total_elapsed
        total_runtime = time.time() - start

        print("Done.")
        print(f"[INFO] Subject {subject_id} runtime: "
              f"{(duration or 0.0):.2f} seconds.")
        print(f"[INFO] Total execution time: {total_runtime:.2f} seconds.")

    else:
        # ---------- Multi-subject / local mode ----------
        if args.repeats or args.repeat_index:
            raise ValueError("Repeat options are only supported with --subject mode.")
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
