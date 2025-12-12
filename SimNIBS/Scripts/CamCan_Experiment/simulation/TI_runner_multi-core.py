#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import argparse
import concurrent.futures
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
    merge_segmentation_maps,
    atomic_replace,
    img_info,
)
import time



#? Set appropriate flags
meshPresent = False
runMNI152 = False
rootDIR = '/mnt/parscratch/users/cop23bi/full-ti-dataset'


def process_subject(subject_entry):
    """Run the TI pipeline for a single subject entry."""
    subject_source = subject_entry
    subject = subject_entry
    subject_start = time.time()

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

    # region Meshing
    if meshPresent:
        print(f"[INFO] ({subject_source}) Mesh present, skipping meshing step.")
    else:
        cmd = [
            "charm",
            subject,  # SUBJECT_ID must be first
            os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
            os.path.join(subject_dir, f"{subject}_T2w.nii.gz"),
            "--forcerun",
	    "--forceqform"
            ]

        try:
            subprocess.run(cmd, cwd=str(subject_dir), check=True)
        except Exception as e:
            print(f"[ERROR] Error creating initial head model for {subject}: {e}")
            return

        # Load images
        custom_seg_map_path = os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks.nii")
        custom_seg_map = nib.load(custom_seg_map_path)

        charm_seg_map_path = os.path.join(subject_dir, f"m2m_sub-{subject.split('-')[-1].upper()}", 'label_prep', 'tissue_labeling_upsampled.nii.gz')
        charm_seg_map = nib.load(charm_seg_map_path)

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

        #? Low to high resampling is not recommended
        # if not (same_shape and same_affine):

        #     print("[INFO] Resampling custom segmentation to CHARM label grid (nearest-neighbor).")
        #     # order=0 enforces nearest-neighbor to preserve labels
        #     # src_img_nn = nib.Nifti1Image(
        #     #     np.rint(custom_seg_map.get_fdata()).astype(np.int16), custom_seg_map.affine, custom_seg_map.header
        #     # )
        #     resampled = resample_from_to(custom_seg_map, charm_seg_map, order=0)
        #     # rsmpl_custom_seg_map = to_int_img(resampled, charm_seg_map)

        # else:
        #     rsmpl_custom_seg_map = to_int_img(custom_seg_map, charm_seg_map)

        #region Re-mesh
        # merged_img, debug = merge_segmentation_maps(resampled, charm_seg_map,
        #     manual_skin_id=5,          # scalp ID in custom segmentation
        #     dilate_envelope_voxels=1,                  # dilate CHARM envelope by this many voxels
        #     background_label=0,              # background ID in custom segmentation
        #     output_path=os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_clipped.nii"),
        #     save_envelope_path=os.path.join(subject_dir,"skin_mask.nii.gz"))


        #? High to low resampling
        if not (same_shape and same_affine):

            print("[INFO] Resampling CHARM segmentation to custom label grid (nearest-neighbor).")
            # order=0 enforces nearest-neighbor to preserve labels
            src_img_nn = nib.Nifti1Image(
                np.rint(charm_seg_map.get_fdata()).astype(np.int16), charm_seg_map.affine, charm_seg_map.header
            )
            resampled = resample_from_to(src_img_nn, custom_seg_map, order=0)

        #region Re-mesh
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
            #os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
            #os.path.join(subject_dir, f"{subject}_T2w.nii.gz"),
            "--mesh"
        ]

        try:
            subprocess.run(remesh_cmd, cwd=str(subject_dir), check=True)
        except Exception as e:
            print(f"Error remeshing head model for {subject}: {e}")


    electrode_size        = [10, 1]       # [radius_mm, thickness_mm]
    electrode_shape       = 'ellipse'
    electrode_conductivity = 0.85

    # Hippocampus montage
    montage_right = ('Fp2', 2, 'P8', -2)
    montage_left  = ('T7', 2, 'P7',  -2)

    # M1 montage
    # montage_right = ('C1', 1.34, 'Cz', -1.34)
    # montage_left  = ('C3', 2.66, 'CP5',  -2.66)

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

    # ———— DEFINE SECOND TDCS MONTAGE ————
    tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
    tdcs2.electrode[0].centre        = montage_left[0]
    tdcs2.electrode[1].centre        = montage_left[2]
    tdcs2.electrode[0].mesh_element_size = 0.1
    tdcs2.electrode[1].mesh_element_size = 0.1

    # ———— RUN SIMULATION ————
    print(f"Running SimNIBS for TI brain-only mesh… ({subject_source})")
    sim.run_simnibs(S)

    # ———— POST-PROCESS ————

    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_1_scalar.msh'))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_2_scalar.msh'))

    # Define tissue tags (replace with actual IDs from your head model)
    gray_tags  = [1002]   # e.g. cortical gray matter tag
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
            subprocess.run(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, labels_path,"--create_label"])
        else:
            subprocess.run(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii'), labels_path,"--create_label"])
    except Exception as e:
        print(f"Error creating label meshes: {e}")

    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            subprocess.run(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, masks_path,"--create_masks"])
        else:
            subprocess.run(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii'), masks_path,"--create_masks"])
    except Exception as e:
        print(f"Error creating mask meshes: {e}")

    try:
        if runMNI152:
            t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
            subprocess.run(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), t1_path, ti_volume_path])
        else:
            subprocess.run(["msh2nii", os.path.join(output_root,'Output',subject,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii'), ti_volume_path])
    except Exception as e:
        print(f"Error creating volumetric mesh: {e}")
    #endregion

    #region Post-process
    # Loads the label file
    label_file_path = os.listdir(volume_labels_path)[0]
    ti_volume_path = os.listdir(volume_base_path)[0]



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

    ti_data = ti_img.get_fdata(dtype=np.float32)
    masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)

    # --- Save outputs ---
    masked_img = nib.Nifti1Image(masked, label_img.affine, label_img.header)
    masked_img.header.set_data_dtype(np.float32)
    nib.save(masked_img, os.path.join(output_root,"ti_brain_only.nii.gz"))

    #endregion

    elapsed = time.time() - subject_start
    print(f"[INFO] Completed TI pipeline for {subject_source} in {elapsed:.2f} seconds.")
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

    args = parser.parse_args()
    start = time.time()

    if args.subject:
        # ---------- Single-subject (Slurm array) mode ----------
        subject_id = args.subject.strip()
        print(f"[INFO] Running TI pipeline for single subject: {subject_id}")
        duration = process_subject(subject_id)
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
