#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-subject SimNIBS TI pipeline.
Refactored from TI_runner.py for HPC array execution.
"""

import os, time, subprocess, numpy as np, nibabel as nib
from nibabel.processing import resample_from_to
from copy import deepcopy
from simnibs import sim_struct, mesh_io, ElementTags, sim
from simnibs.utils import TI_utils as TI
from functions import merge_segmentation_maps, atomic_replace, format_output_dir


def process_subject(subject: str, rootDIR: str, runMNI152: bool = False, meshPresent: bool = False):
    """Run the full TI simulation for a single subject."""
    start = time.time()
    print(f"[START] Subject: {subject}")

    # ---------- Setup paths ----------
    if runMNI152:
        sandboxDIR = rootDIR.split('Jake_Data')[0]
        fnamehead = os.path.join(sandboxDIR, 'simnibs4_exmaples', 'm2m_MNI152', 'MNI152.msh')
        output_root = os.path.join(rootDIR, subject, 'anat', 'SimNIBS')
        subject_dir = os.path.join(rootDIR, subject, 'anat')
    else:
        fnamehead = os.path.join(rootDIR, subject, 'anat', f'm2m_{subject}', f'{subject}.msh')
        output_root = os.path.join(rootDIR, subject, 'anat', 'SimNIBS')
        subject_dir = os.path.join(rootDIR, subject, 'anat')

    os.makedirs(output_root, exist_ok=True)

    # ---------- Meshing ----------
    if meshPresent:
        print("[INFO] Mesh present, skipping meshing step.")
    else:
        cmd = [
            "charm",
            subject,
            os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
            os.path.join(subject_dir, f"{subject}_T2w.nii.gz"),
            "--forcerun",
        ]
        try:
            subprocess.run(cmd, cwd=str(subject_dir), check=True)
        except Exception as e:
            print(f"[ERROR] CHARM meshing failed for {subject}: {e}")
            return

        # --- Merge segmentation maps ---
        custom_seg_map = nib.load(os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks.nii"))
        charm_seg_map  = nib.load(os.path.join(subject_dir, f"m2m_{subject}", "label_prep", "tissue_labeling_upsampled.nii.gz"))

        same_shape = custom_seg_map.shape == charm_seg_map.shape
        same_affine = np.allclose(custom_seg_map.affine, charm_seg_map.affine, atol=1e-5)
        if not (same_shape and same_affine):
            print("[INFO] Resampling CHARM segmentation to custom label grid (nearest-neighbor).")
            src_img_nn = nib.Nifti1Image(
                np.rint(charm_seg_map.get_fdata()).astype(np.int16),
                charm_seg_map.affine,
                charm_seg_map.header,
            )
            resampled = resample_from_to(src_img_nn, custom_seg_map, order=0)
        else:
            resampled = charm_seg_map

        merged_img, _ = merge_segmentation_maps(
            custom_seg_map, resampled,
            manual_skin_id=5, dilate_envelope_voxels=1, background_label=0,
            output_path=os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_clipped.nii"),
            save_envelope_path=os.path.join(subject_dir, "skin_mask.nii.gz")
        )
        merged_seg_img_path = os.path.join(subject_dir, f"{subject}_T1w_ras_1mm_T1andT2_masks_merged.nii")
        nib.save(merged_img, merged_seg_img_path)

        atomic_replace(merged_seg_img_path,
                       os.path.join(subject_dir, f"m2m_{subject}", "label_prep", "tissue_labeling_upsampled.nii.gz"),
                       force_int=True, int_dtype="uint16")

        remesh_cmd = [
            "charm",
            subject,
            os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
            os.path.join(subject_dir, f"{subject}_T2w.nii.gz"),
            "--mesh",
        ]
        try:
            subprocess.run(remesh_cmd, cwd=str(subject_dir), check=True)
        except Exception as e:
            print(f"[ERROR] Remeshing failed for {subject}: {e}")
            return

    # ---------- Simulation ----------
    electrode_size = [10, 1]
    electrode_shape = "ellipse"
    electrode_conductivity = 0.85
    montage_right = ('Fp2', 2, 'P8', -2)
    montage_left  = ('T7', 2, 'P7', -2)

    S = sim_struct.SESSION()
    S.fnamehead = fnamehead
    S.pathfem = os.path.join(output_root, 'Output', subject)
    os.makedirs(S.pathfem, exist_ok=True)
    format_output_dir(S.pathfem)
    S.element_size = 0.1
    S.map_to_vol = True

    tdcs1 = S.add_tdcslist()
    tdcs1.cond[2].value = electrode_conductivity
    tdcs1.currents = [montage_right[1], montage_right[3]]

    el1 = tdcs1.add_electrode()
    el1.channelnr = 1
    el1.centre = montage_right[0]
    el1.shape = electrode_shape
    el1.dimensions = [electrode_size[0]*2]*2
    el1.thickness = electrode_size[1]

    el2 = tdcs1.add_electrode()
    el2.channelnr = 2
    el2.centre = montage_right[2]
    el2.shape = electrode_shape
    el2.dimensions = [electrode_size[0]*2]*2
    el2.thickness = electrode_size[1]

    tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
    tdcs2.electrode[0].centre = montage_left[0]
    tdcs2.electrode[1].centre = montage_left[2]

    print("[RUN] SimNIBS...")
    sim.run_simnibs(S)

    # ---------- Post-processing ----------
    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_1_scalar.msh"))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f"{subject}_TDCS_2_scalar.msh"))
    tags_keep = np.hstack((np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
                           np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1)))
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)
    TImax = TI.get_maxTI(m1.field["E"].value, m2.field["E"].value)
    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(TImax, "TImax")
    out_path = os.path.join(S.pathfem, "TI.msh")
    mesh_io.write_msh(mout, out_path)
    print(f"[OK] Saved TI mesh to {out_path}")

    end = time.time()
    print(f"[DONE] {subject} | elapsed = {end - start:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run TI simulation for one subject")
    p.add_argument("--subject", required=True)
    p.add_argument("--rootDIR", required=True)
    p.add_argument("--runMNI152", action="store_true")
    p.add_argument("--meshPresent", action="store_true")
    args = p.parse_args()

    process_subject(args.subject, args.rootDIR, args.runMNI152, args.meshPresent)

