#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import os
import json
import shutil
import numpy as np
import simnibs as sim
import subprocess
import nibabel as nib

from nibabel.processing import resample_from_to

from copy import deepcopy
from simnibs import sim_struct, mesh_io, ElementTags
from simnibs.utils import TI_utils as TI
from functions import *
import time



#region Parameters
# Mesh and output
start = time.time()

#? Set appropriate flags
meshPresent = True
runMNI152 = True

rootDIR     = '/home/boyan/sandbox/Jake_Data/camcan_test_run/main_data'
# fnamehead    = '/home/boyan/sandbox/Jake_Data/Charm_tests/sub-CC110087_localMap/anat/m2m_sub-CC110087_T1w.nii.gz/sub-CC110087_T1w.nii.gz.msh'
t = os.listdir(rootDIR)[0]
for subject in [t]:
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
# region Meshing
    if meshPresent:
        print("[INFO] Mesh present, skipping meshing step.")
    else:
        cmd = [
            "charm",
            subject,  # SUBJECT_ID must be first
            os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
            os.path.join(subject_dir, f"{subject}_T2w.nii.gz"),
            "--forcerun"
            ]
        
        try:
            subprocess.run(cmd, cwd=str(subject_dir), check=True)
        except Exception as e:
            print(f"[ERROR] Error creating initial head model for {subject}: {e}")
            continue
        
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
            os.path.join(subject_dir, f"{subject}_T1w.nii.gz"),
            os.path.join(subject_dir, f"{subject}_T2w.nii.gz"),
            "--mesh"
        ]
        
        try:
            subprocess.run(remesh_cmd, cwd=str(subject_dir), check=True)
        except Exception as e:
            print(f"Error remeshing head model for {subject}: {e}")

    #endregion
    electrode_shape = 'rect'
    electrode_thickness = 5.0  # mm
    electrode_conductivity = 0.85

    width_values = np.arange(5.0, 10.0 + 1e-6, 2.5)
    height_values = np.arange(30.0, 80.0 + 1e-6, 2.5)

    current_scenarios = [
        {"name": "ratio_1to1", "ratio": "1:1", "pair1_mA": 1.5, "pair2_mA": 1.5},
        {"name": "ratio_1to2", "ratio": "1:2", "pair1_mA": 0.8, "pair2_mA": 1.6},
        {"name": "ratio_1to3", "ratio": "1:3", "pair1_mA": 0.6, "pair2_mA": 1.8},
    ]

    # Hippocampus montage (two electrode pairs)
    montage_right = ('T7', 'C5')
    montage_left  = ('T8', 'C6')

    for width in width_values:
        for height in height_values:
            electrode_dims = [float(width), float(height)]
            width_tag = f"{width:.1f}".replace('.', 'p')
            height_tag = f"{height:.1f}".replace('.', 'p')

            for scenario in current_scenarios:
                scenario_dir = os.path.join(
                    output_root,
                    'Output',
                    subject,
                    scenario['name'],
                    f"w{width_tag}_h{height_tag}"
                )
                if os.path.isdir(scenario_dir):
                    shutil.rmtree(scenario_dir)
                os.makedirs(scenario_dir, exist_ok=True)

                S = sim_struct.SESSION()
                S.open_in_gmsh = False
                S.fnamehead    = fnamehead
                S.pathfem      = scenario_dir
                S.element_size = 0.1
                S.map_to_vol   = True

                tdcs1 = S.add_tdcslist()
                tdcs1.cond[2].value = electrode_conductivity
                pair1_current_A = scenario['pair1_mA'] * 1e-3
                tdcs1.currents = [pair1_current_A, -pair1_current_A]

                el1 = tdcs1.add_electrode()
                el1.channelnr  = 1
                el1.centre     = montage_right[0]
                el1.shape      = electrode_shape
                el1.dimensions = electrode_dims
                el1.thickness  = electrode_thickness

                el2 = tdcs1.add_electrode()
                el2.channelnr  = 2
                el2.centre     = montage_right[1]
                el2.shape      = electrode_shape
                el2.dimensions = electrode_dims
                el2.thickness  = electrode_thickness

                tdcs2 = S.add_tdcslist(deepcopy(tdcs1))
                pair2_current_A = scenario['pair2_mA'] * 1e-3
                tdcs2.currents = [pair2_current_A, -pair2_current_A]
                tdcs2.electrode[0].centre        = montage_left[0]
                tdcs2.electrode[1].centre        = montage_left[1]
                tdcs2.electrode[0].mesh_element_size = 0.1
                tdcs2.electrode[1].mesh_element_size = 0.1

                metadata = {
                    "scenario": scenario['name'],
                    "current_ratio": scenario['ratio'],
                    "pair1_current_mA": scenario['pair1_mA'],
                    "pair2_current_mA": scenario['pair2_mA'],
                    "electrode_dimensions_mm": {
                        "width": electrode_dims[0],
                        "height": electrode_dims[1],
                        "thickness": electrode_thickness
                    },
                    "montage": {
                        "pair1": {"anode": montage_right[0], "cathode": montage_right[1]},
                        "pair2": {"anode": montage_left[0], "cathode": montage_left[1]}
                    }
                }
                with open(os.path.join(scenario_dir, "sweep_metadata.json"), "w") as fh:
                    json.dump(metadata, fh, indent=2)

                print(
                    "[RUN] Scenario {name} | dims {w:.1f}×{h:.1f} mm | "
                    "currents {p1:.1f} mA vs {p2:.1f} mA".format(
                        name=scenario['name'],
                        w=electrode_dims[0],
                        h=electrode_dims[1],
                        p1=scenario['pair1_mA'],
                        p2=scenario['pair2_mA'])
                )

                # ———— RUN SIMULATION ————
                sim.run_simnibs(S)

                # ———— POST-PROCESS ————
                m1 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_1_scalar.msh'))
                m2 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_TDCS_2_scalar.msh'))

                tags_keep = np.hstack((
                    np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
                    np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1)
                ))

                m1 = m1.crop_mesh(tags = tags_keep)
                m2 = m2.crop_mesh(tags = tags_keep)

                E1_vec = m1.field['E']
                E2_vec = m2.field['E']

                TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

                mout = deepcopy(m1)
                mout.elmdata = []
                mout.add_element_field(TImax, 'TImax')

                out_path = os.path.join(S.pathfem, 'TI.msh')
                mesh_io.write_msh(mout, out_path)
                print(f"Saved gray+white TI mesh to: {out_path}")

                #region Saving Results 
                volume_masks_path = os.path.join(S.pathfem,'Volume_Maks')
                volume_base_path = os.path.join(S.pathfem,'Volume_Base')
                volume_labels_path = os.path.join(S.pathfem,'Volume_Labels')
                os.makedirs(volume_masks_path, exist_ok=True)
                os.makedirs(volume_base_path, exist_ok=True)
                os.makedirs(volume_labels_path, exist_ok=True)

                labels_path = os.path.join(volume_labels_path, "TI_Volumetric_Labels")
                masks_path = os.path.join(volume_masks_path, "TI_Volumetric_Masks")
                ti_volume_path = os.path.join(volume_base_path, "TI_Volumetric_Base")

                print('Exporting volumetric meshes...')
                try:
                    if runMNI152:
                        t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
                        subprocess.run(["msh2nii", os.path.join(S.pathfem,'TI.msh'), t1_path, labels_path,"--create_label"])
                    else:
                        subprocess.run(["msh2nii", os.path.join(S.pathfem,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii.gz'), labels_path,"--create_label"])
                except Exception as e:
                    print(f"Error creating label meshes: {e}")

                try:
                    if runMNI152:
                        t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
                        subprocess.run(["msh2nii", os.path.join(S.pathfem,'TI.msh'), t1_path, masks_path,"--create_masks"])
                    else:
                        subprocess.run(["msh2nii", os.path.join(S.pathfem,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii.gz'), masks_path,"--create_masks"])
                except Exception as e:
                    print(f"Error creating mask meshes: {e}")

                try:
                    if runMNI152:
                        t1_path = os.path.join(os.path.dirname(fnamehead),'T1.nii.gz')
                        subprocess.run(["msh2nii", os.path.join(S.pathfem,'TI.msh'), t1_path, ti_volume_path])
                    else:
                        subprocess.run(["msh2nii", os.path.join(S.pathfem,'TI.msh'), os.path.join(f'{subject_dir}',f'{subject}_T1w.nii.gz'), ti_volume_path])
                except Exception as e:
                    print(f"Error creating volumetric mesh: {e}")
                #endregion

                #region Post-process
                label_files = sorted(
                    [f for f in os.listdir(volume_labels_path) if f.endswith(('.nii', '.nii.gz'))]
                )
                if not label_files:
                    raise FileNotFoundError(f"No label NIfTI found in {volume_labels_path}")
                label_file_path = label_files[0]

                ti_files = sorted(
                    [f for f in os.listdir(volume_base_path) if f.endswith(('.nii', '.nii.gz'))]
                )
                if not ti_files:
                    raise FileNotFoundError(f"No TI NIfTI found in {volume_base_path}")
                ti_volume_file = ti_files[0]

                label_img = nib.load(os.path.join(volume_labels_path,label_file_path))
                data = label_img.get_fdata(dtype=np.float32)
                hdr = label_img.header

                print(f'—— Label image info for: {label_file_path} ———')
                print("shape:", data.shape)
                print("voxel sizes (mm):", hdr.get_zooms()[:3])
                print("units:", hdr.get_xyzt_units())
                print(f'———'*19)

                ti_img = nib.load(os.path.join(volume_base_path,ti_volume_file))
                ti_data = ti_img.get_fdata(dtype=np.float32)
                ti_hdr = ti_img.header

                print(f'—— TI image info for: {ti_volume_file} ———')
                print("shape:", ti_data.shape)
                print("voxel sizes (mm):", ti_hdr.get_zooms()[:3])
                print("units:", ti_hdr.get_xyzt_units())
                print(f'———'*19)

                labels = np.asarray(label_img.dataobj).astype(np.int32, copy=False)
                GM_LABELS = {2}
                WM_LABELS = {1}
                brain_mask = np.isin(labels, list(GM_LABELS | WM_LABELS))

                masked = np.where(brain_mask, ti_data, np.nan).astype(np.float32)
                masked_img = nib.Nifti1Image(masked, label_img.affine, label_img.header)
                masked_img.header.set_data_dtype(np.float32)
                nib.save(masked_img, os.path.join(S.pathfem, "ti_brain_only.nii.gz"))
                #endregion
print("Done.")
end = time.time()
print(f"Execution time: {end - start} seconds")
