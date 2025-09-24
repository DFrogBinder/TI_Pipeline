#!/home/boyan/SimNIBS-4.5/bin/simnibs_python
# -*- coding: utf-8 -*-
import os
import numpy as np
import simnibs as sim
import subprocess
import nibabel as nib

from nibabel.processing import resample_from_to

from copy import deepcopy
from simnibs import sim_struct, mesh_io, ElementTags
from simnibs.utils import TI_utils as TI
from functions import merge_segmentation_maps, atomic_replace

def close_gmsh_windows():
    """Kill any lingering Gmsh GUI windows."""
    stop_flag = True
    while stop_flag:
        try:
            result = subprocess.run(
                ['bash', '-c', 'xdotool search --name "Gmsh" windowkill']
            )
            if result.returncode == 1:
                stop_flag = False
        except Exception:
            stop_flag = False


def format_output_dir(directory_path: str) -> None:
    """Delete all files in a folder (leave subfolders intact)."""
    if not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        return
    for fname in os.listdir(directory_path):
        fpath = os.path.join(directory_path, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"Deleted {fpath}")
def generate_mesh_from_nii( output_path: str, T1_path: str, T2_path: str = None) -> str:
    try:
        subprocess.run(["charm", T1_path, T2_path, output_path])
        return True
    except Exception as e:
        print(f"Error creating volumetric mesh: {e}")
        return False

#region Parameters
# Mesh and output
rootDIR     = '/home/boyan/sandbox/Jake_Data/Example_data'
fnamehead    = '/home/boyan/sandbox/Jake_Data/Charm_tests/sub-CC110087_localMap/anat/m2m_sub-CC110087_T1w.nii.gz/sub-CC110087_T1w.nii.gz.msh'
SplitFolderPath = fnamehead.rsplit('/', 4)[1]
output_root  = os.path.join('/home/boyan/sandbox/TI_Pipeline/SimNIBS/',SplitFolderPath)

for subject in os.listdir(rootDIR):

    subject_dir = os.path.join(rootDIR, subject, 'anat')
# region Meshing
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
        print(f"Error creating initial head model for {subject}: {e}")
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
    if not (same_shape and same_affine):
        
        print("[INFO] Resampling custom segmentation to CHARM label grid (nearest-neighbor).")
        # order=0 enforces nearest-neighbor to preserve labels
        # src_img_nn = nib.Nifti1Image(
        #     np.rint(custom_seg_map.get_fdata()).astype(np.int16), custom_seg_map.affine, custom_seg_map.header
        # )
        resampled = resample_from_to(custom_seg_map, charm_seg_map, order=0)
        # rsmpl_custom_seg_map = to_int_img(resampled, charm_seg_map)
    
    else:
        rsmpl_custom_seg_map = to_int_img(custom_seg_map, charm_seg_map)
        
    #? High to low resampling
    # if not (same_shape and same_affine):
        
    #     print("[INFO] Resampling CHARM segmentation to custom label grid (nearest-neighbor).")
    #     # order=0 enforces nearest-neighbor to preserve labels
    #     src_img_nn = nib.Nifti1Image(
    #         np.rint(charm_seg_map.get_fdata()).astype(np.int16), charm_seg_map.affine, charm_seg_map.header
    #     )
    #     resampled = resample_from_to(src_img_nn, custom_seg_map, order=0)
    #     rsmpl_custom_seg_map = to_int_img(custom_seg_map, custom_seg_map)
    #     charm_seg_map = to_int_img(resampled, custom_seg_map)
    
    #region Re-mesh
    merged_img, mask_img, dbg = merge_segmentation_maps(
    resampled,
    charm_seg_map,
    manual_skin_id=5,
    charm_skin_ids=(5,),             # add IDs if CHARM splits outer head
    skin_mask_out=os.path.join(subject_dir,"skin_mask.nii.gz"),
    return_mask=True,
    mask_preserve_labels=True,      # set True to keep CHARM's actual IDs in the mask
    return_debug=True,
    )
    
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

    
    electrode_size        = [10, 1]       # [radius_mm, thickness_mm]
    electrode_shape       = 'ellipse'
    electrode_conductivity = 0.85

    montage_right = ('F8', 1.9, 'P4', -1.9)
    montage_left  = ('FT9', 1.9, 'CP5',  -1.9)

    # Brain tissue tags (adjust if your labeling differs)
    brain_tags = np.hstack((np.arange(1, 100), np.arange(1001, 1100)))
    #region Simulation
    # ———— SET UP SESSION ————
    S = sim_struct.SESSION()
    S.fnamehead    = fnamehead
    S.pathfem      = os.path.join(output_root, 'TI_brain_only')
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
    print("Running SimNIBS for TI brain-only mesh…")
    sim.run_simnibs(S)

    # ———— POST-PROCESS ————
    m1 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_T1w.nii.gz_TDCS_1_scalar.msh'))
    m2 = mesh_io.read_msh(os.path.join(S.pathfem, f'{subject}_T1w.nii.gz_TDCS_2_scalar.msh'))

    # Define tissue tags (replace with actual IDs from your head model)
    gray_tags  = [1002]   # e.g. cortical gray matter tag
    white_tags = [1003]   # e.g. subcortical/white matter tag

    tags_keep = np.hstack((np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1), np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1)))

    # Crop to gray + white matter only
    m1=m1.crop_mesh(tags = tags_keep)
    m2=m2.crop_mesh(tags = tags_keep)

    # Extract field vectors on gray+white mesh
    E1_vec = m1.field['E']
    E2_vec = m2.field['E']

    # Compute TI metric
    TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

    # Build output mesh from gray+white region
    mout = deepcopy(m1)
    mout.elmdata = []

    # Add magnitude and TI fields
    mout.add_element_field(E1_vec.norm(), 'magnE - pair 1')
    mout.add_element_field(E2_vec.norm(), 'magnE - pair 2')
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
        subprocess.run(["msh2nii", os.path.join(output_root,'TI_brain_only','TI.msh'), "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz", labels_path,"--create_label"])
    except Exception as e:
        print(f"Error creating label meshes: {e}")

    try:
        subprocess.run(["msh2nii", os.path.join(output_root,'TI_brain_only','TI.msh'), "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz", masks_path,"--create_masks"])
    except Exception as e:
        print(f"Error creating mask meshes: {e}")

    try:
        subprocess.run(["msh2nii", os.path.join(output_root,'TI_brain_only','TI.msh'), "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz", ti_volume_path])
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

    print(f'—— Label image info for: {label_file_path} ———')
    print("shape:", data.shape)
    print("voxel sizes (mm):", hdr.get_zooms()[:3])
    print("units:", hdr.get_xyzt_units())
    print(f'———'*19)

    ti_img = nib.load(os.path.join(volume_base_path,ti_volume_path))
    ti_data = ti_img.get_fdata(dtype=np.float32)  # read into RAM as float32
    ti_affine = ti_img.affine                 
    ti_hdr = ti_img.header

    print(f'—— TI image info for: {ti_volume_path} ———')
    print("shape:", ti_data.shape)
    print("voxel sizes (mm):", ti_hdr.get_zooms()[:3])
    print("units:", ti_hdr.get_xyzt_units())
    print(f'———'*19)

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
    print("Done.")
