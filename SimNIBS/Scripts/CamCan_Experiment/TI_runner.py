# -*- coding: utf-8 -*-
import os
import numpy as np
import simnibs as sim
import subprocess

from copy import deepcopy
from simnibs import sim_struct, mesh_io
from simnibs.utils import TI_utils as TI



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


#
# ———— USER PARAMETERS ————
#

# Mesh and output
fnamehead    = '/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/MNI152.msh'
output_root  = '/home/boyan/sandbox/TI_Pipeline/SimNIBS/SingleTI_Output'

electrode_size        = [10, 1]       # [radius_mm, thickness_mm]
electrode_shape       = 'ellipse'
electrode_conductivity = 0.85

montage_right = ('FC3', 0.6, 'FC1', -0.6)
montage_left  = ('FCz', 1.9, 'F2',  -1.9)

# Brain tissue tags (adjust if your labeling differs)
brain_tags = np.hstack((np.arange(1, 100), np.arange(1001, 1100)))

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

m1 = mesh_io.read_msh(os.path.join(S.pathfem, 'MNI152_TDCS_1_scalar.msh'))
m2 = mesh_io.read_msh(os.path.join(S.pathfem, 'MNI152_TDCS_2_scalar.msh'))

# Define tissue tags (replace with actual IDs from your head model)
gray_tags  = [1002]   # e.g. cortical gray matter tag
white_tags = [1003]   # e.g. subcortical/white matter tag

# Crop to gray + white matter only
m1_gw = m1.crop_mesh(tags=gray_tags)
m2_gw = m2.crop_mesh(tags=gray_tags)

# Extract field vectors on gray+white mesh
E1_vec = m1_gw.field['E']
E2_vec = m2_gw.field['E']

# Compute TI metric
TImax = TI.get_maxTI(E1_vec.value, E2_vec.value)

# Build output mesh from gray+white region
mout = deepcopy(m1_gw)
mout.elmdata = []

# Add magnitude and TI fields
mout.add_element_field(E1_vec.norm(), 'magnE - pair 1')
mout.add_element_field(E2_vec.norm(), 'magnE - pair 2')
mout.add_element_field(TImax,       'TImax')

# Write out the gray+white TI mesh
out_path = os.path.join(S.pathfem, 'TI.msh')
mesh_io.write_msh(mout, out_path)
print(f"Saved gray+white TI mesh to: {out_path}")


# volume_masks_path = os.path.join(S.pathfem,'Volume_Maks')
# if not os.path.isdir(volume_masks_path):
#     os.mkdir(volume_masks_path)
    
# volume_base_path = os.path.join(S.pathfem,'Volume_Base')
# if not os.path.isdir(volume_base_path):
#     os.mkdir(volume_base_path)
    
# volume_labels_path = os.path.join(S.pathfem,'Volume_Labels')
# if not os.path.isdir(volume_labels_path):
#     os.mkdir(volume_labels_path)

# # Create the volumetric TI mesh with labels
# try:
#     # Command to search and close all windows containing 'SimNIBS' in their title
#     subprocess.run(["msh2nii", output_root, "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz", os.path.join(volume_labels_path, "TI_Volumetric_Labels"),"--create_label"])
#     subprocess.run(["msh2nii", output_root, "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz", os.path.join(volume_masks_path, "TI_Volumetric_Masks"),"--create_masks"])
#     subprocess.run(["msh2nii", output_root, "/home/boyan/sandbox/simnibs4_exmaples/m2m_MNI152/T1.nii.gz", os.path.join(volume_base_path, "TI_Volumetric_Base")])
# except Exception as e:
#     print(f"Error creating volumetric mesh: {e}")

print("Done.")
