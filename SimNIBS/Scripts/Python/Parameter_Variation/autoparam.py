# -*- coding: utf-8 -*-
import os
import numpy as np
import simnibs as sim
import itertools
import subprocess

from copy import deepcopy
from datetime import datetime
from simnibs import sim_struct, run_simnibs, mesh_io
from simnibs.utils import TI_utils as TI

def format_output_dir(directory_path: str) -> None:
    if not os.path.isdir(directory_path):
        print(f"The provided path {directory_path} is not a directory.")
        return
    
    files_in_directory = os.listdir(directory_path)
    files_to_delete = [file for file in files_in_directory if os.path.isfile(os.path.join(directory_path, file))]
    
    if not files_to_delete:
        print("No files found in the directory.")
        return
    
    for file in files_to_delete:
        file_path = os.path.join(directory_path, file)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    
    print("All files deleted.")

# Parameters to be tested
electrode_sizes = [1, 1.5, 2]  # cm
electrode_thickness = [1,1.5, 2]
current_intensities = [1, 1.5, 2]  # mA
positions = [
    ('FC4', 'P4', 'FC3', 'P3')
    # ('F3', 'P3', 'F4', 'P4'),
    # ('Fz', 'Pz', 'F8', 'T6')
]

# Define general parameters
fnamehead = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/simnibs4_examples/m2m_MNI152/MNI152.msh'
base_pathfem = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs'  # Directory for the simulation


# Iterate over all combinations of parameters
for size, current, (pos1a, pos1b, pos2a, pos2b) in itertools.product(electrode_sizes, current_intensities, positions):
    # Create session
    S = sim.sim_struct.SESSION()
    S.fnamehead = fnamehead
    S.pathfem = os.path.join(base_pathfem, f'{size}cm_{current}mA_{pos1a}-{pos1b}_{pos2a}-{pos2b}')
    os.makedirs(S.pathfem, exist_ok=True)
    
    format_output_dir(S.pathfem)
    
    S.map_to_vol = True
    
    # Define electrodes and currents
    tdcs = S.add_tdcslist()
    tdcs.currents = [current / 1000, -current / 1000]  # Convert mA to A

    # Define first electrode pair
    electrode1 = tdcs.add_electrode()
    electrode1.channelnr = 1
    electrode1.centre = pos1a
    electrode1.shape = 'ellipse'
    electrode1.dimensions = [size * 10, size * 10]  # Convert cm to mm
    electrode1.thickness = 2

    electrode2 = tdcs.add_electrode()
    electrode2.channelnr = 2
    electrode2.centre = pos1b
    electrode2.shape = 'ellipse'
    electrode2.dimensions = [size * 10, size * 10]  # Convert cm to mm
    electrode2.thickness = 2
    
    # Define second electrode pair
    tdcs2 = S.add_tdcslist(deepcopy(tdcs))
    tdcs2.electrode[0].centre = pos2a
    tdcs2.electrode[1].centre = pos2b
    
    # Run simulation
    sim.run_simnibs(S)
    
    # Post-process results
    m1 = sim.read_msh(os.path.join(S.pathfem, 'MNI152_TDCS_1_scalar.msh'))
    m2 = sim.read_msh(os.path.join(S.pathfem, 'MNI152_TDCS_2_scalar.msh'))

    tags_keep = np.hstack((np.arange(1, 100), np.arange(1001, 1100)))
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    ef1 = m1.field['E']
    ef2 = m2.field['E']
    TImax = TI.get_maxTI(ef1.value, ef2.value)

    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(ef1.norm(), 'magnE - pair 1')
    mout.add_element_field(ef2.norm(), 'magnE - pair 2')
    mout.add_element_field(TImax, 'TImax')
    sim.write_msh(mout, os.path.join(S.pathfem, 'TI.msh'))
    v = mout.view(
        visible_tags=[1002, 1006],
        visible_fields='TImax',
    )
    ti_export_path = os.path.join(S.pathfem, 'TI.msh')
    v.write_opt(ti_export_path)
    subprocess.run(["msh2nii", ti_export_path, "/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/simnibs4_examples/m2m_MNI152/T1.nii.gz", os.path.join(S.pathfem, "MNI152_TDCS_TI")])


    print(f"Simulation completed for size {size} cm, current {current} mA, positions {pos1a}-{pos1b}, {pos2a}-{pos2b}. Results saved in {S.pathfem}")

