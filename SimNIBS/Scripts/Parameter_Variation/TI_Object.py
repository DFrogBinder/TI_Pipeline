# -*- coding: utf-8 -*-
"""
 example script that runs two simnibs tDCS simulations
 and calculates maximal amplitude of the TI envelope from the E-fields
 
 Created on Thu Jun 23 17:41:21 2022

@author: axthi
"""

from copy import deepcopy
import os
import numpy as np  
import simnibs as sim
# from simnibs import sim_struct, run_simnibs, mesh_io
from simnibs.utils import TI_utils as TI


"""
     set up and run simulations for the two electrode pairs
"""
class TI_Simulator:
    def __init__(self,mesh_path:str,output_path:str, volume_map:bool=True) -> None:
        # specify general parameters
        self.Session = sim.sim_struct.SESSION()
        
        if os.path.isdir(mesh_path):
            self.Session.subpath = mesh_path
        elif os.path.isfile(mesh_path):
            self.Session.fnamehead = mesh_path
        else:
            raise ValueError(f"The provided path '{mesh_path}' does not exist or is not a regular file or directory.")
        
        self.Session.pathfem = output_path
        
        # Allows for volumetric mesh generation
        self.Session.map_to_vol = volume_map
        
        # S.fnamehead = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/m2m_CustomSphere/mysphere.msh'
        # S.subpath = '/home/cogitatorprime/sandbox/SimNIBS/simnibs4_examples/m2m_ernie'  # m2m-folder of the subject
        # S.pathfem = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs'  # Directory for the simulation

        
    def format_output_dir(self,directory_path:str ) -> None: 
        # Check if the provided path is a directory
        if not os.path.isdir(directory_path):
            print(f"The provided path {directory_path} is not a directory.")
            return
        
        # Get the list of files in the directory
        files_in_directory = os.listdir(directory_path)
        
        # Filter out only files (not directories)
        files_to_delete = [file for file in files_in_directory if os.path.isfile(os.path.join(directory_path, file))]
        
        if not files_to_delete:
            print("No files found in the directory.")
            return
        
        # Delete each file in the directory
        for file in files_to_delete:
            file_path = os.path.join(directory_path, file)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        
        print("All files deleted.")

    def SimulationSetup(self,electrode_shape:str='ellipse',electrode_size:list=[30,30],
                        electrode_thickness:float=2, electrode_possition:list=['FT8','TP8','FT7','TP7'],
                        tissue_cond:float=2):
        # specify first electrode pair
        self.tdcs = self.Session.add_tdcslist()
        self.tdcs.currents = [0.002, -0.002]  # Current flow though each channel (A)
        self.tdcs.cond[16].value = tissue_cond # [S/m]
        self.tdcs.cond[16].name = 'custom_tissue'
        
        self.electrode = self.tdcs.add_electrode()
        self.electrode.channelnr = 1
        self.electrode.centre = electrode_possition[0]  
        self.electrode.shape = electrode_shape
        self.electrode.dimensions = electrode_size  # diameter in [mm]
        self.electrode.thickness = electrode_thickness # 2 mm thickness

        self.electrode = self.tdcs.add_electrode()
        self.electrode.channelnr = 2
        self.electrode.centre = electrode_possition[1]
        self.electrode.shape = electrode_shape
        self.electrode.dimensions = electrode_size
        self.electrode.thickness = electrode_thickness

        

        # specify second electrode pair
        tdcs = self.Session.add_tdcslist(deepcopy(self.tdcs))
        tdcs.electrode[0].centre = electrode_possition[2]
        tdcs.electrode[1].centre = electrode_possition[4]

    def RunSimulation(self):
        sim.run_simnibs(self.Session)
        
    def PostProcessing(self):
        """
            generate the TI field from the simulation results
        """
        m1 = sim.read_msh(os.path.join(self.Session.pathfem, 'sphere_TDCS_1_scalar.msh'))
        m2 = sim.read_msh(os.path.join(self.Session.pathfem, 'sphere_TDCS_2_scalar.msh'))

        print('The two scalar tdcs meshes have been loaded!')

        # remove all tetrahedra and triangles belonging to the electrodes so that
        # the two meshes have same number of elements
        tags_keep = np.hstack((np.arange(1,100), np.arange(1001,1100)))
        self.m1=m1.crop_mesh(tags = tags_keep)
        self.m2=m2.crop_mesh(tags = tags_keep)

        print('Removed electrode geometry from both meshes!')


        # calculate the maximal amplitude of the TI envelope
        self.ef1=self.m1.field['E']
        self.ef2=self.m2.field['E']
        self.TImax = TI.get_maxTI(self.ef1.value, self.ef2.value)

        # make a new mesh for visualization of the field strengths
        # and the amplitude of the TI envelope
        mout = deepcopy(m1)
        mout.elmdata = []
        mout.add_element_field(self.ef1.norm(), 'magnE - pair 1')
        mout.add_element_field(self.ef2.norm(), 'magnE - pair 2')                    
        mout.add_element_field(self.TImax,'TImax')
        sim.write_msh(mout,os.path.join(self.Session.pathfem, 'TI.msh'))
        v = mout.view(
            visible_tags=[1002, 1006],
            visible_fields='TImax',    
            )
        v.write_opt(os.path.join(self.Session.pathfem, 'TI.msh'))
        sim.open_in_gmsh(os.path.join(self.Session.pathfem, 'TI.msh'), True)
            
            

