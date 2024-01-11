
import os
import yaml
import pymesh
import sys
import scipy.io as sio
import numpy as np
meshingPath = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/Core'
print("Real Path:", os.path.realpath(meshingPath))
print("Is Path Exist:", os.path.exists(os.path.realpath(meshingPath)))

sys.path.append(os.path.realpath(meshingPath))
print("Sys Path:", sys.path)
import Meshing.MeshOperations as MeshOps


def main_function(settings_file, stl_models_path, electrode_attributes_path,save_dir):
    folders = np.sort(next(os.walk(stl_models_path))[1])

    # Electrodes to ommit based on the imported settings file
    electrodes_to_omit=['Nz', 'N2', 'AF10', 'F10', 'FT10', 'T10(M2)', 'TP10', 'PO10', 'I2', 'Iz', 'I1', 'PO9', 'TP9', 'T9(M1)', 'FT9', 'F9', 'AF9', 'N1', 'P10']

    with open(os.path.realpath(settings_file)) as stream:
        settings = yaml.safe_load(stream)

    for folder in folders:
        if folder == 'meshed':
            continue
        elif folder == 'Raw':
            continue
        print("############")
        print("Model " + folder)
        print("############\n")
        standard_electrodes = sio.loadmat(os.path.join(electrode_attributes_path, '10-10_elec_' + folder + '.mat'))
        elec_attributes = {
            'names': [name[0][0] for name in standard_electrodes['electrodeStruct2']['ElectrodeNames']],
            'coordinates': standard_electrodes['electrodeStruct2']['ElectrodePts'],
            'ids': settings['SfePy']['electrodes']['10-10-mod'],
            'width': 4,
            'radius': 4,
            'elements': 200,
        }

        skin_stl = pymesh.load_mesh(os.path.join(stl_models_path, folder, 'skin_fixed.stl'))

        geometry_list = [
                        'skin_fixed.stl', 
                         'skull_fixed.stl', 
                         'csf_fixed.stl', 
                        # 'gm_fixed.stl', 
                        # 'wm_fixed.stl', 
                        # 'cerebellum_fixed.stl', 
                        # 'ventricles_fixed.stl'
                         ]
        
        meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
        meshing.load_surface_meshes(os.path.join(stl_models_path, folder), geometry_list)
        meshing.phm_model_meshing(os.path.join(save_dir, 'meshed_model_10-10_' + folder + '.poly'), electrodes_to_omit=electrodes_to_omit)

    
stl_models_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Sphere_CAD/'
setting_yml_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/sim_settings.yml'
el_attributes_dir = '/home/cogitatorprime/sandbox/TI_Pipeline/mesh2eeg'
save_dir = '/home/cogitatorprime/sandbox/TI_Pipeline'
main_function(setting_yml_path, stl_models_path, el_attributes_dir, save_dir)
