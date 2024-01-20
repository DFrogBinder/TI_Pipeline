'''
---------------------------------------
Meshing of the spherical layered model
---------------------------------------
'''

import os
import sys
import pymesh
import yaml
import inspect
setting_yml_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/sim_settings.yml'
meshingPath = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/Core'
base_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Sphere_CAD/Fixed'

print("Real Path:", os.path.realpath(meshingPath),'\n')
print("Is Path Exist:", os.path.exists(os.path.realpath(meshingPath)),'\n')

sys.path.append(os.path.realpath(meshingPath))
print("Sys Path:", sys.path,'\n')
import Meshing.MeshOperations as MeshOps

def InspectMethods(object):
    for attribute_name in dir(object):
        attribute = getattr(object, attribute_name)
        if inspect.ismethod(attribute) or inspect.isfunction(attribute):
            print(attribute_name)

with open(os.path.realpath(setting_yml_path)) as stream:
        settings = yaml.safe_load(stream)
# Change the angles to your desired configuration
elec_attributes = {
        'electrodes': {
        'B_VCC': {
        'theta': 258.5217,
        'phi': 0
    },
        'B_GND': {
        'theta': 326.2893,
        'phi': 0
    },
        'D_VCC': {
        'theta': 101.4783,
        'phi': 0
    },
        'D_GND': {
        'theta': 33.7107,
        'phi': 0,
    },
 },
    'ids': settings['SfePy']['electrodes']['sphere'],
    'cylinder_radius': 4,
    'cylinder_width': 3,
    'skin_radius': 0,
    'elements': 200,
}
skin_stl = pymesh.load_mesh(os.path.join(base_path, 'skin_fixed.stl'))

# InspectMethods(MeshOps)

meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
meshing.load_surface_meshes(base_path, ['skin_fixed.stl', 'skull_fixed.stl', 'csf_fixed.stl', 'brain_fixed.stl'])
meshing.sphere_model_meshing(os.path.join(base_path, 'meshed_model_sphere.poly'))

print('All models are have been loaded...')
