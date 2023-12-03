'''
---------------------------------------
Meshing of the spherical layered model
---------------------------------------
'''

import os
import sys
import pymesh
meshingPath='/Users/boyanivanov/sandbox/TI_Pipeline/tTIS/Code/Core/'
sys.path.append(os.path.realpath(meshingPath))
from Meshing import MeshOperations as MeshOps
import inspect

def InspectMethods(object):
    for attribute_name in dir(object):
        attribute = getattr(object, attribute_name)
        if inspect.ismethod(attribute) or inspect.isfunction(attribute):
            print(attribute_name)

base_path = 'Sphere_CAD/Sphere'
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
        'phi': 0
    },
 },
    'cylinder_radius': 4,
    'cylinder_width': 3,
    'skin_radius': 0,
    'elements': 200,
}
skin_stl = pymesh.load_mesh(os.path.join(base_path, 'spheres_skin.stl'))

# InspectMethods(MeshOps)

meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
meshing.load_surface_meshes(base_path, ['spheres_skin.stl', 'spheres_skull.stl', 'spheres_csf.stl', 'spheres_brain.stl'])
meshing.sphere_model_meshing(os.path.join(base_path, 'meshed_model_sphere.poly'))

print('All models are have been loaded...')
