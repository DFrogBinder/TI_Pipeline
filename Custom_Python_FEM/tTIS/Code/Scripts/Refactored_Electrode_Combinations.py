from __future__ import absolute_import
import os
import gc
import sys
import yaml
import numpy as np
import pyvista as pv

def main(settings_file, meshf, model, csv_save_dir):
    with open(os.path.realpath(settings_file)) as stream:
        settings = yaml.safe_load(stream)

    if os.name == 'nt':
        extra_path = '_windows'
    else:
        extra_path = ''

    sys.path.append(os.path.realpath(settings['SfePy']['lib_path' + extra_path]))

    import FEM.Solver as slv

    settings['SfePy'][model]['mesh_file' + extra_path] = meshf
    model_id = os.path.basename(meshf).split('.')[0].split('_')[-1]

    ## Read the mesh with pyvista to get the area ids and AAL regions
    msh = pv.UnstructuredGrid(meshf)
    brain_regions_mask = np.isin(msh['cell_scalars'], [4]) # Refers to region ID in the setting.yml file

    cell_ids_brain = msh['cell_scalars'][brain_regions_mask]
    aal_regions = msh['AAL_regions'][brain_regions_mask]
    cell_volumes_brain = msh.compute_cell_sizes().cell_arrays['Volume'][brain_regions_mask]

    region_volumes_brain = {}
    for region in np.unique(aal_regions):
        roi = np.where(aal_regions == region)[0]
        region_volumes_brain[int(region)] = np.sum(cell_volumes_brain[roi])
    del msh
    region_volumes_brain = np.array(region_volumes_brain)

    electrodes = settings['SfePy']['electrodes']['10-10-mod']
    e_field_values_brain = []

    solve = slv.Solver(settings, 'SfePy', 'sphere')
    solve.load_mesh(model)

    for electrode in electrodes.items():
        if electrode[0] == 'P9':
            continue
        solve.essential_boundaries.clear()
        solve.fields.clear()
        solve.field_variables.clear()

        solve.define_field_variable('potential', 'voltage', out_of_range_assign_region='Skin', out_of_range_group_threshold=71)

        solve.define_essential_boundary(electrode[0], electrode[1]['id'], 'potential', current=1)
        solve.define_essential_boundary('P9', 71, 'potential', current=-1)

        solve.solver_setup(600, 1e-12, 5e-12, verbose=True)
        solution = solve.run_solver(save_results=False, post_process_calculation=True)

        e_field_base = solution['e_field_(potential)'].data[:, 0, :, 0]
        if isinstance(e_field_values_brain, list):
            e_field_values_brain = e_field_base[brain_regions_mask]
        else:
            e_field_values_brain = np.append(e_field_values_brain, e_field_base[brain_regions_mask], axis=0)

        del solution
        gc.collect()

    del solve
    gc.collect

    np.savez_compressed(os.path.join(csv_save_dir, model_id + '_fields_brain'), e_field=e_field_values_brain.reshape((61, -1, 3)), cell_ids=cell_ids_brain, aal_regions=aal_regions, volumes=region_volumes_brain)

settings_file='/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/sim_settings.yml'
meshf='/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Export_Save_Dir/meshed_model_sphere.1.vtk'
model='simple_brain'
csv_save_dir='/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Export_Save_Dir/'
main(settings_file, meshf, model, csv_save_dir)