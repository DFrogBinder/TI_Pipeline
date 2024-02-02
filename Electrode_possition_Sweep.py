import importlib
import os
import gc
import sys
import yaml
from tqdm import tqdm

with open(os.path.realpath('/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)
    
sys.path.append(os.path.realpath(settings['SfePy']['lib_path']))
import FEM.Solver as slv

# settings['SfePy']['simple_brain']['mesh_file'] = 'model path prefix' + model + 'model suffix'
# Change the combinations as you wish and/or add others
electrodes = [[26, 16, 22, 12], [25, 15, 23, 13]]
output_dir_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Simple_Export_Save_Dir'

for electrode in tqdm(electrodes):
    solve = slv.Solver(settings, 'SfePy', 'sphere')
    solve.load_mesh('simple_brain')
    solve.define_field_variable('potential_base', 'voltage')
    solve.define_field_variable('potential_df', 'voltage')
    solve.define_essential_boundary('B_VCC', electrode[0], 'potential_base', 150.0)
    solve.define_essential_boundary('B_GND', electrode[1], 'potential_base', -150.0)
    solve.define_essential_boundary('D_VCC', electrode[2], 'potential_df', 150.0)
    solve.define_essential_boundary('D_GND', electrode[3], 'potential_df', -150.0)
    solve.solver_setup(600, 1e-10, 1e-8, verbose=True)
    state = solve.run_solver(save_results=True, output_dir=output_dir_path,
    output_file_name='fem_model-name_' + "{}-{}-{}-{}".format(*electrode),post_process_calculation=False)
    del state
    solve.clear_all()
    gc.collect()