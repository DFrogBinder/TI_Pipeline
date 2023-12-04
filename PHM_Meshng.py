import importlib
import os
import gc
import sys
import yaml
LibPath='/Users/boyanivanov/sandbox/TI_Pipeline/tTIS/Code/Core/'
sys.path.append(os.path.realpath(LibPath))
from FEM import Solver as slv

with open(os.path.realpath('tTIS/Code/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

sys.path.append(os.path.realpath(settings['SfePy']['lib_path']))

models = ['103414', '105014', '105115', '110411', '111716', '113619', '117122', '163129', '196750']

for model in models:
    settings['SfePy']['real_brain']['mesh_file'] = 'model path prefix' + model + 'model suffix'
    solve = slv.Solver(settings, 'SfePy', '10-20')
    solve.load_mesh('real_brain')
    solve.define_field_variable('potential_base', 'voltage')
    solve.define_field_variable('potential_df', 'voltage')
    solve.define_essential_boundary('Base_VCC', 26, 'potential_base', 150.0)
    solve.define_essential_boundary('Base_GND', 16, 'potential_base', -150.0)
    solve.define_essential_boundary('DF_VCC', 22, 'potential_df', 150.0)
    solve.define_essential_boundary('DF_GND', 12, 'potential_df', -150.0)
    solve.solver_setup(600, 1e-10, 1e-8, verbose=True)
    state = solve.run_solver(save_results=True, output_dir='output directory path',
    output_file_name='model prefix' + model + 'model suffix')
    del state
    solve.clear_all()
    gc.collect()