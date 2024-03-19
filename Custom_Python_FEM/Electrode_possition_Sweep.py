import importlib
import os
import gc
import sys
import yaml
import logging
from tqdm import tqdm

with open(os.path.realpath('/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Code/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)
    
sys.path.append(os.path.realpath(settings['SfePy']['lib_path']))
import FEM.Solver as slv
import Logger.log_config as log

# Change the combinations as you wish and/or add others
# electrodes = [[26, 16, 22, 12], [25, 15, 23, 13]]
            #   , [12,22,16,26], [12,13,10,11]]
electrodes = [[12,13,10,11]]

# Initialize logging configuration
log.setup_logging()

# Get a logger object for this module
logger = logging.getLogger(__name__)


output_dir_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Simple_Export_Save_Dir'

for electrode in tqdm(electrodes):
    solve = slv.Solver(settings, 'SfePy', 'sphere', logger=logger)
    solve.load_mesh('simple_brain')
    solve.define_field_variable('potential_base', 'voltage')
    solve.define_field_variable('potential_df', 'voltage')
    solve.define_essential_boundary('B_VCC', electrode[0], 'potential_base', current=150.0)
    solve.define_essential_boundary('B_GND', electrode[1], 'potential_base', current=-150.0)
    solve.define_essential_boundary('D_VCC', electrode[2], 'potential_df', current=150.0)
    solve.define_essential_boundary('D_GND', electrode[3], 'potential_df', current=-150.0)
    solve.solver_setup(800, 1e-15, 1e-10, verbose=True)
    state = solve.run_solver(save_results=True, output_dir=output_dir_path,
    output_file_name='fem_model-name_' + "{}-{}-{}-{}".format(*electrode),post_process_calculation=True)
    del state
    solve.clear_all()
    gc.collect()