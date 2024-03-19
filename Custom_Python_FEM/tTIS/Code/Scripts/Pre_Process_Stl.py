import os
import trimesh
import pymeshfix 
import numpy as np

def main(source_directory, save_directory):
    for filename in os.listdir(source_directory):
        if filename.endswith('.stl'):
            sphere = filename.split('_')[-1].split('.')[0]
            pymeshfix.clean_from_file(os.path.join(source_directory,filename),
                                      os.path.join(save_directory,sphere + '_fixed.stl'))

Source_dir = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Sphere_CAD/Raw'
Output_dir = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Sphere_CAD/Fixed'
main(Source_dir, Output_dir)
