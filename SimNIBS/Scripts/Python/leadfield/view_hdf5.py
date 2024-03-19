import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def print_structure(hdf5_file):
    def recursive_list(group, prefix=''):
        for key in group.keys():
            item = group[key]
            path = f'{prefix}/{key}'
            print(path)
            if isinstance(item, h5py.Dataset):
                print(f' - Dataset shape: {item.shape}, Dataset dtype: {item.dtype}')
            elif isinstance(item, h5py.Group):
                recursive_list(item, path)
    print('+'*60)            
    recursive_list(hdf5_file)
    print('+'*60)
    
def visualize_leadfields(hdf5_path):
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Adapt these paths to the structure of your specific file
        # Assuming 'leadfields' is the dataset you want to visualize
        
        print_structure(f)
        leadfields = f['/mesh_leadfield/leadfields/tdcs_leadfield'][()]
        
        # Assuming the leadfields are stored in a 3D space (for X, Y, Z)
        # Example: leadfields.shape might be (num_positions, 3)
        # If your data is structured differently, you'll need to adjust the code
        
        # Simple visualization of the first 3D vector in the leadfields
        # This part heavily depends on your specific data structure
        x = leadfields[:,:, 0]  # X coordinates
        y = leadfields[:,:, 1]  # Y coordinates
        z = leadfields[:,:, 2]  # Z coordinates

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.quiver(0, 0, 0, x, y, z)  # Plot vectors from origin to points (for simple visualization)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Leadfields Visualization')
        plt.show()

# Replace 'your_simnibs_output.hdf5' with the path to your SimNIBS HDF5 file
visualize_leadfields('/home/cogitatorprime/sandbox/SimNIBS/Scripts/Python/leadfield/leadfield_output/ernie_leadfield_EEG10-10_UI_Jurak_2007.hdf5')

