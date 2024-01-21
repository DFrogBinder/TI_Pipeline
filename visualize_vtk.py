import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

def visualize_vtk_file(vtk_file_path):
    """
    Visualize the given .vtk file using Matplotlib.
    
    :param vtk_file_path: Path to the .vtk file.
    """
    # Read the .vtk file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    # Extract the points from the .vtk file
    points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Visualize a .vtk file with Matplotlib")
    parser.add_argument("vtk_file_path", type=str, help="Path to the .vtk file")
    args = parser.parse_args()
    visualize_vtk_file(args.vtk_file_path)
