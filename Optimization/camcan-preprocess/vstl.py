import pyvista as pv
import argparse
import os

def visualize_stl(file_path):
    """Load and visualize an STL file using PyVista."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # Load the STL file
        mesh = pv.read(file_path)

        # Create a plotter and add the mesh
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, opacity=0.7)

        # Set plot properties
        plotter.add_axes()
        plotter.show_grid()
        plotter.view_isometric()

        # Show the mesh
        plotter.show()
    except Exception as e:
        print(f"Error loading STL: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an STL file using PyVista.")
    parser.add_argument("file", type=str, help="Path to the STL file")
    args = parser.parse_args()

    visualize_stl(args.file)

