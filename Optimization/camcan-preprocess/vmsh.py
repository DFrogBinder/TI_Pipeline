import pyvista as pv
import argparse
import os

def visualize_mesh(file_path):
    """Load and visualize a .msh file using PyVista."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        # Load the mesh
        mesh = pv.read(file_path)

        # Create a plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, opacity=0.7)

        # Set plot properties
        plotter.add_axes()
        plotter.show_grid()
        plotter.view_isometric()

        # Display mesh
        plotter.show()
    except Exception as e:
        print(f"Error loading mesh: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .msh file using PyVista.")
    parser.add_argument("file", type=str, help="Path to the .msh file")
    args = parser.parse_args()

    visualize_mesh(args.file)

