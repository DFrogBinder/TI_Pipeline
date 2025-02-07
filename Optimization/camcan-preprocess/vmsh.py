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
        
        # Add the mesh with improved visibility
        plotter.add_mesh(
            mesh, 
            show_edges=True, 
            opacity=0.8, 
            color="white", 
            edge_color="black", 
            line_width=2.0,
            lighting=True
        )

        # Add a wireframe overlay
        plotter.add_mesh(mesh, style="wireframe", color="cyan")

        # Set plot properties
        plotter.set_background("darkgray")  # Change background for contrast
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

