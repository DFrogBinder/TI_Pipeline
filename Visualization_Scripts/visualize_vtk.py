import numpy as np
import pyvista as pv

# Load the .vtk file and slice it
mesh = pv.read('/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Simple_Export_Save_Dir/fem_model-name_12-13-10-11.vtk').slice(normal=[0, 0, 1], origin=[0, 0, 0])
all_fields = mesh.array_names
print(f'All fields found in the mesh: \n {all_fields}')
# Assuming 'mat_id' is stored as a cell array, and you have electric field data in point arrays
# Ensure that 'mat_id' exists and contains the appropriate IDs
if 'mat_id' in mesh.cell_arrays:
    # Create a mask for cells with mat_id = 4
    brain_mask = mesh.cell_arrays['mat_id'] == 4
    electodes_mask = np.isin(mesh.cell_arrays['mat_id'],[10,11,12,13])
    
    # Extract only the part of the mesh with mat_id = 4
    region_mesh = mesh.extract_cells(brain_mask)
    electrode_mesh = mesh.extract_cells(electodes_mask)
else:
    print("mat_id not found in the mesh. Please check your mesh data.")
    region_mesh = None

# Proceed if we successfully extracted the region_mesh
if region_mesh:
    fields = ['potential_base', 'potential_df']  # Update this list with the actual electric field data names
    colormaps = ['viridis', 'plasma']  # Define colormaps for visualization
    opacities = [1, 1]  # Define opacities for each field
    
    plotter = pv.Plotter()
    
    for i, field in enumerate(fields):
        if field in region_mesh.point_arrays:
            # Directly visualize the electric field data without transformation
            # plotter.add_mesh(region_mesh, scalars=field, cmap=colormaps[i % len(colormaps)], opacity=opacities[i % len(opacities)])
            plotter.add_mesh(region_mesh, scalars=field, cmap='coolwarm', opacity=opacities[i % len(opacities)])
            plotter.add_scalar_bar(title=field, title_font_size=22, label_font_size=20, n_labels=3, shadow=True, font_family='arial')
        else:
            print(f"Field '{field}' not found in the region mesh.")
    # Adds the electrodes to the plot
    plotter.add_mesh(electrode_mesh)
    plotter.show()