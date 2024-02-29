import numpy as np
import pyvista as pv

# Load the .vtk file and slice it
mesh = pv.read('/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Simple_Export_Save_Dir/fem_model-name_12-13-10-11.vtk').slice(normal=[0, 0, 1], origin=[0, 0, 0])

# Assuming 'field1', 'field2', 'field3', and 'field4' are the names of your fields
# fields = ['e_field_(potential_base)', 'e_field_(potential_df)']
fields = ['potential_df','potential_base']

# Define fields and colormaps
colormaps = ['viridis', 'plasma', 'inferno', 'magma']

# Adjust opacities here if needed
opacities = [0.5, 0.5, 0.5, 0.5]  # Example opacities, adjust based on visualization needs

plotter = pv.Plotter()

for i, field in enumerate(fields):
    if field in mesh.array_names:
        # Adjust the offset and transformation as necessary
        offset = 1e-6
        field_data = np.log10(mesh[field] + offset)
        
        # Check for valid data (not all NaN)
        if not np.all(np.isnan(field_data)):
            plotter.add_mesh(mesh, scalars=field_data, cmap=colormaps[i % len(colormaps)], opacity=opacities[i % len(opacities)])
            plotter.add_scalar_bar(title=f'Log10({field})', title_font_size=22, label_font_size=20, n_labels=3, shadow=True, font_family='arial')
        else:
            print(f"Warning: Field '{field}' contains only NaN values after transformation.")
    else:
        print(f"Field '{field}' not found in the mesh.")

plotter.show()
