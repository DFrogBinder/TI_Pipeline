import numpy as np
import pyvista as pv

# Function to create a symmetric log transformation
def symlog(data, linthresh=1):
    sign_data = np.sign(data)
    abs_data = np.abs(data)
    result = sign_data * np.log1p(abs_data / linthresh)
    return result

# Load the .vtk file and slice it
mesh = pv.read('/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Simple_Export_Save_Dir/fem_model-name_12-13-10-11.vtk').slice(normal=[0, 0, 1], origin=[0, 0, 0])

if 'mat_id' in mesh.cell_arrays:
    brain_mask = mesh.cell_arrays['mat_id'] == 4  # 1-4 tissue types | 10-13 electrodes
    base_electrodes_mask = np.isin(mesh.cell_arrays['mat_id'], [10, 11])
    df_electrodes_mask = np.isin(mesh.cell_arrays['mat_id'], [12, 13])

    assert np.any(brain_mask), 'Brain Mask is empty!'
    
    region_mesh = mesh.extract_cells(brain_mask)
    base_electrode_mesh = mesh.extract_cells(base_electrodes_mask)
    df_electrode_mesh = mesh.extract_cells(df_electrodes_mask)

    fields = ['e_field_(potential_base)', 'e_field_(potential_df)']
    colormaps = ['viridis', 'plasma', 'inferno', 'coolwarm']
    linthresh = 0.1  # Threshold for symlog transformation, adjust based on your data
    
    for i, field in enumerate(fields):
        if field in region_mesh.cell_arrays:
            
            # Apply symmetric log transformation to the scalar values
            # scalars = symlog(region_mesh.cell_arrays[field], linthresh)
            
            # Determine the shift needed to make the smallest value positive (if min_value is negative)
            shift = int(abs(np.min(region_mesh.cell_arrays[field]))) + 1   # Adding 1 to ensure the minimum shifted value is positive

            # Apply the shift-and-log transformation
            shifted_and_logged_data = np.log10(region_mesh.cell_arrays[field] + shift)
            region_mesh.cell_arrays[field] = shifted_and_logged_data  # Update the mesh with the transformed data

            # Create a new Plotter instance for each field
            SubPlot_Number = 4
            plotter = pv.Plotter(shape=(int(SubPlot_Number/2), int(SubPlot_Number/2)))
            for i in range(SubPlot_Number):
                r, c = divmod(i, int(SubPlot_Number/2))  # Calculate row and column index
                plotter.subplot(r, c)
                colormap_index = i % len(colormaps)  # Ensure we loop through the colormaps
                plotter.add_mesh(region_mesh, scalars=field, cmap=colormaps[colormap_index], opacity=1)
                plotter.add_scalar_bar(title=f'SymLog {field}', title_font_size=22, label_font_size=20, n_labels=3, shadow=True, font_family='arial')

                # Determine the correct electrode mesh to use
                # TODO: Check if 'not' keyword is correct here
                electrode_mesh = df_electrode_mesh if 'base' in field else base_electrode_mesh
                plotter.add_mesh(electrode_mesh, color='red', opacity=1)
                
            # Show the plot for the current field
            plotter.show()

else:
    print("mat_id not found in the mesh. Please check your mesh data.")
