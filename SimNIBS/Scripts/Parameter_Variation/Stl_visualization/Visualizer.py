from paraview.simple import *

# List of STL file paths
stl_files = [
    "path_to_mesh_1.stl",
    "path_to_mesh_2.stl",
    "path_to_mesh_3.stl",
    "path_to_mesh_4.stl"
]

# Colors for each mesh (RGBA format, values between 0 and 1)
colors = [
    [1, 0, 0, 1],  # Red
    [0, 1, 0, 1],  # Green
    [0, 0, 1, 1],  # Blue
    [1, 1, 0, 1]   # Yellow
]

# Glyph parameters
glyph_scale_factor = 0.5

# Render view
render_view = GetActiveViewOrCreate('RenderView')

# Loop through each STL file and set up visualization
for idx, stl_file in enumerate(stl_files):
    # Load the STL file
    stl_reader = STLReader(FileNames=[stl_file])

    # Create a glyph for visualization
    glyph = Glyph(Input=stl_reader, GlyphType='Sphere')
    glyph.GlyphType.Radius = 0.05  # Set the size of glyphs
    glyph.ScaleFactor = glyph_scale_factor

    # Set the display properties
    display = Show(glyph, render_view)
    display.DiffuseColor = colors[idx][:3]  # Set color
    display.Opacity = colors[idx][3]  # Set opacity

# Adjust camera
render_view.ResetCamera()

# Render the view
Render()

# Save the state if needed (optional)
# SaveState("paraview_visualization.pvsm")

