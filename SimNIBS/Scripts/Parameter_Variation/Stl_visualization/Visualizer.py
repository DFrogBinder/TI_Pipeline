from paraview.simple import *
import os

# Check if the paraview.simple module is initialized
paraview.simple._DisableFirstRenderCameraReset()

# List of STL file paths
stl_files = [
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/0p2v_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/40p_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/60p_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/80p_cutoff_thresholded_volume.stl"
]

# Colors for each mesh (RGB format, values between 0 and 1)
colors = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0]   # Yellow
]

# Ensure paths are valid
for path in stl_files:
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit()

# Get the active render view or create one
render_view = GetActiveViewOrCreate('RenderView')

# Loop through each STL file and visualize
for idx, stl_file in enumerate(stl_files):
    print(f"Loading {stl_file}...")

    # Load the STL file
    stl_reader = STLReader(FileNames=[stl_file])

    # Apply Transform (Optional, for scaling or adjusting placement)
    transform = Transform(Input=stl_reader)
    transform.Transform.Scale = [1.0, 1.0, 1.0]  # Adjust scale if necessary
    transform.Transform.Translate = [0, 0, idx * 10]  # Separate meshes for visibility

    # Show the mesh
    display = Show(transform, render_view)
    display.Representation = "Surface"  # Use surface rendering
    display.DiffuseColor = colors[idx]  # Assign color

# Reset the camera to fit all objects
render_view.ResetCamera()

# Render the view
Render()

# Keep the render window open
print("Render window is active. Press Ctrl+C in the terminal to close.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Closing render window.")