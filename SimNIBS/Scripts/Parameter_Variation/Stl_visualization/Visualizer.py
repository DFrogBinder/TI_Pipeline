import os
import tkinter as tk
from paraview.simple import *


# Check if the paraview.simple module is initialized
paraview.simple._DisableFirstRenderCameraReset()

# Get screen size
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# List of STL file paths
stl_files = [
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/0p2v_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/40p_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/60p_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output/2cm_1.5mA_F6-P6_F5-P5_rect/Analysis/nifti/80p_cutoff_thresholded_volume.stl"
]

# Colors for each mesh (RGB format, values between 0 and 1)
colors = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0]   # Yellow
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

    # Show the mesh as points
    display = Show(transform, render_view)
    display.Representation = "Points"  # Render as point cloud
    display.PointSize = 0.5  # Adjust the point size for better visibility
    
    # Get all possible values for the 'Representation' property
    try:
        representation_values = display.Representation.Available
        print(f"All possible representation values: {representation_values}")
    except AttributeError:
        print("Unable to retrieve possible values for 'Representation'.")

    # Assign a distinct color
    display.AmbientColor = colors[idx]  # Set the color of the points
    display.Opacity = 0.5

# Reset the camera to fit all objects
render_view.ResetCamera()

# Set the render window size to a percentage of the screen size
render_view = GetActiveViewOrCreate('RenderView')
render_view.ViewSize = [int(screen_width * 0.8), int(screen_height * 0.8)]  # 80% of screen size

# Set the background to black (RGB values: 0, 0, 0)
render_view.Background = [0.0, 0.0, 0.0]

# Ensure any gradient-related settings are cleared (if present)
render_view.Background2 = [0.0, 0.0, 0.0] 

# Render the view
Render()

# Keep the render window open
print("Render window is active. Press Ctrl+C in the terminal to close.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Closing render window.")