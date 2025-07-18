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
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Post-processing/Output/1cm_1mA_FC6-CP6_FC5-CP5_ellipse/Analysis/nifti/0p2v_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Post-processing/Output/1cm_1mA_FC6-CP6_FC5-CP5_ellipse/Analysis/nifti/40p_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Post-processing/Output/1cm_1mA_FC6-CP6_FC5-CP5_ellipse/Analysis/nifti/60p_cutoff_thresholded_volume.stl",
    "/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Post-processing/Output/1cm_1mA_FC6-CP6_FC5-CP5_ellipse/Analysis/nifti/80p_cutoff_thresholded_volume.stl"
]

# Colors for each mesh (RGB format, values between 0 and 1)
colors = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0]   # Yellow
]

# Ensure paths are valid and removes non-existent files
stl_files = [path for path in stl_files if os.path.exists(path)]

# Get the active render view or create one
render_view = GetActiveViewOrCreate('RenderView')

# Loop through each STL file and visualize
for idx, stl_file in enumerate(stl_files):
    print(f"Loading {stl_file}...")

    # Load the STL file
    try:
        stl_reader = STLReader(FileNames=[stl_file])
    except:
        print(f"Error: Unable to load STL file: {stl_file}")
        continue
    
    # Apply Transform (Optional, for scaling or adjusting placement)
    transform = Transform(Input=stl_reader)
    transform.Transform.Scale = [1.0, 1.0, 1.0]  # Adjust scale if necessary
    transform.Transform.Translate = [0, 0, idx * 10]  # Separate meshes for visibility

    # Show the mesh as points
    display = Show(transform, render_view)
    display.Representation = "Points"  # Render as point cloud
    display.PointSize = 0.6  # Adjust the point size for better visibility
    
    # Assign a distinct color
    display.AmbientColor = colors[idx]  # Set the color of the points
    display.Opacity = 0.7

# Get all possible values for the 'Representation' property
try:
    representation_values = display.Representation.Available
    print(f"All possible representation values: {representation_values}")
except AttributeError:
    print("Unable to retrieve possible values for 'Representation'.")

# Reset the camera to fit all objects
render_view.ResetCamera()

# Set the render window size to a percentage of the screen size
render_view = GetActiveViewOrCreate('RenderView')
render_view.ViewSize = [int(screen_width * 0.9), int(screen_height * 0.9)]  # 80% of screen size

LoadPalette(paletteName='BlackBackground')

# Render the view
Render()
SaveScreenshot('image.png',quality=600, view=render_view)
# # Keep the render window open
# print("Render window is active. Press Ctrl+C in the terminal to close.")
# try:
#     while True:
#         pass
# except KeyboardInterrupt:
#     print("Closing render window.")