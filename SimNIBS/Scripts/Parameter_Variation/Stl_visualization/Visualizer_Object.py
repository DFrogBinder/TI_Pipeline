import os
import sys
import json
import tkinter as tk

from tqdm import tqdm
from paraview.simple import *

class Visualizer:
    def __init__(self,sim_dir):
        # Check if the paraview.simple module is initialized
        paraview.simple._DisableFirstRenderCameraReset()

        # Get screen size
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        self.sim_dir = sim_dir
        
    def Render_Stl(self):
        
        simulations = os.listdir(self.sim_dir)
        
        for simulation in tqdm(simulations):
            
            # List of STL file paths
            stl_files = [
                os.path.join(self.sim_dir,simulation,'Analysis','nifti','0p2v_cutoff_thresholded_volume.stl'),
                os.path.join(self.sim_dir,simulation,'Analysis','nifti','40p_cutoff_thresholded_volume.stl'),
                os.path.join(self.sim_dir,simulation,'Analysis','nifti','60p_cutoff_thresholded_volume.stl'),
                os.path.join(self.sim_dir,simulation,'Analysis','nifti','80p_cutoff_thresholded_volume.stl')
            ]

            # Colors for each mesh (RGB format, values between 0 and 1)
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0]   # Yellow
            ]

            # Filter out non-existent files
            stl_files = [file for file in stl_files if os.path.exists(file)]

            # Get the active render view or create one
            render_view = GetActiveViewOrCreate('RenderView')

            created_objects = []
            
            # Loop through each STL file and visualize
            for idx, stl_file in enumerate(stl_files):
                tqdm.write(f"Loading {stl_file}...")

                # Load the STL file
                stl_reader = STLReader(FileNames=[stl_file])
                created_objects.append(stl_reader)  # Track STLReader for cleanup


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
                tqdm.write(f"All possible representation values: {representation_values}")
            except AttributeError:
                tqdm.write("Unable to retrieve possible values for 'Representation'.")

            # Reset the camera to fit all objects
            render_view.ResetCamera()

            # Set the render window size to a percentage of the screen size
            render_view = GetActiveViewOrCreate('RenderView')
            render_view.ViewSize = [int(self.screen_width * 0.8), int(self.screen_height * 0.8)]  # 80% of screen size

            LoadPalette(paletteName='BlackBackground')

            # Render the view
            Render()
            SaveScreenshot(f'{simulation}.png',quality=600, view=render_view)
            Delete()
            
        
def main():
    # Read the JSON input passed via stdin
    input_data = sys.stdin.read()
    
    # Optionally, parse the JSON input into a Python object
    try:
        parsed_data = json.loads(input_data)
        tqdm.write(f"Parsed input data type: {parsed_data}")
    except json.JSONDecodeError as e:
        tqdm.write(f"Error decoding JSON: {e}")
    
    Client = Visualizer(parsed_data)
    Client.Render_Stl()    
    return 

if __name__ == "__main__":
    main()