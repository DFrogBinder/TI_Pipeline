import meshio
import os

def get_resolution(path):

    output_dir = os.listdir(path)
    for case in output_dir:
        
        filepath = os.path.join(path,case)
        # Load the solved FEM mesh
        mesh = meshio.read(os.path.join(filepath,'TI.msh'))

        # Extract electrode geometry based on specific tags
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":  # Example for surface elements
                electrode_elements = cell_block.data
                print(f'Electrode elements: {electrode_elements}')

        # Optionally, write the electrode geometry to a new mesh file
        electrode_mesh = meshio.Mesh(
            points=mesh.points,
            cells=[("triangle", electrode_elements)]
        )
        meshio.write("electrode_geometry.msh", electrode_mesh)
get_resolution('/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Output')