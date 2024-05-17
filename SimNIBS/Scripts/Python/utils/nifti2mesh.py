import argparse
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import meshio

def convert_nii_to_msh(input_path, output_path, threshold=0.5):
    # Load the NIfTI file
    nii = nib.load(input_path)
    data = nii.get_fdata()

    # Create a mesh from the NIfTI data
    verts, faces, _, _ = marching_cubes(data, level=threshold)

    # Prepare mesh data
    cells = [("triangle", faces)]

    # Create the mesh object
    mesh = meshio.Mesh(points=verts, cells=cells)

    # Save the mesh to a .msh file
    mesh.write(output_path)
    print(f"Mesh saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert NIfTI (.nii.gz) files to .msh files.')
    parser.add_argument('input_path', type=str, help='Path to the input NIfTI file (.nii.gz)')
    parser.add_argument('output_path', type=str, help='Path to the output .msh file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value for surface extraction (default: 0.5)')
    
    args = parser.parse_args()
    convert_nii_to_msh(args.input_path, args.output_path, args.threshold)

