import numpy as np
import nibabel as nib
from skimage import measure
from stl import mesh

# Load the NIfTI file
nifti_path = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs/1.5cm_1.5mA_AF4-PO4_AF3-PO3_ellipse/nifti/combined_thalamus_data.nii'
img = nib.load(nifti_path)
data = img.get_fdata()

# Convert data to binary (assuming the binary threshold is clear, i.e., 0 is background and others are foreground)
binary_data = np.where(data > 0, 1, 0)

# Generate a mesh using marching cubes algorithm from skimage
verts, faces, normals, values = measure.marching_cubes(binary_data, level=0.5)

# Create the mesh object
your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        your_mesh.vectors[i][j] = verts[f[j], :]

# Write the mesh to file
your_mesh.save('your_mesh.stl')