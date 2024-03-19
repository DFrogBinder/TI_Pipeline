import numpy as np
import nibabel as nib

def create_sphere(radius, center, grid_size):
    """Generate a sphere within a 3D grid."""
    x = np.arange(0, grid_size[0])
    y = np.arange(0, grid_size[1])
    z = np.arange(0, grid_size[2])
    x, y, z = np.meshgrid(x, y, z)

    # Equation of a sphere (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2
    sphere = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius ** 2
    return sphere.astype(np.int)

def generate_nifti_with_spheres(radii, grid_size=(200, 200, 200)):
    """Generate a NIfTI file with nested spheres."""
    center = (grid_size[0]//2, grid_size[1]//2, grid_size[2]//2)
    layers = np.zeros(grid_size, dtype=np.int)

    for i, radius in enumerate(radii, start=1):
        sphere = create_sphere(radius, center, grid_size)
        layers[sphere == 1] = i

    # Convert the voxel data to a NIfTI image
    nifti_img = nib.Nifti1Image(layers, affine=np.eye(4))
    nib.save(nifti_img, 'nested_spheres.nii.gz')

# Radii of the spheres in mm (converted to voxels if necessary)
radii = [85.96, 81.61, 74.21, 72.32]
generate_nifti_with_spheres(radii)

