import numpy as np
import cv2
import os
import nibabel as nib

# Step 0: Load and slice NIfTI data
def load_and_slice_nifti(file_path, output_folder):
    # Load the NIfTI file
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    
    # Assuming the slicing is along the z-axis, adjust if necessary
    for i in range(data.shape[2]):
        slice = data[:, :, i]
        
        # Normalize and convert to uint8 for image saving
        normalized_slice = cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(normalized_slice)
        
        # Save each slice as an image
        cv2.imwrite(os.path.join(output_folder, f'slice_{i:04d}.png'), image)

# Continue with existing functions and main setup
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def apply_threshold(image, threshold_value):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def stack_images(images):
    return np.stack(images, axis=-1)

def calculate_volume(binary_volume, voxel_size):
    labeled_volume, num_features = label(binary_volume)
    region_volume = np.sum(labeled_volume > 0) * voxel_size
    return region_volume

def main(nifti_path, output_folder, threshold_value, voxel_size):
    # Process NIfTI file and create slices
    load_and_slice_nifti(nifti_path, output_folder)
    
    # Load processed images and calculate volume
    images = load_images_from_folder(output_folder)
    binary_images = [apply_threshold(img, threshold_value) for img in images]
    binary_volume = stack_images(binary_images)
    volume = calculate_volume(binary_volume, voxel_size)
    return volume

# Example usage
nifti_path = 'path/to/your/data.nii.gz'
output_folder = 'path/to/your/image_slices'
threshold_value = 128  # Example threshold value
voxel_size = 1.0  # Example voxel size in cubic units (e.g., 1 mm^3 if images are 1mm thick)
volume = main(nifti_path, output_folder, threshold_value, voxel_size)
print(f"Calculated volume of the region: {volume} cubic units")
