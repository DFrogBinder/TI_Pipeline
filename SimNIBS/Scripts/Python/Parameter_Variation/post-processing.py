import numpy as np
import cv2
import os
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import label

def save_images(images, destination_path):
    """
    Save a list of images to a specified directory using OpenCV.

    Args:
    images (list): List of images where each image is a numpy array.
    destination_path (str): The path where images should be saved.

    Returns:
    None
    """
    # Ensure the destination path exists, if not create it
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Iterate over the list of images
    for idx, img in enumerate(images):
        # Define the path for each image
        image_path = os.path.join(destination_path, f'image_{idx+1}.png')
        # Save the image using OpenCV
        cv2.imwrite(image_path, img)
        print(f"Saved: {image_path}")
        
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
nifti_path = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs/1cm_1mA_FC4-P4_FC3-P3/MNI152_TDCS_TI_TImax.nii.gz'
output_folder = './'
threshold_value = 1 # Example threshold value
voxel_size = 1.0  # Example voxel size in cubic units (e.g., 1 mm^3 if images are 1mm thick)
volume = main(nifti_path, output_folder, threshold_value, voxel_size)
print(f"Calculated volume of the region: {volume} cubic units")