import numpy as np
import cv2
import os
from skimage import measure
from scipy.ndimage import label

# Step 1: Load the 2D Image Slices
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Step 2: Apply Threshold Filter
def apply_threshold(image, threshold_value):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

# Step 3: Extract Specific Region
def extract_region(binary_image):
    return binary_image  # The binary image itself is the mask of the specific region

# Step 4: 3D Reconstruction
def stack_images(images):
    return np.stack(images, axis=-1)

# Step 5: Volume Calculation
def calculate_volume(binary_volume, voxel_size):
    labeled_volume, num_features = label(binary_volume)
    region_volume = np.sum(labeled_volume > 0) * voxel_size
    return region_volume

# Main function to process images and calculate volume
def main(folder, threshold_value, voxel_size):
    images = load_images_from_folder(folder)
    binary_images = [apply_threshold(img, threshold_value) for img in images]
    binary_volume = stack_images(binary_images)
    volume = calculate_volume(binary_volume, voxel_size)
    return volume

# Example usage
folder = 'path/to/your/image_slices'
threshold_value = 128  # Example threshold value
voxel_size = 1.0  # Example voxel size in cubic units
volume = main(folder, threshold_value, voxel_size)
print(f"Calculated volume of the region: {volume} cubic units")

