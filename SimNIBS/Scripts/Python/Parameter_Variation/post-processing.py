import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import nibabel as nib
import tifffile as tiff
from scipy.ndimage import label

def SetupDirPath(path):
    if os.path.isdir(path):
        return True
    else:
        try:
            os.mkdir(path)
            print(f'Folder {path} created!')    
            return True 
        except:
            print(f'Failed to create folder {path}!')
            return False
        

# Step 0: Load and slice NIfTI data
def prepare_base_nifti(file_path, masks_dir, output_folder, plot_flag=True):
    # Load the NIfTI file
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    
    # Load White and Gray Matter Masks
    masks_list = os.listdir(masks_dir)
    for mask in masks_list:
        if '1' in mask and '10' not in mask:
            wm = nib.load(os.path.join(masks_dir, mask)).get_fdata().astype(np.int16)
        elif '2' in mask:
            gm = nib.load(os.path.join(masks_dir, mask)).get_fdata().astype(np.int16)
        else:
            continue

    # Combine the masks
    combined_mask = np.logical_or(wm, gm).astype(np.int16)
    
    # Apply the mask - This zeros out all parts of the image not covered by the mask
    brain = data * combined_mask
    
    if plot_flag:
        slice_number = 190
        
        # Create a figure with 4 subplots
        fig, axs = plt.subplots(2, 2)

        # Display each image
        axs[0, 0].imshow(wm[:,:,slice_number], cmap='viridis')
        axs[0, 0].set_title('White Matter Mask')
        axs[0, 0].axis('off')  # Hide axes for clarity

        axs[0, 1].imshow(gm[:,:,slice_number], cmap='viridis')
        axs[0, 1].set_title('Gray Matter Mask')
        axs[0, 1].axis('off')

        im1 = axs[1, 0].imshow(data[:,:,slice_number], cmap='viridis')
        axs[1, 0].set_title('Base Volume Mesh')
        axs[1, 0].axis('off')
        # Add a colorbar for the third subplot
        cbar1 = fig.colorbar(im1, ax=axs[1, 0])
        cbar1.set_label('Intensity')

        im2 = axs[1, 1].imshow(brain[:,:,slice_number], cmap='viridis')
        axs[1, 1].set_title('Extracted Brain Mesh')
        axs[1, 1].axis('off')
        # Add a colorbar for the fourth subplot
        cbar2 = fig.colorbar(im2, ax=axs[1, 1])
        cbar2.set_label('Intensity')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    
    # Create a new NIfTI image from the mask
    label_img = nib.Nifti1Image(brain, nifti.affine, nifti.header)
    label_img_path = os.path.join(output_folder, 'white_gray_matter.nii.gz')
    nib.save(label_img, label_img_path)

    
    # Assuming the slicing is along the z-axis (axial)
    for i in range(brain.shape[2]):
        # Ensure no alteration in orientation during slicing
        slice = brain[:, :, i]

        # Normalize and convert to uint8 for image saving
        # normalized_slice = cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(slice)
        
        # Save each slice as an image
        cv2.imwrite(os.path.join(output_folder, f'slice_{i:04d}.png'), image)
        
    return brain

# Step 1: Load images, apply threshold, and ensure orientation
def load_images_and_threshold(image_data, folder, threshold_value, binary_output_folder):
    binary_images = []
    for slice, counter in zip(image_data, range(len(image_data))):
        if slice is not None:
            # Create a mask where every pixel above the threshold is True; others are False
            mask = slice > threshold_value
            # Use the mask to keep only the pixels above the threshold; set others to 0
            thresholded_image = np.where(mask, slice, np.nan)

            # _, binary_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
            binary_images.append(thresholded_image)
            # Save the thresholded binary image for inspection, ensuring no change in orientation
            tiff.imwrite(os.path.join(binary_output_folder, f'binary_{counter}.tiff'), thresholded_image)

    return binary_images

# Step 2: Stack images to form a volume
def stack_images(images):
    return np.stack(images, axis=-1)

# Step 3: Calculate volume
def calculate_volume(binary_volume, voxel_size):
    
    # Count the non-NaN values
    non_nan_count = np.sum(~np.isnan(binary_volume))

    total_volume = non_nan_count * voxel_size
    max_intensity = np.max(binary_volume[~np.isnan(binary_volume)])
    min_intensity = np.min(binary_volume[~np.isnan(binary_volume)])
    return total_volume, max_intensity, min_intensity

# Main function to process images and calculate volume
def main(nifti_path, masks_dir, output_folder, binary_output_folder, threshold_value, voxel_size):
    brain_only_images = prepare_base_nifti(nifti_path, masks_dir, output_folder)
    
    # Load processed images, apply threshold, and save binary images
    brain_only_images = load_images_and_threshold(brain_only_images, output_folder, threshold_value, binary_output_folder)
    binary_volume = stack_images(brain_only_images)
    
    # Create an identity affine matrix (this is a simple placeholder)
    affine = np.eye(4)

    # Create the NIfTI image
    img = nib.Nifti1Image(binary_volume, affine)
    nib.save(img,'thresholded_volume.nii.gz')
    volume = calculate_volume(binary_volume, voxel_size)
    return volume

OutputDir = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs/'
simulations = os.listdir(OutputDir)

for simulation in simulations:
    
    # Inputs
    base_nifti_path = os.path.join(OutputDir, simulation,'Volume_Base','TI_Volumetric_Base_TImax.nii.gz')
    masks_nifti_dir = os.path.join(OutputDir, simulation,'Volume_Maks')

    # Output Destinations
    output_folder = os.path.join(OutputDir, simulation, 'Slices')
    if not SetupDirPath(output_folder):
        print(f'Failed to create pathing for {simulation}! Continuing to next study...')
        continue
        
    
    binary_output = os.path.join(OutputDir, simulation,'Binary_Volumes')
    if not SetupDirPath(binary_output):
        print(f'Failed to create pathing for {simulation}! Continuing to next study...')
        continue

    # Parameter Values
    threshold_value = 0.15
    voxel_size = 1.0  # Example voxel size in cubic units (e.g., 1 mm^3 if images are 1mm thick)

    volume, max_intensity, min_intensity = main(base_nifti_path, masks_nifti_dir, output_folder, binary_output, threshold_value, voxel_size)
    print(f"Calculated volume of the region: {volume} cubic units")
