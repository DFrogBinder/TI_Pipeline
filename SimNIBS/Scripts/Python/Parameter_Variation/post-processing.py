import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import nibabel as nib
import tifffile as tiff
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import pyvista as pv
from tqdm import tqdm
from scipy.ndimage import label
from nilearn import datasets, image
from nilearn.image import resample_to_img


def Extract_Thalamus(base_path,output_path):
    # Load the AAL atlas
    aal_dataset = datasets.fetch_atlas_aal(version='SPM12', data_dir='./aal_SPM12')
    atlas_filename = aal_dataset.maps
    labels = aal_dataset.labels

    # Load the atlas image
    atlas_img = nib.load(atlas_filename)
 
    # Find indices for the left and right thalamus
    thalamus_left_idx = aal_dataset.indices[labels.index('Thalamus_L')]
    thalamus_right_idx = aal_dataset.indices[labels.index('Thalamus_R')]

    # Create masks for the left and right thalamus
    thalamus_left_mask = image.math_img(f"img == {thalamus_left_idx}", img=atlas_img)
    thalamus_right_mask = image.math_img(f"img == {thalamus_right_idx}", img=atlas_img)
    combined_thalamus_mask = image.math_img("img1 + img2", img1=thalamus_left_mask, img2=thalamus_right_mask)

    
    nifti = nib.load(base_path)
    # Resample simulation data to match the atlas space
    resampled_sim_data = resample_to_img(source_img=nifti, target_img=atlas_img, interpolation='nearest')

    
    # Apply the masks to the simulation data
    thalamic_left_data = image.math_img("a * b", a=resampled_sim_data, b=thalamus_left_mask)
    thalamic_right_data = image.math_img("a * b", a=resampled_sim_data, b=thalamus_right_mask)
    combined_thalamic_data = image.math_img("a * b", a=resampled_sim_data, b=combined_thalamus_mask)


    # Save the masked data or proceed with analysis
    nib.save(thalamic_left_data, os.path.join(output_path, 'thalamic_left_data.nii'))
    nib.save(thalamic_right_data, os.path.join(output_path, 'thalamic_right_data.nii'))
    nib.save(combined_thalamic_data, os.path.join(output_path, 'combined_thalamus_data.nii'))

    
    
    return thalamus_left_mask, thalamus_right_mask, combined_thalamus_mask

def nifti_to_mesh(nifti_file_path, output_mesh_path):
    # Load the NIfTI file
    img = nib.load(nifti_file_path)
    data = img.get_fdata()
    # Convert the float data to int (assuming binary, you can use 0 and 1)
    data_int = (data > 0).astype(np.int16)
    sitk_img = sitk.GetImageFromArray(data_int)
    
    # Get the contour of the segmentation
    contour_img = sitk.LabelContour(sitk_img, fullyConnected=True, backgroundValue=0)

    # Convert SimpleITK image to a numpy array and prepare for PyVista
    vtk_img = sitk.GetArrayFromImage(contour_img)
    # Create structured grid with correct dimensions and spacing
    grid = pv.StructuredGrid(*vtk_img.shape[::-1])  # PyVista expects dimensions in x, y, z order
    grid.points = np.stack(np.mgrid[0:vtk_img.shape[2], 0:vtk_img.shape[1], 0:vtk_img.shape[0]], axis=-1).reshape(-1, 3) * img.header.get_zooms() + img.affine[:3, 3]
    grid.point_data['values'] = vtk_img.T.ravel()  # Transpose to match VTK indexing

    # Extract the surface
    surface = grid.extract_surface()

    # Save the mesh
    surface.save(output_mesh_path)
    
def GenerateHeatmap(data):
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create a pivot table
    pivot_table = df.pivot_table(values='Total_Volume', index='Electrode_Size', columns='Input_current', aggfunc=np.sum)

    # Set a style and context for better visual appeal
    sns.set(style="whitegrid", context='talk')

    # Create a more professional looking heatmap
    plt.figure(figsize=(10, 8))  # Adjust the size as needed
    professional_heatmap = sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='g', linewidths=.5, linecolor='grey', cbar_kws={'label': 'Total Volume'})

    # Adding more descriptive titles and labels
    plt.title('Heatmap of Total Volume by Electrode Size and Input Current', pad=20)
    plt.xlabel('Input Current')
    plt.ylabel('Electrode Size')

    # Improve readability by configuring tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.show()

def Stats2CSV(volume, max_intensity, min_intensity,electrode_size,input_current,
              pair_1_pos,pair_2_pos,path):
    # Create a dictionary with the data
    data = {
        'Total Volume': [volume],
        'Minimum Value': [min_intensity],
        'Maximum Value': [max_intensity],
        'Electrode Size': [electrode_size],
        'Input Current': [input_current],
        'Pair 1 Position': [pair_1_pos],
        'Pair 2 Position': [pair_2_pos]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_file_path =os.path.join(path,'Stats.csv')  
    df.to_csv(csv_file_path, index=False)
    
    return data

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
def prepare_base_nifti(file_path, masks_dir, output_folder, nifti_output, plot_flag=False):
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
    label_img_path = os.path.join(nifti_output, 'white_gray_matter.nii.gz')
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
def load_images_and_threshold(image_data, folder, binary_output_folder):
    binary_images = []
    threshold_value = np.max(image_data)*0.8 # Sets threshold value as 90% of max e-field
    
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
def main(nifti_path, masks_dir, output_folder, binary_output_folder,nifti_output, voxel_size):
    brain_only_images = prepare_base_nifti(nifti_path, masks_dir, output_folder, nifti_output)
    
    # Load processed images, apply threshold, and save binary images
    brain_only_images = load_images_and_threshold(brain_only_images, output_folder, binary_output_folder)
    binary_volume = stack_images(brain_only_images)
    _,_,_ = Extract_Thalamus(nifti_path,nifti_output)
    # Create an identity affine matrix (this is a simple placeholder)
    affine = np.eye(4)

    # Create the NIfTI image
    img = nib.Nifti1Image(binary_volume, affine)
    nib.save(img,os.path.join(nifti_output,'thresholded_volume.nii.gz'))
    volume = calculate_volume(binary_volume, voxel_size)
    return volume



'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++Execution Starts Here++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

stat_data = {
    'Electrode_Size': [],
    'Pair_1_Pos':[],
    'Pair_2_Pos':[],
    'Input_current':[],
    'Total_Volume': [],
    'Minimum_Value': [],
    'Maximum_Value': []
    }


OutputDir = '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Outputs/'
simulations = os.listdir(OutputDir)

for simulation in tqdm(simulations):

    # Extract Parameters from folder name
    sim_paramters = simulation.split('_')
    electrode_size = sim_paramters[0]
    intensity = sim_paramters[1]
    pair_1_pos = sim_paramters[2]
    pair_2_pos = sim_paramters[3]
    
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
    
    nifti_output = os.path.join(OutputDir, simulation,'nifti')
    if not SetupDirPath(nifti_output):
        print(f'Failed to create pathing for {simulation}! Continuing to next study...')
        continue

    # Parameter Values
    # threshold_value = 0.19
    voxel_size = 1.0  # Example voxel size in cubic units (e.g., 1 mm^3 if images are 1mm thick)

    volume, max_intensity, min_intensity = main(base_nifti_path, masks_nifti_dir, 
                                                output_folder, binary_output, nifti_output, 
                                                voxel_size)
    
    Stats2CSV(volume, 
              max_intensity, 
              min_intensity,
              electrode_size,
              intensity,
              pair_1_pos,
              pair_2_pos,
              os.path.join(OutputDir,simulation)
              )
    
    
    
    # Update Stats Dict For Plotting later
    stat_data['Electrode_Size'].append(electrode_size)
    stat_data['Input_current'].append(intensity)
    stat_data['Pair_1_Pos'].append(pair_1_pos)
    stat_data['Pair_2_Pos'].append(pair_2_pos)
    stat_data['Maximum_Value'].append(max_intensity)
    stat_data['Total_Volume'].append(volume)
    stat_data['Minimum_Value'].append(min_intensity)
    
    # print(f"Calculated volume of the region: {volume} cubic units")
GenerateHeatmap(stat_data)
