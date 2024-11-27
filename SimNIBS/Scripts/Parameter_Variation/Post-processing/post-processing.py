import numpy as np
import cv2
import os
import sys
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
from skimage import measure
from stl import mesh

def nii2msh(data,export_path):
    # Load the NIfTI file
    # img = nib.load(nifti_path)
    # data = img.get_fdata()

    # Convert data to binary (assuming the binary threshold is clear, i.e., 0 is background and others are foreground)
    binary_data = np.where(data > 0, 1, 0)

    if not np.all(np.isnan(binary_data)) and np.any(binary_data):
        # Generate a mesh using marching cubes algorithm from skimage
        verts, faces, normals, values = measure.marching_cubes(binary_data, level=0.5)

        # Create the mesh object
        your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                your_mesh.vectors[i][j] = verts[f[j], :]

        # Write the mesh to file
        your_mesh.save(export_path)
        
        return 1
    else:
        tqdm.write(f'Failed to generate mesh from nifti file for case: {export_path}')
        return

def Extract_stats_csv(OutputDirContents, save_data=False):
    
    # Init empty dataframe for stats storage
    pd_columns = [ '80_Percent_Cutoff_Volume',
        '60_Percent_Cutoff_Volume',
        '40_Percent_Cutoff_Volume',
        'Maximum Value',
        'Electrode Size',
        'Electrode Shape',
        'Input Current',
        'Pair 1 Position',
        'Pair 2 Position',
        'Max_Thalamus_R',
        'Max_Thalamus_L',
        'Max_Thalamus']
    StatDF = pd.DataFrame(columns=pd_columns)
    
    studies = os.listdir(OutputDirContents)
    
    for study in studies:
        
        # Skips stats file
        if 'csv' in study:
            continue
        
        csv_path = os.path.join(OutputDirContents,study,'Stats.csv')
        try:
            stats = pd.read_csv(csv_path)
            StatDF = pd.concat([StatDF,stats],ignore_index=True) # Append stats to the empty dataframe
        except:
            tqdm.write(f'Failed to load file {csv_path}',file=sys.stderr)
    
    # Checks how many stat reports failed to load (if any)
    if len(StatDF) != len(studies):
        tqdm.write(f'{len(studies) - len(StatDF)} failed to load!',file=sys.stderr)
    
    if save_data:
        try:
            StatDF.to_csv(os.path.join(OutputDirContents,'All_Stats.csv'), sep=',', index=False)
            tqdm.write(f'All statistical data has been saved.',file=sys.stderr)
        except:
            tqdm.write(f'There was problem saving colated statistical data!',file=sys.stderr)
            
    return StatDF

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

    
    
    return np.max(thalamic_left_data.get_fdata()), np.max(thalamic_right_data.get_fdata()), np.max(combined_thalamic_data.get_fdata())

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

def Stats2CSV(volume_80,volume_60,volume_40, volume_0p2, max_intensity,electrode_size,electrode_shape,
              input_current, pair_1_pos,pair_2_pos,thal_L, thal_R, thal_ALL,path):
    # Create a dictionary with the data
    data = {
        '80_Percent_Cutoff_Volume': [volume_80],
        '60_Percent_Cutoff_Volume': [volume_60],
        '40_Percent_Cutoff_Volume': [volume_40],
        '0.2v_Cutoff_Volume': [volume_0p2],
        'Maximum Value': [max_intensity],
        'Electrode Size': [electrode_size],
        'Electrode Shape': [electrode_shape],
        'Input Current': [input_current],
        'Pair 1 Position': [pair_1_pos],
        'Pair 2 Position': [pair_2_pos],
        'Max_Thalamus_R': [thal_R],
        'Max_Thalamus_L': [thal_L],
        'Max_Thalamus': [thal_ALL]
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
            tqdm.write(f'Folder {path} created!',file=sys.stderr)    
            return True 
        except:
            tqdm.write(f'Failed to create folder {path}!',file=sys.stderr)
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


def load_images_and_threshold(image_data, folder, binary_output_folder):
    
    binary_images = {'80p_cutoff':[], 
                      '60p_cutoff':[], 
                      '40p_cutoff':[],
                      '0p2v_cutoff':[]
                      }
    
    threshold_list = {'80p_cutoff':np.max(image_data)*0.8, 
                      '60p_cutoff':np.max(image_data)*0.6, 
                      '40p_cutoff':np.max(image_data)*0.4,
                      '0p2v_cutoff': 0.2
                      }
    
    # Check and create folders named after dictionary keys
    for key in threshold_list:
        path = os.path.join(binary_output_folder, key)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            continue
            
    
    for name, threshold in threshold_list.items():
        for slice, counter in zip(image_data, range(len(image_data))):
            if slice is not None:
                # Create a mask where every pixel above the threshold is True; others are False
                mask = slice > threshold
                # Use the mask to keep only the pixels above the threshold; set others to 0
                thresholded_image = np.where(mask, slice, np.nan)
                binary_images[name].append(thresholded_image)
                # Save the thresholded binary image for inspection, ensuring no change in orientation
                tiff.imwrite(os.path.join(binary_output_folder,name, f'binary_{name}_{counter}.tiff'), thresholded_image)

    return binary_images

def stack_images(images):
    image_stacks = {'80p_cutoff':[], 
                    '60p_cutoff':[], 
                    '40p_cutoff':[],
                    '0p2v_cutoff':[]
                    }
    for key in images:
        image_stacks[key].append(np.stack(images[key], axis=-1))
        # return np.stack(images, axis=-1)
    return image_stacks

def calculate_volume(binary_volume, voxel_size, nifti_path, binary_volume_path):
    
    if np.all(np.isnan(binary_volume)):
        cut = binary_volume_path.split('/')[-1].split('.')[0]
        case = nifti_path.split('/')[-2]
        tqdm.write(f'All values in the {cut} are NaNs for case {case}')
        return

    
    stats = {
        'total_volume':None,
        'max_intensity':None,
        'min_intensity':None
    }
    
    # Load the thalamus and volume images
    thalamus_img = nib.load(os.path.join(nifti_path,'combined_thalamus_data.nii'))
    binary_img = nib.load(binary_volume_path)
    
    resampled_thalamus = resample_to_img(source_img=thalamus_img, target_img=binary_img, interpolation='nearest')
    
    # Invert the thalamus mask to exclude thalamus region
    non_thalamus_mask = image.math_img("img == 0", img=resampled_thalamus)
    
    # Apply the non-thalamus mask to the binary volume
    masked_binary_data = image.math_img("a * b", a=binary_img, b=non_thalamus_mask)
    
    # Count the non-NaN values
    non_nan_count = np.sum(~np.isnan(binary_volume))

    
    #TODO: What the fuck is this - fix it
    stats['total_volume'] = non_nan_count * voxel_size
    # stats['max_intensity'] = np.max(binary_volume[~np.isnan(binary_volume)])
    stats['max_intensity'] = np.nanmax(masked_binary_data.get_fdata())
    stats['min_intensity'] = np.nanmin(binary_volume[~np.isnan(binary_volume)])
    return stats

# Main function to process images and calculate volume
def main(nifti_path, masks_dir, output_folder, binary_output_folder,nifti_output, voxel_size):
    brain_only_images = prepare_base_nifti(nifti_path, masks_dir, output_folder, nifti_output)
    
    # Extracts left and rigth thalamus regions via the aal brain atlas
    thalamic_left_data, thalamic_right_data, combined_thalamic_data = Extract_Thalamus(nifti_path,nifti_output)
    
    # Load processed images, apply threshold, and save binary images
    brain_only_images = load_images_and_threshold(brain_only_images, output_folder, binary_output_folder)
    binary_volume = stack_images(brain_only_images)
    
    # Create an identity affine matrix (this is a simple placeholder)
    affine = np.eye(4)

    metrics = {'80p_cutoff':None, 
                '60p_cutoff':None, 
                '40p_cutoff':None,
                '0p2v_cutoff':None
                }
    # Create the NIfTI image
    for cutoff in binary_volume:
        img = nib.Nifti1Image(binary_volume[cutoff][0], affine)
        bin_volume_name = os.path.join(nifti_output,f'{cutoff}_thresholded_volume.nii.gz')
        nib.save(img,bin_volume_name)
        
        # Generate a 3d model of the stimulated area
        nii2msh(binary_volume[cutoff][0],os.path.join(nifti_output,f'{cutoff}_thresholded_volume.stl'))
        metrics[cutoff] = calculate_volume(binary_volume[cutoff][0], voxel_size, nifti_output, bin_volume_name)
        
    
    return metrics,thalamic_left_data, thalamic_right_data, combined_thalamic_data



'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++Execution Starts Here++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
# Set font sizes for all figures via rcParams
plt.rcParams['axes.labelsize'] = 28  # Sets the default axes labels size
plt.rcParams['xtick.labelsize'] = 14  # Sets the x-axis tick labels size
plt.rcParams['ytick.labelsize'] = 14  # Sets the y-axis tick labels size
plt.rcParams['axes.titlesize'] = 30  # Sets the default title size

stat_data = {
    'Electrode_Size': [],
    'Electrode_Shape': [],
    'Pair_1_Pos':[],
    'Pair_2_Pos':[],
    'Input_current':[],
    '80_Percent_Cutoff_Volume': [],
    '60_Percent_Cutoff_Volume': [],
    '40_Percent_Cutoff_Volume': [],
    '0.2v_Cutoff_Volume':[],
    'Maximum_Value': [],
    'Max_Thalamus_R': [],
    'Max_Thalamus_L': [],
    'Max_Thalamus': []
    }


OutputDir = '/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Parameter_Variation/Output'
simulations = os.listdir(OutputDir)
for simulation in tqdm(simulations,file=sys.stdout,desc="Progress"):
    
    # Skips stats file
    if 'csv' in simulation:
        continue
    
    # Extract Parameters from folder name
    sim_paramters = simulation.split('_')
    electrode_size = sim_paramters[0]
    electrode_shape = sim_paramters[-1]
    intensity = sim_paramters[1]
    pair_1_pos = sim_paramters[2]
    pair_2_pos = sim_paramters[3]
    
    # Inputs
    base_nifti_path = os.path.join(OutputDir, simulation,'Volume_Base','TI_Volumetric_Base_TImax.nii.gz')
    masks_nifti_dir = os.path.join(OutputDir, simulation,'Volume_Maks')

    # Output Destinations
    output_folder = os.path.join(OutputDir, simulation, 'Slices')
    if not SetupDirPath(output_folder):
        tqdm.write(f'Failed to create pathing for {simulation}! Continuing to next study...',file=sys.stderr)
        continue
        
    
    binary_output = os.path.join(OutputDir, simulation,'Binary_Volumes')
    if not SetupDirPath(binary_output):
        tqdm.write(f'Failed to create pathing for {simulation}! Continuing to next study...',file=sys.stderr)
        continue
    
    nifti_output = os.path.join(OutputDir, simulation,'nifti')
    if not SetupDirPath(nifti_output):
        tqdm.write(f'Failed to create pathing for {simulation}! Continuing to next study...',file=sys.stderr)
        continue

    # Parameter Values
    # threshold_value = 0.19
    voxel_size = 1.0  # Example voxel size in cubic units (e.g., 1 mm^3 if images are 1mm thick)

    volumes, thal_L,thal_R, thal_ALL = main(base_nifti_path, masks_nifti_dir, 
                                                output_folder, binary_output, nifti_output, 
                                                voxel_size)
    
    # Extract stats from volume dict
    volume_80p = volumes.get('80p_cutoff', {}).get('total_volume')
    volume_60p = volumes.get('60p_cutoff', {}).get('total_volume')
    volume_40p = volumes.get('40p_cutoff', {}).get('total_volume')
    volume_0p2 = volumes.get('0p2v_cutoff', {}).get('total_volume')
    
    # This is the same for all cutoffs 
    max_intensity = volumes.get('80p_cutoff', {}).get('max_intensity')
    
    Stats2CSV(volume_80p,
              volume_60p, 
              volume_40p, 
              volume_0p2, 
              max_intensity, 
              electrode_size,
              electrode_shape,
              intensity,
              pair_1_pos,
              pair_2_pos,
              thal_L,
              thal_R,
              thal_ALL,
              os.path.join(OutputDir,simulation)
              )
    
    
    
    # Update Stats Dict For Plotting later
    stat_data['Electrode_Size'].append(electrode_size)
    stat_data['Electrode_Shape'].append(electrode_shape)
    stat_data['Input_current'].append(intensity)
    stat_data['Pair_1_Pos'].append(pair_1_pos)
    stat_data['Pair_2_Pos'].append(pair_2_pos)
    stat_data['Max_Thalamus_R'].append(thal_R)
    stat_data['Max_Thalamus_L'].append(thal_L)
    stat_data['Max_Thalamus'].append(thal_ALL)
    stat_data['80_Percent_Cutoff_Volume'].append(volume_80p)
    stat_data['60_Percent_Cutoff_Volume'].append(volume_60p)
    stat_data['40_Percent_Cutoff_Volume'].append(volume_40p)
    stat_data['0.2v_Cutoff_Volume'].append(volume_0p2)
    
    
    tqdm.write(f"Finished processing case: {simulation}")

stat_data = Extract_stats_csv(OutputDir, save_data=True)
# GenerateHeatmap(stat_data)
