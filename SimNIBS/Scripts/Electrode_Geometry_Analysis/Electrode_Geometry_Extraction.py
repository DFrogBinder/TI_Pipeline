import os
import gmsh
import pandas as pd
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def plot_data(df):
    # Define the sizes and shapes we are interested in
    sizes = ['1cm', '1.5cm', '2cm']
    shapes = ['rect', 'ellipse']
    
    # Extract size and shape from 'entry_name' and add them as columns
    df['size'] = df['Name'].str.extract(r'(\d+\.?\d*cm)')
    df['shape'] = df['Name'].str.extract(r'(rect|ellipse)')
    
    # Now you can filter based on each combination
    for size in sizes:
        for shape in shapes:
            subset = df[(df['size'] == size) & (df['shape'] == shape)]
            if not subset.empty:
                print(f"Subset for size {size} and shape {shape}:")
 
                # Group by 'size' and 'shape' and plot both swarm and box plots within each group for El_1-1 and El_1-2
                for (size, shape), group in subset.groupby(['size', 'shape']):
                    plt.figure(figsize=(12, 6))
                    
                    # Set the main title for both plots
                    plt.suptitle(f'Plots for size {size} and shape {shape}', fontsize=16)
                    
                    # Swarm Plot
                    plt.subplot(1, 2, 1)
                    sns.swarmplot(data=group[['El_1-1', 'El_1-2']])
                    plt.title(f'Swarm Plot')
                    plt.ylabel('Value')
                    
                    # Box Plot
                    plt.subplot(1, 2, 2)
                    sns.boxplot(data=group[['El_1-1', 'El_1-2']])
                    plt.title(f'Box Plot')
                    plt.ylabel('Value')
                    
                    plt.tight_layout()
                    plt.show()
    return
def process_case(case, path, progress_list, lock, core_index):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    datalist = pd.DataFrame()
    dirpath = os.path.join(path, case)
    filepath = os.path.join(dirpath, "MNI152_TDCS_2_scalar.msh")
    
    electrode_elements = {}
    try:
        # Load the mesh file using gmsh
        gmsh.open(filepath)
    except Exception as e:
        return None, f"Error reading mesh file {filepath}: {e}"
    
    entities = gmsh.model.getEntities()
    total_steps = len(entities) if entities else 1  # For estimating progress
    for i, entity in enumerate(entities):
        entity_dim, entity_tag = entity
        if entity_dim == 2:  # Surface elements are 2D
            try:
                element_types, element_tags, node_tags = gmsh.model.mesh.getElements(entity_dim, entity_tag)
                for etype, etags in zip(element_types, element_tags):
                    if etype == 2:  # 2 represents triangular elements (commonly used for surfaces)
                        physical_tags = gmsh.model.getPhysicalGroupsForEntity(entity_dim, entity_tag)
                        for tag in physical_tags:
                            if tag == 1501 or tag == 1502 or tag == 2101 or tag == 2102:  # Extract the 
                                if tag not in electrode_elements:
                                    electrode_elements[tag] = []
                                electrode_elements[tag].extend(etags)
            except Exception as e:
                return None, f"Error processing entity {entity_tag} in file {filepath}: {e}"

        # Update individual core's progress in the custom progress bar
        with lock:
            progress_list[core_index] = (i + 1) / total_steps  # Track percent progress
            display_progress(progress_list)

    # Prepare the data dictionary
    try:
        data = {
            'Name': case,
            'El_1-1': len(electrode_elements.get(1502, [])),
            'El_1-2': len(electrode_elements.get(1501, [])),
            'El_2-1': len(electrode_elements.get(2101, [])),
            'El_2-2': len(electrode_elements.get(2102, []))
        }
    except KeyError as e:
        return None, f"Missing electrode tag data in file {filepath}: {e}"

    dataframe = pd.DataFrame(data, index=[0])
    csv_name = os.path.join(dirpath, 'Electrode_Geometry.csv')
    dataframe.to_csv(csv_name, index=False)  # Corrected to save CSV properly
    
    gmsh.finalize()

    return dataframe, None

def display_progress(progress_list):
    # Move cursor up to the starting point of custom progress bars
    print("\033[F" * len(progress_list), end="")  # Move up by the number of progress bars
    
    # Display each core's progress without resetting to 0%
    for i, progress in enumerate(progress_list):
        if progress > 0:  # Only display if progress has started
            bar_length = int(progress * 50)  # 50-character long bar
            progress_bar = f"Core {i+1}: [{'#' * bar_length}{'.' * (50 - bar_length)}] {progress * 100:.1f}%"
            print(f"\033[K{progress_bar}")  # Clear the line before printing

def get_resolution(path, core_fraction=6.0):
    output_dir = os.listdir(path)
    if os.path.exists(os.path.join(path,'Combined_Electrode_Geometry.csv')):
        cdf = pd.read_csv(os.path.join(path,'Combined_Electrode_Geometry.csv'))
        plot_data(cdf)
        return cdf
    datalist = pd.DataFrame()
    num_cases = len(output_dir)
    
    # Define the maximum number of cores to use based on fraction
    max_cores = max(1, int(multiprocessing.cpu_count() * core_fraction))
    
    # Initialize a list to track progress for each core (or process)
    manager = multiprocessing.Manager()
    progress_list = manager.list([0] * max_cores)  # Each core starts at 0% progress
    lock = manager.Lock()

    # Create the main progress bar
    progress_bar = tqdm(total=num_cases, dynamic_ncols=True, position=0, leave=True, desc='Processing Cases...')
    
    # Use ProcessPoolExecutor with limited cores
    with ProcessPoolExecutor(max_workers=1) as executor:
        future_to_case = {
            executor.submit(process_case, case, path, progress_list, lock, core_index % max_cores): case 
            for core_index, case in enumerate(output_dir)
        }

        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                dataframe, error = future.result()
                if error:
                    tqdm.write(error)
                elif dataframe is not None:
                    datalist = pd.concat([datalist, dataframe], ignore_index=True)
            except Exception as e:
                tqdm.write(f"Unexpected error processing case {case}: {e}")
            finally:
                progress_bar.update(1)

    # Close the main progress bar once done
    progress_bar.close()
    
    # Save the combined DataFrame if needed
    datalist.to_csv(os.path.join(path, 'Combined_Electrode_Geometry.csv'), index=False)
    plot_data(datalist)
# Usage example:
path = '/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Output'

get_resolution(path, core_fraction=0.3)
