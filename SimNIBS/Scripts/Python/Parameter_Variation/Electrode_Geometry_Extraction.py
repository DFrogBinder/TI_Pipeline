import gmsh
import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

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
                            if tag == 1501 or tag == 1502 or tag == 2101 or tag == 2102:  # Adjust based on your electrode tagging
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

def get_resolution(path, core_fraction=1.0):
    output_dir = os.listdir(path)
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
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
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

# Usage example:
get_resolution('/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Output', core_fraction=0.3)
