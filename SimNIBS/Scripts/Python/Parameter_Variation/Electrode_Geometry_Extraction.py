import gmsh
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def create_plots(dataframe):
    print(dataframe)
    return 1

def process_case(case, path):
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
    for entity in entities:
        entity_dim, entity_tag = entity
        if entity_dim == 2:  # Surface elements are 2D
            try:
                element_types, element_tags, node_tags = gmsh.model.mesh.getElements(entity_dim, entity_tag)
                for etype, etags in zip(element_types, element_tags):
                    if etype == 2:  # 2 represents triangular elements (commonly used for surfaces)
                        physical_tags = gmsh.model.getPhysicalGroupsForEntity(entity_dim, entity_tag)
                        for tag in physical_tags:
                            # if tag > 1000:  # Adjust based on your electrode tagging
                                # if tag not in electrode_elements:
                                #     electrode_elements[tag] = []
                                # electrode_elements[tag].extend(etags)
                            if tag == 1501 or tag == 1502 or tag == 2102 or tag == 2102:  # Adjust based on your electrode tagging
                                if tag not in electrode_elements:
                                    electrode_elements[tag] = []
                                electrode_elements[tag].extend(etags)
            except Exception as e:
                return None, f"Error processing entity {entity_tag} in file {filepath}: {e}"

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

def get_resolution(path,save=True):
    output_dir = os.listdir(path)
    datalist = pd.DataFrame()
    
    # Create the progress bar with dynamic updates
    progress_bar = tqdm(total=len(output_dir), dynamic_ncols=True, position=0, leave=True, desc='Processing Cases...')
    
    # Calculate 30% of available cores
    num_cores = max(1, int(multiprocessing.cpu_count() * 0.5))
    
    # Use ProcessPoolExecutor with a limited number of workers
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_case = {executor.submit(process_case, case, path): case for case in output_dir}

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

    # Close the progress bar once done
    progress_bar.close()
    
    # Save the combined DataFrame if needed
    tqdm.write(f'Combined Electrode Geometry is saved to: {path}')
    datalist.to_csv(os.path.join(path, 'Combined_Electrode_Geometry.csv'), index=False)

# Usage example:
get_resolution('/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Output')
