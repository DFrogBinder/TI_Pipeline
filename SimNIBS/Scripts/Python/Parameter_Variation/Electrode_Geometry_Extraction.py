import gmsh
import pandas as pd
from tqdm import tqdm
import os

def get_resolution(path):
    
    datalist = pd.DataFrame()
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    output_dir = os.listdir(path)
    # Create the progress bar with dynamic updates
    progress_bar = tqdm(total=len(output_dir), dynamic_ncols=True, position=0, leave=True, desc='Processing Cases...')
    
    for case in output_dir:
        dirpath = os.path.join(path, case)
        filepath = os.path.join(dirpath, "MNI152_TDCS_2_scalar.msh")
        
        try:
            # Load the mesh file using gmsh
            gmsh.open(filepath)
        except Exception as e:
            print(f"Error reading mesh file {filepath}: {e}")
            continue
        
        # Get the mesh elements and their tags
        electrode_elements = {}
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
                                # Assuming electrode tags have a specific value, e.g., greater than 1000
                                if tag == 1501 or tag == 1502:  # Adjust the condition based on your electrode tagging
                                    if tag not in electrode_elements:
                                        electrode_elements[tag] = []
                                    electrode_elements[tag].extend(etags)
                except Exception as e:
                    tqdm.write(f"Error processing entity {entity_tag} in file {filepath}: {e}")
                    continue
        
        # Print the number of elements per electrode
        for tag, elements in electrode_elements.items():
            tqdm.write(f"Case: {case}, Electrode {tag}: {len(elements)} elements")

        data = {
            'Name' : case,
            'El_1' : len(electrode_elements[1502]),
            'El_2' : len(electrode_elements[1501])
        }
        
        dataframe = pd.DataFrame(data, index=[0])
        csv_name = os.path.join(dirpath,'Electrode_Geometry.csv')
        dataframe.to_csv()
        
        datalist = pd.concat([datalist, dataframe], ignore_index=True)
        
        # Update the progress bar
        progress_bar.update(1)
    
    # Close the progress bar once done
    progress_bar.close()
    gmsh.finalize()
    

get_resolution('/home/boyan/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/Output')
