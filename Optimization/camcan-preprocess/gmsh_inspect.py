import gmsh
import numpy as np


def get_physical_groups():
    physical_groups = gmsh.model.getPhysicalGroups()
    
    return physical_groups

def get_model_stats(mesh):
    
    # Get mesh nodes and elements count
    node_count = len(mesh.getNodes()[0])
    element_count = len(mesh.getElements()[1][0])
    print("Mesh Statistics:")
    print(f"Number of nodes: {node_count}")
    print(f"Number of elements: {element_count}")
   
def load_msh_file(file_path):
    gmsh.initialize()
    gmsh.open(file_path)

    # Get information about the mesh
    model = gmsh.model
    mesh = model.mesh
    
    get_model_stats(mesh)
    initial_physical_groups = get_physical_groups()
    print("Initial physical groups number:", len(initial_physical_groups))

    # physical_groups = gmsh.model.getPhysicalGroups()
       
    defined_elements_types = gmsh.model.mesh.getElementTypes()
    print(f'Defined elements types for mesh: {defined_elements_types}')

    # Choose which physical group to remove (e.g., if tag is 1 and dimension is 2 for surfaces)
    dimension_to_remove = 2  # 0: Points, 1: Edges, 2: Surfaces, 3: Volumes
    tag_to_remove = [11,12]  # Replace with the actual tag of the region you want to remove

    for tag in tag_to_remove:
                
        entity_dim_tags = gmsh.model.getEntitiesForPhysicalGroup(dimension_to_remove, tag)
        element_types,element_tags = gmsh.model.mesh.getElements(dimension_to_remove, tag)
        
        for element_type, tags in zip(element_types,element_tags):
            for tag in tags:
                gmsh.model.removeElements(element_type,tag)

        for entity_dim, entity_tag in entity_dim_tags:
            try:
                gmsh.model.removeEntities(entity_dim, entity_tag)
            except Exception as e:
                print(f"Warning: Could not remove entity {entity_dim}, {entity_tag}: {e}")
        
        gmsh.model.removePhysicalGroup(dimension_to_remove, tag)

        
    gmsh.write("modified_model.msh")
    print("Modified model saved")
    
    get_model_stats(mesh)
    final_physical_groups = get_physical_groups()
    print("Final physical groups number:", len(final_physical_groups))
    
    gmsh.finalize()

# Example usage
load_msh_file("head.msh")