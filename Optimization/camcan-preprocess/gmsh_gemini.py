import gmsh

gmsh.initialize()

gmsh.open('head.msh')

physical_group_dim = 2
physical_group_tag = 11

# 1. Get the geometric entity associated with the physical group:
entity_dim_tags = gmsh.model.getEntitiesForPhysicalGroup(physical_group_dim, physical_group_tag)

# Check if any entities are associated with the physical group at all
if entity_dim_tags: # If the list is not empty
    entity_dim, entity_tag = entity_dim_tags[0], entity_dim_tags[1] # Extract the dimension and tag

    # 2. Remove the GEOMETRIC ENTITY:
    try:
        gmsh.model.removeEntity(entity_dim, entity_tag)
    except Exception as e:
        print(f"Warning: Could not remove entity {entity_dim}, {entity_tag}: {e}")

    # 3. Get the elements (they should be empty now):
    element_types, element_tags, _ = gmsh.model.mesh.getElements(physical_group_dim, physical_group_tag)

    # 4. Remove the ELEMENTS (if any are left - unlikely):
    if element_tags:
        for element_type, tag_array in zip(element_types, element_tags):
            for tag in tag_array:
                try:
                    gmsh.model.mesh.removeElement(element_type, tag)
                except Exception as e:
                    print(f"Warning: Could not remove element: {e}")

# 5. Remove the physical group itself (optional, but recommended):
gmsh.model.removePhysicalGroup(physical_group_dim, physical_group_tag)

gmsh.finalize()