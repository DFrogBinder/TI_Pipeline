import meshio

def remove_last_two_groups(input_mesh, output_mesh):
    # Read the mesh
    mesh = meshio.read(input_mesh)

    # Check if physical and geometrical groups exist
    if "gmsh:physical" not in mesh.cell_data or "gmsh:geometrical" not in mesh.cell_data:
        raise ValueError("Mesh does not contain physical or geometrical groups.")

    # Get unique physical group IDs
    physical_groups = mesh.cell_data["gmsh:physical"]
    geometrical_groups = mesh.cell_data["gmsh:geometrical"]

    # Extract unique physical and geometrical IDs
    unique_physical = sorted(set(g for group in physical_groups for g in group))
    unique_geometrical = sorted(set(g for group in geometrical_groups for g in group))

    if len(unique_physical) < 2:
        raise ValueError("Mesh has fewer than two physical groups.")

    # Identify the last two physical group IDs to remove
    groups_to_remove = set(unique_physical[-2:])

    # Also find corresponding geometrical groups to remove
    geometrical_to_remove = set()
    for phys_group, geom_group in zip(physical_groups, geometrical_groups):
        for p, g in zip(phys_group, geom_group):
            if p in groups_to_remove:
                geometrical_to_remove.add(g)

    # Remove elements belonging to these groups
    new_cells = []
    new_cell_data = {"gmsh:physical": [], "gmsh:geometrical": []}

    for cell_block, phys_group, geom_group in zip(mesh.cells, physical_groups, geometrical_groups):
        filtered_cells = [
            c for c, p, g in zip(cell_block.data, phys_group, geom_group) 
            if p not in groups_to_remove and g not in geometrical_to_remove
        ]
        filtered_phys = [p for p, g in zip(phys_group, geom_group) if p not in groups_to_remove and g not in geometrical_to_remove]
        filtered_geom = [g for g in geom_group if g not in geometrical_to_remove]

        if filtered_cells:
            new_cells.append(meshio.CellBlock(cell_block.type, filtered_cells))
            new_cell_data["gmsh:physical"].append(filtered_phys)
            new_cell_data["gmsh:geometrical"].append(filtered_geom)

    # Create new mesh
    new_mesh = meshio.Mesh(
        points=mesh.points,
        cells=new_cells,
        cell_data=new_cell_data
    )

    # Write the new mesh to file
    meshio.write(output_mesh, new_mesh,file_format="gmsh")
    print(f"Mesh saved to {output_mesh}")

# Example usage
input_file = "head.msh"
output_file = "filtered_output.msh"
remove_last_two_groups(input_file, output_file)
