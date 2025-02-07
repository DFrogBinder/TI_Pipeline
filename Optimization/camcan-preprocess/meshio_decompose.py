import meshio
import numpy as np
import os

# Path to the .msh file (Change this if necessary)
file_path = "head.msh"

# Load the mesh file
try:
    mesh = meshio.read(file_path)
    print(f"Successfully loaded mesh: {file_path}")
except Exception as e:
    print(f"Error loading mesh: {e}")
    mesh = None

# Ensure mesh is loaded before proceeding
if mesh:
    # Create output directory
    output_dir = os.path.join('Output',os.path.splitext(file_path)[0] + "_msh_export")
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Mesh Element Types and Counts ===\n")
    for cell_type, elements in mesh.cells_dict.items():
        print(f"  - {cell_type}: {len(elements)} elements")

    print("\n=== Extracting Physical Groups and Saving as MSH ===\n")

    # Extract physical groups
    physical_groups = mesh.cell_data_dict.get("gmsh:physical", {})

    # Print all detected physical groups
    print("Physical Groups Found:", physical_groups)

    for cell_type, group_ids in physical_groups.items():
        if cell_type not in mesh.cells_dict:
            print(f"Skipping {cell_type} as it's not found in cells_dict.")
            continue

        unique_groups = np.unique(group_ids)
        print(f"Processing {cell_type} elements with {len(unique_groups)} unique physical groups.")

        for group_id in unique_groups:
            group_indices = np.where(group_ids == group_id)[0]

            if len(group_indices) == 0:
                print(f"  - Skipping Physical Group {group_id} (No elements)")
                continue

            selected_cells = mesh.cells_dict[cell_type][group_indices]
            cell_data_dict = {"gmsh:physical": [group_ids[group_indices].tolist()]} if cell_type in physical_groups else {}

            extracted_mesh = meshio.Mesh(
                points=mesh.points,
                cells=[(cell_type, selected_cells)],
                cell_data=cell_data_dict
            )

            output_file = os.path.join(output_dir, f"physical_group_{group_id}.msh")
            try:
                meshio.write(output_file, extracted_mesh, file_format="gmsh")
                print(f"  - Saved Physical Group {group_id} to {output_file}")
            except Exception as e:
                print(f"  - Error saving Physical Group {group_id}: {e}")

    print("\n=== Extracting and Saving All Elements (Regardless of Groups) ===\n")

    # Extract all elements even if they are not in a physical group
    for cell_type, elements in mesh.cells_dict.items():
        if cell_type not in physical_groups:
            print(f"Extracting {cell_type} elements without physical groups.")

            extracted_mesh = meshio.Mesh(
                points=mesh.points,
                cells=[(cell_type, elements)]
            )

            output_file = os.path.join(output_dir, f"all_{cell_type}.msh")
            try:
                meshio.write(output_file, extracted_mesh, file_format="gmsh")
                print(f"  - Saved all {cell_type} elements to {output_file}")
            except Exception as e:
                print(f"  - Error saving ungrouped elements: {e}")

    print("\n=== Extraction and Export Complete! ===")
