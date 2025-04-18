import meshio
import numpy as np
import os

# Path to the .msh file (Change this to the correct file path)
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
    # Get the directory of the input file
    output_dir = os.path.splitext(file_path)[0] + "_stl_export"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Extracting Physical Groups and Saving as STL ===\n")

    # Extract physical groups
    physical_groups = mesh.cell_data_dict.get("gmsh:physical", {})

    for cell_type, group_ids in physical_groups.items():
        unique_groups = np.unique(group_ids)
        print(f"Processing {cell_type} elements with {len(unique_groups)} unique physical groups.")

        for group_id in unique_groups:
            # Extract indices belonging to this group
            group_indices = np.where(group_ids == group_id)[0]

            # Check if there are elements in this group
            if len(group_indices) == 0:
                print(f"  - Skipping Physical Group {group_id} (No elements)")
                continue

            # Use NumPy slicing instead of list comprehension (memory-efficient)
            selected_cells = mesh.cells_dict[cell_type][group_indices]

            # Create a new mesh for this physical group
            extracted_mesh = meshio.Mesh(
                points=mesh.points,  # Use original points
                cells=[(cell_type, selected_cells)]  # Use sliced array instead of list comprehension
            )

            # Define output file path
            output_file = os.path.join(output_dir, f"physical_group_{group_id}.stl")

            # Save to STL
            try:
                meshio.write(output_file, extracted_mesh)
                print(f"  - Saved Physical Group {group_id} to {output_file}")
            except Exception as e:
                print(f"  - Error saving Physical Group {group_id}: {e}")

    print("\n=== Extraction and Export Complete! ===")