import argparse
import subprocess

def convert_poly_to_msh(poly_file_path, msh_file_path, mesh_size=2.5):
    """
    Convert a .poly file to a .msh file using gmsh.

    Parameters:
    - poly_file_path: Path to the input .poly file.
    - msh_file_path: Path where the output .msh file will be saved.
    - mesh_size: Characteristic length of the mesh elements.

    Returns:
    - None
    """
    gmsh_command = [
        "gmsh", poly_file_path, "-2",
        "-clmax", str(mesh_size), "-o", msh_file_path
    ]
    subprocess.run(gmsh_command, check=True)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert .poly files to .msh format for use with SimNIBS.")
    
    # Add arguments
    parser.add_argument("poly_file", help="Path to the input .poly file")
    parser.add_argument("msh_file", help="Path for the output .msh file")
    parser.add_argument("-s", "--size", type=float, default=2.0, help="Characteristic length of the mesh elements")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert the file
    convert_poly_to_msh(args.poly_file, args.msh_file, args.size)

if __name__ == "__main__":
    main()
