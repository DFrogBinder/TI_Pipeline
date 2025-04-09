import os
import pandas as pd

def save_folders_to_excel(directory, output_excel):
    """
    Scans the given directory for all folders (subdirectories)
    and saves their names into an Excel spreadsheet.
    
    Parameters:
    - directory (str): The path of the directory to scan.
    - output_excel (str): The path (and filename) for the Excel file to create.
    """
    # Get all items in the directory, filter out only those that are directories
    folders = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
    
    # Create a DataFrame with the list of folders
    df = pd.DataFrame(folders, columns=['Folder Name'])
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_excel, index=False)
    print(f"Saved {len(folders)} folder names to {output_excel}")

# Example usage:
if __name__ == "__main__":
    # Specify the directory you want to scan and the output Excel file name
    dir_to_scan = "/home/boyan/sandbox/Jake_Data/ti_dataset"  # Change this path as needed
    output_file = "ti_dataset.xlsx"
    save_folders_to_excel(dir_to_scan, output_file)
