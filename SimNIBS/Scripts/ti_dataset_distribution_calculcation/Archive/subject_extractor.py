import pandas as pd

def extract_subject_subset_multiple_outputs(big_file, subset_file, filtered_output_file, merged_output_file):
    """
    Processes a big multi-sheet Excel file and a subset Excel file containing subject names.
    Outputs two Excel files:
    
    1. filtered_output_file: Each sheet contains only the rows from the big dataset that 
       match the subject names in the subset (simple filtering).
       
    2. merged_output_file: Each sheet is a left merge between the subset and the corresponding
       sheet of the big dataset. This ensures every subject in the subset appears in each sheet,
       with missing data as NaN where a subject does not exist.
    
    Parameters:
    - big_file (str): Path to the Excel file with multiple sheets where the first column 
                      contains the subject identifiers.
    - subset_file (str): Path to the Excel file (single sheet) containing subject identifiers
                         in its first column.
    - filtered_output_file (str): Path for the output Excel file with filtered sheets.
    - merged_output_file (str): Path for the output Excel file with merged (left join) sheets.
    """
    # Load the subset file and standardize subject names
    subset_df = pd.read_excel(subset_file)
    subject_col_subset = subset_df.columns[0]
    subset_df[subject_col_subset] = subset_df[subject_col_subset].astype(str).str.strip()
    # Extract a unique list of subject names (in case you need it)
    subject_names = subset_df[subject_col_subset].unique().tolist()
    
    # Read all sheets from the big dataset (as a dictionary of DataFrames)
    all_sheets = pd.read_excel(big_file, sheet_name=None)
    
    # Dictionaries to hold output data for both methods
    filtered_sheets = {}
    merged_sheets = {}
    
    for sheet_name, df in all_sheets.items():
        if not df.empty:
            # Use the first column of the sheet as the subject identifier.
            subject_col_big = df.columns[0]
            # Convert the subject names in the big dataset to strings and strip extra spaces.
            df[subject_col_big] = df[subject_col_big].astype(str).str.strip()
            
            # --- Method 1: Original Filtering ---
            filtered_df = df[df[subject_col_big].isin(subject_names)]
            filtered_sheets[sheet_name] = filtered_df
            
            # --- Method 2: Left Merge ---
            # Left merge using the subject names from the subset, ensuring every subject appears.
            merged_df = pd.merge(subset_df, df, how='left',
                                 left_on=subject_col_subset, right_on=subject_col_big)
            # Optionally remove the duplicate subject column if the names of the subject columns differ.
            if subject_col_subset != subject_col_big:
                merged_df.drop(columns=[subject_col_big], inplace=True)
            merged_sheets[sheet_name] = merged_df
        else:
            # If a sheet is empty, pass it as-is.
            filtered_sheets[sheet_name] = df
            merged_sheets[sheet_name] = df
    
    # Write the filtered data to an Excel file.
    with pd.ExcelWriter(filtered_output_file, engine='openpyxl') as writer:
        for sheet_name, filtered_df in filtered_sheets.items():
            filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Write the merged (left join) data to a separate Excel file.
    with pd.ExcelWriter(merged_output_file, engine='openpyxl') as writer:
        for sheet_name, merged_df in merged_sheets.items():
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Filtered output saved to {filtered_output_file}")
    print(f"Merged output saved to {merged_output_file}")

# Example usage:
if __name__ == "__main__":
    # Specify the paths for the big dataset, subset file, and desired output file.
    big_dataset_file = "CamCAN.xlsx"
    subset_file = "ti_subjects.xlsx"
    output_file = "ti_dataset-with-data.xlsx"
    
    filtered_output_file = "filtered_subject_subset.xlsx"
    merged_output_file = "merged_subject_subset.xlsx"
    
    extract_subject_subset_multiple_outputs(big_dataset_file, subset_file,
                                              filtered_output_file, merged_output_file)
