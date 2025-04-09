import pandas as pd
import matplotlib.pyplot as plt

def calculate_demographics_distribution(file_path, gender_col='gender_code', age_col='age'):
    """
    Reads an Excel file with potentially multiple sheets, checks each sheet for the presence of 
    gender and age data, and prints the following for each relevant sheet:
    
      - The gender distribution (as counts and percentages)
      - Basic descriptive statistics for the age column
      - A histogram of the age distribution

    Parameters:
      file_path (str): Path to the Excel file.
      gender_col (str): Column name for the gender information.
      age_col (str): Column name for the age information.
    """
    # Read all sheets from the Excel file into a dictionary of DataFrames.
    sheets = pd.read_excel(file_path, sheet_name=None)
    
    for sheet_name, df in sheets.items():
        print(f"Processing sheet: {sheet_name}")
        # Check if the sheet has both the gender and age columns
        if gender_col in df.columns and age_col in df.columns:
            
            # Calculate and print the gender distribution (counts)
            gender_counts = df[gender_col].value_counts(dropna=False)
            print("Gender distribution (counts):")
            print(gender_counts)
            
            # Calculate and print the gender distribution (percentages)
            gender_percentages = df[gender_col].value_counts(normalize=True, dropna=False) * 100
            print("\nGender distribution (percentages):")
            print(gender_percentages.round(2))
            
            # Calculate descriptive statistics for age
            print("\nAge distribution:")
            age_stats = df[age_col].describe()
            print(age_stats)
            
            # Optional: Generate an age distribution histogram
            plt.figure()
            df[age_col].hist(bins=20)
            plt.title(f"Age Distribution - {sheet_name}")
            plt.xlabel(age_col)
            plt.ylabel("Frequency")
            # plt.show()
            plt.savefig(f"{sheet_name}_age_distribution.png")  # Save the histogram as an image
            plt.close()
            
            print("\n" + "-"*40 + "\n")
        else:
            print(f"Sheet '{sheet_name}' does not have both columns '{gender_col}' and '{age_col}'. Skipping.\n")

# Example usage:
if __name__ == "__main__":
    # Specify the path to the Excel file that contains the dataset.
    file_path = "filtered_subject_subset.xlsx"  # or the name of the file you want to analyze
    calculate_demographics_distribution(file_path)
