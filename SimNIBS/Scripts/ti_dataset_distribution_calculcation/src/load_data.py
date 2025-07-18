import pandas as pd

def filter_subjects(
    big_file: str,
    subset_file: str,
    output_filtered: str,
    output_merged: str,
    id_col: str = None
) -> tuple[str, str]:
    """
    Filters and merges subject data.

    - big_file:       Excel workbook with multiple sheets (e.g. CamCAN.xlsx)
    - subset_file:    Excel file whose FIRST column lists 'Subjects'
    - output_filtered: path for the filtered‐only workbook
    - output_merged:  path for the left‐merged workbook
    - id_col:         name of the ID column to use; defaults to the first
                      column name in subset_file.

    Returns (output_filtered, output_merged).
    """
    # 1) load your subset of IDs
    subset_df = pd.read_excel(subset_file)
    if id_col is None:
        id_col = subset_df.columns[0]
    elif id_col not in subset_df.columns:
        raise KeyError(
            f"ID column '{id_col}' not found in subset file; "
            f"available columns are {list(subset_df.columns)}"
        )

    # 2) read all sheets from big_file
    sheets = pd.read_excel(big_file, sheet_name=None)

    # 3) write the filtered workbook
    with pd.ExcelWriter(output_filtered, engine='openpyxl') as writer:
        for name, df in sheets.items():
            # rename whatever the first column is to id_col
            raw_id = df.columns[0]
            df = df.rename(columns={raw_id: id_col})
            # now you can filter safely
            filtered_df = df[df[id_col].isin(subset_df[id_col])]
            filtered_df.to_excel(writer, sheet_name=name, index=False)

    # 4) write the merged workbook
    with pd.ExcelWriter(output_merged, engine='openpyxl') as writer:
        for name, df in sheets.items():
            raw_id = df.columns[0]
            df = df.rename(columns={raw_id: id_col})
            merged_df = subset_df.merge(df, on=id_col, how='left')
            merged_df.to_excel(writer, sheet_name=name, index=False)

    return output_filtered, output_merged

