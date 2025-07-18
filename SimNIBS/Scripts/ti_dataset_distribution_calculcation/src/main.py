import os
import pandas as pd
from .load_data    import filter_subjects
from .describe     import compute_numeric_stats, compute_categorical_stats
from .visualize    import plot_histogram, plot_bar

def main():
    # --- Paths ---
    raw_dir       = os.path.join('data','raw')
    processed_dir = os.path.join('data','processed')
    tables_dir    = os.path.join('outputs','tables')
    plots_dir     = os.path.join('outputs','plots')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(tables_dir,    exist_ok=True)
    os.makedirs(plots_dir,     exist_ok=True)

    # --- Step 1: filter & merge ---
    filtered, merged = filter_subjects(
        big_file    = os.path.join(raw_dir, 'CamCAN.xlsx'),
        subset_file = os.path.join(raw_dir, 'ti_subjects.xlsx'),
        output_filtered = os.path.join(processed_dir, 'filtered_subject_subset.xlsx'),
        output_merged   = os.path.join(processed_dir, 'merged_subject_subset.xlsx')
    )

    # --- Step 2: load the merged sheet you care about ---
    # If you have multiple sheets, you could loop here.
    df = pd.read_excel(merged, sheet_name=0)

    # --- Step 3: descriptive stats ---
    numeric_cols = ['age']            # add more numeric columns here
    cat_cols     = ['gender_code']    # add more categorical columns here

    num_stats = compute_numeric_stats(df, numeric_cols)
    num_stats.to_csv(os.path.join(tables_dir, 'numeric_stats.csv'))

    cat_stats = compute_categorical_stats(df, cat_cols)
    for col, stats_df in cat_stats.items():
        stats_df.to_csv(os.path.join(tables_dir, f'{col}_stats.csv'))

    # --- Step 4: plots ---
    for col in numeric_cols:
        plot_histogram(df, col, os.path.join(plots_dir, f'{col}_distribution.png'))
    for col in cat_cols:
        plot_bar(df, col, os.path.join(plots_dir, f'{col}_counts.png'))

    print("✅ Analysis complete. Check outputs/ for tables & plots.")

if __name__ == '__main__':
    main()
