import matplotlib.pyplot as plt

def plot_histogram(df, col, output_path, bins=30):
    plt.figure()
    df[col].dropna().hist(bins=bins)
    plt.title(f'{col} distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_bar(df, col, output_path):
    plt.figure()
    df[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'{col} counts')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
