import pandas as pd
from scipy.stats import skew, kurtosis

def compute_numeric_stats(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Returns a DataFrame of: count, mean, median, std, var,
    min, 25%, 75%, max, skewness, kurtosis for each numeric column.
    """
    stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        stats[col] = {
            'count': s.count(),
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'variance': s.var(),
            'min': s.min(),
            '25%': s.quantile(0.25),
            '75%': s.quantile(0.75),
            'max': s.max(),
            'skewness': skew(s),
            'kurtosis': kurtosis(s)
        }
    return pd.DataFrame(stats).T

def compute_categorical_stats(df: pd.DataFrame, cat_cols: list) -> dict:
    """
    Returns a dict mapping each categorical column to a DataFrame of:
    count and proportion for each category (incl. NaN).
    """
    out = {}
    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        props  = df[col].value_counts(normalize=True, dropna=False)
        out[col] = pd.DataFrame({'count': counts, 'proportion': props})
    return out
