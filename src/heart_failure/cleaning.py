"""Clinical data-cleaning transformations.

Move validated rules from the original notebook here as small, documented functions.
"""

import pandas as pd


def drop_columns_above_missingness(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns whose missing-data proportion is greater than `threshold`."""
    columns = df.columns[df.isna().mean() > threshold]
    return df.drop(columns=columns)


def drop_rows_above_missingness(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop rows whose missing-data proportion is greater than `threshold`."""
    return df.loc[df.isna().mean(axis=1) <= threshold].copy()
