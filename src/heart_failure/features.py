"""Feature preprocessing and train/test splitting."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target: str, test_size: float, seed: int):
    """Create a stratified train/test split."""
    features = df.drop(columns=target)
    outcome = df[target]
    return train_test_split(features, outcome, test_size=test_size, random_state=seed, stratify=outcome)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Build a preprocessing transformer fitted only within the modelling pipeline."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)])
