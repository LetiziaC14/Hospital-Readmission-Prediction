"""Feature preprocessing mirroring the notebook without train/test leakage."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ORDINAL_LEVELS = {
    "gender": ["Male", "Female"],
    "admission.way": ["NonEmergency", "Emergency"],
    "ageCat": [
        "(21,29]", "(29,39]", "(39,49]", "(49,59]",
        "(59,69]", "(69,79]", "(79,89]", "(89,110]",
    ],
    "Killip.grade": ["I", "II", "III", "IV"],
    "NYHA.cardiac.function.classification": ["II", "III", "IV"],
}

ONE_HOT_REFERENCE = {
    "DestinationDischarge": "Unknown",
    "admission.ward": "Others",
    "occupation": "Others",
    "discharge.department": "Others",
    "type.of.heart.failure": None,
}


def split_features_target(df: pd.DataFrame, target: str, test_size: float, seed: int):
    """Create the common reproducible stratified train/test split."""
    features = df.drop(columns=target)
    outcome = df[target].astype(int)
    return train_test_split(
        features, outcome, test_size=test_size, random_state=seed, stratify=outcome
    )


@dataclass
class NotebookPreprocessor:
    """Encode, MICE-impute and scale using parameters learned on training data."""

    seed: int = 42
    max_iter: int = 25
    encoded_columns: list[str] = field(default_factory=list)
    scaled_columns: list[str] = field(default_factory=list)
    imputer: IterativeImputer | None = None
    scaler: StandardScaler | None = None

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, levels in ORDINAL_LEVELS.items():
            if column in result:
                categorical = pd.Categorical(result[column], categories=levels, ordered=True)
                result[column] = pd.Series(categorical.codes, index=result.index).replace(-1, np.nan)
        for column, reference in ONE_HOT_REFERENCE.items():
            if column not in result:
                continue
            result = pd.get_dummies(result, columns=[column], prefix=column, dtype=int)
            reference_column = f"{column}_{reference}"
            if reference is not None and reference_column in result:
                result = result.drop(columns=reference_column)
        remaining = result.select_dtypes(exclude="number").columns.tolist()
        if remaining:
            raise TypeError(f"Categorical columns without an encoding rule: {remaining}")
        return result.astype(float)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = self._encode(frame)
        self.encoded_columns = encoded.columns.tolist()
        self.imputer = IterativeImputer(random_state=self.seed, max_iter=self.max_iter)
        imputed = pd.DataFrame(
            self.imputer.fit_transform(encoded), columns=self.encoded_columns, index=encoded.index
        )
        self.scaled_columns = imputed.columns[imputed.nunique(dropna=False) > 2].tolist()
        self.scaler = StandardScaler()
        if self.scaled_columns:
            imputed.loc[:, self.scaled_columns] = self.scaler.fit_transform(
                imputed[self.scaled_columns]
            )
        return imputed

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None or self.scaler is None or not self.encoded_columns:
            raise RuntimeError("NotebookPreprocessor must be fitted before transform().")
        encoded = self._encode(frame).reindex(columns=self.encoded_columns, fill_value=0)
        imputed = pd.DataFrame(
            self.imputer.transform(encoded), columns=self.encoded_columns, index=encoded.index
        )
        if self.scaled_columns:
            imputed.loc[:, self.scaled_columns] = self.scaler.transform(
                imputed[self.scaled_columns]
            )
        return imputed
