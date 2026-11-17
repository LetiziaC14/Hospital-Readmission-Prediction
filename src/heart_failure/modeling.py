"""Definitions and fitting of candidate prediction pipelines."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def make_pipeline(name: str, preprocessor, seed: int) -> Pipeline:
    """Create one named modelling pipeline from the project comparison."""
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(class_weight="balanced", random_state=seed),
        "svm": SVC(class_weight="balanced", probability=True, random_state=seed),
    }
    if name not in estimators:
        raise ValueError(f"Unknown pipeline {name!r}. Choose from: {', '.join(estimators)}.")
    return Pipeline([("preprocessor", preprocessor), ("model", estimators[name])])
