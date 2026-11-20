"""Feature-selection and modelling strategies from the original notebook."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor

PIPELINES = [
    "lasso_logistic",
    "elasticnet_svm",
    "random_forest_gini",
    "random_forest_permutation",
]


@dataclass
class TrainingResult:
    model: object
    selected_features: list[str]
    best_params: dict = field(default_factory=dict)
    cv_score: float | None = None
    selection_details: dict = field(default_factory=dict)


def _drop_high_vif(frame: pd.DataFrame, threshold: float = 10.0):
    selected = frame.copy()
    dropped = []
    while selected.shape[1] > 1:
        values = np.asarray(selected, dtype=float)
        vif = pd.Series(
            [variance_inflation_factor(values, i) for i in range(values.shape[1])],
            index=selected.columns,
        ).replace([np.inf, -np.inf], np.inf)
        if vif.max() <= threshold:
            break
        feature = str(vif.idxmax())
        dropped.append(feature)
        selected = selected.drop(columns=feature)
    return selected, dropped


def train_lasso_logistic(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int, selection_frequency: float = 0.60
) -> TrainingResult:
    X_vif, dropped = _drop_high_vif(X_train)
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 0, 100), cv=10, random_state=seed, max_iter=10000
    ).fit(X_vif, y_train)
    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=15, random_state=seed)
    counts = np.zeros(X_vif.shape[1], dtype=float)
    y_array = np.asarray(y_train)
    for train_index, _ in repeated_cv.split(X_vif, y_array):
        selector = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=seed)
        selector.fit(X_vif.iloc[train_index], y_array[train_index])
        counts += selector.coef_ != 0
    frequencies = counts / repeated_cv.get_n_splits()
    features = X_vif.columns[frequencies >= selection_frequency].tolist()
    if not features:
        features = [str(X_vif.columns[np.argmax(frequencies)])]
    model = LogisticRegression(max_iter=10000, class_weight="balanced", random_state=seed)
    model.fit(X_train[features], y_train)
    return TrainingResult(
        model=model,
        selected_features=features,
        best_params={"lasso_alpha": float(lasso_cv.alpha_)},
        selection_details={
            "vif_dropped": dropped,
            "selection_frequency": {
                str(feature): float(frequency)
                for feature, frequency in zip(X_vif.columns, frequencies)
            },
        },
    )


def train_elasticnet_svm(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_jobs: int = -1
) -> TrainingResult:
    selector = ElasticNetCV(
        alphas=np.logspace(-4, 0, 20), l1_ratio=np.arange(0.1, 0.9, 0.1),
        cv=5, random_state=seed, max_iter=10000, tol=1e-4, n_jobs=n_jobs,
    ).fit(X_train, y_train)
    features = X_train.columns[selector.coef_ != 0].tolist()
    if not features:
        features = [str(X_train.columns[np.argmax(np.abs(selector.coef_))])]
    search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
        {"C": np.logspace(-2, 2, 10), "gamma": np.logspace(-3, 1, 5)},
        scoring="balanced_accuracy", cv=5, n_jobs=n_jobs,
    ).fit(X_train[features], y_train)
    return TrainingResult(
        model=search.best_estimator_, selected_features=features,
        best_params={
            "elasticnet_alpha": float(selector.alpha_),
            "elasticnet_l1_ratio": float(selector.l1_ratio_),
            **search.best_params_,
        },
        cv_score=float(search.best_score_),
    )


def _tune_random_forest(X_train, y_train, seed: int, n_jobs: int):
    return GridSearchCV(
        RandomForestClassifier(random_state=seed, class_weight="balanced"),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=3, n_jobs=n_jobs, scoring="balanced_accuracy",
    ).fit(X_train, y_train)


def train_random_forest_gini(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_jobs: int = -1
) -> TrainingResult:
    search = _tune_random_forest(X_train, y_train, seed, n_jobs)
    selector = RFECV(
        estimator=clone(search.best_estimator_), step=1, cv=5,
        scoring="balanced_accuracy", n_jobs=n_jobs,
    ).fit(X_train, y_train)
    features = X_train.columns[selector.support_].tolist()
    model = clone(search.best_estimator_).set_params(oob_score=True).fit(
        X_train[features], y_train
    )
    return TrainingResult(
        model=model, selected_features=features, best_params=search.best_params_,
        cv_score=float(search.best_score_),
        selection_details={"rfecv_features": int(selector.n_features_)},
    )


def train_random_forest_permutation(
    X_train: pd.DataFrame, y_train: pd.Series, seed: int, n_jobs: int = -1,
    feature_count: int = 40,
) -> TrainingResult:
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
    )
    search = _tune_random_forest(X_fit, y_fit, seed, n_jobs)
    validation_model = search.best_estimator_
    importance = permutation_importance(
        validation_model, X_validation, y_validation, n_repeats=30,
        random_state=seed, scoring="balanced_accuracy", n_jobs=n_jobs,
    )
    ranking = pd.Series(
        importance.importances_mean, index=X_train.columns
    ).sort_values(ascending=False)
    features = ranking.head(min(feature_count, len(ranking))).index.tolist()
    model = clone(search.best_estimator_).set_params(oob_score=True).fit(
        X_train[features], y_train
    )
    validation_score = balanced_accuracy_score(
        y_validation, validation_model.predict(X_validation)
    )
    return TrainingResult(
        model=model, selected_features=features, best_params=search.best_params_,
        cv_score=float(search.best_score_),
        selection_details={
            "validation_balanced_accuracy": float(validation_score),
            "permutation_importance": {
                str(feature): float(value) for feature, value in ranking.items()
            },
        },
    )


def train_model(
    name: str, X_train: pd.DataFrame, y_train: pd.Series, seed: int,
    n_jobs: int = -1, selection_frequency: float = 0.60,
    permutation_features: int = 40,
) -> TrainingResult:
    """Dispatch one of the four notebook modelling strategies."""
    if name == "lasso_logistic":
        return train_lasso_logistic(X_train, y_train, seed, selection_frequency)
    if name == "elasticnet_svm":
        return train_elasticnet_svm(X_train, y_train, seed, n_jobs)
    if name == "random_forest_gini":
        return train_random_forest_gini(X_train, y_train, seed, n_jobs)
    if name == "random_forest_permutation":
        return train_random_forest_permutation(
            X_train, y_train, seed, n_jobs, permutation_features
        )
    raise ValueError(f"Unknown pipeline {name!r}. Choose from: {', '.join(PIPELINES)}")
