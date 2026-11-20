"""Reusable metrics, plots and feature-importance exports."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve,
)


def prediction_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores from a fitted binary classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Calculate accuracy, balanced accuracy, ROC-AUC and PR-AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
    }


def save_confusion_matrix(y_true, y_pred, output_path: str | Path, title: str) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, normalize="true", cmap="Blues", values_format=".1%"
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=200)
    plt.close(display.figure_)


def save_curve_comparison(curves: dict, output_path: str | Path) -> None:
    """Save the combined ROC and precision-recall plot from the notebook."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, values in curves.items():
        fpr, tpr, _ = roc_curve(values["y_true"], values["score"])
        precision, recall, _ = precision_recall_curve(values["y_true"], values["score"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
        axes[1].plot(recall, precision, label=f"{name} (AUC={auc(recall, precision):.4f})")
    baseline = float(next(iter(curves.values()))["y_true"].mean())
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.75)
    axes[1].hlines(baseline, 0, 1, color="black", linestyle="--", linewidth=0.75)
    axes[0].set(title="ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[1].set(
        title="Precision-Recall Curves", xlabel="Recall", ylabel="Precision",
        xlim=(0, 1), ylim=(0, 1),
    )
    for axis in axes:
        axis.grid(True)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def feature_importance(
    name: str, model, X_test: pd.DataFrame, y_test: pd.Series, seed: int, n_jobs: int
) -> pd.DataFrame:
    """Return coefficients, Gini importance or permutation importance."""
    if name == "lasso_logistic":
        values = np.abs(model.coef_[0])
    elif name == "random_forest_gini":
        values = model.feature_importances_
    else:
        values = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=n_jobs
        ).importances_mean
    return pd.DataFrame({"feature": X_test.columns, "importance": values}).sort_values(
        "importance", ascending=False
    )
