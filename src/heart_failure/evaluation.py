"""Reusable metric and plot functions for model assessment."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Calculate the common binary-classification metrics used in the report."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def save_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    """Save a labelled confusion matrix to `output_path`."""
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=200)
    plt.close(display.figure_)
