"""Evaluate saved notebook models on the common held-out test set."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from heart_failure.config import load_config
from heart_failure.evaluation import (
    compute_metrics,
    feature_importance,
    prediction_scores,
    save_confusion_matrix,
    save_curve_comparison,
)
from heart_failure.features import split_features_target
from heart_failure.modeling import PIPELINES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=[*PIPELINES, "all"], default="all")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame = pd.read_csv(config["paths"]["processed_data"], index_col=0)
    _, X_test_raw, _, y_test = split_features_target(
        frame, config["target_column"], config["data"]["test_size"], config["seed"]
    )
    selected = PIPELINES if args.pipeline == "all" else [args.pipeline]
    reports = Path(config["paths"]["reports_dir"])
    metrics_dir, figures_dir = reports / "metrics", reports / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for name in selected:
        artifact_path = Path(config["paths"]["models_dir"]) / f"{name}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing {artifact_path}; run train.py --pipeline {name} first.")
        artifact = joblib.load(artifact_path)
        X_test = artifact["preprocessor"].transform(X_test_raw)
        X_selected = X_test[artifact["selected_features"]]
        model = artifact["model"]
        prediction = model.predict(X_selected)
        score = prediction_scores(model, X_selected)
        metrics = compute_metrics(y_test, prediction, score)
        rows.append({"model": name, **metrics})
        curves[name] = {"y_true": y_test, "score": score}
        save_confusion_matrix(
            y_test, prediction, figures_dir / f"{name}_confusion_matrix.png", name
        )
        importance = feature_importance(
            name, model, X_selected, y_test, config["seed"],
            config["training"]["n_jobs"],
        )
        importance.to_csv(metrics_dir / f"{name}_feature_importance.csv", index=False)
    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(metrics_dir / "model_comparison.csv", index=False)
    with (metrics_dir / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(results.to_dict(orient="records"), file, indent=2)
    save_curve_comparison(curves, figures_dir / "model_curve_comparison.png")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
