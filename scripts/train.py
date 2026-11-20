"""Train one or all feature-selection/model pipelines from the notebook."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from heart_failure.config import load_config
from heart_failure.features import NotebookPreprocessor, split_features_target
from heart_failure.modeling import PIPELINES, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=[*PIPELINES, "all"], default="all")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed, target = config["seed"], config["target_column"]
    frame = pd.read_csv(config["paths"]["processed_data"], index_col=0)
    X_train_raw, _, y_train, _ = split_features_target(
        frame, target, config["data"]["test_size"], seed
    )
    preprocessor = NotebookPreprocessor(seed=seed)
    X_train = preprocessor.fit_transform(X_train_raw)
    selected = PIPELINES if args.pipeline == "all" else [args.pipeline]
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    summary_path = models_dir / "training_summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
    else:
        summary = {}
    for name in selected:
        print(f"Training {name} ...", flush=True)
        result = train_model(
            name, X_train, y_train, seed,
            n_jobs=config["training"]["n_jobs"],
            selection_frequency=config["training"]["lasso_selection_frequency"],
            permutation_features=config["training"]["permutation_features"],
        )
        artifact = {
            "name": name, "model": result.model, "preprocessor": preprocessor,
            "selected_features": result.selected_features,
            "best_params": result.best_params, "cv_score": result.cv_score,
            "selection_details": result.selection_details,
            "target": target, "seed": seed,
            "test_size": config["data"]["test_size"],
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
            },
        }
        destination = models_dir / f"{name}.joblib"
        joblib.dump(artifact, destination)
        summary[name] = {
            "selected_feature_count": len(result.selected_features),
            "selected_features": result.selected_features,
            "best_params": result.best_params,
            "cv_score": result.cv_score,
        }
        print(f"Saved {destination}")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(summary), file, indent=2)


if __name__ == "__main__":
    main()
