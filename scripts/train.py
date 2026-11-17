"""Train one or all candidate prediction pipelines."""

import argparse


PIPELINES = ["logistic_regression", "random_forest", "svm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=[*PIPELINES, "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = PIPELINES if args.pipeline == "all" else [args.pipeline]
    raise NotImplementedError(f"Training scaffold created for: {', '.join(selected)}.")


if __name__ == "__main__":
    main()
