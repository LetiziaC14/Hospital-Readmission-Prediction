"""Evaluate saved models on the held-out test set."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="all")
    parser.parse_args()
    raise NotImplementedError("Connect this script after train.py saves the test split and fitted models.")


if __name__ == "__main__":
    main()
