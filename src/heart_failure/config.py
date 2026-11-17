"""Configuration loading shared by command-line scripts."""

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load the project YAML configuration."""
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
