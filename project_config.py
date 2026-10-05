"""Shared TOML configuration helpers for project entry points."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("rb") as file:
        return tomllib.load(file)


def config_value(config: dict, section: str, key: str):
    if section not in config:
        raise KeyError(f"Configuration is missing the [{section}] section")
    if key not in config[section]:
        raise KeyError(f"Configuration is missing {section}.{key}")
    return config[section][key]


def config_path(config: dict, section: str, key: str) -> Path:
    path = Path(config_value(config, section, key))
    return path if path.is_absolute() else PROJECT_ROOT / path
