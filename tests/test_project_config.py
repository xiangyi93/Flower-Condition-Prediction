from pathlib import Path

import pytest

from project_config import PROJECT_ROOT, config_path, config_value, load_config


def test_load_config_reads_an_absolute_toml_file(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.toml"
    config_file.write_text('[pipeline]\ndevice = "cpu"\n', encoding="utf-8")

    config = load_config(config_file)

    assert config_value(config, "pipeline", "device") == "cpu"


def test_load_config_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
        load_config(tmp_path / "missing.toml")


def test_config_value_reports_missing_section_and_key() -> None:
    with pytest.raises(KeyError, match=r"\[segmentation\]"):
        config_value({}, "segmentation", "epochs")

    with pytest.raises(KeyError, match="segmentation.epochs"):
        config_value({"segmentation": {}}, "segmentation", "epochs")


def test_config_path_resolves_relative_paths_from_project_root() -> None:
    config = {"segmentation": {"checkpoint_path": "models/best_model.pth"}}

    assert config_path(config, "segmentation", "checkpoint_path") == (
        PROJECT_ROOT / "models" / "best_model.pth"
    )
