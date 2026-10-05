from argparse import Namespace
from pathlib import Path

import pytest

from Recognition.YOLOv11 import train_classifier, training_device, training_kwargs


def make_args(tmp_path: Path, device: str = "auto") -> Namespace:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    for split in ("train", "val"):
        directory = data_dir / split / "flower"
        directory.mkdir(parents=True)
        (directory / "sample.png").write_bytes(b"fixture")
    return Namespace(
        data_dir=data_dir,
        base_model="yolo11n-cls.pt",
        epochs=50,
        image_size=224,
        batch_size=16,
        patience=10,
        device=device,
        project_dir=tmp_path / "runs" / "classify",
        run_name="flower_bloom_model-3",
    )


def test_training_kwargs_uses_shared_config_values(tmp_path: Path) -> None:
    kwargs = training_kwargs(make_args(tmp_path, device="cpu"))

    assert kwargs["data"] == str(tmp_path / "dataset")
    assert kwargs["project"] == str(tmp_path / "runs" / "classify")
    assert kwargs["name"] == "flower_bloom_model-3"
    assert kwargs["device"] == "cpu"


def test_training_device_maps_auto_and_cuda() -> None:
    assert training_device("auto") is None
    assert training_device("cuda") == 0
    with pytest.raises(ValueError, match="Unsupported training device"):
        training_device("mps")


def test_train_classifier_passes_configured_arguments_to_yolo(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    class FakeYOLO:
        def __init__(self, base_model: str) -> None:
            calls["base_model"] = base_model

        def train(self, **kwargs) -> None:
            calls["kwargs"] = kwargs
            best = tmp_path / "runs/classify/actual_run2/weights/best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"checkpoint")
            self.trainer = Namespace(best=best)

    monkeypatch.setattr("Recognition.YOLOv11.YOLO", FakeYOLO)
    args = make_args(tmp_path)

    checkpoint_path = train_classifier(args)

    assert calls["base_model"] == "yolo11n-cls.pt"
    assert "device" not in calls["kwargs"]
    assert checkpoint_path == tmp_path / "runs/classify/actual_run2/weights/best.pt"
