from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from pipeline_predict import (
    CLASS_COLORS_RGB,
    colour_mask,
    predict_mask,
    probability_dict,
    require_file,
    resolve_device,
    run_pipeline,
)
from Recognition.preprocessing import bloom_transforms


class ConstantSegmentationModel(torch.nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        logits = torch.zeros((batch_size, 4, height, width), device=pixel_values.device)
        logits[:, 2] = 1.0
        return logits


def test_predict_mask_restores_the_input_image_size() -> None:
    image = np.zeros((15, 23, 3), dtype=np.uint8)

    mask = predict_mask(ConstantSegmentationModel(), image, torch.device("cpu"))

    assert mask.shape == (15, 23)
    assert np.all(mask == 2)


def test_colour_mask_uses_the_declared_four_class_colours() -> None:
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    assert np.array_equal(colour_mask(mask), CLASS_COLORS_RGB[mask])


def test_probability_dict_retains_every_classifier_probability() -> None:
    result = SimpleNamespace(
        probs=SimpleNamespace(data=torch.tensor([0.1, 0.9])),
        names={0: "cherry_bud", 1: "cherry_full"},
    )

    probabilities = probability_dict(result)

    assert probabilities["cherry_bud"] == pytest.approx(0.1)
    assert probabilities["cherry_full"] == pytest.approx(0.9)


def test_require_file_and_cpu_device_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input image does not exist"):
        require_file(tmp_path / "missing.jpg", "Input image")

    assert resolve_device("cpu").type == "cpu"


@pytest.mark.parametrize("threshold", [0.0, 0.02])
def test_pipeline_exports_four_regions_and_reports_skips(tmp_path, monkeypatch, threshold):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (20, 20), (30, 100, 20)).save(image_path)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.touch()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # cherry, off-center, sufficient area
    mask[19, 19] = 2  # daylily, nonzero but below the configured threshold
    names = {i: f"{s}_{t}" for i, (s, t) in enumerate(
        (s, t) for s in ("cherry", "daylily", "hydrangeas") for t in ("green", "half", "full")
    )}
    calls = []

    class FakeYOLO:
        def __init__(self, path):
            self.names = names
            self.model = SimpleNamespace(transforms=bloom_transforms(32))

        def predict(self, tensor, **kwargs):
            calls.append(tensor)
            assert tensor.shape == (1, 3, 32, 32)
            assert kwargs["device"] == "cpu"
            p = torch.tensor([.01, .01, .90, .01, .01, .02, .01, .01, .02])
            return [SimpleNamespace(names=names, probs=SimpleNamespace(data=p, top1=2, top1conf=p[2]))]

    monkeypatch.setattr("pipeline_predict.BloomYOLO", FakeYOLO)
    monkeypatch.setattr("pipeline_predict.load_segmentation_model", lambda *args: None)
    monkeypatch.setattr("pipeline_predict.predict_mask", lambda *args: mask)
    output = run_pipeline(SimpleNamespace(
        image=image_path, seg_model=checkpoint, cls_model=checkpoint,
        output_dir=tmp_path / "output", device="cpu", min_area_ratio=threshold,
    ))
    assert set(output["outputs"]["segments"]) == {"background", "cherry", "daylily", "hydrangeas"}
    for path in output["outputs"]["segments"].values():
        assert Path(path).is_file()
    cherry = output["classifications"]["cherry"]
    assert cherry["stage"] == "full"
    assert cherry["confidence"] == pytest.approx(.94)
    assert cherry["foreground_ratio"] == .25
    assert Path(cherry["preprocessed_image"]).is_file()
    absent = output["classifications"]["hydrangeas"]
    assert absent["skip_code"] == "all_black"
    assert absent["stage"] is None and absent["confidence"] is None
    daylily = output["classifications"]["daylily"]
    assert daylily["status"] == ("skipped" if threshold else "ok")
    if threshold:
        assert daylily["skip_code"] == "area_too_small"
    else:
        assert daylily["warning"]
    assert len(calls) == (1 if threshold else 2)
    assert Path(output["outputs"]["summary_csv"]).is_file()
