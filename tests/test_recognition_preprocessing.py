from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.cfg import get_cfg

from Recognition.classifier import (
    BloomDataset,
    BloomPredictor,
    BloomValidator,
    BloomYOLO,
)
from Recognition.preprocessing import (
    ForegroundLetterbox,
    bloom_transforms,
    rgb_on_black,
)


def plant_image(position=(0, 0)):
    image = Image.new("RGB", (160, 120))
    plant = Image.new("RGB", (20, 40), (20, 120, 30))
    plant.paste((240, 20, 30), (4, 5, 10, 15))
    image.paste(plant, position)
    return image


def test_plant_at_edge_and_center_has_same_preprocessing():
    transform = ForegroundLetterbox(64)
    np.testing.assert_array_equal(transform(plant_image()), transform(plant_image((70, 50))))


def test_disconnected_regions_are_both_preserved():
    image = Image.new("RGB", (200, 120))
    image.paste((255, 0, 0), (0, 0, 20, 20))
    image.paste((0, 255, 0), (160, 90, 190, 120))
    array = np.asarray(ForegroundLetterbox(224)(image))
    assert np.any(array[:, :, 0] > 200)
    assert np.any(array[:, :, 1] > 200)


def test_aspect_ratio_preserved_without_stretching():
    image = Image.new("RGB", (200, 200))
    image.paste((255, 255, 255), (20, 80, 120, 130))
    array = np.asarray(ForegroundLetterbox(100, margin=0)(image))
    assert array.shape == (100, 100, 3)
    assert (array[25:75] == 255).all()
    assert not array[:25].any()
    assert not array[75:].any()


def test_transparent_hidden_colors_do_not_expand_foreground():
    image = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
    image.paste((0, 200, 0, 255), (30, 30, 50, 70))
    visible = Image.new("RGB", (100, 100))
    visible.paste((0, 200, 0), (30, 30, 50, 70))
    np.testing.assert_array_equal(ForegroundLetterbox()(image), ForegroundLetterbox()(visible))


def test_all_black_is_explicit_error_not_a_green_training_example():
    with pytest.raises(ValueError, match="No foreground"):
        ForegroundLetterbox()(Image.new("RGB", (32, 32)))


def test_training_only_adds_horizontal_flip():
    image = plant_image((120, 70))
    val = bloom_transforms(64)(image)
    assert torch.equal(val, bloom_transforms(64, augment=True, hflip=0)(image))
    assert torch.equal(val.flip(-1), bloom_transforms(64, augment=True, hflip=1)(image))


def test_dataset_and_predictor_use_identical_rgb_tensors(tmp_path):
    folder = tmp_path / "full"
    folder.mkdir()
    image = plant_image((130, 70)).convert("RGBA")
    image.save(folder / "plant.png")
    args = get_cfg(overrides={"imgsz": 64, "workers": 0})
    dataset = BloomDataset(str(tmp_path), args)
    predictor = BloomPredictor(overrides={"imgsz": 64})
    predictor.model = Namespace(fp16=False, device=torch.device("cpu"))
    predictor.device = torch.device("cpu")
    predictor.transforms = bloom_transforms(64)
    bgr = np.asarray(rgb_on_black(image))[:, :, ::-1].copy()
    assert torch.equal(dataset[0]["img"], predictor.preprocess([bgr])[0])


def test_real_checkpoint_reload_and_resized_inference_keep_custom_transform(tmp_path):
    model = BloomYOLO("yolo11n-cls.yaml")
    model.model.transforms = bloom_transforms(64)
    model.save(tmp_path / "bloom.pt")
    reloaded = BloomYOLO(tmp_path / "bloom.pt")
    result = reloaded.predict(plant_image(), imgsz=96, device="cpu", verbose=False)[0]
    assert result.probs is not None
    assert isinstance(reloaded.predictor.transforms.transforms[0], ForegroundLetterbox)
    assert reloaded.predictor.transforms.transforms[0].size == 96


def test_real_training_and_standalone_validation_share_geometry(tmp_path):
    for split in ("train", "val"):
        for label in ("green", "full"):
            folder = tmp_path / "data" / split / label
            folder.mkdir(parents=True)
            for index in range(2):
                plant_image((index * 30, index * 20)).save(folder / f"{index}.png")
    model = BloomYOLO("yolo11n-cls.yaml")
    model.train(
        data=str(tmp_path / "data"), epochs=1, imgsz=32, batch=4, workers=0,
        device="cpu", pretrained=False, amp=False, plots=False, verbose=False,
        project=str(tmp_path / "runs"), name="smoke", auto_augment=None,
    )
    assert isinstance(model.trainer.validator, BloomValidator)
    train_transform = model.trainer.train_loader.dataset.torch_transforms.transforms[0]
    val_transform = model.trainer.test_loader.dataset.torch_transforms.transforms[0]
    assert isinstance(train_transform, ForegroundLetterbox)
    assert train_transform == val_transform
    restored = BloomYOLO(model.trainer.best)
    metrics = restored.val(
        data=str(tmp_path / "data"), imgsz=32, batch=4, workers=0, device="cpu",
        plots=False, verbose=False, project=str(tmp_path / "runs"), name="validation",
    )
    assert metrics.top1 >= 0
