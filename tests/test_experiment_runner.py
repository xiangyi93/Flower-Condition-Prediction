import numpy as np
import pytest
import torch
from PIL import Image

from Segmentation.experiment_runner import (
    SegmentationDataset,
    evaluation_transform,
    pixel_ap_metrics,
    train_model,
    training_transform,
)


def test_pixel_ap_uses_scores_and_excludes_absent_class() -> None:
    targets = np.array([0, 0, 1, 1])
    scores = np.array(
        [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.6, 0.4, 0.0]]
    )
    assert np.all(scores.argmax(axis=1) == 0)
    assert pixel_ap_metrics(targets, scores) == {
        "AP_per_class": [1.0, 1.0, None],
        "mAP": 1.0,
    }


def test_training_augmentation_repeats_with_seed() -> None:
    first, second = training_transform(17), training_transform(17)
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    mask = np.zeros((12, 16), dtype=np.uint8)
    for _ in range(6):
        assert torch.equal(
            first(image=image, mask=mask)["image"],
            second(image=image, mask=mask)["image"],
        )


def test_missing_masks_fail_without_writing_to_dataset(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Mask directory"):
        SegmentationDataset(tmp_path, evaluation_transform())
    assert list(tmp_path.iterdir()) == []


def test_training_preserves_existing_run(tmp_path) -> None:
    original = tmp_path / "training_log.csv"
    original.write_text("completed experiment", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Use a new --output-dir"):
        train_model(lambda: torch.nn.Conv2d(3, 4, 1), "test", tmp_path, tmp_path)
    assert original.read_text(encoding="utf-8") == "completed experiment"


def test_training_and_evaluation_round_trip(tmp_path, monkeypatch) -> None:
    from Segmentation import experiment_runner

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(experiment_runner, "IMAGE_SIZE", 16)
    source = tmp_path / "data"
    (source / "masks").mkdir(parents=True)
    mask = np.tile(np.arange(4, dtype=np.uint8), (16, 4))
    for index in range(5):
        Image.fromarray(np.full((16, 16, 3), index * 20, dtype=np.uint8)).save(
            source / f"{index}.png"
        )
        Image.fromarray(mask).save(source / "masks" / f"{index}.png")
    initial_weights = []

    def factory():
        model = torch.nn.Conv2d(3, 4, 1)
        initial_weights.append(model.weight.detach().clone())
        return model

    first = experiment_runner.train_model(
        factory, "test", source, tmp_path / "run1", epochs=1, batch_size=2
    )
    experiment_runner.train_model(
        factory, "test", source, tmp_path / "run2", epochs=1, batch_size=2
    )
    assert torch.equal(initial_weights[0], initial_weights[1])
    assert first.is_file()
    result = experiment_runner.evaluate_checkpoint(
        factory, first, source, tmp_path / "evaluation", batch_size=2
    )
    assert np.asarray(result["confusion_matrix"]).sum() == 5 * 16 * 16
    assert result["Macro_F1"] == result["mean_dice"]
    assert result["mAP"] is not None
    assert (tmp_path / "evaluation" / "pixel_scores.npz").is_file()
