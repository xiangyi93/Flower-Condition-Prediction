import random

import numpy as np
import pytest
import torch

from Segmentation.train import (
    mean_iou_from_confusion_matrix,
    resume_training,
    segmentation_confusion_matrix,
    set_seed,
    training_checkpoint,
)


def test_global_confusion_matrix_calculates_four_class_miou() -> None:
    targets = np.array([[[0, 1], [2, 3]]])
    predictions = np.array([[[0, 1], [2, 0]]])

    matrix = segmentation_confusion_matrix(predictions, targets)

    assert matrix.shape == (4, 4)
    assert mean_iou_from_confusion_matrix(matrix) == pytest.approx(0.625)


def test_set_seed_repeats_python_numpy_and_torch_values() -> None:
    set_seed(123)
    first_values = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    second_values = (random.random(), np.random.rand(), torch.rand(1).item())

    assert second_values == pytest.approx(first_values)


def test_resume_training_restores_complete_checkpoint(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(7)
    original_weight = model.weight.detach().clone()
    checkpoint_path = tmp_path / "last_checkpoint.pth"
    torch.save(
        training_checkpoint(
            model,
            optimizer,
            3,
            0.75,
            42,
            generator.get_state(),
            {"num_classes": 4},
        ),
        checkpoint_path,
    )
    with torch.no_grad():
        model.weight.zero_()

    state = resume_training(str(checkpoint_path), model, optimizer, "cpu")

    assert state["start_epoch"] == 3
    assert state["best_val_miou"] == pytest.approx(0.75)
    assert torch.equal(model.weight, original_weight)
    assert torch.equal(state["data_loader_rng_state"], generator.get_state())


def test_resume_training_accepts_legacy_model_only_weights(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "legacy_weights.pth"
    torch.save(model.state_dict(), checkpoint_path)

    state = resume_training(str(checkpoint_path), model, optimizer, "cpu")

    assert state["start_epoch"] == 0
    assert state["best_val_miou"] == float("-inf")
