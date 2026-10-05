import numpy as np
import pytest
import torch

from Segmentation.metrics import DiceLoss, get_miou


def test_dice_loss_is_near_zero_for_confident_correct_predictions() -> None:
    targets = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 4, 2, 2), -10.0)
    for row in range(2):
        for column in range(2):
            logits[0, targets[0, row, column], row, column] = 10.0

    assert DiceLoss()(logits, targets).item() == pytest.approx(0.0, abs=1e-6)


def test_miou_uses_all_four_segmentation_classes() -> None:
    targets = np.array([[[0, 1], [2, 3]]])
    predictions = np.array([[[0, 1], [2, 0]]])

    assert get_miou(predictions, targets, num_classes=4) == pytest.approx(0.625)


def test_miou_ignores_classes_absent_from_both_prediction_and_target() -> None:
    targets = np.array([[[1, 1]]])
    predictions = np.array([[[1, 1]]])

    assert get_miou(predictions, targets, num_classes=4) == pytest.approx(1.0)
