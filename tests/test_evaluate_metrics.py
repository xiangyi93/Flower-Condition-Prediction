import numpy as np
import pytest

from Segmentation.evaluate import CLASS_NAMES, metrics_from_confusion_matrix


def test_metrics_from_confusion_matrix_calculates_dataset_level_values() -> None:
    matrix = np.array(
        [
            [5, 1, 0, 0],
            [1, 3, 0, 0],
            [0, 0, 2, 1],
            [0, 0, 1, 4],
        ]
    )

    metrics = metrics_from_confusion_matrix(matrix)

    assert metrics["pixel_accuracy"] == pytest.approx(14 / 18)
    assert metrics["per_class"]["background"]["iou"] == pytest.approx(5 / 7)
    assert metrics["per_class"]["daylily"]["dice"] == pytest.approx(2 / 3)
    assert set(metrics["per_class"]) == set(CLASS_NAMES)


def test_metrics_from_empty_confusion_matrix_reports_zero_pixel_accuracy() -> None:
    metrics = metrics_from_confusion_matrix(np.zeros((4, 4), dtype=np.int64))

    assert metrics["pixel_accuracy"] == 0.0
    assert np.isnan(metrics["mIoU"])
