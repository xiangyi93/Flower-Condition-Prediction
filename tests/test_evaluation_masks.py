import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from Segmentation.evaluate import ensure_evaluation_masks


def write_labelme_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "imageHeight": 8,
                "imageWidth": 8,
                "shapes": [
                    {
                        "label": "cherry",
                        "points": [[1, 1], [6, 1], [6, 6], [1, 6]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_ensure_evaluation_masks_creates_derived_masks_from_labelme_json(tmp_path: Path) -> None:
    write_labelme_json(tmp_path / "flower.json")

    mask_dir = ensure_evaluation_masks(tmp_path)

    mask = np.asarray(Image.open(mask_dir / "flower.png"))
    assert mask_dir == tmp_path / "masks"
    assert mask.shape == (8, 8)
    assert 1 in np.unique(mask)


def test_ensure_evaluation_masks_uses_existing_mask_directory(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    assert ensure_evaluation_masks(tmp_path) == mask_dir


def test_ensure_evaluation_masks_requires_labelme_json_when_masks_are_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no Labelme JSON files"):
        ensure_evaluation_masks(tmp_path)
