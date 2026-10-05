from pathlib import Path

import pytest

from Segmentation.data import collect_image_mask_pairs


def test_collect_image_mask_pairs_matches_basename_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "masks").mkdir()
    for name in ("b.jpeg", "a.jpg", "ignored.txt"):
        (tmp_path / name).touch()
    for name in ("b.png", "a.png"):
        (tmp_path / "masks" / name).touch()

    pairs = collect_image_mask_pairs(tmp_path, tmp_path / "masks")

    assert [(Path(image).name, Path(mask).name) for image, mask in pairs] == [
        ("a.jpg", "a.png"),
        ("b.jpeg", "b.png"),
    ]


def test_collect_image_mask_pairs_rejects_missing_mask(tmp_path: Path) -> None:
    (tmp_path / "masks").mkdir()
    (tmp_path / "flower.jpg").touch()

    with pytest.raises(ValueError, match="missing masks for: flower"):
        collect_image_mask_pairs(tmp_path, tmp_path / "masks")


def test_collect_image_mask_pairs_rejects_empty_image_directory(tmp_path: Path) -> None:
    (tmp_path / "masks").mkdir()

    with pytest.raises(ValueError, match="No supported images found"):
        collect_image_mask_pairs(tmp_path, tmp_path / "masks")
