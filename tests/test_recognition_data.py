import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from Recognition.prepare_data import file_digest, prepare_dataset, resolve_training_data


def source_dataset(root: Path) -> Path:
    for split, count in (("trainset", 5), ("testset", 2)):
        for label, offset in (("green", 0), ("full", 50)):
            folder = root / split / label
            folder.mkdir(parents=True)
            for index in range(count):
                value = offset + index + (100 if split == "testset" else 0)
                Image.new("RGB", (8, 8), (value, 0, 0)).save(folder / f"{index}.png")
    return root


def inventory(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): file_digest(p) for p in root.rglob("*.png")}


def test_preparation_preserves_source_and_test_and_is_repeatable(tmp_path):
    source = source_dataset(tmp_path / "source")
    before = inventory(source)
    output = prepare_dataset(source, tmp_path / "prepared")
    assert inventory(source) == before
    assert len(list((output / "train").rglob("*.png"))) == 8
    assert len(list((output / "val").rglob("*.png"))) == 2
    assert inventory(output / "test") == inventory(source / "testset")
    train_hashes = set(inventory(output / "train").values())
    val_hashes = set(inventory(output / "val").values())
    test_hashes = set(inventory(output / "test").values())
    assert not train_hashes & val_hashes
    assert not (train_hashes | val_hashes) & test_hashes
    assert prepare_dataset(source, output) == output
    other = prepare_dataset(source, tmp_path / "other")
    assert inventory(other) == inventory(output)


def test_duplicates_remain_in_same_split(tmp_path):
    source = source_dataset(tmp_path / "source")
    shutil.copy2(source / "trainset/green/0.png", source / "trainset/green/duplicate.png")
    output = prepare_dataset(source, tmp_path / "prepared")
    records = json.loads((output / "split_manifest.json").read_text())["images"]
    duplicates = [r for r in records if r["source"] in {"trainset/green/0.png", "trainset/green/duplicate.png"}]
    assert len({r["target"].split("/")[0] for r in duplicates}) == 1


@pytest.mark.parametrize("target", ["testset/green/0.png", "testset/full/0.png"])
def test_cross_split_or_label_duplicates_rejected_before_writing(tmp_path, target):
    source = source_dataset(tmp_path / "source")
    shutil.copy2(source / "trainset/green/0.png", source / target)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Duplicate image"):
        prepare_dataset(source, output)
    assert not output.exists()


def test_conflicting_training_labels_are_reported_and_kept_out_of_validation(tmp_path):
    source = source_dataset(tmp_path / "source")
    shutil.copy2(source / "trainset/green/0.png", source / "trainset/full/0.png")
    with pytest.warns(UserWarning, match="conflicting training labels"):
        output = prepare_dataset(source, tmp_path / "prepared")
    assert (output / "train/green/0.png").is_file()
    assert (output / "train/full/0.png").is_file()
    assert len(json.loads((output / "split_manifest.json").read_text())["conflicting_train_hashes"]) == 1


@pytest.mark.parametrize("change", ["source", "prepared", "seed"])
def test_stale_or_modified_prepared_data_is_not_silently_reused(tmp_path, change):
    source = source_dataset(tmp_path / "source")
    output = prepare_dataset(source, tmp_path / "prepared")
    if change != "seed":
        image = next((source if change == "source" else output).rglob("*.png"))
        Image.new("RGB", (8, 8), (255, 255, 255)).save(image)
    with pytest.raises(ValueError, match="use a new --prepared-dir"):
        prepare_dataset(source, output, seed=43 if change == "seed" else 42)


def test_missing_validation_cannot_fall_back_to_test(tmp_path):
    source = source_dataset(tmp_path / "source")
    (source / "trainset").rename(source / "train")
    (source / "testset").rename(source / "test")
    with pytest.raises(ValueError, match="Missing classification split"):
        resolve_training_data(source)


def test_real_ultralytics_reads_nine_class_layout(tmp_path):
    from ultralytics.data.utils import check_cls_dataset

    source = tmp_path / "source"
    for index in range(9):
        for split, values in (("trainset", (0, 1)), ("testset", (2,))):
            folder = source / split / f"class_{index}"
            folder.mkdir(parents=True)
            for value in values:
                Image.new("RGB", (12, 12), (index, value, 0)).save(folder / f"{value}.png")
    output = resolve_training_data(source)
    data = check_cls_dataset(output)
    assert data["nc"] == 9
    assert set(data["names"].values()) == {f"class_{i}" for i in range(9)}
    assert Path(data["val"]) == output / "val"
    assert Path(data["test"]) == output / "test"
