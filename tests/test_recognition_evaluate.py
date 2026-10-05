import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from Recognition.evaluate import audit_data, evaluate, metric_tables, prediction_row
from Recognition.preprocessing import bloom_transforms


def test_stage_probabilities_are_marginalized_not_top1_suffix():
    names = ["cherry_green", "cherry_full", "daylily_green", "daylily_full"]
    row = prediction_row(Path("image.png"), "cherry_full", names, [.4, .3, .05, .25])
    assert row["predicted_class"] == "cherry_green"
    assert row["predicted_stage"] == "full"
    assert row["stage_confidence"] == pytest.approx(.55)
    assert row["correct_class"] == 0 and row["correct_stage"] == 1


def test_metrics_and_invalid_input_denominator():
    names = ["cherry_green", "cherry_full", "daylily_green", "daylily_full"]
    rows = [
        prediction_row(Path("a"), "cherry_full", names, [.4, .3, .05, .25]),
        prediction_row(Path("b"), "cherry_green", names, [1, 0, 0, 0]),
        {"status": "invalid_image", "true_class": "cherry_green", "true_species": "cherry", "true_stage": "green"},
    ]
    summary, per_class, matrices = metric_tables(rows, names)
    assert summary[0]["accuracy"] == .5
    assert summary[0]["accuracy_all_inputs"] == pytest.approx(1 / 3)
    assert summary[0]["coverage"] == pytest.approx(2 / 3)
    assert summary[2]["accuracy"] == 1
    assert summary[2]["macro_f1"] == pytest.approx(2 / 3)
    green = next(r for r in per_class if r["task"] == "class" and r["label"] == "cherry_green")
    assert green["precision"] == .5 and green["recall"] == 1
    assert green["f1"] == pytest.approx(2 / 3)
    assert matrices["class"][1][1, 0] == 1


def test_exact_cross_split_duplicates_are_reported(tmp_path):
    for split in ("train", "test"):
        folder = tmp_path / split / "cherry_green"
        folder.mkdir(parents=True)
        Image.new("RGB", (32, 32), "green").save(folder / "same.png")
    manifest, duplicates = audit_data(tmp_path)
    assert len(manifest) == len(duplicates) == 2


def test_csv_export_handles_bad_images_and_model_label_order(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for label, color in (("cherry_green", "green"), ("cherry_full", "red")):
        folder = root / "test" / label
        folder.mkdir(parents=True)
        Image.new("RGB", (32, 32), color).save(folder / "plant.png")
    Image.new("RGB", (32, 32)).save(root / "test/cherry_green/black.png")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")

    class FakeModel:
        task = "classify"
        names = {0: "cherry_green", 1: "cherry_full"}
        model = SimpleNamespace(transforms=bloom_transforms(32))

        def __init__(self, path):
            pass

        def predict(self, tensors, **kwargs):
            assert tensors.shape[1:] == (3, 32, 32)
            return [SimpleNamespace(probs=SimpleNamespace(data=torch.tensor([.9, .1]))) for _ in tensors]

    monkeypatch.setattr("Recognition.evaluate.BloomYOLO", FakeModel)
    output = tmp_path / "reports"
    summary = evaluate(checkpoint, root, output, device="cpu", batch_size=1)
    assert summary[0]["invalid_images"] == 1
    assert summary[0]["accuracy"] == .5
    assert (output / "predictions.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    with (output / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(r["predicted_class"] == "cherry_green" for r in rows if r["status"] == "ok")
    with pytest.raises(FileExistsError):
        evaluate(checkpoint, root, output, device="cpu")


def test_missing_test_split_never_falls_back_to_validation(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")
    (tmp_path / "data/val").mkdir(parents=True)
    with pytest.raises(ValueError, match="Missing classification split"):
        evaluate(checkpoint, tmp_path / "data", tmp_path / "reports", device="cpu")
