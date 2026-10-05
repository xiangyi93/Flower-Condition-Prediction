import json

import numpy as np
import pytest
from PIL import Image

from label_region_export.export_label_regions import LABELS, export_dataset, read_sample


def sample(folder, name, colour, label="cherry", width=6):
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 6), colour).save(folder / f"{name}.png")
    data = {"imageWidth": width, "imageHeight": 6, "shapes": [
        {"label": label, "shape_type": "polygon", "points": [[1, 1], [3, 1], [3, 3], [1, 3]]}
    ]}
    (folder / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_regions_numbering_and_originals(tmp_path):
    train, test, output = [tmp_path / name for name in ("train", "test", "out")]
    sample(train, "image10", (90, 80, 70))
    sample(train, "image2", (30, 20, 10))
    sample(test, "image1", (60, 50, 40), "daylily")
    originals = {p: p.read_bytes() for source in (train, test) for p in source.iterdir()}
    assert export_dataset(train, test, output) == {"train": 2, "test": 1}
    assert len(list(output.rglob("*.png"))) == 12
    regions = {label: np.array(Image.open(output / "train" / label / "000001.png"))
               for label in LABELS}
    assert regions["cherry"][2, 2].tolist() == [30, 20, 10]
    assert regions["background"][0, 0].tolist() == [30, 20, 10]
    assert not regions["hydrangeas"].any()
    assert not regions["daylily"].any()
    assert np.array_equal(sum(r.astype(int) for r in regions.values()),
                          np.array(Image.open(train / "image2.png")))
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert (output / "manifest.csv").is_file()
    with pytest.raises(FileExistsError):
        export_dataset(train, test, output)


@pytest.mark.parametrize("problem,match", [
    ("duplicate", "duplicate"), ("unknown", "unknown label"),
    ("dimensions", "dimensions"), ("missing", "missing JSON"),
])
def test_preflight_prevents_partial_output(tmp_path, problem, match):
    train, test, output = [tmp_path / name for name in ("train", "test", "out")]
    sample(train, "a", (10, 20, 30))
    sample(test, "b", (10, 20, 30) if problem == "duplicate" else (40, 50, 60),
           label="typo" if problem == "unknown" else "cherry",
           width=7 if problem == "dimensions" else 6)
    if problem == "missing":
        (test / "b.json").unlink()
        sample(test, "c", (70, 80, 90))
    with pytest.raises(ValueError, match=match):
        export_dataset(train, test, output)
    assert not output.exists()


def test_dry_run_and_nested_output(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    sample(train, "a", (1, 2, 3))
    sample(test, "b", (4, 5, 6))
    assert export_dataset(train, test, tmp_path / "out", True) == {"train": 1, "test": 1}
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="contain each other"):
        export_dataset(train, test, train / "out")


def test_linestrip_is_not_filled_polygon(tmp_path):
    Image.new("RGB", (60, 60), (100, 100, 100)).save(tmp_path / "a.png")
    annotation = tmp_path / "a.json"
    annotation.write_text(json.dumps({"imageHeight": 60, "imageWidth": 60,
        "shapes": [{"label": "cherry", "shape_type": "linestrip",
                    "points": [[5, 5], [50, 5], [50, 50]]}]}), encoding="utf-8")
    _, _, mask = read_sample(annotation)
    assert mask[5, 25] == LABELS["cherry"]
    assert mask[25, 35] == LABELS["background"]
