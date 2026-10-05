"""Export four RGB regions per image using Labelme annotations, without a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tomllib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"background": 0, "cherry": 1, "hydrangeas": 3, "daylily": 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path: Path) -> list:
    return [(0, int(part)) if part.isdigit() else (1, part.casefold())
            for part in re.split(r"(\d+)", path.as_posix())]


def read_sample(annotation: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    data = json.loads(annotation.read_text(encoding="utf-8-sig"))
    candidates = sorted(
        (p for p in annotation.parent.iterdir()
         if p.stem == annotation.stem and p.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if len(candidates) != 1:
        raise ValueError(f"{annotation}: expected one same-stem image, found {len(candidates)}")
    image_path = candidates[0]
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if (data["imageHeight"], data["imageWidth"]) != (height, width):
        raise ValueError(f"{annotation}: annotation dimensions do not match image")
    mask = np.zeros((height, width), dtype=np.uint8)
    # Match the project's getMask.py: later polygons win in overlapping areas.
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in LABELS:
            raise ValueError(f"{annotation}: unknown label {label!r}")
        shape_type = shape.get("shape_type", "polygon")
        if shape_type not in {"polygon", "linestrip"}:
            raise ValueError(f"{annotation}: unsupported shape type {shape_type!r}")
        points = np.asarray(shape["points"], dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2
                or len(points) < (3 if shape_type == "polygon" else 2)
                or not np.isfinite(points).all()):
            raise ValueError(f"{annotation}: invalid shape points")
        if shape_type == "polygon":
            cv2.fillPoly(mask, [points.astype(np.int32)], LABELS[label])
        else:
            # Labelme's default linestrip rasterization: open line, width 10.
            line_mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(line_mask).line([tuple(p) for p in points], fill=1, width=10)
            mask[np.array(line_mask, dtype=bool)] = LABELS[label]
    return image_path, rgb, mask


def collect_annotations(source: Path) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    annotations = sorted(source.rglob("*.json"), key=natural_key)
    if not annotations:
        raise ValueError(f"No Labelme JSON files found in {source}")
    # Derived masks are not source images. All other images must have labels.
    unlabelled = [p for p in source.rglob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES
                  and "masks" not in p.relative_to(source).parts
                  and not p.with_suffix(".json").is_file()]
    if unlabelled:
        raise ValueError(f"Images missing JSON annotations: {unlabelled[:5]}")
    return annotations


def export_dataset(train: Path, test: Path, output: Path, dry_run: bool = False) -> dict:
    sources = {"train": train.resolve(), "test": test.resolve()}
    output = output.resolve()
    if (sources["train"].is_relative_to(sources["test"])
            or sources["test"].is_relative_to(sources["train"])):
        raise ValueError("Train and test source directories must be separate")
    for source in sources.values():
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output and source directories must not contain each other")
    if output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite: {output}")
    samples = {split: collect_annotations(source) for split, source in sources.items()}
    seen: dict[str, tuple[str, Path]] = {}
    rows = []
    # Validate every sample and exact decoded-pixel leakage before writing anything.
    for split, split_annotations in samples.items():
        for number, annotation in enumerate(split_annotations, 1):
            image_path, rgb, mask = read_sample(annotation)
            digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Train/test duplicate image: {seen[digest][1]} and {image_path}")
            seen[digest] = (split, image_path)
            rows.append({"split": split, "filename": f"{number:06d}.png",
                         "source_image": str(image_path), "source_json": str(annotation),
                         "pixel_sha256": digest,
                         **{f"{label}_pixels": int(np.count_nonzero(mask == index))
                            for label, index in LABELS.items()}})
    summary = {
        split: len(split_annotations)
        for split, split_annotations in samples.items()
    }
    if dry_run:
        return summary
    for split in samples:
        for label in LABELS:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    for row in rows:
        _, rgb, mask = read_sample(Path(row["source_json"]))
        for label, index in LABELS.items():
            region = np.where((mask == index)[..., None], rgb, 0).astype(np.uint8)
            Image.fromarray(region).save(output / row["split"] / label / row["filename"])
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.toml")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Recognition/label_regions")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    args = parser.parse_args()
    config = {}
    if args.train_dir is None or args.test_dir is None:
        config = tomllib.loads(args.config.read_text(encoding="utf-8"))["segmentation"]
    train = args.train_dir if args.train_dir is not None else ROOT / config["train_data"]
    test = args.test_dir if args.test_dir is not None else ROOT / config["test_data"]
    summary = export_dataset(train, test, args.output_dir, args.dry_run)
    print(f"{'Validated' if args.dry_run else 'Exported'}: {summary}")
    if not args.dry_run:
        print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
