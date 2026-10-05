"""Prepare YOLO classification splits without changing source images."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def class_images(split: Path) -> dict[str, list[Path]]:
    if not split.is_dir():
        raise ValueError(f"Missing classification split: {split}")
    classes = {}
    for directory in sorted(split.iterdir()):
        if directory.is_dir():
            images = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                raise ValueError(f"Empty classification class: {directory}")
            classes[directory.name] = images
    if not classes:
        raise ValueError(f"No classification classes in {split}")
    if any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in split.iterdir()):
        raise ValueError(f"Images must be inside class directories: {split}")
    return classes


def validate_yolo_layout(root: Path) -> None:
    train = class_images(root / "train")
    for split in ("val", "test"):
        if split == "test" and not (root / split).exists():
            continue
        if set(class_images(root / split)) != set(train):
            raise ValueError(f"Class names differ between train and {split}: {root}")


def prepare_dataset(source: Path, destination: Path, val_fraction: float = 0.2, seed: int = 42) -> Path:
    """Stratify by class, keep identical files together, and reserve testset."""
    source, destination = source.resolve(), destination.resolve()
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Prepared directory must be separate from the source dataset")
    train = class_images(source / "trainset")
    test = class_images(source / "testset")
    if set(train) != set(test):
        raise ValueError("trainset and testset must contain the same class names")

    records = []
    digests = {}
    origins = defaultdict(set)
    for split, classes in (("trainset", train), ("testset", test)):
        for name, paths in classes.items():
            for path in paths:
                digest = file_digest(path)
                digests[path] = digest
                origins[digest].add((split, name))
    conflicts = []
    for digest, identities in origins.items():
        if len({split for split, _ in identities}) > 1:
            raise ValueError(f"Duplicate image crosses trainset/testset: {sorted(identities)}")
        if len(identities) > 1:
            if any(split == "testset" for split, _ in identities):
                raise ValueError(f"Duplicate image has conflicting test labels: {sorted(identities)}")
            conflicts.append(digest)
    if conflicts:
        warnings.warn("Identical images have conflicting training labels; keeping all copies in train. See split_manifest.json.", stacklevel=2)
    rng = random.Random(seed)
    for name in train:
        groups = defaultdict(list)
        for split, paths in (("trainset", train[name]), ("testset", test[name])):
            for path in paths:
                digest = digests[path]
                record = {"source": path.relative_to(source).as_posix(), "sha256": digest}
                if split == "trainset":
                    groups[digest].append(record)
                else:
                    record["target"] = (Path("test") / path.relative_to(source / split)).as_posix()
                    records.append(record)
        if len(groups) < 2:
            raise ValueError(f"Class {name} needs at least two distinct training images for train/val")
        keys = sorted(key for key in groups if key not in conflicts)
        if not keys:
            raise ValueError(f"Class {name} has no unambiguous images for validation")
        rng.shuffle(keys)
        target_count = max(1, round(len(train[name]) * val_fraction))
        val_keys, val_count = set(), 0
        candidates = keys if len(keys) < len(groups) else keys[:-1]
        for key in candidates:  # Always retain at least one group for training.
            if val_count >= target_count:
                break
            val_keys.add(key)
            val_count += len(groups[key])
        for key in sorted(groups):
            split = "val" if key in val_keys else "train"
            for record in groups[key]:
                record["target"] = (Path(split) / Path(record["source"]).relative_to("trainset")).as_posix()
                records.append(record)

    records.sort(key=lambda record: record["source"])
    manifest = {"version": 1, "source": str(source), "val_fraction": val_fraction, "seed": seed, "conflicting_train_hashes": sorted(conflicts), "images": records}
    manifest_path = destination / "split_manifest.json"
    if destination.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Prepared data is incomplete or source/split settings changed; use a new --prepared-dir: {destination}")
        expected = {record["target"] for record in records}
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        if actual != expected or any(file_digest(destination / r["target"]) != r["sha256"] for r in records):
            raise ValueError(f"Prepared images changed; use a new --prepared-dir: {destination}")
    else:
        destination.mkdir(parents=True)
        for record in records:
            target = destination / record["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / record["source"], target)
            if file_digest(target) != record["sha256"]:
                raise ValueError(f"Image changed during preparation: {record['source']}")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validate_yolo_layout(destination)
    return destination


def resolve_training_data(source: Path, destination: Path | None = None, val_fraction: float = 0.2, seed: int = 42) -> Path:
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Classification dataset does not exist: {source}")
    if (source / "trainset").exists() or (source / "testset").exists():
        if any((source / name).exists() for name in ("train", "val", "test")):
            raise ValueError(f"Mixed source and YOLO split layouts: {source}")
        return prepare_dataset(source, destination or source.with_name(source.name + "_prepared"), val_fraction, seed)
    validate_yolo_layout(source)
    return source.resolve()
