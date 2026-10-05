"""Shared training and evaluation loop for segmentation comparison models."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from Segmentation.data import collect_image_mask_pairs
from Segmentation.evaluate import (
    CLASS_NAMES,
    metrics_from_confusion_matrix,
    save_reports,
)
from Segmentation.metrics import DiceLoss
from Segmentation.train import (
    mean_iou_from_confusion_matrix,
    segmentation_confusion_matrix,
    set_seed,
)

IMAGE_SIZE = 224
NUM_CLASSES = len(CLASS_NAMES)
CLASS_WEIGHTS = (1.0, 1.0, 10.0, 10.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationDataset(Dataset):
    """Read image/mask pairs without changing the source dataset."""

    def __init__(self, image_dir: Path, transform: A.Compose):
        self.samples = collect_image_mask_pairs(
            str(image_dir), str(image_dir / "masks")
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.int64)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long(), Path(image_path).name


def training_transform(seed: int = 42) -> A.Compose:
    transform = A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    transform.set_random_seed(seed)
    return transform


def evaluation_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _loaders(
    train_data: Path,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    train_source = SegmentationDataset(train_data, training_transform(seed))
    validation_source = SegmentationDataset(train_data, evaluation_transform())
    indices = list(range(len(train_source)))
    train_indices, validation_indices = train_test_split(
        indices, test_size=val_split, random_state=seed
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(train_source, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        Subset(validation_source, validation_indices), shuffle=False, **common
    )
    names = [Path(image).name for image, _ in train_source.samples]
    return (
        train_loader,
        validation_loader,
        [names[index] for index in train_indices],
        [names[index] for index in validation_indices],
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            matrix += segmentation_confusion_matrix(
                predictions.detach().cpu().numpy(), masks.detach().cpu().numpy()
            )
    return total_loss / len(loader), mean_iou_from_confusion_matrix(matrix)


def train_model(
    model_factory: Callable[[], nn.Module],
    model_name: str,
    train_data: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
) -> Path:
    """Train with the same split, loss and model-selection rule as the main model."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must both be at least 1")
    if learning_rate <= 0 or not 0 < val_split < 1:
        raise ValueError(
            "learning_rate must be positive and val_split must be between 0 and 1"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Training output already contains files: {output_dir}. Use a new --output-dir."
        )
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    train_loader, validation_loader, train_names, validation_names = _loaders(
        train_data, batch_size, val_split, seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seed": seed,
                "val_split": val_split,
                "train": train_names,
                "validation": validation_names,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    weights = torch.tensor(CLASS_WEIGHTS, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss().to(device)
    best_miou = float("-inf")
    best_epoch = 0
    best_path = output_dir / "best_model.pth"
    log_path = output_dir / "training_log.csv"
    started = time.perf_counter()

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_mIoU", "val_loss", "val_mIoU"])
        print(
            f"Training {model_name} on {device}: "
            f"{len(train_names)} train / {len(validation_names)} validation"
        )
        for epoch in range(1, epochs + 1):
            train_loss, train_miou = _run_epoch(
                model, train_loader, ce_loss, dice_loss, device, optimizer
            )
            val_loss, val_miou = _run_epoch(
                model, validation_loader, ce_loss, dice_loss, device, None
            )
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_miou:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_miou:.6f}",
                ]
            )
            file.flush()
            print(
                f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f}, "
                f"mIoU {train_miou:.4f} | val loss {val_loss:.4f}, mIoU {val_miou:.4f}"
            )
            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)

    elapsed = time.perf_counter() - started
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epochs,
            "best_val_miou": best_miou,
        },
        output_dir / "last_checkpoint.pth",
    )
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "best_val_mIoU": best_miou,
                "best_epoch": best_epoch,
                "initialization_and_augmentation_seeded": True,
                "elapsed_seconds": elapsed,
                "device": str(device),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable
                ),
                "total_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            },
            file,
            indent=2,
        )
    print(f"Best validation mIoU: {best_miou:.4f}; checkpoint: {best_path}")
    return best_path


def pixel_ap_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    """Exact one-vs-rest pixel AP, matching the existing research report."""
    values = [
        float(average_precision_score(targets == index, probabilities[:, index]))
        if np.any(targets == index)
        else None
        for index in range(probabilities.shape[1])
    ]
    valid = [value for value in values if value is not None]
    return {"AP_per_class": values, "mAP": float(np.mean(valid)) if valid else None}


def evaluate_checkpoint(
    model_factory: Callable[[], nn.Module],
    checkpoint_path: Path,
    test_data: Path,
    output_dir: Path,
    batch_size: int = 4,
) -> dict:
    """Evaluate from the global pixel confusion matrix used by the existing evaluator."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    # Pair collection fails if masks are absent; evaluation must not create source files.
    dataset = SegmentationDataset(test_data, evaluation_transform())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    global_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image_results: list[dict] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, masks, names in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu().numpy()
            probabilities = logits.softmax(dim=1).permute(0, 2, 3, 1).cpu().numpy()
            targets = masks.numpy()
            all_targets.append(targets.reshape(-1))
            all_probabilities.append(probabilities.reshape(-1, NUM_CLASSES))
            for prediction, target, name in zip(predictions, targets, names):
                matrix = confusion_matrix(
                    target.ravel(), prediction.ravel(), labels=range(NUM_CLASSES)
                )
                global_matrix += matrix
                per_image_results.append(
                    {"image": name, **metrics_from_confusion_matrix(matrix)}
                )
    elapsed = time.perf_counter() - started
    metrics = metrics_from_confusion_matrix(global_matrix)
    metrics.update(
        pixel_ap_metrics(np.concatenate(all_targets), np.concatenate(all_probabilities))
    )
    metrics["Dice"] = metrics["mean_dice"]
    metrics["Macro_F1"] = metrics["mean_dice"]
    metrics["confusion_matrix"] = global_matrix.tolist()
    metrics["support"] = global_matrix.sum(axis=1).tolist()
    for index, name in enumerate(CLASS_NAMES):
        metrics["per_class"][name]["ap"] = metrics["AP_per_class"][index]
    metrics["foreground_mIoU"] = float(
        np.nanmean([metrics["per_class"][name]["iou"] for name in CLASS_NAMES[1:]])
    )
    metrics["inference_seconds"] = elapsed
    metrics["images_per_second"] = len(dataset) / elapsed
    metrics["timing_scope"] = (
        "Data loading, preprocessing and inference; excludes model loading and AP calculation. Not a warmed-up latency benchmark."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_reports(output_dir, metrics, per_image_results, global_matrix)
    np.savez_compressed(
        output_dir / "pixel_scores.npz",
        target=np.concatenate(all_targets),
        probabilities=np.concatenate(all_probabilities),
        image_names=np.array([Path(image).name for image, _ in dataset.samples]),
    )
    with (output_dir / "evaluation_protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "image_size": IMAGE_SIZE,
                "class_names": CLASS_NAMES,
                "ap_definition": "Exact non-interpolated one-vs-rest pixel AP; average only classes with positive ground truth; all test pixels, no sampling.",
                "aggregation": "Global pixel confusion matrix, not average of per-image metrics.",
                "test_images": [
                    {
                        "image": str(Path(image).resolve()),
                        "image_sha256": _sha256(Path(image)),
                        "mask_sha256": _sha256(Path(mask)),
                    }
                    for image, mask in dataset.samples
                ],
            },
            file,
            indent=2,
        )
    with (output_dir / "complete_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for name in (
            "mIoU",
            "foreground_mIoU",
            "pixel_accuracy",
            "mean_dice",
            "Macro_F1",
            "mAP",
        ):
            writer.writerow([name, metrics[name]])
        writer.writerow([])
        writer.writerow(["class", "iou", "dice", "precision", "recall", "ap"])
        for name, values in metrics["per_class"].items():
            writer.writerow(
                [
                    name,
                    *[
                        values[key]
                        for key in ("iou", "dice", "precision", "recall", "ap")
                    ],
                ]
            )
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(
        f"Test mIoU {metrics['mIoU']:.4f} | foreground mIoU "
        f"{metrics['foreground_mIoU']:.4f} | pixel accuracy "
        f"{metrics['pixel_accuracy']:.4f} | mean Dice {metrics['mean_dice']:.4f}"
    )
    print(f"Reports saved to: {output_dir}")
    return metrics


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
