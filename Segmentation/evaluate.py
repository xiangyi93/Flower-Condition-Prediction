"""Evaluate a semantic-segmentation checkpoint on an image/mask dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from project_config import DEFAULT_CONFIG_PATH, config_path, config_value, load_config
from Segmentation.data import collect_image_mask_pairs
from Segmentation.getMask import labelme_json_to_dataset
from Segmentation.train import DINOv3SemanticSeg

CLASS_NAMES = ["background", "cherry", "daylily", "hydrangeas"]


class TestDataset(Dataset):
    def __init__(self, img_dir: str | Path, mask_dir: str | Path, transform: A.Compose):
        self.samples = collect_image_mask_pairs(img_dir, mask_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long(), Path(image_path).name


def evaluation_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def ensure_evaluation_masks(input_dir: Path) -> Path:
    """Return evaluation masks, generating derived masks from Labelme JSON when needed."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Evaluation image directory does not exist: {input_dir}")

    mask_dir = input_dir / "masks"
    if mask_dir.is_dir():
        return mask_dir

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_dir}. "
            "Cannot create masks because no Labelme JSON files were found."
        )

    print(f"Mask directory not found. Creating derived masks from {len(json_files)} Labelme JSON files.")
    labelme_json_to_dataset(str(input_dir))
    if not mask_dir.is_dir():
        raise RuntimeError(f"Mask creation did not produce the expected directory: {mask_dir}")
    return mask_dir


def metrics_from_confusion_matrix(matrix: np.ndarray) -> dict:
    """Calculate dataset-level segmentation metrics from a confusion matrix."""
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive

    union = true_positive + false_positive + false_negative
    cardinality = 2 * true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    dice = np.divide(
        2 * true_positive,
        cardinality,
        out=np.full_like(true_positive, np.nan),
        where=cardinality > 0,
    )
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.full_like(true_positive, np.nan),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.full_like(true_positive, np.nan),
        where=(true_positive + false_negative) > 0,
    )

    valid_iou = iou[~np.isnan(iou)]
    valid_dice = dice[~np.isnan(dice)]
    total_pixels = matrix.sum()
    pixel_accuracy = float(true_positive.sum() / total_pixels) if total_pixels else 0.0
    return {
        "mIoU": float(valid_iou.mean()) if valid_iou.size else float("nan"),
        "pixel_accuracy": pixel_accuracy,
        "mean_dice": float(valid_dice.mean()) if valid_dice.size else float("nan"),
        "per_class": {
            class_name: {
                "iou": float(iou[index]),
                "dice": float(dice[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Pixel-level confusion matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_reports(
    output_dir: Path, overall_metrics: dict, per_image_results: list[dict], matrix: np.ndarray
) -> None:
    with (output_dir / "evaluation_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["mIoU", f"{overall_metrics['mIoU']:.4f}"])
        writer.writerow(["pixel_accuracy", f"{overall_metrics['pixel_accuracy']:.4f}"])
        writer.writerow(["mean_dice", f"{overall_metrics['mean_dice']:.4f}"])
        writer.writerow([])
        writer.writerow(["class", "iou", "dice", "precision", "recall"])
        for class_name, metrics in overall_metrics["per_class"].items():
            writer.writerow(
                [
                    class_name,
                    f"{metrics['iou']:.4f}",
                    f"{metrics['dice']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                ]
            )

    with (output_dir / "per_image_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "mIoU", "pixel_accuracy", "mean_dice"])
        writer.writeheader()
        for result in per_image_results:
            writer.writerow(
                {
                    "image": result["image"],
                    "mIoU": f"{result['mIoU']:.4f}",
                    "pixel_accuracy": f"{result['pixel_accuracy']:.4f}",
                    "mean_dice": f"{result['mean_dice']:.4f}",
                }
            )

    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png")


def evaluate_model(
    model_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 1,
) -> dict:
    model_path = Path(model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint does not exist: {model_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    mask_dir = ensure_evaluation_masks(input_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3SemanticSeg(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    dataset = TestDataset(input_dir, mask_dir, evaluation_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    global_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    per_image_results = []

    with torch.no_grad():
        for images, masks, image_names in loader:
            predictions = torch.argmax(model(images.to(device)), dim=1).cpu().numpy()
            targets = masks.numpy()
            for prediction, target, image_name in zip(predictions, targets, image_names):
                image_matrix = confusion_matrix(
                    target.ravel(), prediction.ravel(), labels=range(len(CLASS_NAMES))
                )
                global_matrix += image_matrix
                image_metrics = metrics_from_confusion_matrix(image_matrix)
                per_image_results.append({"image": image_name, **image_metrics})

    overall_metrics = metrics_from_confusion_matrix(global_matrix)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_reports(output_dir, overall_metrics, per_image_results, global_matrix)

    print(f"Evaluation results saved to: {output_dir}")
    print(f"mIoU: {overall_metrics['mIoU']:.4f}")
    print(f"Pixel accuracy: {overall_metrics['pixel_accuracy']:.4f}")
    print(f"Mean Dice: {overall_metrics['mean_dice']:.4f}")
    return overall_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a semantic-segmentation model.")
    parser.add_argument("model_path", type=Path, nargs="?", help="Path to best_model.pth.")
    parser.add_argument("input_dir", type=Path, nargs="?", help="Directory containing images and masks/.")
    parser.add_argument("output_dir", type=Path, nargs="?", help="Directory for CSV and confusion-matrix outputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    args.model_path = args.model_path or config_path(config, "segmentation", "checkpoint_path")
    args.input_dir = args.input_dir or config_path(config, "segmentation", "test_data")
    args.output_dir = args.output_dir or config_path(config, "evaluation", "output_dir")
    if args.batch_size is None:
        args.batch_size = int(config_value(config, "evaluation", "batch_size"))
    return args


if __name__ == "__main__":
    arguments = parse_args()
    evaluate_model(
        arguments.model_path,
        arguments.input_dir,
        arguments.output_dir,
        batch_size=arguments.batch_size,
    )
