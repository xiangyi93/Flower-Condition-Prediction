"""Train the U-Net + ResNet34 comparison model."""

from __future__ import annotations

import argparse
from pathlib import Path

from Segmentation.experiment_runner import train_model
from unet_resnet34.model import UNetResNet34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("Segmentation/datasets/train_data/traindata0929"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("unet_resnet34/output/training")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        lambda: UNetResNet34(pretrained=not args.no_pretrained),
        "U-Net + ResNet34 (ImageNet pretrained)"
        if not args.no_pretrained
        else "U-Net + ResNet34 (random initialization)",
        args.train_data,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.val_split,
        args.seed,
    )
