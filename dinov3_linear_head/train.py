"""Train a frozen DINOv3 backbone with a linear segmentation head."""

from __future__ import annotations

import argparse
from pathlib import Path

from dinov3_linear_head.model import DINOv3LinearSeg
from Segmentation.experiment_runner import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("Segmentation/datasets/train_data/traindata0929"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dinov3_linear_head/output/training")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        DINOv3LinearSeg,
        "Frozen DINOv3 + linear 1x1 segmentation head",
        args.train_data,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.val_split,
        args.seed,
    )
