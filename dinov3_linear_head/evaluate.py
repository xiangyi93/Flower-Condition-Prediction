"""Evaluate the DINOv3 linear-probe segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from dinov3_linear_head.model import DINOv3LinearSeg
from Segmentation.experiment_runner import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("dinov3_linear_head/output/training/best_model.pth"),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("Segmentation/datasets/test_img/testdata0929"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dinov3_linear_head/output/evaluation")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        DINOv3LinearSeg,
        args.checkpoint,
        args.test_data,
        args.output_dir,
        args.batch_size,
    )
