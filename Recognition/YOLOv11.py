"""Train the YOLO flower-stage classifier using the shared TOML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from project_config import DEFAULT_CONFIG_PATH, config_path, config_value, load_config
from Recognition.classifier import BloomYOLO as YOLO
from Recognition.prepare_data import resolve_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLO flower-stage classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Use an already cleaned train/val/test root directly")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data without loading or training a model")
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    args.data_dir = args.data_dir or config_path(config, "classification", "train_data")
    args.base_model = config_value(config, "classification", "base_model")
    args.project_dir = config_path(config, "classification", "training_project_dir")
    args.run_name = config_value(config, "classification", "training_run_name")
    args.image_size = args.image_size or int(config_value(config, "classification", "image_size"))
    args.batch_size = int(config_value(config, "classification", "batch_size"))
    args.patience = int(config_value(config, "classification", "patience"))
    if args.epochs is None:
        args.epochs = int(config_value(config, "classification", "epochs"))
    if args.device is None:
        args.device = config_value(config, "classification", "device")
    return args


def training_device(requested_device: str) -> str | int | None:
    if requested_device == "auto":
        return None
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        return 0
    raise ValueError(f"Unsupported training device: {requested_device}")


def training_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "data": str(args.data_dir),
        "epochs": args.epochs,
        "imgsz": args.image_size,
        "batch": args.batch_size,
        "patience": args.patience,
        "project": str(args.project_dir),
        "name": args.run_name,
        "auto_augment": None,
        "erasing": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "scale": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }
    device = training_device(args.device)
    if device is not None:
        kwargs["device"] = device
    return kwargs


def prepare_training_data(args: argparse.Namespace) -> Path:
    return resolve_training_data(
        args.data_dir,
        getattr(args, "prepared_dir", None),
        getattr(args, "val_fraction", 0.2),
        getattr(args, "split_seed", 42),
    )


def train_classifier(args: argparse.Namespace) -> Path:
    data_dir = prepare_training_data(args)

    model = YOLO(args.base_model)
    kwargs = training_kwargs(args)
    kwargs["data"] = str(data_dir)
    model.train(**kwargs)
    best_model_path = Path(model.trainer.best)
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Training did not produce a best checkpoint: {best_model_path}")
    print(f"Training complete. Best checkpoint: {best_model_path}")
    return best_model_path


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        print(f"Prepared classification dataset: {prepare_training_data(args)}")
    else:
        train_classifier(args)


if __name__ == "__main__":
    main()
