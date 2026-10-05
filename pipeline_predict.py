"""Run semantic segmentation followed by flower-stage classification on one image."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from project_config import DEFAULT_CONFIG_PATH, config_path, config_value, load_config
from Recognition.classifier import BloomYOLO
from Recognition.preprocessing import ForegroundLetterbox, rgb_on_black
from Segmentation.train import DINOv3SemanticSeg

CLASS_NAMES = ["background", "cherry", "daylily", "hydrangeas"]
CLASS_COLORS_RGB = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [255, 255, 0],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment one image and classify each detected flower type."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--seg-model",
        type=Path,
        default=None,
        help="Segmentation checkpoint (defaults to segmentation.checkpoint_path in config).",
    )
    parser.add_argument(
        "--cls-model",
        type=Path,
        default=None,
        help="Classification checkpoint (defaults to classification.checkpoint_path in config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results (default: output/pipeline/<image stem>).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=None,
        help="Skip classification when a flower mask covers less than this image ratio.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Inference device for both segmentation and classification.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    args.seg_model = args.seg_model or config_path(config, "segmentation", "checkpoint_path")
    args.cls_model = args.cls_model or config_path(config, "classification", "checkpoint_path")
    args.pipeline_output_root = config_path(config, "pipeline", "output_root")
    if args.min_area_ratio is None:
        args.min_area_ratio = float(config_value(config, "pipeline", "min_area_ratio"))
    if args.device is None:
        args.device = config_value(config, "pipeline", "device")
    if not 0.0 <= args.min_area_ratio <= 1.0:
        parser.error("--min-area-ratio must be between 0 and 1.")
    return args


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def segmentation_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_segmentation_model(model_path: Path, device: torch.device) -> DINOv3SemanticSeg:
    model = DINOv3SemanticSeg(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def predict_mask(
    model: DINOv3SemanticSeg, image_rgb: np.ndarray, device: torch.device
) -> np.ndarray:
    tensor = segmentation_transform()(image=image_rgb)["image"].unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor.to(device))
    prediction_224 = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    height, width = image_rgb.shape[:2]
    return cv2.resize(
        prediction_224.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )


def colour_mask(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS_RGB[mask]


def probability_dict(result) -> dict[str, float]:
    probabilities = result.probs.data.detach().cpu().tolist()
    return {
        str(result.names[index]): float(probability)
        for index, probability in enumerate(probabilities)
    }


def run_pipeline(args: argparse.Namespace) -> dict:
    if not 0 <= args.min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be between 0 and 1")
    require_file(args.image, "Input image")
    require_file(args.seg_model, "Segmentation checkpoint")
    require_file(args.cls_model, "Classification checkpoint")

    output_dir = args.output_dir or args.pipeline_output_root / args.image.stem
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / "preprocessed"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    with Image.open(args.image) as image:
        image_rgb = np.array(rgb_on_black(image))
    segmentation_model = load_segmentation_model(args.seg_model, device)
    classification_model = BloomYOLO(str(args.cls_model))
    transforms = getattr(classification_model.model, "transforms", None)
    if transforms is None or not isinstance(transforms.transforms[0], ForegroundLetterbox):
        raise ValueError("YOLO checkpoint must use the trained foreground preprocessing")
    expected = {f"{species}_{stage}" for species in CLASS_NAMES[1:] for stage in ("green", "half", "full")}
    if set(classification_model.names.values()) != expected:
        raise ValueError("YOLO checkpoint must contain all nine species/stage classes")
    mask = predict_mask(segmentation_model, image_rgb, device)

    colour = colour_mask(mask)
    overlay = cv2.addWeighted(image_rgb, 0.65, colour, 0.35, 0.0)
    Image.fromarray(mask, mode="L").save(output_dir / "segmentation_mask.png")
    Image.fromarray(colour).save(output_dir / "segmentation_colour.png")
    Image.fromarray(overlay).save(output_dir / "segmentation_overlay.png")

    total_pixels = mask.size
    classifications = {}
    segment_paths = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_mask = mask == class_index
        area_ratio = float(binary_mask.sum() / total_pixels)
        segment_rgb = np.where(binary_mask[..., None], image_rgb, 0).astype(np.uint8)
        segment_path = segments_dir / f"{class_name}.png"
        Image.fromarray(segment_rgb).save(segment_path)
        segment_paths[class_name] = str(segment_path)
        if class_name == "background":
            continue
        foreground_pixels = int(np.any(segment_rgb != 0, axis=2).sum())
        foreground_ratio = foreground_pixels / total_pixels
        prepared_path = prepared_dir / f"{class_name}.png"
        # Remove only our own stale diagnostic output when reusing an output directory.
        if prepared_path.is_file():
            prepared_path.unlink()
        record = {
            "area_ratio": area_ratio, "foreground_ratio": foreground_ratio,
            "foreground_pixels": foreground_pixels, "min_area_ratio": args.min_area_ratio,
            "segment_image": str(segment_path), "preprocessed_image": None,
            "status": "skipped", "reason": None, "stage": None,
            "stage_label": None, "confidence": None, "warning": None,
        }
        classifications[class_name] = record
        if foreground_pixels == 0:
            record["reason"] = "全黑：未取得有效植物影像"
            record["skip_code"] = "all_black"
            continue
        if foreground_ratio < args.min_area_ratio:
            record["reason"] = "有效前景占原圖比例過小，無法判斷花況"
            record["skip_code"] = "area_too_small"
            continue
        tensor = transforms(Image.fromarray(segment_rgb))
        Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(prepared_path)
        result = classification_model.predict(
            tensor.unsqueeze(0), imgsz=transforms.transforms[0].size,
            device=str(device), verbose=False,
        )[0]
        probabilities = probability_dict(result)
        top1_index = result.probs.top1
        # Match Recognition.evaluate: marginalize joint probabilities over species.
        stage_probabilities = {
            stage: sum(probabilities[f"{species}_{stage}"] for species in CLASS_NAMES[1:])
            for stage in ("green", "half", "full")
        }
        stage = max(stage_probabilities, key=stage_probabilities.get)
        predicted_species = str(result.names[top1_index]).rsplit("_", 1)[0]
        record.update({
            "status": "ok", "stage": stage,
            "stage_label": {"green": "未開花", "half": "半開", "full": "盛開"}[stage],
            "confidence": stage_probabilities[stage],
            "stage_probabilities": stage_probabilities,
            "preprocessed_image": str(prepared_path),
            "warning": "DINO 與 YOLO Top-1 的植物種類不一致，請人工確認" if predicted_species != class_name else None,
            "top1_class": str(result.names[top1_index]),
            "top1_probability": float(result.probs.top1conf.item()),
            "probabilities": probabilities,
        })

    output = {
        "input_image": str(args.image),
        "segmentation_model": str(args.seg_model),
        "classification_model": str(args.cls_model),
        "device": str(device),
        "confidence_definition": "Sum joint-class probabilities over species for the selected stage; not bloom percentage or calibrated accuracy",
        "outputs": {
            "mask": str(output_dir / "segmentation_mask.png"),
            "colour": str(output_dir / "segmentation_colour.png"),
            "overlay": str(output_dir / "segmentation_overlay.png"),
            "segments": segment_paths,
            "summary_csv": str(output_dir / "summary.csv"),
        },
        "classifications": classifications,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as result_file:
        json.dump(output, result_file, ensure_ascii=False, indent=2)
    fields = ["species", "status", "stage", "stage_label", "confidence", "reason", "warning", "foreground_ratio", "area_ratio", "segment_image", "preprocessed_image"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for species, record in classifications.items():
            writer.writerow({"species": species, **{key: record.get(key) for key in fields[1:]}})
    return output


def main() -> None:
    output = run_pipeline(parse_args())
    print(f"Segmentation output: {output['outputs']['overlay']}")
    print(f"Classification probabilities: {Path(output['outputs']['overlay']).parent / 'results.json'}")
    for flower_name, result in output["classifications"].items():
        if result["status"] == "ok":
            print(
                f"{flower_name}: {result['stage_label']} "
                f"(模型信心 {result['confidence']:.2%})"
            )
            if result["warning"]:
                print(f"  警示：{result['warning']}")
        else:
            print(f"{flower_name}: 無法判斷；{result['reason']} (前景比例 {result['foreground_ratio']:.4%})")


if __name__ == "__main__":
    main()
