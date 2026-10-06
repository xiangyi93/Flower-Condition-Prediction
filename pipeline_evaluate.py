"""Evaluate the complete DINOv3 -> filtering -> YOLO bloom pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from label_region_export.export_label_regions import collect_annotations, read_sample
from pipeline_predict import (
    CLASS_NAMES,
    load_segmentation_model,
    predict_mask,
    probability_dict,
    resolve_device,
)
from project_config import DEFAULT_CONFIG_PATH, config_path, config_value, load_config
from Recognition.classifier import BloomYOLO
from Recognition.preprocessing import ForegroundLetterbox, rgb_on_black
from Segmentation.evaluate import metrics_from_confusion_matrix

SPECIES_ORDER = ("cherry", "hydrangeas", "daylily")
STAGE_ORDER = ("green", "half", "full")
STAGE_ZH = {"green": "未開花", "half": "半開", "full": "盛開"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_sample_number(filename: str) -> int:
    """Extract the six-digit source number retained in classification crops."""
    match = re.search(r"(?:^|__)(\d{6})(?:\.|__)", filename)
    if not match:
        raise ValueError(f"Cannot recover source number from classification image: {filename}")
    return int(match.group(1))


def load_stage_truth(test_root: Path) -> tuple[dict[tuple[int, str], str], dict[tuple[int, str], Path]]:
    truth: dict[tuple[int, str], str] = {}
    paths: dict[tuple[int, str], Path] = {}
    if not test_root.is_dir():
        raise FileNotFoundError(f"Classification test split does not exist: {test_root}")
    for class_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        try:
            species, stage = class_dir.name.rsplit("_", 1)
        except ValueError as error:
            raise ValueError(f"Invalid classification directory: {class_dir.name}") from error
        if species not in SPECIES_ORDER or stage not in STAGE_ORDER:
            raise ValueError(f"Unexpected classification directory: {class_dir.name}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = (parse_sample_number(path.name), species)
            if key in truth:
                raise ValueError(f"Duplicate stage truth for source/species {key}: {path}")
            truth[key] = stage
            paths[key] = path
    if not truth:
        raise ValueError(f"No classification truth images found in {test_root}")
    return truth, paths


def validate_reference_crop(
    crop_path: Path, image_rgb: np.ndarray, true_mask: np.ndarray, class_index: int
) -> tuple[bool, float]:
    """Confirm that a labelled crop is the matching ground-truth region."""
    with Image.open(crop_path) as image:
        crop = np.asarray(rgb_on_black(image))
    expected = np.where((true_mask == class_index)[..., None], image_rgb, 0).astype(np.uint8)
    if crop.shape != expected.shape:
        return False, 0.0
    matching = np.all(crop == expected, axis=2)
    return bool(matching.all()), float(matching.mean())


def validate_classifier(model: BloomYOLO):
    transforms = getattr(model.model, "transforms", None)
    if transforms is None or not isinstance(transforms.transforms[0], ForegroundLetterbox):
        raise ValueError("YOLO checkpoint must use the trained foreground preprocessing")
    expected = {f"{species}_{stage}" for species in SPECIES_ORDER for stage in STAGE_ORDER}
    if set(model.names.values()) != expected:
        raise ValueError("YOLO checkpoint must contain all nine species/stage classes")
    return transforms


def stage_from_result(result) -> dict:
    probabilities = probability_dict(result)
    stage_probabilities = {
        stage: sum(probabilities[f"{species}_{stage}"] for species in SPECIES_ORDER)
        for stage in STAGE_ORDER
    }
    stage = max(STAGE_ORDER, key=stage_probabilities.get)
    top1 = int(result.probs.top1)
    top1_class = str(result.names[top1])
    return {
        "predicted_stage": stage,
        "stage_confidence": stage_probabilities[stage],
        "top1_class": top1_class,
        "top1_probability": float(result.probs.top1conf.item()),
        "predicted_species": top1_class.rsplit("_", 1)[0],
        **{f"prob_{name}": value for name, value in stage_probabilities.items()},
    }


def binary_counts(rows: list[dict], threshold: float) -> dict[str, int | float]:
    counts = Counter()
    for row in rows:
        predicted = row["foreground_pixels"] > 0 and row["foreground_ratio"] >= threshold
        truth = row["gt_present"]
        counts[(truth, predicted)] += 1
    tp, fn = counts[(True, True)], counts[(True, False)]
    fp, tn = counts[(False, True)], counts[(False, False)]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def stage_metrics(rows: list[dict], threshold: float) -> dict[str, float | int]:
    labelled = [row for row in rows if row["true_stage"]]
    eligible = [
        row for row in labelled
        if row["foreground_pixels"] > 0 and row["foreground_ratio"] >= threshold
    ]
    correct = sum(row["predicted_stage"] == row["true_stage"] for row in eligible)
    conditional_accuracy = correct / len(eligible) if eligible else 0.0
    system_accuracy = correct / len(labelled) if labelled else 0.0
    macro_f1 = (
        f1_score(
            [row["true_stage"] for row in eligible],
            [row["predicted_stage"] for row in eligible],
            labels=list(STAGE_ORDER),
            average="macro",
            zero_division=0,
        )
        if eligible else 0.0
    )
    return {
        "labelled": len(labelled), "eligible": len(eligible), "skipped": len(labelled) - len(eligible),
        "coverage": len(eligible) / len(labelled) if labelled else 0.0,
        "correct": correct, "conditional_accuracy": conditional_accuracy,
        "conditional_macro_f1": float(macro_f1), "system_accuracy": system_accuracy,
    }


def threshold_rows(rows: list[dict], thresholds: list[float]) -> list[dict]:
    output = []
    for threshold in thresholds:
        output.append({"threshold": threshold, **binary_counts(rows, threshold), **stage_metrics(rows, threshold)})
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"fieldnames are required for an empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(matrix: np.ndarray, xlabels: list[str], ylabels: list[str], title: str, path: Path) -> None:
    width = max(6.0, len(xlabels) * 1.5)
    figure, axis = plt.subplots(figsize=(width, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    axis.set_yticks(range(len(ylabels)), ylabels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    cutoff = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > cutoff else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def metrics_for_subset(rows: list[dict], threshold: float) -> dict:
    presence = binary_counts(rows, threshold)
    stages = stage_metrics(rows, threshold)
    return {"presence": presence, "stages": stages}


def classification_breakdown(rows: list[dict]) -> tuple[dict, dict, dict]:
    by_species = {}
    for species in SPECIES_ORDER:
        labelled = [row for row in rows if row["species"] == species and row["true_stage"]]
        eligible = [row for row in labelled if row["status"] == "ok"]
        correct = sum(row["predicted_stage"] == row["true_stage"] for row in eligible)
        by_species[species] = {
            "labelled": len(labelled), "eligible": len(eligible), "correct": correct,
            "conditional_accuracy": correct / len(eligible) if eligible else 0.0,
            "system_accuracy": correct / len(labelled) if labelled else 0.0,
        }
    by_stage = {}
    for stage in STAGE_ORDER:
        labelled = [row for row in rows if row["true_stage"] == stage]
        eligible = [row for row in labelled if row["status"] == "ok"]
        correct = sum(row["predicted_stage"] == stage for row in eligible)
        by_stage[stage] = {
            "labelled": len(labelled), "eligible": len(eligible), "correct": correct,
            "conditional_recall": correct / len(eligible) if eligible else 0.0,
            "system_recall": correct / len(labelled) if labelled else 0.0,
        }
    eligible = [row for row in rows if row["true_stage"] and row["status"] == "ok"]
    wrong = [row for row in eligible if row["predicted_stage"] != row["true_stage"]]
    correct = [row for row in eligible if row["predicted_stage"] == row["true_stage"]]
    confidence = {
        "wrong": len(wrong),
        "wrong_at_least_0_90": sum(row["stage_confidence"] >= 0.9 for row in wrong),
        "wrong_mean": float(np.mean([row["stage_confidence"] for row in wrong])) if wrong else 0.0,
        "wrong_max": max((row["stage_confidence"] for row in wrong), default=0.0),
        "correct_mean": float(np.mean([row["stage_confidence"] for row in correct])) if correct else 0.0,
    }
    return by_species, by_stage, confidence


def evaluate_pipeline(
    segmentation_model_path: Path,
    classification_model_path: Path,
    image_root: Path,
    classification_test_root: Path,
    output_dir: Path,
    min_area_ratio: float,
    requested_device: str = "auto",
) -> dict:
    if not 0 <= min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be between 0 and 1")
    for path, description in (
        (segmentation_model_path, "Segmentation checkpoint"),
        (classification_model_path, "Classification checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}")
    annotations = collect_annotations(image_root)
    truth, truth_paths = load_stage_truth(classification_test_root)
    if max(number for number, _ in truth) > len(annotations):
        raise ValueError("Classification source number exceeds the segmentation test set size")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(requested_device)
    segmentation_model = load_segmentation_model(segmentation_model_path, device)
    classification_model = BloomYOLO(str(classification_model_path))
    transforms = validate_classifier(classification_model)

    pixel_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    rows: list[dict] = []
    pending: list[tuple[dict, torch.Tensor]] = []
    reference_audit: list[dict] = []

    for number, annotation in enumerate(annotations, 1):
        image_path, image_rgb, true_mask = read_sample(annotation)
        predicted_mask = predict_mask(segmentation_model, image_rgb, device)
        pixel_matrix += confusion_matrix(
            true_mask.ravel(), predicted_mask.ravel(), labels=range(len(CLASS_NAMES))
        )
        total_pixels = true_mask.size
        for species in SPECIES_ORDER:
            class_index = CLASS_NAMES.index(species)
            gt_pixels = int(np.count_nonzero(true_mask == class_index))
            predicted_binary = predicted_mask == class_index
            predicted_pixels = int(predicted_binary.sum())
            segment_rgb = np.where(predicted_binary[..., None], image_rgb, 0).astype(np.uint8)
            foreground_pixels = int(np.any(segment_rgb != 0, axis=2).sum())
            key = (number, species)
            row = {
                "source_number": number, "image": image_path.name, "image_stem": image_path.stem,
                "species": species, "true_stage": truth.get(key), "gt_present": gt_pixels > 0,
                "gt_pixels": gt_pixels, "gt_area_ratio": gt_pixels / total_pixels,
                "predicted_pixels": predicted_pixels, "predicted_area_ratio": predicted_pixels / total_pixels,
                "foreground_pixels": foreground_pixels, "foreground_ratio": foreground_pixels / total_pixels,
                "status": "all_black" if foreground_pixels == 0 else "predicted",
                "predicted_stage": None, "stage_confidence": None, "top1_class": None,
                "top1_probability": None, "predicted_species": None,
                "species_agreement": None, "prob_green": None, "prob_half": None, "prob_full": None,
            }
            rows.append(row)
            if foreground_pixels:
                pending.append((row, transforms(Image.fromarray(segment_rgb))))
            if key in truth_paths:
                exact, fraction = validate_reference_crop(
                    truth_paths[key], image_rgb, true_mask, class_index
                )
                reference_audit.append({
                    "source_number": number, "image": image_path.name, "species": species,
                    "stage": truth[key], "crop": str(truth_paths[key]),
                    "exact_pixel_match": exact, "matching_pixel_fraction": fraction,
                })

    # A tiny labelled region may intentionally have no bloom-stage annotation and
    # still belongs in segmentation/presence evaluation.  The reverse direction
    # is invalid: a stage label cannot refer to a species absent from the mask.
    inconsistent = [row for row in rows if row["true_stage"] and not row["gt_present"]]
    if inconsistent:
        examples = [(row["image"], row["species"], row["true_stage"], row["gt_pixels"])
                    for row in inconsistent[:5]]
        raise ValueError(f"Stage truth and segmentation presence disagree: {examples}")

    if pending:
        tensors = torch.stack([tensor for _, tensor in pending])
        results = classification_model.predict(
            tensors, imgsz=transforms.transforms[0].size, device=str(device), verbose=False
        )
        if len(results) != len(pending):
            raise RuntimeError("YOLO result count does not match the number of segmented regions")
        for (row, _), result in zip(pending, results):
            row.update(stage_from_result(result))
            row["species_agreement"] = row["predicted_species"] == row["species"]

    for row in rows:
        if row["foreground_pixels"] == 0:
            row["status"] = "all_black"
        elif row["foreground_ratio"] < min_area_ratio:
            row["status"] = "area_too_small"
        else:
            row["status"] = "ok"
        row["stage_correct"] = bool(
            row["true_stage"] and row["status"] == "ok"
            and row["predicted_stage"] == row["true_stage"]
        )

    segmentation_metrics = metrics_from_confusion_matrix(pixel_matrix)
    thresholds = sorted(set([0.0, 0.0001, 0.0005, min_area_ratio, 0.002, 0.005, 0.01]))
    sweep = threshold_rows(rows, thresholds)
    primary = metrics_for_subset(rows, min_area_ratio)
    by_species, by_stage, confidence = classification_breakdown(rows)

    stage_eligible = [row for row in rows if row["true_stage"] and row["status"] == "ok"]
    stage_matrix = confusion_matrix(
        [row["true_stage"] for row in stage_eligible],
        [row["predicted_stage"] for row in stage_eligible], labels=list(STAGE_ORDER),
    )
    stage_system_matrix = np.zeros((3, 4), dtype=np.int64)
    for row in (item for item in rows if item["true_stage"]):
        true_index = STAGE_ORDER.index(row["true_stage"])
        predicted_index = (STAGE_ORDER.index(row["predicted_stage"])
                           if row["status"] == "ok" else 3)
        stage_system_matrix[true_index, predicted_index] += 1

    display_seg_order = [0, CLASS_NAMES.index("cherry"), CLASS_NAMES.index("hydrangeas"), CLASS_NAMES.index("daylily")]
    display_seg_matrix = pixel_matrix[np.ix_(display_seg_order, display_seg_order)]
    display_seg_labels = [CLASS_NAMES[index] for index in display_seg_order]
    plot_matrix(display_seg_matrix, display_seg_labels, display_seg_labels,
                "DINOv3 pixel confusion matrix", output_dir / "segmentation_confusion.png")
    plot_matrix(stage_matrix, list(STAGE_ORDER), list(STAGE_ORDER),
                "YOLO bloom-stage confusion (eligible regions)", output_dir / "stage_confusion.png")
    plot_matrix(stage_system_matrix, [*STAGE_ORDER, "skipped"], list(STAGE_ORDER),
                "End-to-end bloom-stage confusion", output_dir / "stage_system_confusion.png")

    write_csv(output_dir / "per_image_species.csv", rows)
    write_csv(output_dir / "reference_crop_audit.csv", reference_audit)
    write_csv(output_dir / "threshold_sweep.csv", sweep)
    write_csv(output_dir / "segmentation_confusion.csv", [
        {"true_class": CLASS_NAMES[row], **{CLASS_NAMES[column]: int(pixel_matrix[row, column])
         for column in range(len(CLASS_NAMES))}}
        for row in range(len(CLASS_NAMES))
    ])
    write_csv(output_dir / "stage_confusion.csv", [
        {"true_stage": STAGE_ORDER[row], **{STAGE_ORDER[column]: int(stage_matrix[row, column])
         for column in range(len(STAGE_ORDER))}}
        for row in range(len(STAGE_ORDER))
    ])

    summary = {
        "scope": {
            "images": len(annotations), "image_species_pairs": len(rows),
            "stage_labelled_pairs": sum(bool(row["true_stage"]) for row in rows),
            "gt_present_pairs": sum(bool(row["gt_present"]) for row in rows),
            "gt_present_without_stage": sum(
                bool(row["gt_present"]) and not row["true_stage"] for row in rows
            ),
            "device": str(device), "min_area_ratio": min_area_ratio,
            "segmentation_model": str(segmentation_model_path),
            "classification_model": str(classification_model_path),
        },
        "segmentation": segmentation_metrics,
        "pipeline": primary,
        "classification_by_species": by_species,
        "classification_by_stage": by_stage,
        "confidence_diagnostic": confidence,
        "reference_crop_audit": {
            "audited": len(reference_audit),
            "exact_matches": sum(row["exact_pixel_match"] for row in reference_audit),
            "minimum_matching_pixel_fraction": min(
                (row["matching_pixel_fraction"] for row in reference_audit), default=0.0
            ),
        },
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "species_agreement": {
            species: {
                "n": len(species_rows := [row for row in rows if row["species"] == species and row["status"] == "ok"]),
                "agree": sum(row["species_agreement"] for row in species_rows),
            }
            for species in SPECIES_ORDER
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(output_dir / "report.md", summary, sweep)
    return summary


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_markdown_report(path: Path, summary: dict, sweep: list[dict]) -> None:
    segmentation = summary["segmentation"]
    pipeline = summary["pipeline"]
    stages = pipeline["stages"]
    presence = pipeline["presence"]
    audit = summary["reference_crop_audit"]
    by_species = summary["classification_by_species"]
    by_stage = summary["classification_by_stage"]
    confidence = summary["confidence_diagnostic"]
    lines = [
        "# 完整影像流程測試評估報告", "",
        "## 評估範圍", "",
        f"- 測試原圖：{summary['scope']['images']} 張，逐張執行 DINOv3 分割、面積過濾與 YOLOv11 花況辨識。",
        f"- 植物判斷單位：{summary['scope']['image_species_pairs']} 組原圖 × 植物；其中 {summary['scope']['gt_present_pairs']} 組人工遮罩含植物，{summary['scope']['stage_labelled_pairs']} 組有花況真值。",
        f"- 面積門檻：`{summary['scope']['min_area_ratio']}`；裝置：`{summary['scope']['device']}`。",
        "- 信心值是 9 個聯合類別中，同花況跨植物機率的總和，不是盛開百分比。", "",
        "## 主要結果", "",
        "| 指標 | 結果 |", "|---|---:|",
        f"| DINOv3 mIoU | {percent(segmentation['mIoU'])} |",
        f"| DINOv3 mean Dice | {percent(segmentation['mean_dice'])} |",
        f"| DINOv3 pixel accuracy | {percent(segmentation['pixel_accuracy'])} |",
        f"| 有植物區域送入 YOLO 的涵蓋率 | {stages['eligible']}/{stages['labelled']} ({percent(stages['coverage'])}) |",
        f"| YOLO 條件式花況正確率 | {stages['correct']}/{stages['eligible']} ({percent(stages['conditional_accuracy'])}) |",
        f"| YOLO 條件式花況 macro-F1 | {percent(stages['conditional_macro_f1'])} |",
        f"| 端到端花況正確率 | {stages['correct']}/{stages['labelled']} ({percent(stages['system_accuracy'])}) |",
        f"| 植物存在判斷 precision / recall | {percent(presence['precision'])} / {percent(presence['recall'])} |", "",
        "條件式花況正確率只計算成功通過分割與門檻的影像；端到端正確率把漏分割及面積過小也算錯，較接近實際部署表現。", "",
        "## 分割類別結果", "", "| 類別 | IoU | Dice | Precision | Recall |", "|---|---:|---:|---:|---:|",
    ]
    for name in ("background", *SPECIES_ORDER):
        values = segmentation["per_class"][name]
        lines.append(f"| {name} | {percent(values['iou'])} | {percent(values['dice'])} | {percent(values['precision'])} | {percent(values['recall'])} |")
    lines += ["", "## 花況辨識分項", "", "| 植物 | 可判斷／有真值 | 條件式正確率 | 端到端正確率 |", "|---|---:|---:|---:|"]
    for species in SPECIES_ORDER:
        values = by_species[species]
        lines.append(
            f"| {species} | {values['eligible']}/{values['labelled']} | "
            f"{percent(values['conditional_accuracy'])} | {percent(values['system_accuracy'])} |"
        )
    lines += ["", "| 真實花況 | 判對／可判斷 | 條件式 Recall | 端到端 Recall |", "|---|---:|---:|---:|"]
    for stage in STAGE_ORDER:
        values = by_stage[stage]
        lines.append(
            f"| {stage}（{STAGE_ZH[stage]}） | {values['correct']}/{values['eligible']} | "
            f"{percent(values['conditional_recall'])} | {percent(values['system_recall'])} |"
        )
    lines += [
        "",
        f"共有 {confidence['wrong']} 組通過門檻但花況判錯，其中 {confidence['wrong_at_least_0_90']} 組信心仍達 90%；錯誤案例最高信心為 {percent(confidence['wrong_max'])}。現有 confidence 不宜直接當成可靠度或自動放行門檻。",
    ]
    lines += ["", "## 花況辨識分項", "", "| 植物 | 可判斷／有真值 | 條件式正確率 | 端到端正確率 |", "|---|---:|---:|---:|"]
    for species in SPECIES_ORDER:
        values = by_species[species]
        lines.append(
            f"| {species} | {values['eligible']}/{values['labelled']} | "
            f"{percent(values['conditional_accuracy'])} | {percent(values['system_accuracy'])} |"
        )
    lines += ["", "| 真實花況 | 判對／可判斷 | 條件式 Recall | 端到端 Recall |", "|---|---:|---:|---:|"]
    for stage in STAGE_ORDER:
        values = by_stage[stage]
        lines.append(
            f"| {stage}（{STAGE_ZH[stage]}） | {values['correct']}/{values['eligible']} | "
            f"{percent(values['conditional_recall'])} | {percent(values['system_recall'])} |"
        )
    lines += [
        "",
        f"共有 {confidence['wrong']} 組通過門檻但花況判錯，其中 {confidence['wrong_at_least_0_90']} 組信心仍達 90%；錯誤案例最高信心為 {percent(confidence['wrong_max'])}。現有 confidence 不宜直接當成可靠度或自動放行門檻。",
    ]
    lines += ["", "## 面積門檻敏感度", "", "| 門檻 | 涵蓋率 | 條件式正確率 | 端到端正確率 | Presence precision | Presence recall |", "|---:|---:|---:|---:|---:|---:|"]
    for row in sweep:
        lines.append(
            f"| {row['threshold']:.4f} | {percent(row['coverage'])} | "
            f"{percent(row['conditional_accuracy'])} | {percent(row['system_accuracy'])} | "
            f"{percent(row['precision'])} | {percent(row['recall'])} |"
        )
    lines += [
        "", "## 資料與可追溯性檢查", "",
        f"- 花況標籤影像與相同編號、相同植物的人工分割區域逐像素比對：{audit['exact_matches']}/{audit['audited']} 完全一致；最低像素一致率 {percent(audit['minimum_matching_pixel_fraction'])}。",
        f"- 有 {summary['scope']['gt_present_without_stage']} 組人工遮罩含植物但沒有花況標籤；保留在分割與植物存在評估，不納入花況準確率分母。",
        "- 評估直接呼叫正式 pipeline 的 DINO `predict_mask`，YOLO 則讀取 checkpoint 內保存的 `ForegroundLetterbox`。分類訓練、驗證與推論使用相同裁切、縮放及填黑幾何；只有訓練集額外使用水平翻轉。",
        "## 結論與判讀", "",
        "- 端到端正確率同時反映分割門檻與花況分類造成的錯誤，應與涵蓋率及條件式正確率一併判讀。",
        "- 面積門檻應使用 validation 或獨立校正集決定，不應依同一 test 的最佳結果調整。",
        "- 各花種與各階段的分項結果可用來安排後續資料補強與錯誤分析。",
        "- 實際應用應保留 `all_black`、`area_too_small` 與 DINO／YOLO 花種不一致警示；高 confidence 仍可能判錯。", "",
        "## 輸出檔案", "",
        "- `per_image_species.csv`：每張原圖、每種植物的完整中間結果與錯誤來源。",
        "- `threshold_sweep.csv`：面積門檻敏感度。",
        "- `reference_crop_audit.csv`：花況標籤與原圖人工遮罩的映射稽核。",
        "- `segmentation_confusion.png`、`stage_confusion.png`、`stage_system_confusion.png`：混淆矩陣圖。",
        "- `summary.json`：機器可讀的完整摘要。", "",
        "## 判讀限制", "",
        "- 目前測試集只有 37 張原圖，且 full、half 類別樣本少；整體數字容易受少數圖片影響。",
        "- 花況真值來自人工分割後的分類測試集，因此可以評估現有場景的端到端流程，但不能替代新地點、新拍攝日期的外部測試。",
        "- 本次只評估離線圖片；實際串接 DINO 三張輸出時仍需保留全黑、比例過小與低信心警示。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seg-model", type=Path)
    parser.add_argument("--cls-model", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--classification-test", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/pipeline_evaluation"))
    parser.add_argument("--min-area-ratio", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    args.seg_model = args.seg_model or config_path(config, "segmentation", "checkpoint_path")
    args.cls_model = args.cls_model or config_path(config, "classification", "checkpoint_path")
    args.images = args.images or config_path(config, "segmentation", "test_data")
    args.classification_test = args.classification_test or (
        config_path(config, "classification", "train_data") / "test"
    )
    args.min_area_ratio = (args.min_area_ratio if args.min_area_ratio is not None
                           else float(config_value(config, "pipeline", "min_area_ratio")))
    args.device = args.device or config_value(config, "pipeline", "device")
    return args


def main() -> None:
    args = parse_args()
    summary = evaluate_pipeline(
        args.seg_model, args.cls_model, args.images, args.classification_test,
        args.output_dir, args.min_area_ratio, args.device,
    )
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    print(f"End-to-end stage accuracy: {summary['pipeline']['stages']['system_accuracy']:.4f}")


if __name__ == "__main__":
    main()
