"""Evaluate a bloom checkpoint on held-out data and export Excel-compatible CSVs."""

from __future__ import annotations

import argparse
import csv
import platform
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import ultralytics
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from project_config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, config_path, load_config
from Recognition.classifier import BloomYOLO
from Recognition.prepare_data import class_images, file_digest
from Recognition.preprocessing import ForegroundLetterbox, bloom_transforms

STAGES = ["green", "half", "full"]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_label(label: str) -> tuple[str, str]:
    species, stage = label.rsplit("_", 1)
    if stage not in STAGES:
        raise ValueError(f"Unsupported bloom label: {label}")
    return species, stage


def prediction_row(path: Path, truth: str, names: list[str], probabilities) -> dict:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(names),) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid probabilities")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-3):
        raise ValueError("Classification probabilities must sum to one")
    top1 = int(probabilities.argmax())
    species_names = sorted({split_label(name)[0] for name in names})
    stage_probs = {stage: 0.0 for stage in STAGES}
    species_probs = dict.fromkeys(species_names, 0.0)
    for name, probability in zip(names, probabilities):
        species, stage = split_label(name)
        stage_probs[stage] += float(probability)
        species_probs[species] += float(probability)
    true_species, true_stage = split_label(truth)
    predicted_stage = max(stage_probs, key=stage_probs.get)
    predicted_species = max(species_probs, key=species_probs.get)
    return {
        "image": str(path), "status": "ok", "error": "", "true_class": truth,
        "predicted_class": names[top1], "confidence": float(probabilities[top1]),
        "correct_class": int(names[top1] == truth),
        "true_species": true_species, "predicted_species": predicted_species,
        "correct_species": int(predicted_species == true_species),
        "true_stage": true_stage, "predicted_stage": predicted_stage,
        "stage_from_top1_class": split_label(names[top1])[1],
        "stage_confidence": stage_probs[predicted_stage],
        "correct_stage": int(predicted_stage == true_stage),
        **{f"prob_{name}": float(probability) for name, probability in zip(names, probabilities)},
        **{f"stage_prob_{stage}": value for stage, value in stage_probs.items()},
    }


def metric_tables(rows: list[dict], names: list[str]):
    """Invalid images stay in the total/coverage denominator, never silently disappear."""
    species_names = sorted({split_label(name)[0] for name in names})
    tasks = [("class", names, rows), ("species", species_names, rows), ("stage", STAGES, rows)]
    tasks += [(f"stage/{species}", STAGES, [r for r in rows if r["true_species"] == species]) for species in species_names]
    summary, per_class, matrices = [], [], {}
    for task, labels, task_rows in tasks:
        key = task.split("/")[0]
        evaluated = [r for r in task_rows if r["status"] == "ok"]
        truth = [r[f"true_{key}"] for r in evaluated]
        predicted = [r[f"predicted_{key}"] for r in evaluated]
        matrix = confusion_matrix(truth, predicted, labels=labels) if evaluated else np.zeros((len(labels), len(labels)), dtype=int)
        if "/" not in task:
            matrices[task] = (labels, matrix)
        if evaluated:
            precision, recall, f1, support = precision_recall_fscore_support(
                truth, predicted, labels=labels, zero_division=0
            )
        else:
            precision, recall, f1, support = (np.zeros(len(labels)) for _ in range(4))
        count = len(evaluated)
        correct = int(np.trace(matrix))
        summary.append({
            "task": task, "total_images": len(task_rows), "evaluated_images": count,
            "invalid_images": len(task_rows) - count,
            "coverage": count / len(task_rows) if task_rows else 0,
            "correct_images": correct, "accuracy": correct / count if count else "",
            "accuracy_all_inputs": correct / len(task_rows) if task_rows else "",
            "macro_precision": float(precision.mean()) if count else "",
            "macro_recall": float(recall.mean()) if count else "",
            "macro_f1": float(f1.mean()) if count else "",
            "balanced_accuracy": float(recall[support > 0].mean()) if count else "",
            "weighted_f1": float(np.dot(f1, support) / count) if count else "",
        })
        for index, label in enumerate(labels):
            per_class.append({
                "task": task, "label": label, "support": int(support[index]),
                "predicted_count": int(matrix[:, index].sum()),
                "true_positive": int(matrix[index, index]),
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]),
            })
    return summary, per_class, matrices


def audit_data(root: Path):
    manifest, grouped = [], defaultdict(list)
    for split in ("train", "val", "test"):
        if not (root / split).is_dir():
            continue
        for label, paths in class_images(root / split).items():
            for path in paths:
                record = {"split": split, "class": label, "image": str(path), "sha256": file_digest(path)}
                manifest.append(record)
                grouped[record["sha256"]].append(record)
    leaked = [row for group in grouped.values() if len({r["split"] for r in group}) > 1 for row in group]
    return manifest, leaked


def evaluate(model_path: Path, data: Path, output: Path, split="test", device="auto", batch_size=16):
    model_path, data, output = model_path.resolve(), data.resolve(), output.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if batch_size < 1 or split not in {"val", "test"}:
        raise ValueError("Use a positive batch size and an explicit val/test split")
    images = class_images(data / split)  # No fallback from test to val.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing evaluation: {output}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    model = BloomYOLO(str(model_path))
    if model.task != "classify":
        raise ValueError("Expected a classification checkpoint")
    names = [model.names[index] for index in range(len(model.names))]
    if set(images) != set(names):
        raise ValueError(f"Dataset/model class mismatch: dataset={sorted(images)}, model={names}")
    for name in names:
        split_label(name)
    saved = getattr(model.model, "transforms", None)
    if saved is None or not isinstance(saved.transforms[0], ForegroundLetterbox):
        raise ValueError("Checkpoint lacks the current foreground preprocessing; choose the new bloom model")
    size = saved.transforms[0].size
    transform = bloom_transforms(size)
    if transform.transforms[0] != saved.transforms[0]:
        raise ValueError("Checkpoint preprocessing differs from current defaults")
    output.mkdir(parents=True, exist_ok=True)
    manifest, leaked = audit_data(data)
    audit_fields = ["split", "class", "image", "sha256"]
    write_csv(output / "dataset_manifest.csv", manifest, audit_fields)
    write_csv(output / "leakage_check.csv", leaked, audit_fields)
    if leaked:
        raise ValueError(f"Exact duplicate images cross dataset splits; inspect {output / 'leakage_check.csv'}")
    rows, pending, tensors = [], [], []

    def flush():
        if not tensors:
            return
        results = model.predict(
            torch.stack(tensors), imgsz=size, device=0 if device == "cuda" else "cpu",
            verbose=False, save=False,
        )
        if len(results) != len(pending):
            raise RuntimeError("Prediction count does not match input batch")
        for (path, label), result in zip(pending, results):
            rows.append(prediction_row(path, label, names, result.probs.data.cpu().numpy()))
        pending.clear()
        tensors.clear()

    for label, paths in images.items():
        for path in paths:
            try:
                with Image.open(path) as image:
                    tensor = transform(image)
            except (OSError, ValueError) as error:
                species, stage = split_label(label)
                rows.append({"image": str(path), "status": "invalid_image", "error": str(error),
                             "true_class": label, "true_species": species, "true_stage": stage})
                continue
            tensors.append(tensor)
            pending.append((path, label))
            if len(tensors) == batch_size:
                flush()
        print(f"Evaluated input class: {label}", flush=True)
    flush()
    rows.sort(key=lambda row: row["image"])
    fields = list(prediction_row(Path("example"), names[0], names, np.ones(len(names)) / len(names)))
    write_csv(output / "predictions.csv", rows, fields)
    errors = [r for r in rows if r["status"] != "ok" or not r["correct_class"] or not r["correct_stage"]]
    write_csv(output / "errors.csv", errors, fields)
    summary, per_class, matrices = metric_tables(rows, names)
    write_csv(output / "summary.csv", summary, list(summary[0]))
    write_csv(output / "per_class_metrics.csv", per_class, list(per_class[0]))
    for task, (labels, matrix) in matrices.items():
        table = [{"true_label": label, **dict(zip(labels, map(int, matrix[index])))} for index, label in enumerate(labels)]
        write_csv(output / f"confusion_{task}.csv", table, ["true_label", *labels])
    metadata = {
        "model": str(model_path), "model_sha256": file_digest(model_path),
        "data": str(data), "split": split, "device": device, "image_size": size,
        "timestamp": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "preprocessing": repr(saved), "leakage_check": "SHA256 exact files only; near duplicates/source groups not checked",
        "stage_rule": "Sum joint-class probabilities over species; argmax in green,half,full order",
        "species_rule": "Sum joint-class probabilities over stages",
        "macro_rule": "All listed labels included; undefined precision/recall/F1 set to 0; balanced_accuracy excludes zero-support labels",
        "invalid_rule": "Invalid inputs excluded from accuracy; included in coverage and accuracy_all_inputs",
    }
    write_csv(output / "metadata.csv", [{"key": k, "value": v} for k, v in metadata.items()], ["key", "value"])
    for row in summary[:3]:
        print(row)
    print(f"CSV reports: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = args.data_dir or config_path(load_config(args.config), "classification", "train_data")
    run_name = args.model.parent.parent.name
    output = args.output_dir or PROJECT_ROOT / "Recognition/evaluation" / f"{run_name}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    evaluate(args.model, data, output, args.split, args.device, args.batch_size)


if __name__ == "__main__":
    main()
