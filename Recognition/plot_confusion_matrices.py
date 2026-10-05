"""Render Recognition confusion-matrix CSV files as annotated PNG heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STAGE_ORDER = ["green", "half", "full"]
SPECIES_ORDER = ["cherry", "hydrangeas", "daylily"]


def ordered_labels(labels: list[str], chart_name: str) -> list[str]:
    if chart_name == "confusion_stage.csv":
        expected = STAGE_ORDER
        return [label for label in expected if label in labels] + [label for label in labels if label not in expected]
    if chart_name == "confusion_species.csv":
        expected = SPECIES_ORDER
        return [label for label in expected if label in labels] + [label for label in labels if label not in expected]
    if chart_name == "confusion_class.csv":
        expected = [f"{species}_{stage}" for species in SPECIES_ORDER for stage in STAGE_ORDER]
        return [label for label in expected if label in labels] + [label for label in labels if label not in expected]
    return labels


def render(csv_path: Path, output_path: Path, title: str, chart_name: str) -> None:
    table = pd.read_csv(csv_path, index_col=0)
    labels = ordered_labels(list(table.index), chart_name)
    table = table.loc[labels, labels]
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(max(5, len(table.columns) * 1.25), max(4.5, len(table.index) * 0.8)))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(1, values.max()))
    fig.colorbar(image, ax=ax, label="Image count")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = values.max() / 2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = int(values[row, col])
            ax.text(col, row, value, ha="center", va="center", color="white" if value > threshold else "black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    charts = {
        "confusion_class.csv": ("confusion_class.png", "9-Class: Species + Bloom Stage"),
        "confusion_stage.csv": ("confusion_stage.png", "Bloom Stage"),
        "confusion_species.csv": ("confusion_species.png", "Plant Species"),
    }
    for source, (target, title) in charts.items():
        render(args.report_dir / source, args.report_dir / target, title, source)
    print(f"Charts written to {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
