"""Predict with the same foreground preprocessing used for bloom training."""

import argparse
import json
from pathlib import Path

from PIL import Image

from Recognition.classifier import BloomYOLO
from Recognition.preprocessing import rgb_on_black


def predict_bloom_state(image_path, model_path):
    model = BloomYOLO(str(model_path))
    with Image.open(image_path) as image:
        result = model.predict(rgb_on_black(image), verbose=False)[0]
    return {
        "class": result.names[result.probs.top1],
        "confidence": float(result.probs.top1conf.item()),
        "probabilities": {
            result.names[index]: float(value)
            for index, value in enumerate(result.probs.data.cpu().tolist())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint trained with foreground preprocessing")
    args = parser.parse_args()
    print(json.dumps(predict_bloom_state(args.image, args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
