"""Create a before/after contact sheet without changing training images."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from Recognition.preprocessing import ForegroundLetterbox, rgb_on_black


def preview(images, output: Path, size=224):
    width, row_height = 800, 300
    sheet = Image.new("RGB", (width, row_height * len(images)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    transform = ForegroundLetterbox(size)
    for index, path in enumerate(images):
        with Image.open(path) as image:
            before = ImageOps.contain(rgb_on_black(image), (360, 250))
            after = transform(image)
        y = index * row_height
        draw.text((12, y + 8), f"{path.parent.name} / {path.name}", fill="white")
        sheet.paste(before, (12 + (360 - before.width) // 2, y + 34))
        sheet.paste(after, (460, y + 40))
        draw.text((420, y + 18), f"Foreground crop + padding ({size}px)", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("output/recognition_preprocessing.jpg"))
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    preview(args.images, args.output, args.size)


if __name__ == "__main__":
    main()
