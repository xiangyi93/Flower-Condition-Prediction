from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_image_mask_pairs(img_dir: str, mask_dir: str) -> list[tuple[str, str]]:
    """Return image/mask pairs matched by basename, failing on incomplete data."""
    image_root = Path(img_dir)
    mask_root = Path(mask_dir)

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_root}")

    images = {
        path.stem: path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        path.stem: path
        for path in mask_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }

    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing masks for: {', '.join(missing_masks[:5])}")
        if missing_images:
            details.append(f"missing images for: {', '.join(missing_images[:5])}")
        raise ValueError("Image/mask basenames do not match; " + "; ".join(details))

    if not images:
        raise ValueError(f"No supported images found in: {image_root}")

    return [(str(images[name]), str(masks[name])) for name in sorted(images)]
