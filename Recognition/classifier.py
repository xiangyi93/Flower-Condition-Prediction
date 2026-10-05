"""Ultralytics adapters that keep bloom preprocessing consistent in every mode."""

from copy import copy

from PIL import Image
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.classify import (
    ClassificationPredictor,
    ClassificationTrainer,
    ClassificationValidator,
)

from Recognition.preprocessing import bloom_transforms


class BloomDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        self.torch_transforms = bloom_transforms(args.imgsz, augment, args.fliplr)

    def __getitem__(self, index):
        # Use the same RGB/alpha decoding for all splits; never discard alpha with cv2.imread.
        path, label = self.samples[index][:2]
        with Image.open(path) as image:
            try:
                tensor = self.torch_transforms(image)
            except ValueError as error:
                raise ValueError(f"Invalid bloom image {path}: {error}") from error
        return {"img": tensor, "cls": label}


class BloomValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return BloomDataset(img_path, self.args, augment=False, prefix=self.args.split)


class BloomTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return BloomDataset(img_path, self.args, augment=mode == "train", prefix=mode)

    def get_validator(self):
        self.loss_names = ["loss"]
        return BloomValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class BloomPredictor(ClassificationPredictor):
    def setup_source(self, source):
        # The stock predictor falls back to CenterCrop when imgsz changes or on export.
        BasePredictor.setup_source(self, source)
        self.transforms = bloom_transforms(max(self.imgsz))


class BloomYOLO(YOLO):
    """Use this wrapper for train(), val(), and predict() on the bloom dataset."""

    @property
    def task_map(self):
        mapping = super().task_map
        mapping["classify"].update(
            trainer=BloomTrainer, validator=BloomValidator, predictor=BloomPredictor
        )
        return mapping
