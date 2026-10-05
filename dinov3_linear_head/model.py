"""Linear semantic-segmentation probe over frozen DINOv3 patch tokens."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


class DINOv3LinearSeg(nn.Module):
    """Freeze DINOv3 and learn one 1x1 convolution over its patch grid."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = 4,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone or AutoModel.from_pretrained(
            model_name, token=os.environ.get("HF_TOKEN")
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        hidden_size = int(self.backbone.config.hidden_size)
        self.prefix_tokens = 1 + int(
            getattr(self.backbone.config, "num_register_tokens", 4)
        )
        self.linear_head = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output_size = pixel_values.shape[-2:]
        with torch.no_grad():
            tokens = self.backbone(pixel_values).last_hidden_state
        patch_tokens = tokens[:, self.prefix_tokens :, :]
        patch_count = patch_tokens.shape[1]
        height = int(patch_count**0.5)
        if height * height != patch_count:
            raise ValueError(
                f"Expected a square DINOv3 patch grid, received {patch_count} tokens"
            )
        features = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], patch_tokens.shape[2], height, height
        )
        logits = self.linear_head(features)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
