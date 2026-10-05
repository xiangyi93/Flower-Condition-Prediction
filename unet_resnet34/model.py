"""Conventional U-Net decoder on a torchvision ResNet34 encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(
                input_channels + skip_channels,
                output_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.convolutions(torch.cat([inputs, skip], dim=1))


class UNetResNet34(nn.Module):
    """Five-stage ResNet34 encoder with a standard skip-connected U-Net decoder."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_size = inputs.shape[-2:]
        stem = self.stem(inputs)
        encoder1 = self.encoder1(self.pool(stem))
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        decoded = self.decoder4(encoder4, encoder3)
        decoded = self.decoder3(decoded, encoder2)
        decoded = self.decoder2(decoded, encoder1)
        decoded = self.decoder1(decoded, stem)
        decoded = F.interpolate(
            decoded, size=input_size, mode="bilinear", align_corners=False
        )
        return self.final(decoded)
