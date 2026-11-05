"""U-Net with a MobileNetV3-Small encoder for single-class segmentation.

The encoder starts from ImageNet weights when ``pretrained=True`` (the only
network download in the project, and only during training); the decoder is
trained from scratch. Input is a 3-channel RGB image normalised with the
ImageNet statistics, output is one logit channel at the input resolution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ARCHITECTURE = "unet_mobilenetv3_small"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel, padding=kernel // 2, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Bilinear upsample, channel reduction, then fuse the encoder skip."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = ConvBNAct(in_channels, out_channels)
        self.fuse = ConvBNAct(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.reduce(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class UNetMobileNetV3Small(nn.Module):
    """Five-level U-Net: 1/2, 1/4, 1/8, 1/16, 1/32 with skip connections."""

    def __init__(self, *, pretrained: bool = False, classes: int = 1) -> None:
        super().__init__()
        if classes < 1:
            raise ValueError("classes must be >= 1")

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights).features

        self.enc0 = features[0]  # /2,  16
        self.enc1 = features[1]  # /4,  16
        self.enc2 = features[2:4]  # /8,  24
        self.enc3 = features[4:9]  # /16, 48
        self.enc4 = features[9:]  # /32, 576

        self.bottleneck = ConvBNAct(576, 256, kernel=1)
        self.dec4 = DecoderBlock(256, 48, 128)
        self.dec3 = DecoderBlock(128, 24, 64)
        self.dec2 = DecoderBlock(64, 16, 32)
        self.dec1 = DecoderBlock(32, 16, 16)
        self.head = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d4 = self.dec4(self.bottleneck(s4), s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        logits = self.head(d1)
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
