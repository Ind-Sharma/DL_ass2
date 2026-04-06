"""Segmentation model
"""

from typing import Dict

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class UNetDecoder(nn.Module):
    """Expansive path: transposed-conv upsampling + skip concatenation + conv blocks."""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
        )
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
        )
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
        )
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
        )
        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.up1(bottleneck)
        x = torch.cat([x, skips["block5"]], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skips["block4"]], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skips["block3"]], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x, skips["block2"]], dim=1)
        x = self.dec4(x)

        x = self.up5(x)
        x = torch.cat([x, skips["block1"]], dim=1)
        x = self.dec5(x)
        return self.head(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, dropout_p=dropout_p)
        self.decoder = UNetDecoder(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, skips = self.encoder(x, return_features=True)
        return self.decoder(bottleneck, skips)
