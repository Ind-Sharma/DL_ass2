"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG-A (11 weight layers in full VGG: 8 conv + 3 FC) encoder.

    Convolutional stack follows Simonyan & Zisserman (2014) Table 1, configuration **A**:
    64 → pool → 128 → pool → 256×2 → pool → 512×2 → pool → 512×2 → pool.

    BatchNorm and :class:`CustomDropout` are inserted as required by the assignment:
    Conv → BatchNorm → ReLU; dropout is applied on the **downsampled** activations
    after each MaxPool (inverted dropout, training-only), which regularizes deeper
    maps without wiping early low-level features on full-resolution tensors.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        self.dropout_p = dropout_p

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = CustomDropout(dropout_p)

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = CustomDropout(dropout_p)

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = CustomDropout(dropout_p)

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = CustomDropout(dropout_p)

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop5 = CustomDropout(dropout_p)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        f1 = self.block1(x)
        x = self.pool1(f1)
        x = self.drop1(x)

        f2 = self.block2(x)
        x = self.pool2(f2)
        x = self.drop2(x)

        f3 = self.block3(x)
        x = self.pool3(f3)
        x = self.drop3(x)

        f4 = self.block4(x)
        x = self.pool4(f4)
        x = self.drop4(x)

        f5 = self.block5(x)
        x = self.pool5(f5)
        x = self.drop5(x)

        bottleneck = x

        if not return_features:
            return bottleneck

        features: Dict[str, torch.Tensor] = {
            "block1": f1,
            "block2": f2,
            "block3": f3,
            "block4": f4,
            "block5": f5,
        }
        return bottleneck, features


# Course / autograder compatibility (see README import: `from models.vgg11 import VGG11`)
VGG11 = VGG11Encoder
