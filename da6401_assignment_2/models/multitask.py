"""Unified multi-task model
"""

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import UNetDecoder, VGG11UNet
from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    _FC_IN = 512 * 7 * 7

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        dropout_p: float = 0.5,
        localizer_hidden: int = 512,
    ):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
            dropout_p: Dropout probability (shared with sub-model definitions).
            localizer_hidden: Hidden dim for the localization MLP (must match training).
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, dropout_p=dropout_p)

        # Classification head (same layout as :class:`VGG11Classifier` tail).
        self.fc1 = nn.Linear(self._FC_IN, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop_fc1 = CustomDropout(dropout_p)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop_fc2 = CustomDropout(dropout_p)
        self.fc3 = nn.Linear(4096, num_breeds)

        # Localization head (same layout as :class:`VGG11Localizer` tail).
        self.loc_fc1 = nn.Linear(self._FC_IN, localizer_hidden)
        self.loc_relu = nn.ReLU(inplace=True)
        self.loc_drop = CustomDropout(dropout_p)
        self.loc_fc2 = nn.Linear(localizer_hidden, 4)

        self.seg_decoder = UNetDecoder(num_classes=seg_classes, dropout_p=dropout_p)

        # Optional: before submission, paste ``gdown.download(...)`` lines from the course README
        # (requires ``gdown``) to fetch checkpoints from Google Drive.
        self._load_pretrained(
            classifier_path=classifier_path,
            localizer_path=localizer_path,
            unet_path=unet_path,
        )

    def _load_pretrained(self, classifier_path: str, localizer_path: str, unet_path: str) -> None:
        """Load weights from independently trained single-task checkpoints."""
        if os.path.isfile(classifier_path):
            clf = VGG11Classifier(num_classes=self.fc3.out_features)
            state = torch.load(classifier_path, map_location="cpu")
            clf.load_state_dict(state, strict=True)
            self.encoder.load_state_dict(clf.encoder.state_dict())
            self.fc1.load_state_dict(clf.fc1.state_dict())
            self.fc2.load_state_dict(clf.fc2.state_dict())
            self.fc3.load_state_dict(clf.fc3.state_dict())

        if os.path.isfile(localizer_path):
            loc = VGG11Localizer(hidden_dim=self.loc_fc1.out_features)
            state = torch.load(localizer_path, map_location="cpu")
            loc.load_state_dict(state, strict=True)
            self.loc_fc1.load_state_dict(loc.fc1.state_dict())
            self.loc_fc2.load_state_dict(loc.fc2.state_dict())

        if os.path.isfile(unet_path):
            unet = VGG11UNet(num_classes=int(self.seg_decoder.head.out_channels))
            state = torch.load(unet_path, map_location="cpu")
            unet.load_state_dict(state, strict=True)
            self.seg_decoder.load_state_dict(unet.decoder.state_dict())

    def _forward_cls(self, flat: torch.Tensor) -> torch.Tensor:
        x = self.fc1(flat)
        x = self.relu1(x)
        x = self.drop_fc1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop_fc2(x)
        return self.fc3(x)

    def _forward_loc(self, flat: torch.Tensor) -> torch.Tensor:
        x = self.loc_fc1(flat)
        x = self.loc_relu(x)
        x = self.loc_drop(x)
        raw = self.loc_fc2(x)
        xc, yc, w, h = raw.split(1, dim=1)
        w = F.softplus(w) + 1e-3
        h = F.softplus(h) + 1e-3
        return torch.cat([xc, yc, w, h], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, skips = self.encoder(x, return_features=True)
        flat = torch.flatten(bottleneck, 1)
        return {
            "classification": self._forward_cls(flat),
            "localization": self._forward_loc(flat),
            "segmentation": self.seg_decoder(bottleneck, skips),
        }
