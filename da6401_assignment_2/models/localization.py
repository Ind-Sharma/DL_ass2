"""Localization modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    _FC_IN = 512 * 7 * 7

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, hidden_dim: int = 512):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            hidden_dim: Hidden size of the regression MLP.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, dropout_p=dropout_p)
        self.fc1 = nn.Linear(self._FC_IN, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = CustomDropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        feat = self.encoder(x, return_features=False)
        feat = torch.flatten(feat, 1)
        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.drop(feat)
        raw = self.fc2(feat)
        xc, yc, w, h = raw.split(1, dim=1)
        w = F.softplus(w) + 1e-3
        h = F.softplus(h) + 1e-3
        return torch.cat([xc, yc, w, h], dim=1)
