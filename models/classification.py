"""Classification components
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    # VGG expects fixed 224×224 inputs; conv stack output is 7×7×512.
    _FC_IN = 512 * 7 * 7

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, dropout_p=dropout_p)

        self.fc1 = nn.Linear(self._FC_IN, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop_fc1 = CustomDropout(dropout_p)

        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop_fc2 = CustomDropout(dropout_p)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        x = self.encoder(x, return_features=False)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop_fc1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop_fc2(x)
        x = self.fc3(x)
        return x
