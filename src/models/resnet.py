"""
ResNet models for lane detection.

Alternative architecture using the classic ResNet backbone.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

from .base import LaneDetectorBase, ModelConfig, RegressionHead


class ResNet18Lane(LaneDetectorBase):
    """
    ResNet-18 based lane detection model.

    Architecture:
    - Backbone: ResNet-18 (pretrained on ImageNet)
    - Head: Custom regression head with Tanh output

    Args:
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate in regression head
        config: Model configuration
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        config: ModelConfig | None = None,
    ):
        super().__init__(config)

        # Load pretrained backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        # Get the number of output features (512 for ResNet-18)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.config.input_size, self.config.input_size)
            features = self.backbone(dummy)
            in_features = features.shape[1]

        # Custom regression head
        self.head = RegressionHead(
            in_features=in_features,
            hidden_dim=256,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Lane center predictions (B, 1) in range [-1, 1]
        """
        features = self.backbone(x)
        output = self.head(features)
        return output


class ResNet34Lane(LaneDetectorBase):
    """
    ResNet-34 based lane detection model.

    Larger variant of ResNet for potentially better accuracy.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        config: ModelConfig | None = None,
    ):
        super().__init__(config)

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet34(weights=weights)

        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.config.input_size, self.config.input_size)
            features = self.backbone(dummy)
            in_features = features.shape[1]

        self.head = RegressionHead(
            in_features=in_features,
            hidden_dim=256,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.head(features)
        return output
