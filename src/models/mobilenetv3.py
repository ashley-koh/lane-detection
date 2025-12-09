"""
MobileNetV3-Small model for lane detection.

This is the primary recommended architecture for lane detection
on the Jetson Xavier NX due to its excellent speed/accuracy tradeoff.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from .base import LaneDetectorBase, ModelConfig, RegressionHead


class MobileNetV3Lane(LaneDetectorBase):
    """
    MobileNetV3-Small based lane detection model.

    Architecture:
    - Backbone: MobileNetV3-Small (pretrained on ImageNet)
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
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)

        # Extract features (everything except classifier)
        self.backbone = base_model.features

        # Get the number of output features from backbone
        # MobileNetV3-Small outputs 576 features
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


class MobileNetV3LaneV2(LaneDetectorBase):
    """
    Alternative MobileNetV3-Small variant with larger head.

    This version has a slightly larger regression head for
    potentially better accuracy at minimal speed cost.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.4,
        config: ModelConfig | None = None,
    ):
        super().__init__(config)

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)

        self.backbone = base_model.features

        # Larger head with two hidden layers
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.head(features)
        return output
