"""
EfficientNet-Lite model for lane detection.

Alternative architecture with potentially better accuracy
at the cost of slightly higher latency.
"""

import torch
import torch.nn as nn
import timm

from .base import LaneDetectorBase, ModelConfig, RegressionHead


class EfficientNetLiteLane(LaneDetectorBase):
    """
    EfficientNet-Lite based lane detection model.

    Architecture:
    - Backbone: EfficientNet-Lite0 (pretrained on ImageNet)
    - Head: Custom regression head with Tanh output

    Args:
        variant: EfficientNet-Lite variant ('lite0', 'lite1', 'lite2')
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate in regression head
        config: Model configuration
    """

    def __init__(
        self,
        variant: str = "lite0",
        pretrained: bool = True,
        dropout: float = 0.3,
        config: ModelConfig | None = None,
    ):
        super().__init__(config)

        # Map variant names
        model_name = f"efficientnet_{variant}"

        # Load pretrained backbone using timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove global pooling
        )

        # Get the number of output features
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


class EfficientNetB0Lane(LaneDetectorBase):
    """
    EfficientNet-B0 based lane detection model.

    Standard EfficientNet-B0 (not Lite) for comparison.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        config: ModelConfig | None = None,
    ):
        super().__init__(config)

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

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
