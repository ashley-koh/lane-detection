"""
Base model interface for lane detection.

This module defines the base class that all lane detection models
must inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for a lane detection model."""

    input_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    num_channels: int = 3


class LaneDetectorBase(nn.Module, ABC):
    """
    Base class for lane detection models.

    All lane detection models should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Tensor of shape (batch_size, 1) with lane center predictions
            in range [-1, 1]
        """
        pass

    def get_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.config

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Freeze backbone weights (for transfer learning)."""
        if hasattr(self, "backbone") and isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        if hasattr(self, "backbone") and isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = True


class RegressionHead(nn.Module):
    """
    Regression head for lane center prediction.

    Takes features from a backbone and outputs a single value
    representing the lane center position.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
