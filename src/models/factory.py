"""
Model factory for creating lane detection models.

This module provides a unified interface for creating different
model architectures.
"""

from typing import Literal

import torch.nn as nn

from .base import LaneDetectorBase, ModelConfig


# Available architectures
ARCHITECTURES = Literal[
    "mobilenetv3",
    "mobilenetv3_large",
    "mobilenetv3_v2",
    "efficientnet_lite0",
    "efficientnet_lite1",
    "efficientnet_lite2",
    "efficientnet_b0",
    "resnet18",
    "resnet34",
]


def create_model(
    architecture: str,
    pretrained: bool = True,
    dropout: float = 0.3,
    config: ModelConfig | None = None,
) -> LaneDetectorBase:
    """
    Create a lane detection model.

    Args:
        architecture: Model architecture name
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regression head
        config: Optional model configuration

    Returns:
        Lane detection model

    Raises:
        ValueError: If architecture is not supported
    """
    config = config or ModelConfig()

    if architecture == "mobilenetv3":
        from .mobilenetv3 import MobileNetV3Lane

        return MobileNetV3Lane(pretrained=pretrained, dropout=dropout, config=config)

    elif architecture == "mobilenetv3_v2":
        from .mobilenetv3 import MobileNetV3LaneV2

        return MobileNetV3LaneV2(pretrained=pretrained, dropout=dropout, config=config)

    elif architecture == "mobilenetv3_large":
        from .mobilenetv3 import MobileNetV3LargeLane

        return MobileNetV3LargeLane(
            pretrained=pretrained, dropout=dropout, config=config
        )

    elif architecture.startswith("efficientnet_lite"):
        from .efficientnet import EfficientNetLiteLane

        variant = architecture.replace("efficientnet_", "")
        return EfficientNetLiteLane(
            variant=variant, pretrained=pretrained, dropout=dropout, config=config
        )

    elif architecture == "efficientnet_b0":
        from .efficientnet import EfficientNetB0Lane

        return EfficientNetB0Lane(pretrained=pretrained, dropout=dropout, config=config)

    elif architecture == "resnet18":
        from .resnet import ResNet18Lane

        return ResNet18Lane(pretrained=pretrained, dropout=dropout, config=config)

    elif architecture == "resnet34":
        from .resnet import ResNet34Lane

        return ResNet34Lane(pretrained=pretrained, dropout=dropout, config=config)

    else:
        available = [
            "mobilenetv3",
            "mobilenetv3_large",
            "mobilenetv3_v2",
            "efficientnet_lite0",
            "efficientnet_lite1",
            "efficientnet_lite2",
            "efficientnet_b0",
            "resnet18",
            "resnet34",
        ]
        raise ValueError(
            f"Unknown architecture: {architecture}. Available: {available}"
        )


def list_architectures() -> list[str]:
    """List all available architectures."""
    return [
        "mobilenetv3",
        "mobilenetv3_large",
        "mobilenetv3_v2",
        "efficientnet_lite0",
        "efficientnet_lite1",
        "efficientnet_lite2",
        "efficientnet_b0",
        "resnet18",
        "resnet34",
    ]


def get_architecture_info(architecture: str) -> dict:
    """
    Get information about an architecture.

    Args:
        architecture: Architecture name

    Returns:
        Dict with architecture information
    """
    import torch

    model = create_model(architecture, pretrained=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate FLOPS (rough approximation)
    dummy = torch.zeros(1, 3, model.config.input_size, model.config.input_size)

    return {
        "name": architecture,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_size": model.config.input_size,
        "parameter_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }
