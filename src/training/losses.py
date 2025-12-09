"""
Loss functions for lane detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss.

    More robust to outliers than MSE.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta)


class MSELoss(nn.Module):
    """Mean Squared Error loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    """Mean Absolute Error (L1) loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss that penalizes errors more at the extremes.

    Useful when you want the model to be more accurate
    when predicting large steering corrections.
    """

    def __init__(self, edge_weight: float = 2.0):
        super().__init__()
        self.edge_weight = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Weight based on distance from center
        weights = 1.0 + (self.edge_weight - 1.0) * torch.abs(target)
        mse = (pred - target) ** 2
        return (weights * mse).mean()


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name.

    Args:
        loss_type: One of 'smooth_l1', 'mse', 'mae', 'weighted_mse'
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function module
    """
    loss_types = {
        "smooth_l1": SmoothL1Loss,
        "mse": MSELoss,
        "mae": MAELoss,
        "weighted_mse": WeightedMSELoss,
    }

    if loss_type not in loss_types:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available: {list(loss_types.keys())}"
        )

    return loss_types[loss_type](**kwargs)
