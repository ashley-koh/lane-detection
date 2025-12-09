"""
Evaluation metrics for lane detection.
"""

import torch
import numpy as np


class MetricsTracker:
    """
    Track and compute metrics during training/evaluation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated values."""
        self.predictions = []
        self.targets = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Add batch predictions and targets.

        Args:
            pred: Predictions tensor (B, 1) or (B,)
            target: Targets tensor (B, 1) or (B,)
        """
        pred_np = pred.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()

        self.predictions.extend(pred_np)
        self.targets.extend(target_np)

    def compute(self) -> dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        pred = np.array(self.predictions)
        target = np.array(self.targets)

        if len(pred) == 0:
            return {}

        # Compute metrics
        errors = pred - target
        abs_errors = np.abs(errors)

        metrics = {
            "mae": float(np.mean(abs_errors)),
            "mse": float(np.mean(errors**2)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "max_error": float(np.max(abs_errors)),
            "std_error": float(np.std(errors)),
            # Percentage within thresholds
            "within_0.05": float(np.mean(abs_errors <= 0.05) * 100),
            "within_0.1": float(np.mean(abs_errors <= 0.1) * 100),
            "within_0.2": float(np.mean(abs_errors <= 0.2) * 100),
        }

        return metrics


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error."""
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Squared Error."""
    return float(torch.mean((pred - target) ** 2).item())


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Squared Error."""
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def compute_max_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute maximum absolute error."""
    return float(torch.max(torch.abs(pred - target)).item())


def compute_within_threshold(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> float:
    """
    Compute percentage of predictions within threshold of target.

    Args:
        pred: Predictions
        target: Targets
        threshold: Maximum allowed error

    Returns:
        Percentage of predictions within threshold (0-100)
    """
    abs_errors = torch.abs(pred - target)
    within = torch.sum(abs_errors <= threshold).float()
    return float((within / len(abs_errors) * 100).item())
