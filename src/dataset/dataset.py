"""
PyTorch Dataset for lane detection.

This module provides the LaneDataset class for loading
and preprocessing lane detection training data.
"""

from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import get_train_transforms, get_val_transforms


class LaneDataset(Dataset):
    """
    PyTorch Dataset for lane center detection.

    Loads images and their corresponding lane center labels
    from a CSV annotation file.

    Args:
        data_dir: Directory containing images
        annotations_csv: Path to CSV with columns [image_path, lane_center]
        transform: Albumentations transform to apply
        horizontal_flip_prob: Probability of horizontal flip (with label inversion)
        split: Optional split filter ('train', 'val', 'test')
    """

    def __init__(
        self,
        data_dir: str | Path,
        annotations_csv: str | Path,
        transform=None,
        horizontal_flip_prob: float = 0.5,
        split: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.horizontal_flip_prob = horizontal_flip_prob

        # Load annotations
        self.annotations = pd.read_csv(annotations_csv)

        # Filter by split if specified
        if split and "split" in self.annotations.columns:
            self.annotations = self.annotations[
                self.annotations["split"] == split
            ].reset_index(drop=True)

        # Validate annotations
        if "image_path" not in self.annotations.columns:
            raise ValueError("Annotations CSV must have 'image_path' column")
        if "lane_center" not in self.annotations.columns:
            raise ValueError("Annotations CSV must have 'lane_center' column")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.annotations.iloc[idx]

        # Load image
        image_path = self.data_dir / row["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get label
        lane_center = float(row["lane_center"])

        # Apply horizontal flip with label inversion
        # This is done separately from other augmentations because
        # we need to also flip the label
        if (
            self.horizontal_flip_prob > 0
            and random.random() < self.horizontal_flip_prob
        ):
            image = cv2.flip(image, 1)  # Horizontal flip
            lane_center = -lane_center  # Invert label

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor without transforms
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Convert label to tensor
        label = torch.tensor([lane_center], dtype=torch.float32)

        return image, label

    def get_image_path(self, idx: int) -> Path:
        """Get the image path for a given index."""
        row = self.annotations.iloc[idx]
        return self.data_dir / row["image_path"]


class LaneDatasetFromDirectory(Dataset):
    """
    PyTorch Dataset that loads from a split directory structure.

    Expected structure:
    data_dir/
        train/
            images/
            annotations.csv
        val/
            images/
            annotations.csv
        test/
            images/
            annotations.csv

    Args:
        data_dir: Root data directory
        split: One of 'train', 'val', 'test'
        transform: Albumentations transform to apply
        horizontal_flip_prob: Probability of horizontal flip (only for train)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        transform=None,
        horizontal_flip_prob: float = 0.5,
    ):
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.horizontal_flip_prob = horizontal_flip_prob if split == "train" else 0.0

        # Load annotations
        annotations_path = self.data_dir / "annotations.csv"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")

        self.annotations = pd.read_csv(annotations_path)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.annotations.iloc[idx]

        # Load image
        image_path = self.data_dir / row["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get label
        lane_center = float(row["lane_center"])

        # Apply horizontal flip with label inversion
        if (
            self.horizontal_flip_prob > 0
            and random.random() < self.horizontal_flip_prob
        ):
            image = cv2.flip(image, 1)
            lane_center = -lane_center

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = torch.tensor([lane_center], dtype=torch.float32)

        return image, label


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    horizontal_flip_prob: float = 0.5,
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Root data directory with train/val/test splits
        batch_size: Batch size for all loaders
        image_size: Input image size
        num_workers: Number of data loading workers
        horizontal_flip_prob: Flip probability for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = LaneDatasetFromDirectory(
        data_dir, "train", train_transform, horizontal_flip_prob
    )
    val_dataset = LaneDatasetFromDirectory(data_dir, "val", val_transform, 0.0)
    test_dataset = LaneDatasetFromDirectory(data_dir, "test", val_transform, 0.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
