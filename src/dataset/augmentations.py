"""
Data augmentation pipeline for lane detection.

This module provides augmentation transforms optimized for
a small dataset (~400 images) to maximize training effectiveness.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_train_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Aggressive augmentation for small dataset (~400 images).

    Args:
        image_size: Target image size (square)
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)

    Returns:
        Albumentations Compose transform
    """
    return A.Compose(
        [
            # Resize to target size
            A.Resize(image_size, image_size),
            # Horizontal flip - IMPORTANT: handled specially in dataset
            # because we need to flip the label too
            # A.HorizontalFlip(p=0.5),  # Handled in dataset
            # Color augmentations
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0,
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.15,
                        hue=0.05,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            # Simulate different lighting conditions
            A.OneOf(
                [
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_limit=(1, 2),
                        shadow_dimension=5,
                        p=1.0,
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        src_radius=100,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            # Blur augmentations
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.2,
            ),
            # Noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            # Slight geometric transforms (careful not to affect label too much)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
            # Image quality degradation
            A.OneOf(
                [
                    A.ImageCompression(quality_lower=70, quality_upper=95, p=1.0),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
                ],
                p=0.2,
            ),
            # Normalize and convert to tensor
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get validation/test augmentation pipeline.

    Only resize and normalize, no augmentation.

    Args:
        image_size: Target image size (square)
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)

    Returns:
        Albumentations Compose transform
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Get inference-time transforms.

    Same as validation transforms.

    Args:
        image_size: Target image size (square)
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)

    Returns:
        Albumentations Compose transform
    """
    return get_val_transforms(image_size, mean, std)


# Test-time augmentation for potentially better predictions
def get_tta_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> list[A.Compose]:
    """
    Get test-time augmentation transforms.

    Returns multiple transforms for TTA ensemble.

    Args:
        image_size: Target image size (square)
        mean: Normalization mean
        std: Normalization std

    Returns:
        List of transforms for TTA
    """
    base = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # Horizontal flip (remember to flip prediction)
    flip = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return [base, flip]
