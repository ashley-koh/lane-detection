"""
Image cropping and preprocessing utilities.

This module provides functionality to crop and preprocess images
for lane detection training.
"""

import argparse
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
from tqdm import tqdm


# Predefined crop presets
CROP_PRESETS = {
    "bottom-half": lambda h, w: (h // 2, 0, h // 2, w),  # (y, x, height, width)
    "bottom-third": lambda h, w: (2 * h // 3, 0, h // 3, w),
    "bottom-left": lambda h, w: (h // 2, 0, h // 2, w // 2),
    "bottom-right": lambda h, w: (h // 2, w // 2, h // 2, w // 2),
    "center-strip": lambda h, w: (h // 3, 0, h // 3, w),
    "lower-center": lambda h, w: (h // 2, w // 4, h // 2, w // 2),
}


def parse_crop_region(
    region: str,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    """
    Parse crop region from string specification.

    Args:
        region: Either a preset name or 'x,y,width,height' format
        image_height: Height of the source image
        image_width: Width of the source image

    Returns:
        Tuple of (y, x, crop_height, crop_width)
    """
    if region in CROP_PRESETS:
        return CROP_PRESETS[region](image_height, image_width)

    # Parse custom format: x,y,width,height
    try:
        parts = [int(p.strip()) for p in region.split(",")]
        if len(parts) == 4:
            x, y, width, height = parts
            return (y, x, height, width)
        else:
            raise ValueError(
                f"Invalid crop region format: {region}. "
                f"Use 'x,y,width,height' or a preset: {list(CROP_PRESETS.keys())}"
            )
    except ValueError as e:
        raise ValueError(
            f"Invalid crop region: {region}. "
            f"Use 'x,y,width,height' or a preset: {list(CROP_PRESETS.keys())}"
        ) from e


def parse_resize(resize: str) -> tuple[int, int]:
    """
    Parse resize specification from string.

    Args:
        resize: String in format 'WIDTHxHEIGHT' (e.g., '224x224')

    Returns:
        Tuple of (width, height)
    """
    try:
        parts = resize.lower().split("x")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"Invalid resize format: {resize}. Use 'WIDTHxHEIGHT'")
    except ValueError as e:
        raise ValueError(f"Invalid resize format: {resize}. Use 'WIDTHxHEIGHT'") from e


def crop_image(
    image: np.ndarray,
    region: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop an image to the specified region.

    Args:
        image: Input image (H, W, C)
        region: Tuple of (y, x, height, width)

    Returns:
        Cropped image
    """
    y, x, height, width = region
    return image[y : y + height, x : x + width]


def process_image(
    image: np.ndarray,
    crop_region: tuple[int, int, int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Process an image with optional cropping and resizing.

    Args:
        image: Input image
        crop_region: Optional crop region (y, x, height, width)
        resize: Optional resize dimensions (width, height)

    Returns:
        Processed image
    """
    if crop_region is not None:
        image = crop_image(image, crop_region)

    if resize is not None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_LINEAR)

    return image


def process_images(
    input_dir: str | Path,
    output_dir: str | Path,
    crop_region: str | None = None,
    resize: str | None = None,
    image_format: str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Process all images in a directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        crop_region: Crop region specification (preset or 'x,y,w,h')
        resize: Resize specification ('WIDTHxHEIGHT')
        image_format: Output format (None = keep original)
        verbose: Show progress bar

    Returns:
        List of paths to processed images
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [
        f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    # Parse resize if specified
    resize_dims = parse_resize(resize) if resize else None

    processed_paths = []
    iterator = tqdm(image_files, desc="Processing images") if verbose else image_files

    for image_path in iterator:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load {image_path}, skipping")
            continue

        # Parse crop region (needs image dimensions)
        crop_dims = None
        if crop_region:
            h, w = image.shape[:2]
            crop_dims = parse_crop_region(crop_region, h, w)

        # Process image
        processed = process_image(image, crop_dims, resize_dims)

        # Determine output path
        if image_format:
            output_path = output_dir / f"{image_path.stem}.{image_format}"
        else:
            output_path = output_dir / image_path.name

        cv2.imwrite(str(output_path), processed)
        processed_paths.append(output_path)

    if verbose:
        print(f"Processed {len(processed_paths)} images to {output_dir}")

    return processed_paths


def main():
    parser = argparse.ArgumentParser(
        description="Crop and preprocess images for lane detection"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--crop-region",
        type=str,
        default=None,
        help=(
            f"Crop region: preset ({', '.join(CROP_PRESETS.keys())}) "
            "or 'x,y,width,height'"
        ),
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize after crop: 'WIDTHxHEIGHT' (e.g., '224x224')",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["png", "jpg", "jpeg", "bmp"],
        help="Output image format (default: keep original)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available crop presets and exit",
    )

    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable crop presets:")
        print("-" * 40)
        for name, func in CROP_PRESETS.items():
            # Show example for 480x640 image
            y, x, h, w = func(480, 640)
            print(f"  {name}:")
            print(f"    Example (640x480 image): y={y}, x={x}, h={h}, w={w}")
        return

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crop_region=args.crop_region,
        resize=args.resize,
        image_format=args.format,
    )


if __name__ == "__main__":
    main()
