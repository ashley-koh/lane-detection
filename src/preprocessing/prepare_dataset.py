"""
Prepare dataset for annotation.

This module provides functionality to select and prepare images
for annotation in Roboflow.
"""

import argparse
import hashlib
from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm


def compute_image_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute perceptual hash of an image for duplicate detection.

    Args:
        image: Input image
        hash_size: Size of the hash (default 8 = 64-bit hash)

    Returns:
        Hexadecimal hash string
    """
    # Convert to grayscale and resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))

    # Compute difference hash
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert to hex string
    return "".join(str(int(b)) for b in diff.flatten())


def compute_image_variance(image: np.ndarray) -> float:
    """
    Compute variance of an image (measure of visual complexity).

    Args:
        image: Input image

    Returns:
        Variance value
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.var(gray))


def select_diverse_frames(
    image_paths: list[Path],
    num_samples: int,
    hash_threshold: int = 5,
    verbose: bool = True,
) -> list[Path]:
    """
    Select diverse frames by removing near-duplicates.

    Args:
        image_paths: List of image paths
        num_samples: Target number of samples
        hash_threshold: Hamming distance threshold for duplicates
        verbose: Show progress

    Returns:
        List of selected image paths
    """
    if len(image_paths) <= num_samples:
        return image_paths

    # Compute hashes for all images
    hashes = {}
    variances = {}

    iterator = tqdm(image_paths, desc="Computing hashes") if verbose else image_paths
    for path in iterator:
        image = cv2.imread(str(path))
        if image is not None:
            hashes[path] = compute_image_hash(image)
            variances[path] = compute_image_variance(image)

    # Remove near-duplicates
    def hamming_distance(h1: str, h2: str) -> int:
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))

    selected = []
    remaining = list(hashes.keys())

    # Sort by variance (prefer more complex images)
    remaining.sort(key=lambda p: variances[p], reverse=True)

    while remaining and len(selected) < num_samples:
        # Take the next image
        current = remaining.pop(0)
        selected.append(current)

        # Remove near-duplicates of current
        current_hash = hashes[current]
        remaining = [
            p
            for p in remaining
            if hamming_distance(hashes[p], current_hash) > hash_threshold
        ]

    # If we still need more, add random samples from originals
    if len(selected) < num_samples:
        unused = [p for p in image_paths if p not in selected]
        random.shuffle(unused)
        selected.extend(unused[: num_samples - len(selected)])

    return selected[:num_samples]


def prepare_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    num_samples: int | None = None,
    remove_duplicates: bool = True,
    seed: int | None = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Prepare images for annotation by selecting diverse samples.

    Args:
        input_dir: Directory containing processed images
        output_dir: Directory to save selected images
        num_samples: Number of images to select (None = all)
        remove_duplicates: Whether to remove near-duplicate images
        seed: Random seed for reproducibility
        verbose: Show progress

    Returns:
        List of paths to prepared images
    """
    if seed is not None:
        random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    if verbose:
        print(f"Found {len(image_files)} images in {input_dir}")

    # Select images
    if num_samples and remove_duplicates:
        selected = select_diverse_frames(image_files, num_samples, verbose=verbose)
    elif num_samples:
        random.shuffle(image_files)
        selected = image_files[:num_samples]
    else:
        selected = image_files

    # Copy selected images with sequential naming
    prepared_paths = []
    for idx, src_path in enumerate(
        tqdm(selected, desc="Preparing images") if verbose else selected
    ):
        # Use sequential naming for easier annotation
        dst_path = output_dir / f"frame_{idx:04d}{src_path.suffix}"

        # Copy image
        image = cv2.imread(str(src_path))
        cv2.imwrite(str(dst_path), image)
        prepared_paths.append(dst_path)

    # Create manifest file
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# Dataset manifest\n")
        f.write(f"# Source: {input_dir}\n")
        f.write(f"# Samples: {len(prepared_paths)}\n")
        f.write("# Format: new_name -> original_name\n")
        f.write("---\n")
        for idx, src_path in enumerate(selected):
            f.write(f"frame_{idx:04d}{src_path.suffix} -> {src_path.name}\n")

    if verbose:
        print(f"Prepared {len(prepared_paths)} images in {output_dir}")
        print(f"Manifest saved to {manifest_path}")

    return prepared_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare images for annotation")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing processed images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save prepared images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of images to select (default: all)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable near-duplicate removal",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        remove_duplicates=not args.no_dedup,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
