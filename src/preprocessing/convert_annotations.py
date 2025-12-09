"""
Convert Roboflow annotations to lane detection format.

This module handles the conversion of annotations exported from Roboflow
into the normalized lane center format required for training.
"""

import argparse
import json
import random
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm


def load_coco_annotations(annotation_path: Path) -> tuple[dict, dict, dict]:
    """
    Load COCO format annotations from Roboflow.

    Args:
        annotation_path: Path to _annotations.coco.json file

    Returns:
        Tuple of (image_id_to_filename, image_id_to_info, image_id_to_annotations)
    """
    with open(annotation_path) as f:
        coco = json.load(f)

    # Map image IDs to filenames
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # Map image IDs to image info (for dimensions)
    id_to_info = {
        img["id"]: {"width": img["width"], "height": img["height"]}
        for img in coco["images"]
    }

    # Map image IDs to annotations
    id_to_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in id_to_annotations:
            id_to_annotations[img_id] = []
        id_to_annotations[img_id].append(ann)

    return id_to_filename, id_to_info, id_to_annotations


def extract_lane_center_from_keypoint(
    annotation: dict,
    image_width: int,
) -> float | None:
    """
    Extract lane center from keypoint annotation.

    Args:
        annotation: COCO annotation dict with keypoints
        image_width: Width of the image

    Returns:
        Normalized lane center (-1 to 1) or None if invalid
    """
    if "keypoints" in annotation:
        keypoints = annotation["keypoints"]
        # Keypoints format: [x, y, visibility, ...]
        if len(keypoints) >= 2:
            x = keypoints[0]
            # Normalize to [-1, 1] range
            return (x / image_width) * 2 - 1
    return None


def extract_lane_center_from_bbox(
    annotation: dict,
    image_width: int,
) -> float | None:
    """
    Extract lane center from bounding box center.

    Args:
        annotation: COCO annotation dict with bbox
        image_width: Width of the image

    Returns:
        Normalized lane center (-1 to 1) or None if invalid
    """
    if "bbox" in annotation:
        bbox = annotation["bbox"]  # [x, y, width, height]
        # Calculate center x
        center_x = bbox[0] + bbox[2] / 2
        # Normalize to [-1, 1] range
        return (center_x / image_width) * 2 - 1
    return None


def extract_lane_center_from_point(
    annotation: dict,
    image_width: int,
) -> float | None:
    """
    Extract lane center from point annotation (segmentation with single point).

    Args:
        annotation: COCO annotation dict
        image_width: Width of the image

    Returns:
        Normalized lane center (-1 to 1) or None if invalid
    """
    if "segmentation" in annotation:
        seg = annotation["segmentation"]
        if seg and len(seg) > 0:
            # Segmentation can be [[x1, y1, x2, y2, ...]]
            points = seg[0] if isinstance(seg[0], list) else seg
            if len(points) >= 2:
                # Take first x coordinate (or average if multiple points)
                x_coords = points[::2]  # Every other element starting from 0
                x = sum(x_coords) / len(x_coords)
                return (x / image_width) * 2 - 1
    return None


def convert_roboflow_annotations(
    roboflow_dir: str | Path,
    output_dir: str | Path,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Convert Roboflow export to lane detection format.

    Expected Roboflow export structure:
    roboflow_dir/
        train/
            images/
            _annotations.coco.json
        valid/ (optional)
        test/ (optional)

    Or flat structure:
    roboflow_dir/
        *.jpg/png
        _annotations.coco.json

    Args:
        roboflow_dir: Path to Roboflow export directory
        output_dir: Output directory for converted dataset
        split_ratios: (train, val, test) ratios if no predefined splits
        seed: Random seed for splitting
        verbose: Show progress

    Returns:
        Dict mapping split names to annotation CSV paths
    """
    random.seed(seed)

    roboflow_dir = Path(roboflow_dir)
    output_dir = Path(output_dir)

    # Detect Roboflow export structure
    has_splits = (roboflow_dir / "train").exists()

    all_data = []

    if has_splits:
        # Process each split
        for split_name in ["train", "valid", "test"]:
            split_dir = roboflow_dir / split_name
            if not split_dir.exists():
                continue

            # Find annotation file
            ann_file = split_dir / "_annotations.coco.json"
            if not ann_file.exists():
                print(f"Warning: No annotations found in {split_dir}")
                continue

            id_to_filename, id_to_info, id_to_annotations = load_coco_annotations(
                ann_file
            )

            # Find images directory
            images_dir = split_dir
            if (split_dir / "images").exists():
                images_dir = split_dir / "images"

            for img_id, filename in id_to_filename.items():
                info = id_to_info[img_id]
                annotations = id_to_annotations.get(img_id, [])

                # Extract lane center from annotations
                lane_center = None
                for ann in annotations:
                    lane_center = extract_lane_center_from_keypoint(ann, info["width"])
                    if lane_center is None:
                        lane_center = extract_lane_center_from_bbox(ann, info["width"])
                    if lane_center is None:
                        lane_center = extract_lane_center_from_point(ann, info["width"])
                    if lane_center is not None:
                        break

                if lane_center is not None:
                    all_data.append(
                        {
                            "filename": filename,
                            "source_path": images_dir / filename,
                            "lane_center": lane_center,
                            "split": "val" if split_name == "valid" else split_name,
                        }
                    )
    else:
        # Flat structure - need to split ourselves
        ann_file = roboflow_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"No annotations found at {ann_file}")

        id_to_filename, id_to_info, id_to_annotations = load_coco_annotations(ann_file)

        for img_id, filename in id_to_filename.items():
            info = id_to_info[img_id]
            annotations = id_to_annotations.get(img_id, [])

            lane_center = None
            for ann in annotations:
                lane_center = extract_lane_center_from_keypoint(ann, info["width"])
                if lane_center is None:
                    lane_center = extract_lane_center_from_bbox(ann, info["width"])
                if lane_center is None:
                    lane_center = extract_lane_center_from_point(ann, info["width"])
                if lane_center is not None:
                    break

            if lane_center is not None:
                all_data.append(
                    {
                        "filename": filename,
                        "source_path": roboflow_dir / filename,
                        "lane_center": lane_center,
                        "split": None,  # Will be assigned
                    }
                )

        # Assign splits
        random.shuffle(all_data)
        n = len(all_data)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])

        for i, item in enumerate(all_data):
            if i < n_train:
                item["split"] = "train"
            elif i < n_train + n_val:
                item["split"] = "val"
            else:
                item["split"] = "test"

    if not all_data:
        raise ValueError("No valid annotations found!")

    if verbose:
        print(f"Found {len(all_data)} annotated images")

    # Create output structure
    output_paths = {}
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_data = [d for d in all_data if d["split"] == split]
        if not split_data:
            continue

        # Copy images and create CSV
        records = []
        iterator = (
            tqdm(split_data, desc=f"Processing {split}") if verbose else split_data
        )

        for item in iterator:
            src = item["source_path"]
            dst = split_dir / item["filename"]

            if src.exists():
                shutil.copy2(src, dst)
                records.append(
                    {
                        "image_path": item["filename"],
                        "lane_center": item["lane_center"],
                    }
                )

        # Save CSV
        csv_path = split_dir / "annotations.csv"
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        output_paths[split] = csv_path

        if verbose:
            print(f"  {split}: {len(records)} images")

    # Save combined annotations
    combined_csv = output_dir / "annotations.csv"
    all_records = []
    for item in all_data:
        all_records.append(
            {
                "image_path": f"{item['split']}/{item['filename']}",
                "lane_center": item["lane_center"],
                "split": item["split"],
            }
        )
    pd.DataFrame(all_records).to_csv(combined_csv, index=False)

    if verbose:
        print(f"\nDataset saved to {output_dir}")
        print(f"Combined annotations: {combined_csv}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Convert Roboflow annotations to lane detection format"
    )
    parser.add_argument(
        "--roboflow-dir",
        type=str,
        required=True,
        help="Path to Roboflow export directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="80:10:10",
        help="Train:val:test split ratios (e.g., '80:10:10')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    # Parse split ratios
    parts = [float(x) for x in args.split.split(":")]
    total = sum(parts)
    split_ratios = (parts[0] / total, parts[1] / total, parts[2] / total)

    convert_roboflow_annotations(
        roboflow_dir=args.roboflow_dir,
        output_dir=args.output_dir,
        split_ratios=split_ratios,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
