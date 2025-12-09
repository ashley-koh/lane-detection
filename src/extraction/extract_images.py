"""
Extract images from ROS2 bag files.

This module provides functionality to extract images from ROS2 bag files
and save them as individual image files for further processing.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml
from rosbags.image import message_to_cvimage
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm


class BagCorruptedError(Exception):
    """Raised when a ROS2 bag file is corrupted."""

    pass


def check_bag_integrity(bag_path: Path) -> tuple[bool, str]:
    """
    Check if a ROS2 bag file's SQLite database is valid.

    Args:
        bag_path: Path to the ROS2 bag directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Find the .db3 file in the bag directory
    db_files = list(bag_path.glob("*.db3"))
    if not db_files:
        return False, f"No .db3 database file found in {bag_path}"

    db_path = db_files[0]

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Run SQLite integrity check
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()

        conn.close()

        if result[0] == "ok":
            return True, ""
        else:
            # Truncate long error messages
            error_text = result[0]
            lines = error_text.split("\n")
            if len(lines) > 5:
                error_text = (
                    "\n".join(lines[:5]) + f"\n... and {len(lines) - 5} more errors"
                )
            return False, f"SQLite integrity check failed:\n{error_text}"

    except sqlite3.DatabaseError as e:
        return False, f"SQLite database error: {e}"
    except Exception as e:
        return False, f"Error checking bag integrity: {e}"


def read_metadata_yaml(bag_path: Path) -> dict | None:
    """
    Read metadata.yaml from a ROS2 bag directory.

    Args:
        bag_path: Path to the ROS2 bag directory

    Returns:
        Parsed metadata dict or None if not found/invalid
    """
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def list_topics_from_metadata(bag_path: Path) -> list[tuple[str, str, int]] | None:
    """
    Extract topic information from metadata.yaml (fallback for corrupted DBs).

    Args:
        bag_path: Path to the ROS2 bag directory

    Returns:
        List of (topic_name, message_type, message_count) tuples, or None if unavailable
    """
    metadata = read_metadata_yaml(bag_path)
    if not metadata:
        return None

    try:
        topics = []
        rosbag2_info = metadata.get("rosbag2_bagfile_information", {})
        topics_with_types = rosbag2_info.get("topics_with_message_count", [])

        for topic_info in topics_with_types:
            topic_meta = topic_info.get("topic_metadata", {})
            topic_name = topic_meta.get("name", "unknown")
            msg_type = topic_meta.get("type", "unknown")
            msg_count = topic_info.get("message_count", 0)
            topics.append((topic_name, msg_type, msg_count))

        return topics if topics else None
    except Exception:
        return None


def get_image_messages(
    bag_path: Path,
    topic: str,
    sample_rate: int = 1,
) -> Iterator[tuple[int, np.ndarray, int]]:
    """
    Yield images from a ROS2 bag file.

    Args:
        bag_path: Path to the ROS2 bag directory
        topic: Image topic name
        sample_rate: Extract every Nth frame (1 = all frames)

    Yields:
        Tuple of (frame_index, image_array, timestamp_ns)
    """
    # Check bag integrity first
    is_valid, error_msg = check_bag_integrity(bag_path)
    if not is_valid:
        raise BagCorruptedError(
            f"Bag file is corrupted: {error_msg}\n"
            "Recovery suggestions:\n"
            "  - Try re-recording the bag file\n"
            "  - Check if the bag was fully written (recording may have been interrupted)\n"
            "  - Use 'sqlite3 <bag>.db3 \".recover\"' to attempt partial recovery"
        )

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(bag_path) as reader:
        # Register types from the bag
        for connection in reader.connections:
            if connection.msgtype not in typestore.types:
                try:
                    typestore.register(connection.msgdef)
                except Exception:
                    pass

        # Find the target topic
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = [c.topic for c in reader.connections]
            raise ValueError(
                f"Topic '{topic}' not found in bag. Available topics: {available}"
            )

        frame_idx = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if frame_idx % sample_rate == 0:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                # Convert message to OpenCV image
                try:
                    image = message_to_cvimage(msg, "bgr8")
                except Exception:
                    # Handle compressed images or other formats
                    if hasattr(msg, "data"):
                        if hasattr(msg, "format"):
                            # Compressed image
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        else:
                            # Raw image
                            image = np.frombuffer(msg.data, dtype=np.uint8)
                            image = image.reshape((msg.height, msg.width, -1))
                            if image.shape[2] == 1:
                                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    else:
                        raise ValueError(
                            f"Cannot decode message type: {connection.msgtype}"
                        )

                yield frame_idx // sample_rate, image, timestamp

            frame_idx += 1


def extract_images(
    bag_path: str | Path,
    topic: str,
    output_dir: str | Path,
    image_format: str = "png",
    sample_rate: int = 1,
    verbose: bool = True,
) -> list[Path]:
    """
    Extract images from a ROS2 bag file and save them to disk.

    Args:
        bag_path: Path to the ROS2 bag directory
        topic: Image topic name
        output_dir: Directory to save extracted images
        image_format: Output image format (png, jpg, etc.)
        sample_rate: Extract every Nth frame
        verbose: Show progress bar

    Returns:
        List of paths to extracted images

    Raises:
        FileNotFoundError: If the bag path does not exist
        BagCorruptedError: If the bag file database is corrupted
        ValueError: If the topic is not found in the bag
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path not found: {bag_path}")

    # Check bag integrity before proceeding
    is_valid, error_msg = check_bag_integrity(bag_path)
    if not is_valid:
        raise BagCorruptedError(
            f"Bag file is corrupted: {error_msg}\n"
            "Recovery suggestions:\n"
            "  - Try re-recording the bag file\n"
            "  - Check if the bag was fully written (recording may have been interrupted)\n"
            "  - Use 'sqlite3 <bag>.db3 \".recover\"' to attempt partial recovery"
        )

    # Get bag name for prefixing images
    bag_name = bag_path.stem

    # Count total messages for progress bar
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = [c.topic for c in reader.connections]
            raise ValueError(f"Topic '{topic}' not found. Available: {available}")
        total_messages = sum(c.msgcount for c in connections)

    extracted_paths = []
    iterator = get_image_messages(bag_path, topic, sample_rate)

    if verbose:
        iterator = tqdm(
            iterator,
            total=total_messages // sample_rate,
            desc=f"Extracting from {bag_name}",
        )

    # Save metadata
    metadata_path = output_dir / f"{bag_name}_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"bag_path: {bag_path}\n")
        f.write(f"topic: {topic}\n")
        f.write(f"sample_rate: {sample_rate}\n")
        f.write(f"format: {image_format}\n")
        f.write("---\n")
        f.write("frame_idx,filename,timestamp_ns\n")

        for frame_idx, image, timestamp in iterator:
            filename = f"{bag_name}_{frame_idx:06d}.{image_format}"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), image)
            extracted_paths.append(filepath)

            f.write(f"{frame_idx},{filename},{timestamp}\n")

    if verbose:
        print(f"Extracted {len(extracted_paths)} images to {output_dir}")

    return extracted_paths


def list_topics(bag_path: str | Path) -> list[tuple[str, str, int]]:
    """
    List all topics in a ROS2 bag file.

    This function first checks the bag file integrity and falls back to
    reading metadata.yaml if the database is corrupted.

    Args:
        bag_path: Path to the ROS2 bag directory

    Returns:
        List of (topic_name, message_type, message_count) tuples

    Raises:
        BagCorruptedError: If the bag is corrupted and no metadata fallback is available
    """
    bag_path = Path(bag_path)

    # Check bag integrity first
    is_valid, error_msg = check_bag_integrity(bag_path)

    if is_valid:
        # Normal path: read from database
        with Reader(bag_path) as reader:
            topics = [(c.topic, c.msgtype, c.msgcount) for c in reader.connections]
        return topics
    else:
        # Try to fall back to metadata.yaml
        topics = list_topics_from_metadata(bag_path)
        if topics:
            print(f"Warning: Bag database is corrupted ({error_msg})")
            print("         Showing topic information from metadata.yaml instead.")
            print("         Note: Actual message extraction will not be possible.\n")
            return topics
        else:
            raise BagCorruptedError(
                f"Bag file is corrupted: {error_msg}\n"
                "Additionally, could not read topic information from metadata.yaml.\n"
                "Recovery suggestions:\n"
                "  - Try re-recording the bag file\n"
                "  - Check if the bag was fully written (recording may have been interrupted)\n"
                "  - Use 'sqlite3 <bag>.db3 \".recover\"' to attempt partial recovery"
            )


def main():
    parser = argparse.ArgumentParser(description="Extract images from ROS2 bag files")
    parser.add_argument(
        "--bag-path",
        type=str,
        required=True,
        help="Path to ROS2 bag directory",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Image topic name (e.g., /camera/image_raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for extracted images",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "bmp"],
        help="Output image format",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List available topics in the bag and exit",
    )

    args = parser.parse_args()

    try:
        if args.list_topics:
            topics = list_topics(args.bag_path)
            print(f"\nTopics in {args.bag_path}:")
            print("-" * 60)
            for topic, msgtype, count in topics:
                print(f"  {topic}")
                print(f"    Type: {msgtype}")
                print(f"    Messages: {count}")
            return

        if not args.topic:
            parser.error("--topic is required when not using --list-topics")

        extract_images(
            bag_path=args.bag_path,
            topic=args.topic,
            output_dir=args.output_dir,
            image_format=args.format,
            sample_rate=args.sample_rate,
        )
    except BagCorruptedError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    main()
