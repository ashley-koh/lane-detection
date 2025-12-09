#!/usr/bin/env python3
"""
Lane Detection Training Pipeline CLI.

This CLI provides commands for the complete lane detection workflow:
- extract: Extract images from ROS2 bag files
- crop: Crop and preprocess images
- prepare: Select diverse frames for annotation
- convert: Convert Roboflow annotations to training format
- train: Train the lane detection model
- export: Export model to ONNX/TensorRT
- info: Show model architecture information

Example workflow:
    # 1. Extract images from ROS2 bags
    lane-detection extract --bag-path ./bags/driving1 --topic /camera/image_raw

    # 2. Crop images to focus on road
    lane-detection crop --input-dir data/raw --output-dir data/cropped --crop-region bottom-half

    # 3. Prepare diverse frames for annotation
    lane-detection prepare --input-dir data/cropped --output-dir data/to_annotate --num-samples 400

    # 4. (Annotate in Roboflow, download COCO export)

    # 5. Convert annotations
    lane-detection convert --roboflow-dir ./roboflow_export --output-dir data/annotated

    # 6. Train model
    lane-detection train --data-dir data/annotated --epochs 100 --architecture mobilenetv3

    # 7. Export to ONNX and TensorRT
    lane-detection export --checkpoint outputs/checkpoints/best.pt --output model.onnx
    lane-detection export --checkpoint outputs/checkpoints/best.pt --output model.engine --tensorrt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract images from ROS2 bag files."""
    from src.extraction.extract_images import extract_images, list_topics

    if args.list_topics:
        topics = list_topics(args.bag_path)
        print(f"\nTopics in {args.bag_path}:")
        print("-" * 60)
        for topic, msgtype, count in topics:
            print(f"  {topic}")
            print(f"    Type: {msgtype}")
            print(f"    Messages: {count}")
        return 0

    if not args.topic:
        print("Error: --topic is required when not using --list-topics")
        return 1

    extract_images(
        bag_path=args.bag_path,
        topic=args.topic,
        output_dir=args.output_dir,
        image_format=args.format,
        sample_rate=args.sample_rate,
    )
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    """Crop and preprocess images."""
    from src.preprocessing.crop import process_images, CROP_PRESETS

    if args.list_presets:
        print("\nAvailable crop presets:")
        print("-" * 40)
        for name, func in CROP_PRESETS.items():
            y, x, h, w = func(480, 640)
            print(f"  {name}:")
            print(f"    Example (640x480 image): y={y}, x={x}, h={h}, w={w}")
        return 0

    # Validate required arguments when not listing presets
    if not args.input_dir:
        print("Error: --input-dir is required when not using --list-presets")
        return 1
    if not args.output_dir:
        print("Error: --output-dir is required when not using --list-presets")
        return 1

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crop_region=args.crop_region,
        resize=args.resize,
        image_format=args.format,
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare images for annotation."""
    from src.preprocessing.prepare_dataset import prepare_dataset

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        remove_duplicates=not args.no_dedup,
        seed=args.seed,
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert Roboflow annotations."""
    from src.preprocessing.convert_annotations import convert_roboflow_annotations

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
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the lane detection model."""
    import torch
    from torch.utils.data import DataLoader

    from src.models.factory import create_model
    from src.dataset.dataset import LaneDataset
    from src.dataset.augmentations import (
        get_training_augmentations,
        get_validation_augmentations,
    )
    from src.training.trainer import Trainer, TrainingConfig

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create config
    config = TrainingConfig(
        architecture=args.architecture,
        pretrained=not args.no_pretrained,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        loss_type=args.loss,
        dropout=args.dropout,
        freeze_backbone_epochs=args.freeze_backbone,
        backbone_lr_multiplier=args.backbone_lr_mult,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_best=True,
        use_amp=not args.no_amp,
        seed=args.seed,
    )

    # Create datasets
    data_dir = Path(args.data_dir)

    train_aug = get_training_augmentations(args.image_size)
    val_aug = get_validation_augmentations(args.image_size)

    train_dataset = LaneDataset(
        data_dir=data_dir / "train",
        transform=train_aug,
        horizontal_flip_prob=0.5,
    )
    val_dataset = LaneDataset(
        data_dir=data_dir / "val",
        transform=val_aug,
        horizontal_flip_prob=0.0,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
    model = create_model(
        architecture=config.architecture,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )

    print(f"\n{'=' * 60}")
    print(f"Lane Detection Training")
    print(f"{'=' * 60}")
    print(f"Architecture: {config.architecture}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print(f"{'=' * 60}\n")

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export model to ONNX or TensorRT."""
    from src.export.to_onnx import load_checkpoint_and_export

    output_path = Path(args.output)

    # Export to ONNX first
    if args.tensorrt:
        # For TensorRT, we need ONNX intermediate
        onnx_path = output_path.with_suffix(".onnx")
        if not onnx_path.exists():
            print(f"Creating intermediate ONNX: {onnx_path}")
            load_checkpoint_and_export(
                checkpoint_path=args.checkpoint,
                output_path=onnx_path,
                architecture=args.architecture,
                input_size=args.input_size,
            )

        # Convert to TensorRT
        from src.export.to_tensorrt import export_to_tensorrt

        export_to_tensorrt(
            onnx_path=onnx_path,
            output_path=output_path,
            fp16=args.fp16,
            int8=args.int8,
            max_batch_size=args.max_batch_size,
            workspace_size_gb=args.workspace,
        )
    else:
        # Export to ONNX
        load_checkpoint_and_export(
            checkpoint_path=args.checkpoint,
            output_path=output_path,
            architecture=args.architecture,
            input_size=args.input_size,
        )

    print(f"\nModel exported to: {output_path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show model architecture information."""
    from src.models.factory import list_architectures, get_architecture_info

    if args.architecture:
        info = get_architecture_info(args.architecture)
        print(f"\n{info['name']}")
        print("-" * 40)
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        print(f"  Parameter size: {info['parameter_size_mb']:.2f} MB")
        print(f"  Input size: {info['input_size']}x{info['input_size']}")
    else:
        print("\nAvailable architectures:")
        print("-" * 60)
        for arch in list_architectures():
            try:
                info = get_architecture_info(arch)
                print(f"  {arch}:")
                print(
                    f"    Parameters: {info['total_parameters']:,} ({info['parameter_size_mb']:.1f} MB)"
                )
            except Exception as e:
                print(f"  {arch}: (error loading)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Lane Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract images from ROS2 bag files"
    )
    extract_parser.add_argument(
        "--bag-path", type=str, required=True, help="Path to ROS2 bag directory"
    )
    extract_parser.add_argument(
        "--topic", type=str, help="Image topic name (e.g., /camera/image_raw)"
    )
    extract_parser.add_argument(
        "--output-dir", type=str, default="data/raw", help="Output directory"
    )
    extract_parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "bmp"],
        help="Output image format",
    )
    extract_parser.add_argument(
        "--sample-rate", type=int, default=1, help="Extract every Nth frame"
    )
    extract_parser.add_argument(
        "--list-topics", action="store_true", help="List available topics and exit"
    )

    # Crop command
    crop_parser = subparsers.add_parser("crop", help="Crop and preprocess images")
    crop_parser.add_argument(
        "--input-dir", type=str, help="Directory containing images"
    )
    crop_parser.add_argument("--output-dir", type=str, help="Output directory")
    crop_parser.add_argument(
        "--crop-region", type=str, help="Crop preset or 'x,y,width,height'"
    )
    crop_parser.add_argument(
        "--resize", type=str, help="Resize to 'WIDTHxHEIGHT' (e.g., '224x224')"
    )
    crop_parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "jpeg", "bmp"],
        help="Output format",
    )
    crop_parser.add_argument(
        "--list-presets", action="store_true", help="List crop presets and exit"
    )

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare images for annotation"
    )
    prepare_parser.add_argument(
        "--input-dir", type=str, required=True, help="Directory containing images"
    )
    prepare_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    prepare_parser.add_argument(
        "--num-samples", type=int, help="Number of images to select"
    )
    prepare_parser.add_argument(
        "--no-dedup", action="store_true", help="Disable near-duplicate removal"
    )
    prepare_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert Roboflow annotations"
    )
    convert_parser.add_argument(
        "--roboflow-dir", type=str, required=True, help="Roboflow export directory"
    )
    convert_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    convert_parser.add_argument(
        "--split", type=str, default="80:10:10", help="Train:val:test split ratios"
    )
    convert_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to annotated dataset"
    )
    train_parser.add_argument(
        "--architecture", type=str, default="mobilenetv3", help="Model architecture"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    train_parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer",
    )
    train_parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="LR scheduler",
    )
    train_parser.add_argument(
        "--warmup-epochs", type=int, default=5, help="Warmup epochs"
    )
    train_parser.add_argument(
        "--loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "mse", "mae"],
        help="Loss function",
    )
    train_parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    train_parser.add_argument(
        "--freeze-backbone",
        type=int,
        default=0,
        help="Epochs to freeze backbone (0=disabled)",
    )
    train_parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Learning rate multiplier for backbone",
    )
    train_parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    train_parser.add_argument(
        "--image-size", type=int, default=224, help="Input image size"
    )
    train_parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers"
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Checkpoint directory",
    )
    train_parser.add_argument(
        "--log-dir", type=str, default="outputs/logs", help="Tensorboard log directory"
    )
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument(
        "--no-pretrained", action="store_true", help="Don't use pretrained weights"
    )
    train_parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    export_parser.add_argument(
        "--output", type=str, required=True, help="Output file path"
    )
    export_parser.add_argument(
        "--architecture", type=str, default="mobilenetv3", help="Model architecture"
    )
    export_parser.add_argument(
        "--input-size", type=int, default=224, help="Input image size"
    )
    export_parser.add_argument(
        "--tensorrt", action="store_true", help="Export to TensorRT (requires ONNX)"
    )
    export_parser.add_argument(
        "--fp16", action="store_true", default=True, help="Enable FP16 (TensorRT)"
    )
    export_parser.add_argument(
        "--int8", action="store_true", help="Enable INT8 (TensorRT)"
    )
    export_parser.add_argument(
        "--max-batch-size", type=int, default=1, help="Max batch size (TensorRT)"
    )
    export_parser.add_argument(
        "--workspace", type=float, default=1.0, help="Workspace size GB (TensorRT)"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "--architecture", type=str, help="Specific architecture (or all if omitted)"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "extract": cmd_extract,
        "crop": cmd_crop,
        "prepare": cmd_prepare,
        "convert": cmd_convert,
        "train": cmd_train,
        "export": cmd_export,
        "info": cmd_info,
    }

    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.command == "train":
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
