"""
Training loop for lane detection models.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import get_loss_function
from .metrics import MetricsTracker


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    architecture: str = "mobilenetv3"
    pretrained: bool = True

    # Data
    data_dir: str = "data/annotated"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    horizontal_flip_prob: float = 0.5

    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5

    # Loss
    loss_type: str = "smooth_l1"

    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.0

    # Progressive unfreezing
    freeze_backbone_epochs: int = 0  # 0 = don't freeze
    backbone_lr_multiplier: float = 0.1  # Lower LR for backbone

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: str = "outputs/checkpoints"
    save_every: int = 10
    save_best: bool = True

    # Logging
    log_dir: str = "outputs/logs"
    log_every: int = 10

    # Device
    device: str = "auto"
    use_amp: bool = True  # Automatic Mixed Precision

    # Seed
    seed: int = 42


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class Trainer:
    """
    Training loop for lane detection models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Setup loss function
        self.criterion = get_loss_function(config.loss_type)

        # Setup optimizer with different LRs for backbone and head
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Setup AMP (only use GradScaler for CUDA)
        self.use_amp = config.use_amp and self.device.type in ("cuda", "mps")
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if config.use_amp and self.device.type == "cuda"
            else None
        )

        # Setup logging
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(config.log_dir)

        # Setup checkpointing
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _setup_optimizer(self):
        """Setup optimizer with optional different LRs for backbone/head."""
        config = self.config

        # Separate parameters
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # Create parameter groups
        param_groups = [
            {"params": head_params, "lr": config.learning_rate},
        ]

        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": config.learning_rate * config.backbone_lr_multiplier,
                }
            )

        # Create optimizer
        if config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        config = self.config
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = len(self.train_loader) * config.warmup_epochs

        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
            )
        elif config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif config.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        elif config.scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")

        # Warmup scheduler
        if config.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self.warmup_scheduler = None

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Handle backbone freezing
        if self.current_epoch < self.config.freeze_backbone_epochs:
            self.model.freeze_backbone()
        elif self.current_epoch == self.config.freeze_backbone_epochs:
            self.model.unfreeze_backbone()

        metrics = MetricsTracker()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                if self.scaler is not None:
                    # CUDA path with GradScaler
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # MPS path without GradScaler
                    loss.backward()
                    self.optimizer.step()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Update schedulers
            if (
                self.warmup_scheduler
                and self.global_step
                < len(self.train_loader) * self.config.warmup_epochs
            ):
                self.warmup_scheduler.step()
            elif self.scheduler and self.config.scheduler != "plateau":
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            metrics.update(outputs, targets)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to tensorboard
            if self.global_step % self.config.log_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr",
                    self.optimizer.param_groups[0]["lr"],
                    self.global_step,
                )

            self.global_step += 1

        # Compute epoch metrics
        epoch_metrics = metrics.compute()
        epoch_metrics["loss"] = total_loss / num_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()

        metrics = MetricsTracker()
        total_loss = 0.0
        num_batches = 0

        for images, targets in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1
            metrics.update(outputs, targets)

        val_metrics = metrics.compute()
        val_metrics["loss"] = total_loss / num_batches

        return val_metrics

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best.pt"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def train(self) -> dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dict with training history and final metrics
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_mae"].append(train_metrics["mae"])

            # Validate
            val_metrics = self.validate()
            history["val_loss"].append(val_metrics["loss"])
            history["val_mae"].append(val_metrics["mae"])

            # Update plateau scheduler
            if self.scheduler and self.config.scheduler == "plateau":
                self.scheduler.step(val_metrics["loss"])

            # Log to tensorboard
            self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/train_mae", train_metrics["mae"], epoch)
            self.writer.add_scalar("epoch/val_mae", val_metrics["mae"], epoch)

            # Print progress
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(
                f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}"
            )
            print(
                f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}"
            )
            print(f"  Within 0.1: {val_metrics['within_0.1']:.1f}%")

            # Check for best model
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                print("  ** New best model! **")

            # Save checkpoint
            if self.config.save_best and is_best:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt",
                    is_best=True,
                )
            elif (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"epoch_{epoch + 1}.pt",
                )

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            print()

        # Save final model
        self.save_checkpoint(
            Path(self.config.checkpoint_dir) / "final.pt",
        )

        self.writer.close()

        return {
            "history": history,
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.current_epoch,
        }
