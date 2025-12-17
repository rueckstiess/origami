"""ORIGAMI training loop.

Provides training utilities with support for:
- Grammar-constrained loss
- Key-order shuffling / upscaling
- Mixed discrete + continuous loss
- Learning rate scheduling with warmup
"""

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from origami.utils import get_device

from .collator import OrigamiDataCollator
from .dataset import EvalDataset, UpscaledDataset

if TYPE_CHECKING:
    from origami.model.config import TrainingConfig
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import JSONTokenizer


@dataclass
class TrainState:
    """Mutable training state."""

    epoch: int = 0
    global_step: int = 0
    best_eval_loss: float = float("inf")


@dataclass
class TrainMetrics:
    """Metrics from a training epoch or evaluation."""

    loss: float
    num_samples: int
    num_tokens: int
    duration_seconds: float

    @property
    def tokens_per_second(self) -> float:
        """Compute throughput."""
        return self.num_tokens / self.duration_seconds if self.duration_seconds > 0 else 0


class OrigamiTrainer:
    """Training loop for ORIGAMI model.

    Supports:
    - Automatic upscaling and key-order shuffling
    - Grammar-constrained loss (via model)
    - Mixed discrete + continuous loss (via model)
    - Linear warmup learning rate schedule
    - Periodic evaluation and checkpointing

    Example:
        ```python
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_objects,
            eval_data=eval_objects,
            config=TrainingConfig(
                batch_size=32,
                num_epochs=100,
                upscale_factor=10,
            ),
        )
        trainer.train()
        ```

    Attributes:
        model: ORIGAMI model to train
        tokenizer: JSONTokenizer for encoding
        config: Training configuration
        device: Device for training
    """

    def __init__(
        self,
        model: "OrigamiModel",
        tokenizer: "JSONTokenizer",
        train_data: list[dict],
        eval_data: list[dict] | None = None,
        config: "TrainingConfig | None" = None,
        device: torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
        shuffle: bool = True,
    ):
        """Initialize trainer.

        Args:
            model: ORIGAMI model to train
            tokenizer: JSONTokenizer with fitted vocabulary
            train_data: List of JSON objects for training
            eval_data: Optional list of JSON objects for evaluation
            config: Training configuration (uses defaults if None)
            device: Device for training (auto-detects if None)
            checkpoint_dir: Directory for saving checkpoints
            shuffle: Whether to shuffle key order during training (default True).
                     If False, upscaling is disabled since it would just duplicate samples.
        """
        from origami.model.config import TrainingConfig

        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()

        # Auto-detect device (supports CUDA, MPS, CPU)
        self.device = get_device(device)
        self.model.to(self.device)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create datasets
        self.train_dataset = UpscaledDataset(
            train_data,
            tokenizer,
            upscale_factor=self.config.upscale_factor,
            shuffle=shuffle,
        )
        self.eval_dataset = EvalDataset(eval_data, tokenizer) if eval_data else None

        # Create collator
        self.collator = OrigamiDataCollator(
            tokenizer,
            max_length=model.config.max_seq_length,
            device=self.device,
        )

        # Create optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate total training steps for scheduler
        steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        self.total_steps = steps_per_epoch * self.config.num_epochs

        # Create scheduler with linear warmup
        self.scheduler = self._create_scheduler()

        # Training state
        self.state = TrainState()

        # Callbacks (optional)
        self.on_epoch_end: Callable[[int, TrainMetrics], None] | None = None
        self.on_eval_end: Callable[[int, TrainMetrics], None] | None = None

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with linear warmup."""
        warmup_steps = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - step / max(1, self.total_steps))

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> TrainState:
        """Run full training loop.

        Returns:
            Final training state
        """
        print(f"Training on {self.device}")
        print(
            f"Train samples: {len(self.train_dataset)} (base: {self.train_dataset.base_size}, upscale: {self.config.upscale_factor}x)"
        )
        if self.eval_dataset:
            print(f"Eval samples: {len(self.eval_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Total steps: {self.total_steps}")
        print()

        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            metrics = self._train_epoch()

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Loss: {metrics.loss:.4f} - "
                f"Tokens/sec: {metrics.tokens_per_second:.0f}"
            )

            if self.on_epoch_end:
                self.on_epoch_end(epoch, metrics)

            # Evaluation
            if self.eval_dataset and (epoch + 1) % max(1, self.config.save_every_n_epochs) == 0:
                eval_metrics = self.evaluate()
                print(f"  Eval Loss: {eval_metrics.loss:.4f}")

                if self.on_eval_end:
                    self.on_eval_end(epoch, eval_metrics)

                # Save best model (skip if loss is nan)
                if (
                    not math.isnan(eval_metrics.loss)
                    and eval_metrics.loss < self.state.best_eval_loss
                ):
                    self.state.best_eval_loss = eval_metrics.loss
                    if self.checkpoint_dir:
                        self.save_checkpoint("best")

            # Periodic checkpointing
            if self.checkpoint_dir and (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        # Final evaluation if we haven't evaluated at the last epoch
        if self.eval_dataset:
            last_epoch_had_eval = self.config.num_epochs % self.config.save_every_n_epochs == 0
            if not last_epoch_had_eval:
                eval_metrics = self.evaluate()
                print(f"  Final Eval Loss: {eval_metrics.loss:.4f}")
                if (
                    not math.isnan(eval_metrics.loss)
                    and eval_metrics.loss < self.state.best_eval_loss
                ):
                    self.state.best_eval_loss = eval_metrics.loss
                    if self.checkpoint_dir:
                        self.save_checkpoint("best")

        return self.state

    def _train_epoch(self) -> TrainMetrics:
        """Train for one epoch.

        Returns:
            Training metrics for the epoch
        """
        self.model.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Shuffle order of (upscaled) samples
            collate_fn=self.collator,
            drop_last=True,  # Drop incomplete batches for consistent batch size
        )

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {self.state.epoch + 1}", leave=False)
        for batch in pbar:
            loss, num_tokens = self._train_step(batch)

            total_loss += loss
            total_tokens += num_tokens
            num_batches += 1
            self.state.global_step += 1

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

        duration = time.time() - start_time

        return TrainMetrics(
            loss=total_loss / max(1, num_batches),
            num_samples=num_batches * self.config.batch_size,
            num_tokens=total_tokens,
            duration_seconds=duration,
        )

    def _train_step(self, batch: dict[str, Tensor]) -> tuple[float, int]:
        """Execute single training step.

        Args:
            batch: Collated batch dictionary

        Returns:
            Tuple of (loss value, number of tokens)
        """
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(
            input_ids=batch["input_ids"],
            path_types=batch["path_types"],
            path_ids=batch["path_ids"],
            path_lengths=batch["path_lengths"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Count tokens (excluding padding)
        num_tokens = batch["attention_mask"].sum().item()

        return loss.item(), int(num_tokens)

    @torch.no_grad()
    def evaluate(self) -> TrainMetrics:
        """Evaluate on eval dataset.

        Returns:
            Evaluation metrics
        """
        if self.eval_dataset is None:
            raise ValueError("No eval dataset provided")

        self.model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()

        for batch in eval_loader:
            output = self.model(
                input_ids=batch["input_ids"],
                path_types=batch["path_types"],
                path_ids=batch["path_ids"],
                path_lengths=batch["path_lengths"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            total_loss += output.loss.item()
            total_tokens += batch["attention_mask"].sum().item()
            num_batches += 1

        duration = time.time() - start_time

        return TrainMetrics(
            loss=total_loss / max(1, num_batches),
            num_samples=len(self.eval_dataset),
            num_tokens=total_tokens,
            duration_seconds=duration,
        )

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint.

        Saves model weights, optimizer state, scheduler state, training state,
        model config, and tokenizer. The checkpoint can be loaded with
        `OrigamiModel.load()` for inference or `load_checkpoint()` to resume training.

        Args:
            name: Checkpoint name (e.g., "best", "epoch_10")

        Returns:
            Path to saved checkpoint
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                # Model weights and config
                "model_state_dict": self.model.state_dict(),
                "model_config": asdict(self.model.config),
                # Tokenizer state for full reconstruction
                "tokenizer_state": {
                    "vocab": self.tokenizer.vocab.to_dict(),
                    "max_depth": self.tokenizer.max_depth,
                    "max_array_index": self.tokenizer.max_array_index,
                },
                # Training state for resumption
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "state": {
                    "epoch": self.state.epoch,
                    "global_step": self.state.global_step,
                    "best_eval_loss": self.state.best_eval_loss,
                },
                "training_config": asdict(self.config),
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        state_dict = checkpoint["state"]
        self.state.epoch = state_dict["epoch"]
        self.state.global_step = state_dict["global_step"]
        self.state.best_eval_loss = state_dict["best_eval_loss"]
