"""ORIGAMI training loop.

Provides training utilities with support for:
- Grammar-constrained loss
- Key-order shuffling / upscaling
- Mixed discrete + continuous loss
- Learning rate scheduling with warmup
- Callback system for monitoring and customization
- Step-based and epoch-based evaluation scheduling
"""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from origami.utils import get_device

from .callbacks import CallbackHandler, TrainerCallback
from .collator import OrigamiDataCollator
from .dataset import EvalDataset, UpscaledDataset

if TYPE_CHECKING:
    from origami.config import TrainingConfig
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import EncodedBatch, JSONTokenizer


@dataclass
class TrainResult:
    """Mutable training state and result.

    This class tracks training progress during training and contains the final
    result after training completes (whether normally or via interruption).
    """

    # Training progress (updated during training)
    epoch: int = 0
    global_step: int = 0
    best_eval_loss: float = float("inf")
    epoch_step: int = 0
    current_batch_loss: float = 0.0
    current_lr: float = 0.0
    # Completion status (set when training ends)
    completed: bool = False  # True if all epochs finished
    interrupted: bool = False  # True if stopped via KeyboardInterrupt


@dataclass
class EpochStats:
    """Statistics from a training epoch.

    Note: This is distinct from evaluation metrics (dict[str, float]).
    EpochStats tracks training throughput and performance per epoch.
    """

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
        callbacks: list[TrainerCallback] | None = None,
        log_every_n_batches: int = 1,
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
            callbacks: List of TrainerCallback instances for monitoring/customization.
                     Use ProgressCallback for progress bars. Evaluation metrics are
                     computed automatically based on TrainingConfig settings.
            log_every_n_batches: Fire batch callbacks every N batches (default=1).
        """
        from origami.config import TrainingConfig

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

        # Store raw data for evaluator (Evaluator needs original dicts, not tokenized)
        self.train_data = train_data
        self.eval_data = eval_data

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
        self.state = TrainResult()

        # Callback handler
        self.callback_handler = CallbackHandler(
            callbacks or [], log_every_n_batches=log_every_n_batches
        )

        # Create evaluator for unified evaluation (lazy import to avoid circular)
        from origami.inference import OrigamiEvaluator

        self.evaluator = OrigamiEvaluator(
            model=model,
            tokenizer=tokenizer,
            target_key=self.config.target_key,
        )

        # Track last evaluation step to avoid duplicate evals
        self._last_eval_step = -1

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with linear warmup."""
        warmup_steps = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - step / max(1, self.total_steps))

        return LambdaLR(self.optimizer, lr_lambda)

    def _should_evaluate_step(self) -> bool:
        """Check if we should evaluate at the current step.

        Returns True for step-based evaluation when:
        - eval_strategy is "steps"
        - Current step is a multiple of eval_steps
        - We haven't already evaluated at this step
        """
        if self.config.eval_strategy != "steps":
            return False
        if self.state.global_step == 0:
            return False  # Don't evaluate before training starts
        if self.state.global_step == self._last_eval_step:
            return False  # Already evaluated at this step
        return self.state.global_step % self.config.eval_steps == 0

    def _should_evaluate_epoch(self) -> bool:
        """Check if we should evaluate at the current epoch.

        Returns True for epoch-based evaluation when:
        - eval_strategy is "epoch"
        - Current epoch is a multiple of eval_epochs
        """
        if self.config.eval_strategy != "epoch":
            return False
        # epoch is 0-indexed, so check (epoch + 1)
        return (self.state.epoch + 1) % self.config.eval_epochs == 0

    def _run_evaluation(self) -> dict[str, float]:
        """Run unified evaluation using the Evaluator.

        Computes all configured metrics on train and/or eval data.
        Moves model to eval mode, then restores training mode after.

        Returns:
            Dict of metrics with prefixes: {"train_loss": ..., "val_loss": ..., etc}
        """
        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}

        # Evaluate on training data if configured
        if self.config.eval_on_train and self.train_data:
            train_results = self.evaluator.evaluate(
                self.train_data,
                metrics=self.config.eval_metrics,
                sample_size=self.config.eval_sample_size,
                batch_size=self.config.batch_size,
            )
            metrics.update({f"train_{k}": v for k, v in train_results.items()})

        # Evaluate on eval data
        if self.eval_data:
            val_results = self.evaluator.evaluate(
                self.eval_data,
                metrics=self.config.eval_metrics,
                sample_size=self.config.eval_sample_size,
                batch_size=self.config.batch_size,
            )
            metrics.update({f"val_{k}": v for k, v in val_results.items()})

        # Restore training mode
        if was_training:
            self.model.train()

        # Track this evaluation step
        self._last_eval_step = self.state.global_step

        # Fire callback with metrics dict
        self.callback_handler.fire_event("on_evaluate", self, self.state, metrics)

        return metrics

    def _run_evaluation_and_checkpoint(self) -> dict[str, float]:
        """Run evaluation and save best checkpoint if loss improved.

        This consolidates the common pattern of:
        1. Running evaluation
        2. Checking if val_loss improved
        3. Saving "best" checkpoint if configured

        Returns:
            Dict of evaluation metrics
        """
        eval_metrics = self._run_evaluation()

        # Save best model based on val_loss (skip if nan or no val_loss)
        val_loss = eval_metrics.get("val_loss")
        if val_loss is not None and not math.isnan(val_loss):
            if val_loss < self.state.best_eval_loss:
                self.state.best_eval_loss = val_loss
                if self.checkpoint_dir:
                    self.save_checkpoint("best")

        return eval_metrics

    def train(self) -> TrainResult:
        """Run full training loop.

        Handles KeyboardInterrupt gracefully by running final evaluation
        and returning with interrupted=True. The model state is preserved
        and can be saved.

        Returns:
            TrainResult with completion status and training metrics
        """
        self.callback_handler.fire_event("on_train_begin", self, self.state, None)

        try:
            for epoch in range(self.config.num_epochs):
                self.state.epoch = epoch
                metrics = self._train_epoch()

                self.callback_handler.fire_event("on_epoch_end", self, self.state, metrics)

                # Epoch-based evaluation (using unified system)
                if self._should_evaluate_epoch():
                    self._run_evaluation_and_checkpoint()

                # Periodic checkpointing
                if self.checkpoint_dir and (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}")

            # Final evaluation if we haven't evaluated at the last epoch
            if self.config.eval_strategy == "epoch" and self.eval_data:
                last_epoch_had_eval = self.config.num_epochs % self.config.eval_epochs == 0
                if not last_epoch_had_eval:
                    self._run_evaluation_and_checkpoint()

            # Training completed normally
            self.state.completed = True

        except KeyboardInterrupt:
            # Training interrupted - run final evaluation before returning
            self.state.interrupted = True
            if self.eval_data:
                self._run_evaluation_and_checkpoint()
            self.callback_handler.fire_event("on_interrupt", self, self.state, None)

        self.callback_handler.fire_event("on_train_end", self, self.state, None)

        return self.state

    def _train_epoch(self) -> EpochStats:
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

        # Reset epoch step counter
        self.state.epoch_step = 0

        self.callback_handler.fire_event("on_epoch_begin", self, self.state, None)

        for batch in train_loader:
            self.callback_handler.fire_event("on_batch_begin", self, self.state, None)

            loss, num_tokens = self._train_step(batch)

            total_loss += loss
            total_tokens += num_tokens
            num_batches += 1
            self.state.global_step += 1
            self.state.epoch_step += 1

            # Update state with batch-level info for callbacks
            self.state.current_batch_loss = loss
            self.state.current_lr = self.scheduler.get_last_lr()[0]

            self.callback_handler.fire_event("on_batch_end", self, self.state, None)

            # Step-based evaluation (runs within epoch if configured)
            if self._should_evaluate_step():
                self._run_evaluation_and_checkpoint()

        duration = time.time() - start_time

        return EpochStats(
            loss=total_loss / max(1, num_batches),
            num_samples=num_batches * self.config.batch_size,
            num_tokens=total_tokens,
            duration_seconds=duration,
        )

    def _train_step(self, batch: "EncodedBatch") -> tuple[float, int]:
        """Execute single training step.

        Args:
            batch: Collated EncodedBatch

        Returns:
            Tuple of (loss value, number of tokens)
        """
        self.optimizer.zero_grad()

        # Compute grammar mask if model has grammar constraints enabled
        grammar_mask = self.model.compute_grammar_mask(batch.input_ids)

        # Forward pass with explicit grammar mask
        output = self.model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            numeric_values=batch.numeric_values,
            numeric_mask=batch.numeric_mask,
            grammar_mask=grammar_mask,
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
        num_tokens = batch.attention_mask.sum().item()

        return loss.item(), int(num_tokens)

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
