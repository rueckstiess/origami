"""Trainer callbacks for monitoring and customizing training.

Provides a HuggingFace-style callback system for the OrigamiTrainer.
The trainer is silent by default - all output is handled via callbacks.

Example:
    ```python
    from origami.training import OrigamiTrainer, ProgressCallback, MetricsCallback

    trainer = OrigamiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        callbacks=[
            ProgressCallback(),
            MetricsCallback(target_key="category"),
        ],
    )
    trainer.train()
    ```
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from .metrics import exact_match

if TYPE_CHECKING:
    from .trainer import OrigamiTrainer, TrainMetrics, TrainState


class TrainerCallback:
    """Base class for trainer callbacks.

    Subclass this and override the methods you need. All methods receive:
    - trainer: The OrigamiTrainer instance
    - state: Current TrainState (epoch, global_step, etc.)
    - metrics: TrainMetrics from the most recent epoch/eval (may be None)
    """

    def on_train_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called at the end of each batch."""
        pass

    def on_evaluate(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Called after evaluation."""
        pass


class CallbackHandler:
    """Manages multiple callbacks and dispatches events."""

    def __init__(
        self,
        callbacks: list[TrainerCallback],
        log_every_n_batches: int = 1,
    ):
        """Initialize callback handler.

        Args:
            callbacks: List of callbacks to manage.
            log_every_n_batches: Fire batch callbacks every N batches (default=1).
        """
        self.callbacks = callbacks
        self.log_every_n_batches = log_every_n_batches
        self._batch_count = 0

    def fire_event(
        self,
        event: str,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None = None,
    ) -> None:
        """Fire an event to all callbacks.

        Args:
            event: Event name (e.g., "on_epoch_end").
            trainer: The trainer instance.
            state: Current training state.
            metrics: Optional metrics from the event.
        """
        # Handle batch-level throttling
        if event == "on_batch_begin":
            self._batch_count += 1
            if self._batch_count % self.log_every_n_batches != 0:
                return
        elif event == "on_batch_end":
            if self._batch_count % self.log_every_n_batches != 0:
                return
        elif event == "on_epoch_begin":
            self._batch_count = 0  # Reset batch count at epoch start

        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(trainer, state, metrics)


class ProgressCallback(TrainerCallback):
    """Displays tqdm progress bars and training summaries.

    Shows:
    - Progress bar during epoch with loss and learning rate
    - Epoch summary after each epoch
    - Evaluation results when available
    """

    def __init__(self) -> None:
        self._pbar: tqdm | None = None
        self._num_batches: int = 0

    def on_train_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Print training info at start."""
        print(f"Training on {trainer.device}")
        print(
            f"Train samples: {len(trainer.train_dataset)} "
            f"(base: {trainer.train_dataset.base_size}, "
            f"upscale: {trainer.config.upscale_factor}x)"
        )
        if trainer.eval_dataset:
            print(f"Eval samples: {len(trainer.eval_dataset)}")
        print(f"Batch size: {trainer.config.batch_size}")
        print(f"Epochs: {trainer.config.num_epochs}")
        print(f"Total steps: {trainer.total_steps}")
        print()

    def on_epoch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Create progress bar for epoch."""
        self._num_batches = len(trainer.train_dataset) // trainer.config.batch_size
        self._pbar = tqdm(
            total=self._num_batches,
            desc=f"Epoch {state.epoch + 1}",
            leave=False,
        )

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Update progress bar with current batch info."""
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(
                loss=f"{state.current_batch_loss:.4f}",
                lr=f"{state.current_lr:.2e}",
            )

    def on_epoch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Close progress bar and print epoch summary."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        if metrics is not None:
            print(
                f"Epoch {state.epoch + 1}/{trainer.config.num_epochs} - "
                f"Loss: {metrics.loss:.4f} - "
                f"Tokens/sec: {metrics.tokens_per_second:.0f}"
            )

    def on_evaluate(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Print evaluation results."""
        if metrics is not None:
            print(f"\nEval Loss: {metrics.loss:.4f}")


class MetricsCallback(TrainerCallback):
    """Computes and tracks prediction metrics during training.

    Uses the Predictor to evaluate model performance on samples from
    train and eval datasets. Supports any sklearn-compatible metric.

    Example:
        ```python
        from sklearn.metrics import accuracy_score, f1_score

        callback = MetricsCallback(
            target_key="category",
            train_sample_size=200,
            metrics={
                "accuracy": accuracy_score,
                "f1": lambda y, p: f1_score(y, p, average="macro"),
            },
        )
        ```

    Attributes:
        history: Dict mapping metric names to lists of values per epoch.
            Keys are formatted as "{split}_{metric}" (e.g., "train_accuracy").
    """

    def __init__(
        self,
        target_key: str,
        train_sample_size: int = 100,
        eval_sample_size: int = 100,
        compute_every_n_epochs: int = 1,
        metrics: dict[str, Callable[[list, list], float]] | None = None,
    ):
        """Initialize metrics callback.

        Args:
            target_key: The field to predict and evaluate.
            train_sample_size: Number of samples from train set (0 to skip).
            eval_sample_size: Number of samples from eval set (0 to skip).
            compute_every_n_epochs: Compute metrics every N epochs.
            metrics: Dict of {name: metric_fn}. Each metric_fn takes
                (y_true, y_pred) and returns a float. Defaults to exact_match.
        """
        self.target_key = target_key
        self.train_sample_size = train_sample_size
        self.eval_sample_size = eval_sample_size
        self.compute_every_n_epochs = compute_every_n_epochs
        self.metrics = metrics or {"accuracy": exact_match}

        self._predictor = None
        self._train_data: list[dict] | None = None
        self._eval_data: list[dict] | None = None

        # History of metrics per epoch
        self.history: dict[str, list[float]] = {}
        for metric_name in self.metrics:
            self.history[f"train_{metric_name}"] = []
            self.history[f"eval_{metric_name}"] = []

    def on_train_begin(
        self,
        _trainer: OrigamiTrainer,
        _state: TrainState,
        _metrics: TrainMetrics | None,
    ) -> None:
        """Reset caches at the start of training."""
        # Clear cached predictor to ensure fresh state for this training run
        self._predictor = None
        self._train_data = None
        self._eval_data = None

    def _get_predictor(self, trainer: OrigamiTrainer):
        """Lazily create predictor from trainer's model and tokenizer."""
        if self._predictor is None:
            from origami.inference import OrigamiPredictor

            self._predictor = OrigamiPredictor(trainer.model, trainer.tokenizer)
        return self._predictor

    def _get_data(self, trainer: OrigamiTrainer) -> None:
        """Cache references to train/eval data."""
        if self._train_data is None:
            # Trainer always uses UpscaledDataset for train (has base_data)
            self._train_data = trainer.train_dataset.base_data
        if self._eval_data is None and trainer.eval_dataset is not None:
            # Trainer always uses EvalDataset for eval (has data)
            self._eval_data = trainer.eval_dataset.data

    def _sample_data(self, data: list[dict], sample_size: int) -> tuple[list[dict], list[Any]]:
        """Sample data and extract true labels.

        Returns:
            Tuple of (sampled_objects, true_labels)
        """
        if sample_size >= len(data):
            samples = data
        else:
            samples = random.sample(data, sample_size)

        # Extract true labels
        true_labels = [obj[self.target_key] for obj in samples]

        return samples, true_labels

    def on_epoch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        metrics: TrainMetrics | None,
    ) -> None:
        """Compute and log metrics at end of epoch."""
        # Check if we should compute this epoch
        if (state.epoch + 1) % self.compute_every_n_epochs != 0:
            return

        self._get_data(trainer)

        # Move model to CPU for faster prediction, save original device
        original_device = trainer.device
        trainer.model.to("cpu")
        trainer.model.eval()

        predictor = self._get_predictor(trainer)

        results = []

        # Compute train metrics
        if self.train_sample_size > 0 and self._train_data:
            samples, y_true = self._sample_data(self._train_data, self.train_sample_size)
            y_pred = predictor.predict_batch(samples, self.target_key)
            print(f"y_pred", y_pred)

            for name, metric_fn in self.metrics.items():
                value = metric_fn(y_true, y_pred)
                self.history[f"train_{name}"].append(value)
                results.append(f"train_{name}: {value:.4f}")

        # Compute eval metrics
        if self.eval_sample_size > 0 and self._eval_data:
            samples, y_true = self._sample_data(self._eval_data, self.eval_sample_size)
            y_pred = predictor.predict_batch(samples, self.target_key)

            for name, metric_fn in self.metrics.items():
                value = metric_fn(y_true, y_pred)
                self.history[f"eval_{name}"].append(value)
                results.append(f"eval_{name}: {value:.4f}")

        # Restore model to original device and train mode
        trainer.model.to(original_device)
        trainer.model.train()

        # Print results
        if results:
            print(f"  Metrics: {', '.join(results)}")


class TableLogCallback(TrainerCallback):
    """Single-line table format logging, replicating old Origami output style.

    Outputs logs like:
        | step: 10 | epoch: 0 | lr: 1.00e-05 | batch_dt: 23ms | train_loss: 2.2552 |

    When metrics are computed (every `eval_every` steps), also shows:
        | step: 100 | ... | train_acc: 0.3700 | val_loss: 2.2535 | val_acc: 0.3500 |

    Example:
        ```python
        from origami.training import OrigamiTrainer, TableLogCallback

        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            callbacks=[
                TableLogCallback(
                    print_every=10,
                    eval_every=100,
                    target_key="category",
                ),
            ],
        )
        trainer.train()
        ```
    """

    def __init__(
        self,
        print_every: int = 10,
        eval_every: int = 100,
        target_key: str | None = None,
        train_sample_size: int = 100,
        eval_sample_size: int = 100,
    ):
        """Initialize table log callback.

        Args:
            print_every: Print a log line every N batches.
            eval_every: Compute metrics every N batches (must be >= print_every).
            target_key: Field to predict for accuracy. None to skip metrics.
            train_sample_size: Number of samples for train accuracy.
            eval_sample_size: Number of samples for eval metrics.
        """
        import time

        self.print_every = print_every
        self.eval_every = eval_every
        self.target_key = target_key
        self.train_sample_size = train_sample_size
        self.eval_sample_size = eval_sample_size

        # Internal state
        self._time = time
        self._batch_start_time: float = 0.0

        # Cached metrics (shown on each log line until updated)
        self._last_train_acc: float | None = None
        self._last_val_loss: float | None = None
        self._last_val_acc: float | None = None

        # Lazy-loaded components
        self._predictor = None
        self._train_data: list[dict] | None = None
        self._eval_data: list[dict] | None = None

    def on_train_begin(
        self,
        _trainer: OrigamiTrainer,
        _state: TrainState,
        _metrics: TrainMetrics | None,
    ) -> None:
        """Reset state at start of training."""
        self._last_train_acc = None
        self._last_val_loss = None
        self._last_val_acc = None
        self._predictor = None
        self._train_data = None
        self._eval_data = None

    def on_batch_begin(
        self,
        _trainer: OrigamiTrainer,
        _state: TrainState,
        _metrics: TrainMetrics | None,
    ) -> None:
        """Record batch start time."""
        self._batch_start_time = self._time.time()

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainState,
        _metrics: TrainMetrics | None,
    ) -> None:
        """Log batch info and optionally compute metrics."""
        batch_dt = self._time.time() - self._batch_start_time

        # Use global_step for cumulative batch counting
        step = state.global_step

        # Compute metrics if this is an eval batch
        is_eval_step = self.target_key is not None and step % self.eval_every == 0
        if is_eval_step:
            self._compute_metrics(trainer)

        # Print log line if this is a print batch
        if step % self.print_every == 0:
            self._print_log_line(state, batch_dt, show_metrics=is_eval_step)

    def _get_predictor(self, trainer: OrigamiTrainer):
        """Lazily create predictor from trainer's model and tokenizer."""
        if self._predictor is None:
            from origami.inference import OrigamiPredictor

            self._predictor = OrigamiPredictor(trainer.model, trainer.tokenizer)
        return self._predictor

    def _get_data(self, trainer: OrigamiTrainer) -> None:
        """Cache references to train/eval data."""
        if self._train_data is None:
            self._train_data = trainer.train_dataset.base_data
        if self._eval_data is None and trainer.eval_dataset is not None:
            self._eval_data = trainer.eval_dataset.data

    def _compute_metrics(self, trainer: OrigamiTrainer) -> None:
        """Compute train_acc, test_loss, test_acc."""
        self._get_data(trainer)

        # Save original device and move to CPU for prediction
        original_device = trainer.device
        trainer.model.to("cpu")
        trainer.model.eval()

        predictor = self._get_predictor(trainer)

        # Compute train accuracy
        if self.train_sample_size > 0 and self._train_data:
            samples, y_true = self._sample_data(self._train_data, self.train_sample_size)
            y_pred = predictor.predict_batch(samples, self.target_key)
            self._last_train_acc = exact_match(y_true, y_pred)

        # Compute validation loss and accuracy
        if self.eval_sample_size > 0 and self._eval_data:
            samples, y_true = self._sample_data(self._eval_data, self.eval_sample_size)

            # Validation accuracy
            y_pred = predictor.predict_batch(samples, self.target_key)
            self._last_val_acc = exact_match(y_true, y_pred)

            # Validation loss (need to tokenize and run forward pass)
            self._last_val_loss = self._compute_eval_loss(trainer, samples)

        # Restore model to original device and train mode
        trainer.model.to(original_device)
        trainer.model.train()

    def _sample_data(
        self, data: list[dict], sample_size: int
    ) -> tuple[list[dict], list[Any]]:
        """Sample data and extract true labels."""
        if sample_size >= len(data):
            samples = data
        else:
            samples = random.sample(data, sample_size)

        true_labels = [obj[self.target_key] for obj in samples]
        return samples, true_labels

    def _compute_eval_loss(
        self, trainer: OrigamiTrainer, samples: list[dict]
    ) -> float:
        """Compute average loss on samples."""
        import torch

        if not samples:
            print("DEBUG: No samples!")
            return float("nan")

        # Tokenize samples (shuffle=False for deterministic eval)
        tokenized = [trainer.tokenizer.tokenize(obj, shuffle=False) for obj in samples]
        print(f"DEBUG: tokenized {len(tokenized)} samples")

        # Collate into batch
        batch = trainer.collator(tokenized)
        print(f"DEBUG: batch keys = {batch.keys()}")
        print(f"DEBUG: input_ids shape = {batch['input_ids'].shape}")
        print(f"DEBUG: labels shape = {batch['labels'].shape}")

        # Move batch to model's current device (CPU during metrics computation)
        model_device = next(trainer.model.parameters()).device
        print(f"DEBUG: model_device = {model_device}")
        batch = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass
        with torch.no_grad():
            output = trainer.model(
                input_ids=batch["input_ids"],
                path_types=batch["path_types"],
                path_ids=batch["path_ids"],
                path_lengths=batch["path_lengths"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                numeric_values=batch.get("numeric_values"),
                numeric_mask=batch.get("numeric_mask"),
            )

        print(f"DEBUG: output.loss = {output.loss}")
        if output.loss is None:
            return float("nan")

        return output.loss.item()

    def _print_log_line(
        self, state: TrainState, batch_dt: float, show_metrics: bool = False
    ) -> None:
        """Print single-line log output."""
        # Convert batch_dt from seconds to milliseconds
        batch_dt_ms = batch_dt * 1000

        parts = [
            f"step: {state.global_step}",
            f"epoch: {state.epoch}",
            f"lr: {state.current_lr:.2e}",
            f"batch_dt: {batch_dt_ms:.0f}ms",
            f"train_loss: {state.current_batch_loss:.4f}",
        ]

        # Only show metrics on the step they were computed
        if show_metrics:
            if self._last_train_acc is not None:
                parts.append(f"train_acc: {self._last_train_acc:.4f}")
            if self._last_val_loss is not None:
                parts.append(f"val_loss: {self._last_val_loss:.4f}")
            if self._last_val_acc is not None:
                parts.append(f"val_acc: {self._last_val_acc:.4f}")

        print("| " + " | ".join(parts) + " |")
