"""Trainer callbacks for monitoring and customizing training.

Provides a HuggingFace-style callback system for the OrigamiTrainer.
The trainer is silent by default - all output is handled via callbacks.

Example:
    ```python
    from origami.training import OrigamiTrainer, ProgressCallback

    trainer = OrigamiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        callbacks=[ProgressCallback()],
    )
    trainer.train()
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from .trainer import EpochStats, OrigamiTrainer, TrainResult


class TrainerCallback:
    """Base class for trainer callbacks.

    Subclass this and override the methods you need. All methods receive:
    - trainer: The OrigamiTrainer instance
    - state: Current TrainResult (epoch, global_step, etc.)
    - payload: Event-specific data (type varies by event):
        - on_epoch_end: EpochStats with training throughput info
        - on_evaluate: dict[str, float] with evaluation metrics
        - Other events: None
    """

    def on_train_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Called at the start of training. Payload is None."""
        pass

    def on_train_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Called at the end of training. Payload is None."""
        pass

    def on_epoch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Called at the start of each epoch. Payload is None."""
        pass

    def on_epoch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: EpochStats | None,
    ) -> None:
        """Called at the end of each epoch. Payload is EpochStats."""
        pass

    def on_batch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Called at the start of each batch. Payload is None."""
        pass

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Called at the end of each batch. Payload is None."""
        pass

    def on_evaluate(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: dict[str, float],
    ) -> None:
        """Called after evaluation.

        Args:
            trainer: The trainer instance.
            state: Current training state.
            payload: Dict of evaluation metrics with prefixed keys.
                Examples: {"val_loss": 0.5, "val_accuracy": 0.85}
                         {"train_loss": 0.6, "train_accuracy": 0.80, "val_loss": 0.5}
        """
        pass

    def on_best(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: dict[str, float],
    ) -> None:
        """Called when a new best model is found (configurable metric improved).

        By default, triggers when val_loss improves (lower is better). The metric
        can be configured via TrainingConfig.best_metric (e.g., "acc" for accuracy)
        and TrainingConfig.best_metric_direction ("maximize" or "minimize").

        Fires AFTER on_evaluate and AFTER state.best_metric_value is updated.
        The payload contains the same evaluation metrics dict as on_evaluate.

        Use this to save checkpoints with additional state that the trainer
        doesn't know about (e.g., preprocessor, schema in OrigamiPipeline).

        Args:
            trainer: The trainer instance.
            state: Current training state. state.best_metric_value contains the new best,
                state.best_metric_name contains the metric key (e.g., "loss", "acc").
            payload: Dict of evaluation metrics (same as on_evaluate payload).
        """
        pass

    def on_interrupt(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Called when training is interrupted via KeyboardInterrupt.

        Fired BEFORE on_train_end. The state.interrupted flag will be True.
        Use this to print interrupt messages or perform cleanup.
        """
        pass


class CallbackHandler:
    """Manages multiple callbacks and dispatches events."""

    def __init__(self, callbacks: list[TrainerCallback]):
        """Initialize callback handler.

        Args:
            callbacks: List of callbacks to manage.
        """
        self.callbacks = callbacks

    def fire_event(
        self,
        event: str,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any = None,
    ) -> None:
        """Fire an event to all callbacks.

        Args:
            event: Event name (e.g., "on_epoch_end").
            trainer: The trainer instance.
            state: Current training state.
            payload: Event-specific data. Type varies by event:
                - on_epoch_end: EpochStats
                - on_evaluate: dict[str, float]
                - Other events: None
        """
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(trainer, state, payload)


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
        state: TrainResult,
        payload: EpochStats | None,
    ) -> None:
        """Print training info at start."""
        if not trainer.is_main_process:
            return

        # Check if resuming from checkpoint
        if state.global_step > 0:
            print(f"Resuming training from epoch {state.epoch + 1}, step {state.global_step}")
        else:
            print(f"Training on {trainer.device}")

        print(f"Train samples: {len(trainer.train_dataset)}")
        if trainer.eval_dataset:
            print(f"Eval samples: {len(trainer.eval_dataset)}")
        print(f"Batch size: {trainer.config.batch_size}")
        print(f"Epochs: {trainer.config.num_epochs}")
        print(f"Total steps: {trainer.total_steps}")
        print()

    def on_epoch_begin(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: EpochStats | None,
    ) -> None:
        """Create progress bar for epoch."""
        if not trainer.is_main_process:
            return
        # Use trainer's steps_per_epoch which accounts for multi-GPU
        self._num_batches = trainer.steps_per_epoch
        self._pbar = tqdm(
            total=self._num_batches,
            initial=state.epoch_resume_step,  # Start at resume position if mid-epoch
            desc=f"Epoch {state.epoch + 1}",
            leave=False,
            unit="batch",
        )

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: EpochStats | None,
    ) -> None:
        """Update progress bar with current batch info."""
        if not trainer.is_main_process:
            return
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(
                loss=f"{state.current_batch_loss:.4f}",
                lr=f"{state.current_lr:.2e}",
                dt=f"{state.current_batch_dt * 1000:.0f}ms",
            )

    def on_epoch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: EpochStats | None,
    ) -> None:
        """Close progress bar and print epoch summary."""
        if not trainer.is_main_process:
            return
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        if payload is not None:
            print(
                f"Epoch {state.epoch + 1}/{trainer.config.num_epochs} - "
                f"loss: {payload.loss:.4f} - "
                f"tokens/sec: {payload.tokens_per_second:.0f}"
            )

    def on_evaluate(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: dict[str, float],
    ) -> None:
        """Print evaluation results.

        Uses tqdm.write() to properly coordinate with the progress bar
        when step-based evaluation fires mid-epoch.
        """
        if not trainer.is_main_process:
            return
        if payload:
            parts = []
            for key, value in sorted(payload.items()):
                parts.append(f"{key}: {value:.4f}")
            # Use tqdm.write to properly coordinate with progress bar
            tqdm.write(f"Eval: {', '.join(parts)}")

    def on_interrupt(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        payload: Any,
    ) -> None:
        """Handle training interruption."""
        if not trainer.is_main_process:
            return
        # Close progress bar if open
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        # Print interrupt message
        print(f"\nTraining interrupted at epoch {state.epoch + 1}, step {state.global_step}")


class TableLogCallback(TrainerCallback):
    """Single-line table format logging, replicating old Origami output style.

    Outputs logs like:
        | step: 10 | epoch: 0 | lr: 1.00e-05 | batch_dt: 23ms | loss: 2.2552 |

    When evaluation runs (configured via TrainingConfig), prints metrics:
        | Eval: train_accuracy: 0.3700, val_accuracy: 0.3500, val_loss: 2.2535 |

    Example:
        ```python
        from origami.training import OrigamiTrainer, TableLogCallback, accuracy
        from origami.config import TrainingConfig

        config = TrainingConfig(
            eval_strategy="steps",
            eval_steps=100,
            eval_metrics={"acc": accuracy},
            target_key="category",
        )
        trainer = OrigamiTrainer(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            eval_data=eval_data,
            config=config,
            callbacks=[TableLogCallback(print_every=10)],
        )
        trainer.train()
        ```
    """

    def __init__(self, print_every: int = 10):
        """Initialize table log callback.

        Args:
            print_every: Print a log line every N steps.
        """
        import time

        self.print_every = print_every
        self._time = time
        self._batch_start_time: float = 0.0
        # For step-based: defer batch line to combine with eval on same step
        self._pending_batch_parts: list[str] | None = None
        # For epoch-based: store eval metrics for next printed batch line
        self._pending_eval_metrics: dict[str, float] | None = None

    def on_batch_begin(
        self,
        _trainer: OrigamiTrainer,
        _state: TrainResult,
        _payload: Any,
    ) -> None:
        """Record batch start time."""
        self._batch_start_time = self._time.time()

    def on_batch_end(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        _payload: Any,
    ) -> None:
        """Print log line every print_every steps."""
        if not trainer.is_main_process:
            return
        if state.global_step % self.print_every != 0:
            return

        batch_dt_ms = (self._time.time() - self._batch_start_time) * 1000

        parts = [
            f"step: {state.global_step}",
            f"epoch: {state.epoch}",
            f"lr: {state.current_lr:.2e}",
            f"batch_dt: {batch_dt_ms:3.0f}ms",
            f"loss: {state.current_batch_loss:.4f}",
        ]

        # Append any pending eval metrics from previous epoch
        if self._pending_eval_metrics:
            for k, v in sorted(self._pending_eval_metrics.items()):
                parts.append(f"{k}: {v:.4f}")
            self._pending_eval_metrics = None

        # Check if step-based evaluation will happen after this batch
        will_eval_step = (
            trainer.config.eval_strategy == "steps"
            and state.global_step > 0
            and state.global_step % trainer.config.eval_steps == 0
        )

        if will_eval_step:
            # Defer printing - on_evaluate will print combined line
            self._pending_batch_parts = parts
        else:
            print("| " + " | ".join(parts) + " |")

    def on_evaluate(
        self,
        trainer: OrigamiTrainer,
        _state: TrainResult,
        payload: dict[str, float],
    ) -> None:
        """Print evaluation metrics, combined with batch stats if available."""
        if not trainer.is_main_process:
            return
        if not payload:
            return

        if self._pending_batch_parts:
            # Step-based eval: combine batch stats + metrics on same line
            metric_parts = [f"{k}: {v:.4f}" for k, v in sorted(payload.items())]
            parts = self._pending_batch_parts + metric_parts
            print("| " + " | ".join(parts) + " |")
            self._pending_batch_parts = None
        elif trainer.config.eval_strategy == "epoch":
            # Epoch-based eval: store for next printed batch line
            self._pending_eval_metrics = payload
        else:
            # Fallback: standalone line (shouldn't normally happen)
            metric_parts = [f"{k}: {v:.4f}" for k, v in sorted(payload.items())]
            print("| Eval: " + ", ".join(metric_parts) + " |")

    def on_interrupt(
        self,
        trainer: OrigamiTrainer,
        state: TrainResult,
        _payload: Any,
    ) -> None:
        """Print interrupt message in table format."""
        if not trainer.is_main_process:
            return
        print(f"\nTraining interrupted at epoch {state.epoch}, step {state.global_step}")
