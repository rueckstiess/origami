"""ORIGAMI training loop.

Provides training utilities with support for:
- Grammar-constrained loss
- Key-order shuffling for data augmentation
- Mixed discrete + continuous loss
- Learning rate scheduling with warmup
- Callback system for monitoring and customization
- Step-based and epoch-based evaluation scheduling (within epochs)
"""

import gc
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Optional accelerate for multi-GPU training
try:
    from accelerate import Accelerator

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

    class Accelerator:  # type: ignore[no-redef]
        """Stub for type checking when accelerate is not installed."""

        pass


from origami.position_encoding import PATH_TYPE_KEY
from origami.tokenizer.vocabulary import KeyToken
from origami.utils import get_device

from .callbacks import CallbackHandler, TrainerCallback
from .collator import OrigamiDataCollator
from .dataset import OrigamiDataset


def _worker_init_fn(worker_id: int) -> None:
    """Initialize DataLoader worker with single-threaded Numba.

    When using multiple DataLoader workers, each worker would otherwise spawn
    its own Numba thread pool. With N workers each using M Numba threads,
    you get N×M threads competing for CPU cores, causing contention.

    By setting NUMBA_NUM_THREADS=1 in each worker, we let the DataLoader
    workers provide parallelism instead of Numba's internal threading.
    """
    import os

    os.environ["NUMBA_NUM_THREADS"] = "1"


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
    best_eval_loss: float = float(
        "inf"
    )  # Backwards compat: tracks val_loss when best_metric="loss"
    epoch_step: int = 0
    epoch_completed: bool = False  # True after epoch finishes, False at start
    epoch_resume_step: int = 0  # Steps skipped at start of epoch when resuming mid-epoch
    current_batch_loss: float = 0.0
    current_lr: float = 0.0
    current_batch_dt: float = 0.0  # Batch time in seconds
    # Generalized best metric tracking (for configurable best_metric)
    best_metric_value: float = float("inf")  # Initialized based on direction in Trainer.__init__
    best_metric_name: str = "loss"  # Key from eval_metrics (or "loss")
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
    - Key-order shuffling for data augmentation
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
        callbacks: list[TrainerCallback] | None = None,
        training_state: dict | None = None,
        schema: dict | None = None,
        inverse_transform_fn: Callable[[Any, str], Any] | None = None,
    ):
        """Initialize trainer.

        Args:
            model: ORIGAMI model to train
            tokenizer: JSONTokenizer with fitted vocabulary
            train_data: List of JSON objects for training
            eval_data: Optional list of JSON objects for evaluation
            config: Training configuration (uses defaults if None)
            device: Device for training (auto-detects if None)
            callbacks: List of TrainerCallback instances for monitoring/customization.
                     Use ProgressCallback for progress bars. Evaluation metrics are
                     computed automatically based on TrainingConfig settings.
            training_state: Optional training state dict for resuming training.
                     Contains optimizer_state, scheduler_state, epoch, global_step.
            schema: Optional JSON Schema dict for semantic constraints.
                     When provided, schema-based masks are intersected with grammar
                     masks during training to restrict outputs by type/enum/keys.
            inverse_transform_fn: Optional function to inverse-transform predicted values.
                     Signature: (value, target_key) -> transformed_value.
                     Used for scaled numeric values to compute metrics on original scale.
        """
        from origami.config import TrainingConfig

        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()

        # Determine if we should use accelerate
        # Only use accelerate when:
        # 1. It's available and enabled in config
        # 2. No explicit device was passed (user wants auto-detection)
        self._use_accelerate = (
            ACCELERATE_AVAILABLE and self.config.use_accelerate and device is None
        )
        self._accelerator: Accelerator | None = None

        if self._use_accelerate:
            # Use accelerate for device management and distributed training
            self._accelerator = Accelerator(mixed_precision=self.config.mixed_precision)
            self.device = self._accelerator.device
        else:
            # Standard single-device training
            self.device = get_device(device)
            self.model.to(self.device)

        # Enable TF32 for faster float32 matmuls on Ampere+ GPUs (A100, etc.)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Store raw data for evaluator (Evaluator needs original dicts, not tokenized)
        self.train_data = train_data
        self.eval_data = eval_data

        # Create datasets
        self.train_dataset = OrigamiDataset(
            train_data,
            tokenizer,
            shuffle=self.config.shuffle_keys,
        )
        self.eval_dataset = (
            OrigamiDataset(eval_data, tokenizer, shuffle=False) if eval_data else None
        )

        # Create collator
        # Don't move tensors to device in collator when:
        # 1. Using accelerate (it handles tensor placement)
        # 2. Using DataLoader workers (tensors must stay on CPU for pickling)
        use_workers = self.config.dataloader_num_workers > 0
        collator_device = None if self._use_accelerate or use_workers else self.device
        # Create grammar PDA if constrain_grammar is enabled and attach to model.
        # This pre-computes grammar masks during data loading instead of in training loop.
        grammar_pda = None
        if self.config.constrain_grammar:
            from origami.constraints.json_grammar import JSONGrammarPDA

            grammar_pda = JSONGrammarPDA(tokenizer.vocab, max_depth=model.config.max_depth)
            model._grammar_pda = grammar_pda

        # Create SchemaPDA for training masks only if constrain_schema is enabled.
        # By default (constrain_schema=False), schema constraints are only applied
        # during inference (evaluator/generator), not during training.
        schema_pda = None
        if schema is not None and self.config.constrain_schema:
            from origami.constraints import SchemaPDA

            schema_pda = SchemaPDA(schema, tokenizer.vocab, max_depth=model.config.max_depth)
            model._schema_pda = schema_pda

        self.collator = OrigamiDataCollator(
            tokenizer,
            max_length=model.config.max_seq_length,
            device=collator_device,
            grammar_pda=grammar_pda,
            schema_pda=schema_pda,
        )

        # Store schema for inference use (evaluator predictions)
        self._schema = schema
        # Track whether we need to move tensors to device in training loop
        self._move_batch_to_device = use_workers and not self._use_accelerate

        # Create optimizer (use fused AdamW on CUDA for faster updates)
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fused=torch.cuda.is_available(),
        )

        # Calculate training steps
        # total_steps: for LR scheduler - based on full epochs (undivided)
        # steps_per_epoch: for progress bar - actual batches per GPU
        steps_per_epoch_full = max(1, len(self.train_dataset) // self.config.batch_size)
        self.total_steps = steps_per_epoch_full * self.config.num_epochs

        # With accelerate, each GPU processes dataset_size / num_gpus batches per epoch
        if self._use_accelerate:
            self.steps_per_epoch = steps_per_epoch_full // self._accelerator.num_processes
        else:
            self.steps_per_epoch = steps_per_epoch_full

        # Create scheduler with linear warmup
        self.scheduler = self._create_scheduler()

        # Apply torch.compile on CUDA for faster training
        if torch.cuda.is_available():
            self.model = torch.compile(self.model)

        # Wrap with accelerate if enabled
        if self._use_accelerate:
            self.model, self.optimizer, self.scheduler = self._accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )

        # Training state
        self.state = TrainResult()
        self._resume_steps_in_epoch = 0  # Steps to skip when resuming mid-epoch

        # Restore training state if provided (for checkpoint resumption)
        if training_state is not None:
            self._restore_training_state(training_state)

        # Callback handler
        self.callback_handler = CallbackHandler(callbacks or [])

        # Validate and resolve best_metric direction
        self._best_metric_direction = self._resolve_best_metric_direction()

        # Initialize best_metric_value based on direction
        if self._best_metric_direction == "maximize":
            self.state.best_metric_value = float("-inf")
        else:
            self.state.best_metric_value = float("inf")
        self.state.best_metric_name = self.config.best_metric

        # Create evaluator for unified evaluation (lazy import to avoid circular)
        from origami.inference import OrigamiEvaluator

        # Resolve allow_complex_values with auto-detection and warning
        allow_complex_values = self._resolve_allow_complex_values()

        # Apply schema constraints during evaluation to match training loss.
        # Uses lenient UNK settings (allow_unk_key=True, allow_unk_value=True)
        # so eval data with unseen tokens maps to UNK and doesn't cause inf loss.
        self.evaluator = OrigamiEvaluator(
            model=model,
            tokenizer=tokenizer,
            target_key=self.config.target_key,
            inverse_transform=inverse_transform_fn,
            allow_complex_values=allow_complex_values,
            schema=self._schema,
            constrain_schema=self.config.constrain_schema,
            allow_unk_key=True,
            allow_unk_value=True,
        )

        # Track last evaluation step to avoid duplicate evals
        self._last_eval_step = -1

        # Cache target key ID for loss weighting (avoids repeated lookup per batch)
        self._target_key_id: int | None = None
        if self.config.target_key is not None and self.config.target_loss_weight != 1.0:
            target_key_token = KeyToken(self.config.target_key)
            if target_key_token in self.tokenizer.vocab._token_to_id:
                self._target_key_id = self.tokenizer.vocab.encode(target_key_token)

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with linear warmup."""
        warmup_steps = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - step / max(1, self.total_steps))

        return LambdaLR(self.optimizer, lr_lambda)

    @property
    def is_main_process(self) -> bool:
        """True if this is the main process (for logging/checkpointing).

        When using accelerate for distributed training, only one process
        should handle logging and saving to avoid conflicts.
        """
        if self._accelerator is not None:
            return self._accelerator.is_main_process
        return True

    @property
    def unwrapped_model(self) -> "OrigamiModel":
        """Get the unwrapped model (without DDP/torch.compile wrappers).

        When using accelerate, the model is wrapped in DistributedDataParallel.
        When using torch.compile, the model is wrapped in OptimizedModule.
        Use this property to access the underlying model for saving or inference.
        """
        model = self.model
        # Unwrap accelerate's DDP wrapper
        if self._accelerator is not None:
            model = self._accelerator.unwrap_model(model)
        # Unwrap torch.compile's OptimizedModule
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _clear_memory_caches(self) -> None:
        """Clear memory caches to prevent unbounded memory growth.

        This is called periodically during training to free memory that
        PyTorch and Python may be holding onto. On macOS, this helps
        prevent excessive swapping during long training runs.
        """
        # Run Python garbage collection
        gc.collect()

        # Clear PyTorch CUDA cache if applicable
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Clear MPS cache if on Apple Silicon
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _resolve_allow_complex_values(self) -> bool:
        """Resolve allow_complex_values with auto-detection and conflict warning.

        If config.allow_complex_values is None, auto-detects based on whether
        any configured metrics require complex values (arrays/objects).

        If config.allow_complex_values is explicitly False but metrics require
        complex values, emits a warning.

        Returns:
            Resolved boolean value for allow_complex_values.
        """
        from origami.training.metrics import (
            any_metric_requires_complex_values,
            metric_requires_complex_values,
        )

        config_value = self.config.allow_complex_values
        requires_complex = any_metric_requires_complex_values(self.config.eval_metrics)

        if config_value is None:
            # Auto-detect based on metrics
            return requires_complex

        if config_value is False and requires_complex:
            # Explicit False but metrics require complex values - warn
            # Build list of conflicting metrics for the warning message
            conflicting = [
                f"{name} ({fn.__name__})"
                for name, fn in self.config.eval_metrics.items()
                if metric_requires_complex_values(fn)
            ]
            warnings.warn(
                f"allow_complex_values=False but these metrics require complex values: "
                f"{sorted(conflicting)}. Evaluation predictions will be restricted "
                f"to primitive values, which may cause these metrics to report incorrect results.",
                UserWarning,
                stacklevel=3,
            )

        return config_value

    def _resolve_best_metric_direction(self) -> str:
        """Validate best_metric and resolve its optimization direction.

        Validates that best_metric is either "loss" (always available) or a key
        in eval_metrics. Then resolves the direction from config or METRIC_DIRECTION.

        Returns:
            "maximize" or "minimize"

        Raises:
            ValueError: If best_metric is not in eval_metrics keys (except "loss")
            ValueError: If metric direction cannot be resolved (unknown metric without
                explicit best_metric_direction)
        """
        from origami.training.metrics import get_metric_direction

        # Validate best_metric is available (except "loss" which is always computed)
        if self.config.best_metric != "loss":
            if (
                not self.config.eval_metrics
                or self.config.best_metric not in self.config.eval_metrics
            ):
                available = (
                    list(self.config.eval_metrics.keys()) if self.config.eval_metrics else []
                )
                raise ValueError(
                    f"best_metric='{self.config.best_metric}' not found in eval_metrics keys "
                    f"{available}. Use one of these keys, or use best_metric='loss'."
                )

        # Resolve direction
        # For "loss", direction is always "minimize"
        # For other metrics, resolve alias -> function name -> direction
        if self.config.best_metric == "loss":
            metric_name = "loss"
        else:
            metric_spec = self.config.eval_metrics[self.config.best_metric]
            metric_name = (
                metric_spec
                if isinstance(metric_spec, str)
                else getattr(metric_spec, "__name__", "")
            )

        direction = self.config.best_metric_direction or get_metric_direction(metric_name)
        if direction is None:
            raise ValueError(
                f"Metric '{metric_name}' (from best_metric='{self.config.best_metric}') "
                f"has no registered direction. Set best_metric_direction='maximize' or "
                f"'minimize' explicitly in TrainingConfig."
            )

        return direction

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

        # Clear memory after evaluation (can accumulate many intermediate tensors)
        self._clear_memory_caches()

        # Fire callback with metrics dict
        self.callback_handler.fire_event("on_evaluate", self, self.state, metrics)

        return metrics

    def _run_evaluation_and_checkpoint(self) -> dict[str, float]:
        """Run evaluation and fire on_best callback if metric improved.

        This consolidates the common pattern of:
        1. Running evaluation
        2. Checking if best_metric improved (based on direction)
        3. Firing on_best callback if improved

        Returns:
            Dict of evaluation metrics
        """
        eval_metrics = self._run_evaluation()

        # Find the metric value (try val_ prefix first, then train_)
        metric_key = f"val_{self.config.best_metric}"
        metric_value = eval_metrics.get(metric_key)

        if metric_value is None:
            # Fall back to train_ prefix if no eval data
            metric_key = f"train_{self.config.best_metric}"
            metric_value = eval_metrics.get(metric_key)

        if metric_value is not None and not math.isnan(metric_value):
            # Check if improved based on direction
            if self._best_metric_direction == "maximize":
                improved = metric_value > self.state.best_metric_value
            else:
                improved = metric_value < self.state.best_metric_value

            if improved:
                self.state.best_metric_value = metric_value
                self.state.best_metric_name = self.config.best_metric
                # Backwards compat: also update best_eval_loss if tracking loss
                if self.config.best_metric == "loss" and metric_key == "val_loss":
                    self.state.best_eval_loss = metric_value
                # Fire on_best callback (allows external saving with additional state)
                self.callback_handler.fire_event("on_best", self, self.state, eval_metrics)

        return eval_metrics

    def train(self) -> TrainResult:
        """Run full training loop.

        Handles KeyboardInterrupt gracefully by running final evaluation
        and returning with interrupted=True. The model state is preserved
        and can be saved.

        When resuming from a checkpoint, training continues from the next
        epoch after the checkpoint was saved.

        Returns:
            TrainResult with completion status and training metrics
        """
        self.callback_handler.fire_event("on_train_begin", self, self.state, None)

        # Determine starting epoch (resume from checkpoint or fresh start)
        # If resuming and epoch was completed, start from next epoch
        # If resuming mid-epoch, start from saved epoch (will skip to saved step)
        if self.state.global_step > 0:
            start_epoch = self.state.epoch + 1 if self.state.epoch_completed else self.state.epoch
        else:
            start_epoch = 0

        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                self.state.epoch = epoch
                self.state.epoch_completed = False  # Mark epoch as in-progress
                epoch_stats = self._train_epoch()
                self.state.epoch_completed = True  # Mark epoch as completed

                self.callback_handler.fire_event("on_epoch_end", self, self.state, epoch_stats)

                # Epoch-based evaluation (using unified system)
                if self._should_evaluate_epoch():
                    self._run_evaluation_and_checkpoint()

            # Final evaluation if we haven't evaluated recently
            if self.config.eval_strategy != "no" and self.eval_data:
                # Check if we should run final eval (avoid duplicate if just evaluated)
                should_run_final_eval = self.state.global_step != self._last_eval_step
                if should_run_final_eval:
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
            Training statistics for the epoch
        """
        self.model.train()

        num_workers = self.config.dataloader_num_workers
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Shuffle sample order each epoch
            collate_fn=self.collator,
            drop_last=True,  # Drop incomplete batches for consistent batch size
            num_workers=num_workers,
            # Use persistent workers if using multiple workers (avoids spawn overhead)
            persistent_workers=num_workers > 0,
            # Pin memory for faster CPU-GPU transfers on CUDA
            pin_memory=torch.cuda.is_available(),
            # Set single-threaded Numba in workers to avoid thread contention
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

        # Wrap dataloader with accelerate for distributed training
        if self._use_accelerate:
            train_loader = self._accelerator.prepare(train_loader)

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()

        # Reset epoch step counter (or start from resume point)
        self.state.epoch_step = 0
        steps_to_skip = self._resume_steps_in_epoch
        self._resume_steps_in_epoch = 0  # Reset for next epoch
        # Track resume position for callbacks (e.g., progress bar initial position)
        self.state.epoch_resume_step = steps_to_skip

        self.callback_handler.fire_event("on_epoch_begin", self, self.state, None)

        for batch in train_loader:
            # Skip batches when resuming mid-epoch
            if steps_to_skip > 0:
                steps_to_skip -= 1
                self.state.epoch_step += 1
                continue

            self.callback_handler.fire_event("on_batch_begin", self, self.state, None)

            batch_start = time.time()
            loss, num_tokens = self._train_step(batch)
            batch_dt = time.time() - batch_start

            total_loss += loss
            total_tokens += num_tokens
            num_batches += 1
            self.state.global_step += 1
            self.state.epoch_step += 1

            # Update state with batch-level info for callbacks
            self.state.current_batch_loss = loss
            self.state.current_lr = self.scheduler.get_last_lr()[0]
            self.state.current_batch_dt = batch_dt

            self.callback_handler.fire_event("on_batch_end", self, self.state, None)

            # Step-based evaluation (runs within epoch if configured)
            if self._should_evaluate_step():
                self._run_evaluation_and_checkpoint()

        duration = time.time() - start_time

        # Clear memory caches at end of epoch to prevent unbounded growth
        self._clear_memory_caches()

        return EpochStats(
            loss=total_loss / max(1, num_batches),
            num_samples=num_batches * self.config.batch_size,
            num_tokens=total_tokens,
            duration_seconds=duration,
        )

    def _compute_loss_weights(self, batch: "EncodedBatch") -> torch.Tensor | None:
        """Compute normalized loss weights for target value tokens.

        Uses path information already in the batch to identify all tokens
        belonging to the target key's value. This works correctly for both
        primitive values and complex nested values (objects/arrays).

        A token belongs to the target value if its path starts with the target key.
        For example, in {"target": {"nested": 1}}:
        - Key("target") has path () - NOT weighted (it's the key, not the value)
        - OBJ_START has path (Key("target"),) - weighted
        - Key("nested") has path (Key("target"),) - weighted
        - Value(1) has path (Key("target"), Key("nested")) - weighted
        - OBJ_END has path (Key("target"),) - weighted

        Weights are normalized so their sum equals the number of valid tokens,
        maintaining stable gradients regardless of target_loss_weight value.

        Args:
            batch: Collated EncodedBatch with path_types and path_ids tensors

        Returns:
            Tensor of shape (batch, seq_len) with normalized weights,
            or None if no weighting is needed.
        """
        # Use cached target key ID (computed once at init, not every batch)
        if self._target_key_id is None:
            return None

        # Identify tokens inside target value using path information
        # A token is in target value if its path starts with the target key
        in_target_value = (batch.path_types[:, :, 0] == PATH_TYPE_KEY) & (
            batch.path_ids[:, :, 0] == self._target_key_id
        )

        # Create weights: target_weight for target value tokens, 1.0 elsewhere
        target_weight = self.config.target_loss_weight
        weights = torch.where(in_target_value, target_weight, 1.0)

        # Normalize weights so sum equals number of valid tokens
        # This keeps the effective learning rate stable
        valid_mask = batch.attention_mask
        valid_weights = weights * valid_mask
        weight_sum = valid_weights.sum()
        valid_token_count = valid_mask.sum()

        if weight_sum > 0:
            weights = weights * (valid_token_count / weight_sum)

        return weights

    def _batch_to_device(self, batch: "EncodedBatch") -> "EncodedBatch":
        """Move batch tensors to training device.

        Used when DataLoader workers return CPU tensors that need to be
        moved to GPU/MPS for training. Uses non_blocking=True to overlap
        transfers with computation when using pinned memory.
        """
        from origami.tokenizer.json_tokenizer import EncodedBatch

        # Use non_blocking for async transfers (works with pin_memory)
        nb = torch.cuda.is_available()

        return EncodedBatch(
            input_ids=batch.input_ids.to(self.device, non_blocking=nb),
            path_types=batch.path_types.to(self.device, non_blocking=nb),
            path_ids=batch.path_ids.to(self.device, non_blocking=nb),
            path_lengths=batch.path_lengths.to(self.device, non_blocking=nb),
            attention_mask=batch.attention_mask.to(self.device, non_blocking=nb),
            numeric_values=batch.numeric_values.to(self.device, non_blocking=nb),
            numeric_mask=batch.numeric_mask.to(self.device, non_blocking=nb),
            lengths=batch.lengths.to(self.device, non_blocking=nb),
            labels=batch.labels.to(self.device, non_blocking=nb)
            if batch.labels is not None
            else None,
            grammar_mask=(
                batch.grammar_mask.to(self.device, non_blocking=nb)
                if batch.grammar_mask is not None
                else None
            ),
        )

    def _train_step(self, batch: "EncodedBatch") -> tuple[float, int]:
        """Execute single training step.

        Args:
            batch: Collated EncodedBatch

        Returns:
            Tuple of (loss value, number of tokens)
        """
        # Move batch tensors to device if needed (when using DataLoader workers)
        if self._move_batch_to_device:
            batch = self._batch_to_device(batch)

        self.optimizer.zero_grad()

        # Use pre-computed grammar mask from batch if available (computed in DataLoader workers)
        # Otherwise compute it here (slower, blocks GPU)
        if batch.grammar_mask is not None:
            grammar_mask = batch.grammar_mask
        else:
            # Fallback: compute grammar mask if not pre-computed
            # Use unwrapped_model to access methods when wrapped in DDP
            grammar_mask = self.unwrapped_model.compute_grammar_mask(batch.input_ids)

        # Compute loss weights for target value tokens (if configured)
        loss_weights = self._compute_loss_weights(batch)

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
            loss_weights=loss_weights,
        )
        loss = output.loss

        # Backward pass (accelerate-aware)
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Count tokens (excluding padding)
        num_tokens = batch.attention_mask.sum().item()

        return loss.item(), int(num_tokens)

    def get_training_state(self) -> dict:
        """Get current training state for checkpoint resumption.

        Returns a dictionary that can be passed to a new trainer's
        `training_state` parameter to resume training.

        Returns:
            Dict containing optimizer_state, scheduler_state, and training progress.
        """
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_eval_loss": self.state.best_eval_loss,
            "epoch_completed": self.state.epoch_completed,
            "steps_in_epoch": self.state.epoch_step,  # For mid-epoch resumption
            # Generalized best metric tracking
            "best_metric_value": self.state.best_metric_value,
            "best_metric_name": self.state.best_metric_name,
        }

    def _restore_training_state(self, state: dict) -> None:
        """Restore training state from checkpoint.

        Called during __init__ when training_state is provided.

        Args:
            state: Training state dict from get_training_state() or checkpoint.
        """
        # Restore optimizer and scheduler state
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        # Restore training progress
        self.state.epoch = state.get("epoch", 0)
        self.state.global_step = state.get("global_step", 0)
        self.state.best_eval_loss = state.get("best_eval_loss", float("inf"))
        self.state.epoch_completed = state.get(
            "epoch_completed", True
        )  # Default True for backwards compat

        # Restore generalized best metric tracking
        # Default to current state values (set in __init__ based on direction)
        if "best_metric_value" in state:
            self.state.best_metric_value = state["best_metric_value"]
        if "best_metric_name" in state:
            self.state.best_metric_name = state["best_metric_name"]

        # Track steps to skip for mid-epoch resumption
        # Only relevant if epoch was not completed
        if not self.state.epoch_completed:
            self._resume_steps_in_epoch = state.get("steps_in_epoch", 0)
        else:
            self._resume_steps_in_epoch = 0
