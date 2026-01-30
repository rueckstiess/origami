"""Unified evaluation for ORIGAMI models.

Provides a single Evaluator class that computes both loss and prediction-based
metrics on the same data samples, supporting step-based and post-training evaluation.
"""

import random
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from origami.preprocessing.numeric_scaler import ScaledNumeric
from origami.training.collator import OrigamiDataCollator
from origami.training.dataset import OrigamiDataset
from origami.training.metrics import MetricSpec, resolve_metrics

from .predictor import OrigamiPredictor

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import JSONTokenizer


class OrigamiEvaluator:
    """Unified evaluation for loss and prediction-based metrics.

    Loss is always computed. Additional prediction-based metrics can be provided
    as a dict mapping prefixes to metric names. All metrics are computed on the
    same data sample for consistency.

    Example:
        ```python
        evaluator = OrigamiEvaluator(model, tokenizer, target_key="label")

        # Just loss (default, fast)
        results = evaluator.evaluate(data=test_data)
        print(f"Loss: {results['loss']:.4f}")

        # Loss + custom metrics
        results = evaluator.evaluate(
            data=test_data,
            metrics={"acc": "accuracy"},
            sample_size=100,
        )
        print(f"Loss: {results['loss']:.4f}")
        print(f"Accuracy: {results['acc']:.2%}")
        ```

    Attributes:
        model: ORIGAMI model for evaluation
        tokenizer: JSONTokenizer for encoding
        target_key: Key to predict for prediction-based metrics
    """

    def __init__(
        self,
        model: "OrigamiModel",
        tokenizer: "JSONTokenizer",
        target_key: str | None = None,
        inverse_transform: Callable[[str, Any], Any] | None = None,
        allow_complex_values: bool = False,
        schema: dict | None = None,
        constrain_schema: bool = False,
        allow_unk_key: bool = True,
        allow_unk_value: bool = True,
    ):
        """Initialize evaluator.

        Args:
            model: ORIGAMI model for evaluation
            tokenizer: JSONTokenizer with fitted vocabulary
            target_key: Key to predict for prediction-based metrics.
                Required if using any metric other than "loss".
            inverse_transform: Optional function to transform predicted values
                back to original scale. Signature: (leaf_key, value) -> value.
                Used for continuous numeric fields that were scaled during preprocessing.
            allow_complex_values: Whether to allow complex values (objects/arrays)
                during predictions. Default False for backward compatibility.
            schema: Optional JSON Schema dict for semantic constraints.
                Passed through to the internal predictor/generator for predictions.
            constrain_schema: If True and schema is provided, also apply schema
                constraints during loss computation (to match training loss when
                the trainer uses constrain_schema=True).
            allow_unk_key: Whether UNK_KEY is allowed in schema masks. Default
                True for evaluation so unseen keys in eval data don't cause inf loss.
            allow_unk_value: Whether UNK_VALUE is allowed in schema masks. Default
                True for evaluation so unseen values in eval data don't cause inf loss.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_key = target_key
        self.inverse_transform = inverse_transform
        self.allow_complex_values = allow_complex_values
        self._schema = schema
        self._predictor: OrigamiPredictor | None = None

        # Create SchemaPDA for loss computation only if constrain_schema is True.
        # Uses lenient UNK settings by default so eval data with unseen tokens
        # doesn't produce inf loss.
        self._schema_pda = None
        if constrain_schema and schema is not None:
            from origami.constraints import SchemaPDA

            self._schema_pda = SchemaPDA(
                schema,
                tokenizer.vocab,
                max_depth=model.config.max_depth,
                allow_unk_key=allow_unk_key,
                allow_unk_value=allow_unk_value,
            )

    @property
    def device(self) -> torch.device:
        """Get the model's current device."""
        return next(self.model.parameters()).device

    def evaluate(
        self,
        data: list[dict],
        metrics: dict[str, MetricSpec] | None = None,
        sample_size: int | None = None,
        batch_size: int = 32,
        allow_complex_values: bool | None = None,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Compute loss and any additional metrics on the same data sample.

        Args:
            data: List of JSON objects to evaluate on
            metrics: Dict mapping prefixes to metric names or functions.
                Example: {"acc": "accuracy", "f1": "array_f1"}
                Loss is always computed automatically.
            sample_size: If set, randomly sample this many examples.
                None means use all data.
            batch_size: Batch size for loss computation and prediction.
            allow_complex_values: If provided, overrides the instance default.
            verbose: If True, show progress bars during evaluation.

        Returns:
            Dict mapping metric names to their values. Always includes "loss".

        Raises:
            ValueError: If metrics provided but target_key not set.
        """
        # Validate prediction metrics have target_key
        if metrics and self.target_key is None:
            raise ValueError(
                f"target_key required for prediction metrics: {list(metrics.keys())}. "
                "Pass target_key to OrigamiEvaluator constructor."
            )

        # Resolve metrics from string names to functions
        resolved_metrics = resolve_metrics(metrics) if metrics else None

        # Resolve allow_complex_values (parameter overrides instance default)
        effective_allow_complex = (
            allow_complex_values if allow_complex_values is not None else self.allow_complex_values
        )

        # Sample data if requested
        sample = self._sample_data(data, sample_size)

        results: dict[str, float] = {}

        # Always compute loss
        results["loss"] = self._compute_loss(sample, batch_size, verbose=verbose)

        # Compute prediction-based metrics if any provided
        if resolved_metrics:
            y_true, y_pred = self._get_predictions(
                sample, batch_size, effective_allow_complex, verbose=verbose
            )
            for name, metric_fn in resolved_metrics.items():
                results[name] = metric_fn(y_true, y_pred)

        return results

    def _sample_data(self, data: list[dict], sample_size: int | None) -> list[dict]:
        """Sample data if sample_size is specified."""
        if sample_size is None or sample_size >= len(data):
            return data
        return random.sample(data, sample_size)

    def _wrap_progress(self, iterable: Iterator, total: int, desc: str, verbose: bool) -> Iterator:
        """Wrap an iterator with tqdm if verbose, otherwise return as-is."""
        if verbose:
            return tqdm(iterable, total=total, desc=desc)
        return iterable

    @torch.no_grad()
    def _compute_loss(self, data: list[dict], batch_size: int, verbose: bool = False) -> float:
        """Compute average loss over data with grammar and optional schema constraints.

        Uses the same collator-based approach as the trainer to ensure consistent
        loss computation. Grammar mask is always applied. Schema mask is applied
        only when constrain_schema=True (matching training behavior).
        """
        was_training = self.model.training
        self.model.eval()

        # Get grammar PDA from model (same as trainer does)
        grammar_pda = getattr(self.model, "_grammar_pda", None)

        # Create dataset and dataloader with PDA-enabled collator
        dataset = OrigamiDataset(data, self.tokenizer, shuffle=False)
        collator = OrigamiDataCollator(
            self.tokenizer,
            max_length=self.model.config.max_seq_length,
            device=self.device,
            grammar_pda=grammar_pda,
            schema_pda=self._schema_pda,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        total_loss = 0.0
        num_batches = 0
        num_total_batches = (len(data) + batch_size - 1) // batch_size

        for batch in self._wrap_progress(loader, num_total_batches, "Computing loss", verbose):
            output = self.model(
                input_ids=batch.input_ids,
                path_types=batch.path_types,
                path_ids=batch.path_ids,
                path_lengths=batch.path_lengths,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                numeric_values=batch.numeric_values,
                numeric_mask=batch.numeric_mask,
                grammar_mask=batch.grammar_mask,
            )

            total_loss += output.loss.item()
            num_batches += 1

        # Restore training mode if it was on
        if was_training:
            self.model.train()

        return total_loss / max(1, num_batches)

    def _get_predictions(
        self,
        data: list[dict],
        batch_size: int,
        allow_complex_values: bool = False,
        verbose: bool = False,
    ) -> tuple[list[Any], list[Any]]:
        """Get true values and predictions for all samples.

        Uses CPU for prediction as it's faster for autoregressive generation.

        Args:
            data: List of JSON objects to predict on.
            batch_size: Batch size for prediction.
            allow_complex_values: Whether to allow complex values (objects/arrays).
            verbose: If True, show progress bar during prediction.
        """
        if self.target_key is None:
            raise ValueError("target_key required for predictions")

        # Lazy init predictor
        if self._predictor is None:
            self._predictor = OrigamiPredictor(
                self.model,
                self.tokenizer,
                inverse_transform_fn=self.inverse_transform,
                schema=self._schema,
            )

        # Extract true values and prepare them (unwrap ScaledNumeric, inverse transform)
        y_true = [self._prepare_true_value(obj[self.target_key]) for obj in data]

        # Get predictions (predictor handles device management)
        y_pred = self._predictor.predict_batch(
            data,
            target_key=self.target_key,
            batch_size=batch_size,
            allow_complex_values=allow_complex_values,
            verbose=verbose,
        )

        return y_true, y_pred

    def _prepare_true_value(self, value: Any) -> Any:
        """Unwrap ScaledNumeric and apply inverse transform if configured.

        This ensures y_true matches y_pred (which is already inverse-transformed
        by the predictor) for consistent metric computation.
        """
        # Unwrap ScaledNumeric
        if isinstance(value, ScaledNumeric):
            value = value.value

        # Apply inverse transform if configured (to match y_pred)
        if (
            self.inverse_transform
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return self.inverse_transform(value, self.target_key)

        return value


def evaluate(
    model: "OrigamiModel",
    tokenizer: "JSONTokenizer",
    data: list[dict],
    target_key: str | None = None,
    metrics: dict[str, MetricSpec] | None = None,
    sample_size: int | None = None,
    batch_size: int = 32,
    inverse_transform: Callable[[str, Any], Any] | None = None,
    allow_complex_values: bool = False,
    schema: dict | None = None,
    constrain_schema: bool = False,
) -> dict[str, float]:
    """Convenience function for one-shot evaluation.

    Args:
        model: ORIGAMI model for evaluation
        tokenizer: JSONTokenizer with fitted vocabulary
        data: List of JSON objects to evaluate on
        target_key: Key to predict for prediction-based metrics.
            Required if metrics are provided.
        metrics: Dict mapping prefixes to metric names or functions.
            Example: {"acc": "accuracy"}. Loss is always computed.
        sample_size: If set, randomly sample this many examples.
        batch_size: Batch size for evaluation.
        inverse_transform: Optional function to transform predicted values.
        allow_complex_values: Whether to allow complex values (objects/arrays)
            during predictions. Default False.
        schema: Optional JSON Schema dict for semantic constraints.
        constrain_schema: If True and schema is provided, apply schema
            constraints during loss computation (to match training loss).

    Returns:
        Dict mapping metric names to their values. Always includes "loss".

    Example:
        ```python
        from origami.inference import evaluate

        # Just loss
        results = evaluate(model, tokenizer, test_data)
        print(f"Loss: {results['loss']:.4f}")

        # Loss + accuracy
        results = evaluate(
            model, tokenizer, test_data,
            target_key="label",
            metrics={"acc": "accuracy"}
        )
        print(f"Accuracy: {results['acc']:.2%}")
        ```
    """
    evaluator = OrigamiEvaluator(
        model,
        tokenizer,
        target_key,
        inverse_transform,
        allow_complex_values,
        schema,
        constrain_schema,
    )
    return evaluator.evaluate(
        data,
        metrics=metrics,
        sample_size=sample_size,
        batch_size=batch_size,
    )
