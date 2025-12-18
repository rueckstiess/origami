"""Unified evaluation for ORIGAMI models.

Provides a single Evaluator class that computes both loss and prediction-based
metrics on the same data samples, supporting step-based and post-training evaluation.
"""

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from origami.training.collator import OrigamiDataCollator
from origami.training.dataset import EvalDataset

from .predictor import OrigamiPredictor

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import JSONTokenizer

# Type alias for metric functions (sklearn convention)
MetricFn = Callable[[list[Any], list[Any]], float]


class OrigamiEvaluator:
    """Unified evaluation for loss and prediction-based metrics.

    Loss is always computed. Additional prediction-based metrics can be provided
    as a dict mapping names to metric functions. All metrics are computed on the
    same data sample for consistency.

    Example:
        ```python
        from origami.training import accuracy

        evaluator = OrigamiEvaluator(model, tokenizer, target_key="label")

        # Just loss (default, fast)
        results = evaluator.evaluate(data=test_data)
        print(f"Loss: {results['loss']:.4f}")

        # Loss + custom metrics
        results = evaluator.evaluate(
            data=test_data,
            metrics={"acc": accuracy},
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
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_key = target_key
        self.inverse_transform = inverse_transform
        self._predictor: OrigamiPredictor | None = None

    @property
    def device(self) -> torch.device:
        """Get the model's current device."""
        return next(self.model.parameters()).device

    def evaluate(
        self,
        data: list[dict],
        metrics: dict[str, MetricFn] | None = None,
        sample_size: int | None = None,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Compute loss and any additional metrics on the same data sample.

        Args:
            data: List of JSON objects to evaluate on
            metrics: Dict mapping metric names to functions. Each function should
                follow sklearn convention: (y_true, y_pred) -> float.
                Example: {"acc": accuracy, "f1": array_f1}
                Loss is always computed automatically.
            sample_size: If set, randomly sample this many examples.
                None means use all data.
            batch_size: Batch size for loss computation and prediction.

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

        # Sample data if requested
        sample = self._sample_data(data, sample_size)

        results: dict[str, float] = {}

        # Always compute loss
        results["loss"] = self._compute_loss(sample, batch_size)

        # Compute prediction-based metrics if any provided
        if metrics:
            y_true, y_pred = self._get_predictions(sample, batch_size)
            for name, metric_fn in metrics.items():
                results[name] = metric_fn(y_true, y_pred)

        return results

    def _sample_data(
        self, data: list[dict], sample_size: int | None
    ) -> list[dict]:
        """Sample data if sample_size is specified."""
        if sample_size is None or sample_size >= len(data):
            return data
        return random.sample(data, sample_size)

    @torch.no_grad()
    def _compute_loss(self, data: list[dict], batch_size: int) -> float:
        """Compute average loss over data with grammar constraints.

        This mirrors the trainer's evaluate() method to ensure consistent
        loss computation with grammar mask applied.
        """
        was_training = self.model.training
        self.model.eval()

        # Create dataset and dataloader
        dataset = EvalDataset(data, self.tokenizer)
        collator = OrigamiDataCollator(
            self.tokenizer,
            max_length=self.model.config.max_seq_length,
            device=self.device,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            # Compute grammar mask for proper loss evaluation
            grammar_mask = self.model.compute_grammar_mask(batch.input_ids)

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

            total_loss += output.loss.item()
            num_batches += 1

        # Restore training mode if it was on
        if was_training:
            self.model.train()

        return total_loss / max(1, num_batches)

    def _get_predictions(
        self, data: list[dict], batch_size: int
    ) -> tuple[list[Any], list[Any]]:
        """Get true values and predictions for all samples.

        Uses CPU for prediction as it's faster for autoregressive generation.
        """
        if self.target_key is None:
            raise ValueError("target_key required for predictions")

        # Lazy init predictor
        if self._predictor is None:
            self._predictor = OrigamiPredictor(
                self.model,
                self.tokenizer,
                inverse_transform_fn=self.inverse_transform,
            )

        # Extract true values
        y_true = [obj[self.target_key] for obj in data]

        # Get predictions (predictor handles device management)
        y_pred = self._predictor.predict_batch(
            data,
            target_key=self.target_key,
            batch_size=batch_size,
        )

        return y_true, y_pred


def evaluate(
    model: "OrigamiModel",
    tokenizer: "JSONTokenizer",
    data: list[dict],
    target_key: str | None = None,
    metrics: dict[str, MetricFn] | None = None,
    sample_size: int | None = None,
    batch_size: int = 32,
    inverse_transform: Callable[[str, Any], Any] | None = None,
) -> dict[str, float]:
    """Convenience function for one-shot evaluation.

    Args:
        model: ORIGAMI model for evaluation
        tokenizer: JSONTokenizer with fitted vocabulary
        data: List of JSON objects to evaluate on
        target_key: Key to predict for prediction-based metrics.
            Required if metrics are provided.
        metrics: Dict mapping metric names to functions.
            Example: {"acc": accuracy}. Loss is always computed.
        sample_size: If set, randomly sample this many examples.
        batch_size: Batch size for evaluation.
        inverse_transform: Optional function to transform predicted values.

    Returns:
        Dict mapping metric names to their values. Always includes "loss".

    Example:
        ```python
        from origami.inference import evaluate
        from origami.training import accuracy

        # Just loss
        results = evaluate(model, tokenizer, test_data)
        print(f"Loss: {results['loss']:.4f}")

        # Loss + accuracy
        results = evaluate(
            model, tokenizer, test_data,
            target_key="label",
            metrics={"acc": accuracy}
        )
        print(f"Accuracy: {results['acc']:.2%}")
        ```
    """
    evaluator = OrigamiEvaluator(
        model, tokenizer, target_key, inverse_transform
    )
    return evaluator.evaluate(
        data,
        metrics=metrics,
        sample_size=sample_size,
        batch_size=batch_size,
    )
