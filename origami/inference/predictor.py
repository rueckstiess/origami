"""ORIGAMI value predictor.

Predicts values for target keys in JSON documents using a trained ORIGAMI model.

This is a thin wrapper around Generator. It:
1. Prepares input (reorder keys, tokenize, truncate at target)
2. Delegates to Generator for actual generation
3. Applies inverse transformation if needed
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from tqdm.auto import tqdm

from origami.preprocessing import move_target_last
from origami.tokenizer.vocabulary import ValueToken

from .generator import OrigamiGenerator
from .utils import find_target_positions

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import EncodedBatch, JSONTokenizer


class OrigamiPredictor:
    """Predict values for target keys using a trained ORIGAMI model.

    This is a thin wrapper around Generator. It:
    1. Prepares input (reorder keys, truncate at target)
    2. Delegates to Generator for actual generation
    3. Applies inverse transformation if needed

    Example:
        ```python
        predictor = OrigamiPredictor(model, tokenizer)

        # Predict single value
        obj = {"name": "Alice", "age": 30, "city": None}  # city is target
        prediction = predictor.predict(obj, target_key="city")
        # Returns: "NYC"

        # Batch prediction with internal batching
        predictions = predictor.predict_batch(objects, target_key="city", batch_size=64)
        # Returns: ["NYC", "LA", "SF", ...]

        # Get probability distribution
        probs = predictor.predict_proba(obj, target_key="city")
        # Returns: {"NYC": 0.45, "LA": 0.32, "SF": 0.18, ...}

        # Get top-k with probabilities
        top3 = predictor.predict_proba(obj, target_key="city", top_k=3)
        # Returns: [("NYC", 0.45), ("LA", 0.32), ("SF", 0.18)]
        ```
    """

    def __init__(
        self,
        model: "OrigamiModel",
        tokenizer: "JSONTokenizer",
        inverse_transform_fn: Callable[[Any, str], Any] | None = None,
        schema: dict | None = None,
    ):
        """Initialize predictor.

        Args:
            model: Trained ORIGAMI model
            tokenizer: JSONTokenizer with fitted vocabulary
            inverse_transform_fn: Optional function to inverse-transform predictions.
                Signature: (value, target_key) -> transformed_value.
                Used for scaled numeric values that need to be converted back
                to original scale.
            schema: Optional JSON Schema dict for semantic constraints.
                Passed through to the internal OrigamiGenerator.

        Note:
            The Predictor uses the model's current device dynamically.
            Move the model to your desired device before calling predict().
            For standalone use, CPU is typically fastest for ORIGAMI model sizes.
        """
        from origami.training.collator import OrigamiDataCollator

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self._inverse_transform = inverse_transform_fn

        # Create generator for value generation (handles grammar + continuous values)
        self._generator = OrigamiGenerator(model, tokenizer, schema=schema)

        # Create collator for batch creation (include_labels=False for inference)
        self._collator = OrigamiDataCollator(tokenizer, include_labels=False)

    @property
    def device(self) -> torch.device:
        """Get the model's current device dynamically."""
        return next(self.model.parameters()).device

    def predict(
        self,
        obj: dict,
        target_key: str,
        allow_complex_values: bool = False,
    ) -> Any:
        """Predict value for a single object.

        Args:
            obj: JSON object (target_key's current value is ignored)
            target_key: Key to predict (dot notation for nested)
            allow_complex_values: If False (default), restrict to primitive values
                only (strings, numbers, booleans, null). If True, allow objects
                and arrays which may require multiple tokens to generate.

        Returns:
            Predicted value (inverse transformed if scaler configured)
        """
        return self.predict_batch([obj], target_key, allow_complex_values=allow_complex_values)[0]

    @torch.no_grad()
    def predict_batch(
        self,
        objects: list[dict],
        target_key: str,
        batch_size: int = 32,
        allow_complex_values: bool = False,
        verbose: bool = False,
    ) -> list[Any]:
        """Predict values for a batch of objects.

        Handles batching internally for large object lists.

        Args:
            objects: List of JSON objects
            target_key: Key to predict (same for all objects)
            batch_size: Number of objects to process in parallel
            allow_complex_values: If False (default), restrict to primitive values
                only (strings, numbers, booleans, null). If True, allow objects
                and arrays which may require multiple tokens to generate.
            verbose: If True, show progress bar during prediction.

        Returns:
            List of predicted values
        """
        results: list[Any] = []

        num_batches = (len(objects) + batch_size - 1) // batch_size
        batch_iter = range(0, len(objects), batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, total=num_batches, desc="Predicting")

        for start in batch_iter:
            batch_objects = objects[start : start + batch_size]

            # 1. Reorder to place target key last (maximum context)
            reordered = [move_target_last(obj, target_key) for obj in batch_objects]

            # 2. Create batch using collator
            batch = self._collator.collate_objects(reordered, shuffle=False)
            batch = batch.to(self.device)

            # 3. Truncate at target key (exclude value)
            truncated = self._truncate_at_target_key(batch, target_key)

            # 4. Generate using Generator
            values = self._generator.generate_from_batch(
                truncated,
                stop_after_value=True,
                max_tokens=200 if allow_complex_values else 1,
                temperature=0.0,  # Greedy for deterministic predictions
                allow_complex_values=allow_complex_values,
            )

            # 5. Inverse transform if configured
            if self._inverse_transform is not None:
                values = [self._inverse_transform(v, target_key) for v in values]

            results.extend(values)

        return results

    def predict_proba(
        self,
        obj: dict,
        target_key: str,
        values: list[Any] | None = None,
        top_k: int | None = None,
        allow_complex_values: bool = False,
    ) -> dict[Any, float] | list[tuple[Any, float]]:
        """Get probability distribution for a single object.

        Args:
            obj: JSON object
            target_key: Key to predict
            values: If provided, only return probabilities for these values
            top_k: If set, return only top-k values
            allow_complex_values: If False (default), exclude OBJ_START/ARRAY_START
                from the probability distribution.

        Returns:
            If top_k=None: dict mapping values to probabilities
            If top_k set: list of (value, prob) tuples sorted by prob
        """
        return self.predict_proba_batch(
            [obj], target_key, values=values, top_k=top_k, allow_complex_values=allow_complex_values
        )[0]

    @torch.no_grad()
    def predict_proba_batch(
        self,
        objects: list[dict],
        target_key: str,
        values: list[Any] | None = None,
        top_k: int | None = None,
        batch_size: int = 32,
        allow_complex_values: bool = False,
    ) -> list[dict[Any, float] | list[tuple[Any, float]]]:
        """Get probability distributions for a batch of objects.

        Handles batching internally for large object lists.

        Args:
            objects: List of JSON objects
            target_key: Key to predict (same for all objects)
            values: If provided, only return probabilities for these values
            top_k: If set, return only top-k values per object
            batch_size: Number of objects to process in parallel
            allow_complex_values: If False (default), exclude OBJ_START/ARRAY_START
                from the probability distribution.

        Returns:
            List of distributions (one per object).
            Each is dict or list[tuple] depending on top_k.
        """
        all_results: list[dict[Any, float] | list[tuple[Any, float]]] = []

        for start in range(0, len(objects), batch_size):
            batch_objects = objects[start : start + batch_size]

            # 1. Prepare batch
            reordered = [move_target_last(obj, target_key) for obj in batch_objects]
            batch = self._collator.collate_objects(reordered, shuffle=False)
            batch = batch.to(self.device)
            truncated = self._truncate_at_target_key(batch, target_key)

            # 2. Get distributions from Generator
            probs, _continuous_params = self._generator.get_next_token_distribution(
                truncated, allow_complex_values=allow_complex_values
            )

            # 3. Map token probabilities to values for each item
            for i in range(len(batch_objects)):
                result = self._map_probs_to_values(probs[i], values)

                if top_k is not None:
                    sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
                    all_results.append(sorted_items[:top_k])
                else:
                    all_results.append(result)

        return all_results

    def _map_probs_to_values(
        self,
        probs: Tensor,
        specific_values: list[Any] | None = None,
    ) -> dict[Any, float]:
        """Map token probabilities to Python values.

        Args:
            probs: (vocab_size,) probabilities for a single sequence
            specific_values: If provided, only return probs for these values

        Returns:
            Dict mapping values to probabilities
        """
        vocab = self.tokenizer.vocab

        if specific_values is not None:
            # Get probabilities for specific values
            result: dict[Any, float] = {}
            for value in specific_values:
                token = ValueToken(value)
                try:
                    token_id = vocab.encode(token)
                    if token_id == vocab.unk_value_id:
                        result[value] = 0.0
                    else:
                        result[value] = probs[token_id].item()
                except KeyError:
                    result[value] = 0.0
            return result

        # Build distribution over all primitive values
        result = {}
        value_ids = vocab.get_all_primitive_value_ids()
        for token_id in value_ids:
            prob = probs[token_id].item()
            if prob > 1e-6:
                token = vocab.decode(token_id)
                if isinstance(token, ValueToken):
                    result[token.value] = prob
                elif token_id == vocab.num_token_id:
                    # NUM token - include as special marker
                    result["<NUM>"] = prob

        return result

    def _truncate_at_target_key(
        self,
        batch: "EncodedBatch",
        target_key: str,
    ) -> "EncodedBatch":
        """Truncate sequences to end at the target key (excluding its value).

        Args:
            batch: Encoded batch with full sequences
            target_key: Key to find (leaf key if nested)

        Returns:
            New EncodedBatch truncated to end at target key positions
        """
        from origami.tokenizer.json_tokenizer import EncodedBatch

        target_positions = find_target_positions(batch.input_ids, target_key, self.tokenizer.vocab)
        batch_size = batch.input_ids.size(0)

        # Find max length needed (target_pos + 1 for each sequence)
        # target_positions are absolute positions in the left-padded sequence
        max_len = (target_positions + 1).max().item()

        # For left-padded sequences, we need to slice from the right
        # If a sequence is [PAD, PAD, START, key1, val1, key2] with target at pos 5,
        # we want to keep up to and including pos 5, so [:6]
        # But we also need to handle the case where different sequences have
        # different target positions

        # Create new tensors with the truncated length
        new_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        new_path_types = torch.zeros(
            batch_size, max_len, batch.path_types.size(2), dtype=torch.long, device=self.device
        )
        new_path_ids = torch.zeros(
            batch_size, max_len, batch.path_ids.size(2), dtype=torch.long, device=self.device
        )
        new_path_lengths = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        new_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        new_numeric_values = torch.zeros(batch_size, max_len, dtype=torch.float, device=self.device)
        new_numeric_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        new_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            target_pos = target_positions[i].item()
            # Number of tokens to copy (from start up to and including target key)
            num_tokens = target_pos + 1

            # For left-padded sequences, tokens are at the end
            # We copy the last num_tokens from the source to the last num_tokens of dest
            # But we want right-alignment in the new tensor too

            # Source: copy from position 0 to target_pos+1
            # Dest: place at end (right-aligned)
            dest_start = max_len - num_tokens
            new_input_ids[i, dest_start:] = batch.input_ids[i, : target_pos + 1]
            new_path_types[i, dest_start:] = batch.path_types[i, : target_pos + 1]
            new_path_ids[i, dest_start:] = batch.path_ids[i, : target_pos + 1]
            new_path_lengths[i, dest_start:] = batch.path_lengths[i, : target_pos + 1]
            new_attention_mask[i, dest_start:] = batch.attention_mask[i, : target_pos + 1]
            new_numeric_values[i, dest_start:] = batch.numeric_values[i, : target_pos + 1]
            new_numeric_mask[i, dest_start:] = batch.numeric_mask[i, : target_pos + 1]
            new_lengths[i] = num_tokens

        return EncodedBatch(
            input_ids=new_input_ids,
            path_types=new_path_types,
            path_ids=new_path_ids,
            path_lengths=new_path_lengths,
            attention_mask=new_attention_mask,
            numeric_values=new_numeric_values,
            numeric_mask=new_numeric_mask,
            lengths=new_lengths,
            labels=None,  # No labels for inference
        )
