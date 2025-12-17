"""ORIGAMI value predictor.

Predicts values for target keys in JSON documents using a trained ORIGAMI model.
"""

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from origami.preprocessing import move_target_last
from origami.tokenizer.vocabulary import KeyToken, ValueToken

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import EncodedBatch, JSONTokenizer

from .generator import OrigamiGenerator


class OrigamiPredictor:
    """Predict values for target keys using a trained ORIGAMI model.

    The predictor uses the Generator for ALL value prediction, ensuring
    grammar constraints are applied consistently.

    For primitive values, the Generator samples from the grammar-constrained
    distribution. For complex values (objects, arrays), the Generator
    continues until the value is complete.

    Example:
        ```python
        predictor = OrigamiPredictor(model, tokenizer)

        # Predict single value
        obj = {"name": "Alice", "age": 30, "city": None}  # city is target
        prediction = predictor.predict(obj, target_key="city")
        # Returns: "NYC"

        # Batch prediction
        predictions = predictor.predict_batch([obj1, obj2, obj3], target_key="city")
        # Returns: ["NYC", "LA", "SF"]

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
    ):
        """Initialize predictor.

        Args:
            model: Trained ORIGAMI model
            tokenizer: JSONTokenizer with fitted vocabulary

        Note:
            Predictor always runs on CPU as benchmarking shows it's faster
            for the model sizes typically used with ORIGAMI.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Create generator for value generation (handles grammar + continuous values)
        self._generator = OrigamiGenerator(model, tokenizer)

    @torch.no_grad()
    def predict(
        self,
        obj: dict,
        target_key: str,
    ) -> Any:
        """Predict value for a target key.

        Args:
            obj: JSON object. The target_key's current value is ignored.
            target_key: Key to predict (dot notation for nested keys)

        Returns:
            The predicted value

        Raises:
            KeyError: If target_key doesn't exist in obj
        """
        results = self.predict_batch([obj], target_key)
        return results[0]

    @torch.no_grad()
    def predict_batch(
        self,
        objects: list[dict],
        target_key: str,
    ) -> list[Any]:
        """Predict values for a batch of objects.

        Args:
            objects: List of JSON objects
            target_key: Key to predict (same for all objects)

        Returns:
            List of predicted values (one per object)
        """
        # 1. Reorder objects to place target key last for maximum context
        reordered = [move_target_last(obj, target_key) for obj in objects]

        # 2. Tokenize (without shuffle for deterministic results)
        batch = self.tokenizer.encode_batch(reordered, shuffle=False)
        batch = batch.to(self.device)

        # 3. Truncate sequences to end at target key (exclude the value)
        truncated = self._truncate_at_target_key(batch, target_key)

        # 4. Generate value using Generator (handles grammar + continuous values)
        # Use temperature=0 for greedy decoding (deterministic predictions)
        values = self._generator.generate_from_tensors(
            input_ids=truncated.input_ids,
            path_types=truncated.path_types,
            path_ids=truncated.path_ids,
            path_lengths=truncated.path_lengths,
            attention_mask=truncated.attention_mask,
            numeric_values=truncated.numeric_values,  # Pass numeric context for conditioning
            stop_after_value=True,
            max_tokens=100,
            temperature=0.0,  # Greedy decoding for deterministic predictions
        )

        return values

    def predict_proba(
        self,
        obj: dict,
        target_key: str,
        values: list[Any] | None = None,
        top_k: int | None = None,
    ) -> dict[Any, float] | list[tuple[Any, float]]:
        """Get probability distribution over possible values.

        Uses the Generator's get_value_distribution() to get grammar-constrained
        probabilities.

        Args:
            obj: JSON object
            target_key: Key to predict
            values: Specific values to get probabilities for
            top_k: If specified, return only top-k values sorted by probability

        Returns:
            If top_k is None: dict mapping values to probabilities
            If top_k is set: list of (value, prob) tuples, sorted desc by probability
        """
        # 1. Prepare tensors (same as predict_batch)
        reordered = move_target_last(obj, target_key)
        batch = self.tokenizer.encode_batch([reordered], shuffle=False)
        batch = batch.to(self.device)
        truncated = self._truncate_at_target_key(batch, target_key)

        # 2. Get distribution from Generator (grammar-constrained!)
        probs, _continuous_params = self._generator.get_value_distribution(
            input_ids=truncated.input_ids,
            path_types=truncated.path_types,
            path_ids=truncated.path_ids,
            path_lengths=truncated.path_lengths,
            attention_mask=truncated.attention_mask,
            numeric_values=truncated.numeric_values,  # Pass numeric context for conditioning
        )

        # 3. Map token probabilities to values
        vocab = self.tokenizer.vocab
        if values is not None:
            # Get probabilities for specific values
            result = {}
            for value in values:
                token = ValueToken(value)
                try:
                    token_id = vocab.encode(token)
                    if token_id == vocab.unk_value_id:
                        result[value] = 0.0
                    else:
                        result[value] = probs[0, token_id].item()
                except KeyError:
                    result[value] = 0.0
            return result

        # 4. Build distribution over all values
        result = {}
        value_ids = vocab.get_all_primitive_value_ids()
        for token_id in value_ids:
            prob = probs[0, token_id].item()
            if prob > 1e-6:
                token = vocab.decode(token_id)
                if isinstance(token, ValueToken):
                    result[token.value] = prob
                elif token_id == vocab.num_token_id:
                    # NUM token - include as special marker
                    result["<NUM>"] = prob

        # 5. Return top_k if requested
        if top_k is not None:
            sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:top_k]

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

        target_positions = self._find_target_positions(batch.input_ids, target_key)
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
        )

    def _find_target_positions(
        self,
        input_ids: Tensor,
        target_key: str,
    ) -> Tensor:
        """Find position of target key token in each sequence.

        Args:
            input_ids: (batch, seq_len) token IDs
            target_key: The target key to find (leaf key if nested)

        Returns:
            Tensor of shape (batch,) with position indices
        """
        # Get the leaf key (last part of dot-separated path)
        leaf_key = target_key.split(".")[-1]

        # Get the token ID for this key
        key_token = KeyToken(leaf_key)
        key_id = self.tokenizer.vocab.encode(key_token)

        batch_size = input_ids.size(0)
        target_positions = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

        for i in range(batch_size):
            # Find all positions where this key appears
            matches = (input_ids[i] == key_id).nonzero(as_tuple=True)[0]
            if len(matches) == 0:
                raise ValueError(f"Target key '{leaf_key}' not found in sequence {i}")
            # Use the last occurrence (after move_target_last, target is last)
            target_positions[i] = matches[-1]

        return target_positions
