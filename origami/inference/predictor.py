"""ORIGAMI value predictor.

Predicts values for target keys in JSON documents using a trained ORIGAMI model.
"""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from origami.preprocessing import move_target_last
from origami.tokenizer.vocabulary import KeyToken, ValueToken

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import EncodedBatch, JSONTokenizer


class OrigamiPredictor:
    """Predict values for target keys using a trained ORIGAMI model.

    The predictor uses the model's learned distribution to predict the most
    likely value for a specified key, given the rest of the document as context.

    Example:
        ```python
        predictor = OrigamiPredictor(model, tokenizer)

        # Predict single value
        obj = {"name": "Alice", "age": 30, "city": None}  # city is target
        prediction = predictor.predict(obj, target_key="city")
        # Returns: "NYC"

        # Get top-k predictions with probabilities
        predictions = predictor.predict(obj, target_key="city", top_k=3, return_probs=True)
        # Returns: [("NYC", 0.45), ("LA", 0.32), ("SF", 0.18)]

        # Batch prediction
        predictions = predictor.predict_batch([obj1, obj2, obj3], target_key="city")
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

    @torch.no_grad()
    def predict(
        self,
        obj: dict,
        target_key: str,
        top_k: int = 1,
        return_probs: bool = False,
    ) -> Any | list[tuple[Any, float]]:
        """Predict value for a target key.

        Args:
            obj: JSON object. The target_key's current value is ignored.
            target_key: Key to predict (dot notation for nested keys)
            top_k: Number of predictions to return
            return_probs: Whether to include probabilities in output

        Returns:
            If top_k=1 and return_probs=False: single predicted value
            If top_k=1 and return_probs=True: (value, probability) tuple
            If top_k>1: list of (value, probability) tuples

        Raises:
            KeyError: If target_key doesn't exist in obj
        """
        results = self.predict_batch([obj], target_key, top_k=top_k)

        if top_k == 1 and not return_probs:
            return results[0][0][0]  # Just the value
        elif top_k == 1 and return_probs:
            return results[0][0]  # (value, prob) tuple
        else:
            return results[0]  # List of (value, prob) tuples

    @torch.no_grad()
    def predict_batch(
        self,
        objects: list[dict],
        target_key: str,
        top_k: int = 1,
    ) -> list[list[tuple[Any, float]]]:
        """Predict values for a batch of objects.

        Args:
            objects: List of JSON objects
            target_key: Key to predict (same for all objects)
            top_k: Number of predictions per object

        Returns:
            List of prediction lists. Each inner list contains (value, prob) tuples.
        """
        # Reorder objects to place target key last for maximum context
        reordered = [move_target_last(obj, target_key) for obj in objects]

        # Tokenize (without shuffle for deterministic results)
        batch = self.tokenizer.encode_batch(reordered, shuffle=False)
        batch = batch.to(self.device)

        # Forward pass
        output = self.model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )

        # Find target key positions for each sequence
        target_positions = self._find_target_positions(batch.input_ids, target_key)

        # Get logits at target positions (these predict the value token)
        # target_positions point to the key token; the next position predicts the value
        batch_size = batch.input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=self.device)

        # The logits at position i predict token at position i+1
        # So logits at target_key position predict the value
        target_logits = output.logits[batch_indices, target_positions]  # (batch, vocab_size)

        # Convert to probabilities
        probs = F.softmax(target_logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        # Decode predictions
        vocab = self.tokenizer.vocab
        results = []
        for i in range(batch_size):
            predictions = []
            for j in range(top_k):
                token_id = top_indices[i, j].item()
                prob = top_probs[i, j].item()
                token = vocab.decode(token_id)

                # Extract the actual value from the token
                if isinstance(token, ValueToken):
                    value = token.value
                elif token_id == vocab.obj_start_id or token_id == vocab.array_start_id:
                    # Complex value - need to generate the rest
                    value = self._generate_complex_value(
                        batch, i, target_positions[i].item(), token_id
                    )
                else:
                    # Grammar tokens or key tokens - shouldn't happen with proper training
                    # but handle gracefully
                    value = None

                predictions.append((value, prob))
            results.append(predictions)

        return results

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

    def _generate_complex_value(
        self,
        batch: "EncodedBatch",
        batch_idx: int,
        target_pos: int,
        start_token_id: int,
    ) -> Any:
        """Generate a complex value (object or array) using the Generator.

        Args:
            batch: The encoded batch
            batch_idx: Index of the sequence in the batch
            target_pos: Position of the target key in the sequence
            start_token_id: The start token (OBJ_START or ARRAY_START)

        Returns:
            The generated complex value (dict or list)
        """
        from .generator import OrigamiGenerator

        # Create generator (lazy initialization)
        # Note: Generator always runs on CPU for performance
        if not hasattr(self, "_generator"):
            self._generator = OrigamiGenerator(self.model, self.tokenizer)

        # Extract the sequence up to and including the target key position
        # Then add the start token
        seq_len = target_pos + 1
        input_ids = batch.input_ids[batch_idx : batch_idx + 1, :seq_len]
        path_types = batch.path_types[batch_idx : batch_idx + 1, :seq_len]
        path_ids = batch.path_ids[batch_idx : batch_idx + 1, :seq_len]
        path_lengths = batch.path_lengths[batch_idx : batch_idx + 1, :seq_len]

        # Append the start token
        start_token = torch.tensor(
            [[start_token_id]], dtype=torch.long, device=self.device
        )
        input_ids = torch.cat([input_ids, start_token], dim=1)

        # Append path for the start token (same as target key's path)
        new_path_types = path_types[:, -1:, :]
        new_path_ids = path_ids[:, -1:, :]
        new_path_lengths = path_lengths[:, -1:]
        path_types = torch.cat([path_types, new_path_types], dim=1)
        path_ids = torch.cat([path_ids, new_path_ids], dim=1)
        path_lengths = torch.cat([path_lengths, new_path_lengths], dim=1)

        # Initialize path state from the token sequence
        path_states = self._generator._init_path_states_from_tokens(
            input_ids[0].tolist(), num_samples=1
        )
        path_state = path_states[0]

        # Generate the rest of the complex value
        _, value = self._generator.generate_value(
            input_ids,
            path_types,
            path_ids,
            path_lengths,
            path_state,
            max_tokens=100,
        )

        return value

    def predict_proba(
        self,
        obj: dict,
        target_key: str,
        values: list[Any] | None = None,
    ) -> dict[Any, float]:
        """Get probability distribution over possible values.

        Args:
            obj: JSON object
            target_key: Key to predict
            values: Specific values to get probabilities for.
                   If None, returns all values with non-zero probability.

        Returns:
            Dictionary mapping values to their probabilities
        """
        # Reorder and encode
        reordered = move_target_last(obj, target_key)
        batch = self.tokenizer.encode_batch([reordered], shuffle=False)
        batch = batch.to(self.device)

        # Forward pass
        output = self.model(
            input_ids=batch.input_ids,
            path_types=batch.path_types,
            path_ids=batch.path_ids,
            path_lengths=batch.path_lengths,
            attention_mask=batch.attention_mask,
        )

        # Find target position
        target_pos = self._find_target_positions(batch.input_ids, target_key)
        target_logits = output.logits[0, target_pos[0]]  # (vocab_size,)

        # Convert to probabilities
        probs = F.softmax(target_logits, dim=-1)

        if values is not None:
            # Get probabilities for specific values
            vocab = self.tokenizer.vocab
            result = {}
            for value in values:
                token = ValueToken(value)
                try:
                    token_id = vocab.encode(token)
                    # If value maps to UNK_VALUE, it's unknown - return 0
                    if token_id == vocab.unk_value_id:
                        result[value] = 0.0
                    else:
                        result[value] = probs[token_id].item()
                except KeyError:
                    # Unknown value (if vocab not frozen)
                    result[value] = 0.0
            return result
        else:
            # Return all values with meaningful probability
            result = {}
            value_ids = self.tokenizer.vocab.get_all_primitive_value_ids()
            for token_id in value_ids:
                prob = probs[token_id].item()
                if prob > 1e-6:  # Filter very small probabilities
                    token = self.tokenizer.vocab.decode(token_id)
                    if isinstance(token, ValueToken):
                        result[token.value] = prob
            return result
