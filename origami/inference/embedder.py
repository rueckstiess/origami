"""ORIGAMI document embedder.

Extracts dense vector embeddings from JSON documents using a trained ORIGAMI model.
"""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from origami.preprocessing import move_target_last

from .utils import find_target_positions

if TYPE_CHECKING:
    from origami.model.origami_model import OrigamiModel
    from origami.tokenizer.json_tokenizer import JSONTokenizer

PoolingStrategy = Literal["mean", "max", "last", "target"]


class OrigamiEmbedder:
    """Extract embeddings from JSON documents using a trained ORIGAMI model.

    Supports multiple pooling strategies for aggregating hidden states:
    - mean: Average over non-padding positions
    - max: Max-pool over non-padding positions
    - last: Last non-padding token's hidden state
    - target: Hidden state at target key position (before seeing value)

    The 'target' pooling is particularly useful for downstream prediction tasks,
    as it captures the model's representation state just before predicting
    a specific field.

    Example:
        ```python
        embedder = OrigamiEmbedder(model, tokenizer, pooling="mean")

        # Embed single document
        embedding = embedder.embed({"name": "Alice", "age": 30})

        # Embed batch with internal batching
        embeddings = embedder.embed_batch(objects, batch_size=64)

        # Target-specific embedding for classification
        embedder_target = OrigamiEmbedder(model, tokenizer, pooling="target")
        embedding = embedder_target.embed(
            {"name": "Alice", "age": 30, "city": "NYC"},
            target_key="city"
        )
        ```
    """

    def __init__(
        self,
        model: "OrigamiModel",
        tokenizer: "JSONTokenizer",
        pooling: PoolingStrategy = "mean",
    ):
        """Initialize embedder.

        Args:
            model: Trained ORIGAMI model
            tokenizer: JSONTokenizer with fitted vocabulary
            pooling: Pooling strategy for aggregating hidden states

        Note:
            The Embedder uses the model's current device dynamically.
            Move the model to your desired device before calling embed().
        """
        from origami.training.collator import OrigamiDataCollator

        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.model.eval()

        # Create collator for batch creation (include_labels=False for inference)
        self._collator = OrigamiDataCollator(tokenizer, include_labels=False)

    @property
    def device(self) -> torch.device:
        """Get the model's current device dynamically."""
        return next(self.model.parameters()).device

    def embed(
        self,
        obj: dict,
        target_key: str | None = None,
        normalize: bool = True,
        enable_grad: bool = False,
    ) -> Tensor:
        """Embed a single JSON object.

        Args:
            obj: JSON object to embed
            target_key: Required if pooling="target". Dot-separated key path.
            normalize: Whether to L2-normalize the embedding
            enable_grad: If True, compute gradients for fine-tuning.
                If False (default), use torch.no_grad() for inference.

        Returns:
            Embedding tensor of shape (d_model,)

        Raises:
            ValueError: If pooling="target" but target_key not provided
        """
        embeddings = self.embed_batch(
            [obj], target_key=target_key, normalize=normalize, enable_grad=enable_grad
        )
        return embeddings[0]

    def embed_batch(
        self,
        objects: list[dict],
        target_key: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        enable_grad: bool = False,
    ) -> Tensor:
        """Embed multiple JSON objects.

        Handles batching internally for large object lists.

        Args:
            objects: List of JSON objects to embed
            target_key: Required if pooling="target". Dot-separated key path.
                        All objects in batch use the same target_key.
            normalize: Whether to L2-normalize embeddings
            batch_size: Number of objects to process in parallel
            enable_grad: If True, compute gradients for fine-tuning.
                If False (default), use torch.no_grad() for inference.

        Returns:
            Embedding tensor of shape (len(objects), d_model)

        Raises:
            ValueError: If pooling="target" but target_key not provided
        """
        if self.pooling == "target" and target_key is None:
            raise ValueError("target_key is required when pooling='target'")

        # Use no_grad for inference, nullcontext for training with gradients
        grad_context = nullcontext() if enable_grad else torch.no_grad()

        all_embeddings = []

        with grad_context:
            for start in range(0, len(objects), batch_size):
                batch_objects = objects[start : start + batch_size]

                # Reorder if target pooling
                if self.pooling == "target":
                    batch_objects = [move_target_last(obj, target_key) for obj in batch_objects]

                # Create batch using collator
                batch = self._collator.collate_objects(batch_objects, shuffle=False)
                batch = batch.to(self.device)

                # Forward pass
                output = self.model(
                    input_ids=batch.input_ids,
                    path_types=batch.path_types,
                    path_ids=batch.path_ids,
                    path_lengths=batch.path_lengths,
                    attention_mask=batch.attention_mask,
                    numeric_values=batch.numeric_values,
                )

                hidden_states = output.hidden_states  # (batch, seq_len, d_model)

                # Apply pooling strategy
                if self.pooling == "mean":
                    embeddings = self._mean_pool(hidden_states, batch.attention_mask)
                elif self.pooling == "max":
                    embeddings = self._max_pool(hidden_states, batch.attention_mask)
                elif self.pooling == "last":
                    embeddings = self._last_pool(hidden_states, batch.attention_mask)
                elif self.pooling == "target":
                    embeddings = self._target_pool(
                        hidden_states,
                        batch.input_ids,
                        target_key,  # type: ignore
                    )
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")

                # Normalize if requested
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _mean_pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean pooling over non-padding positions.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) - True for valid positions

        Returns:
            Pooled embeddings of shape (batch, d_model)
        """
        # Expand mask for broadcasting
        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Mask out padding positions (set to 0) to avoid NaN * 0 = NaN
        # This is needed because hidden_states may contain NaN at padding positions
        masked_hidden = hidden_states.masked_fill(~mask, 0.0)

        # Sum over valid positions
        summed = masked_hidden.sum(dim=1)  # (batch, d_model)

        # Divide by count of valid positions
        counts = mask.float().sum(dim=1).clamp(min=1)  # (batch, 1)

        return summed / counts

    def _max_pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Max pooling over non-padding positions.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) - True for valid positions

        Returns:
            Pooled embeddings of shape (batch, d_model)
        """
        # Set padding positions to very negative value
        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        masked = hidden_states.masked_fill(~mask, float("-inf"))

        # Max over sequence dimension
        return masked.max(dim=1).values  # (batch, d_model)

    def _last_pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Extract last non-padding token's hidden state.

        Args:
            hidden_states: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len) - True for valid positions

        Returns:
            Last token embeddings of shape (batch, d_model)
        """
        # Find last valid position for each sequence
        # attention_mask is True for valid positions
        lengths = attention_mask.sum(dim=1)  # (batch,)
        last_indices = (lengths - 1).clamp(min=0)  # (batch,)

        # Gather last hidden states
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)

        return hidden_states[batch_indices, last_indices]  # (batch, d_model)

    def _target_pool(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        target_key: str,
    ) -> Tensor:
        """Extract hidden state at target key position.

        The target key's hidden state captures the model's "prediction state"
        just before it sees the value, containing all context from previous tokens.

        Args:
            hidden_states: (batch, seq_len, d_model)
            input_ids: (batch, seq_len) - token IDs
            target_key: The target key to find (leaf key if nested)

        Returns:
            Target position embeddings of shape (batch, d_model)
        """
        target_positions = find_target_positions(input_ids, target_key, self.tokenizer.vocab)

        # Gather hidden states at target positions
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[batch_indices, target_positions]  # (batch, d_model)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.model.config.d_model
